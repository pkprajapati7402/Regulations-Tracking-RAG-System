"""Retrieval service: one object that owns the indexes and serves queries.

Supports the four configurations the project compares:

    bm25 | dense | hybrid | hybrid_rerank

and the audit switch ``include_deprecated``. By default *deprecated* chunks
(those belonging to a superseded document version) are excluded from
retrieval - that is what makes the answer reflect the current rule after an
amendment - but they remain retrievable on request for audit purposes.

Indexes are rebuilt lazily: ingesting a document bumps a revision counter and
the next query rebuilds the snapshot. Rebuilds are guarded by a lock so
concurrent API requests never see a half-built index.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from sqlalchemy.orm import Session

from core.config import settings
from core.db import session_scope
from core.logging_conf import get_logger
from core.models import STATUS_ACTIVE
from retrieval.bm25 import BM25Index
from retrieval.corpus_index import CorpusSnapshot, load_snapshot
from retrieval.dense import DenseIndex
from retrieval.hybrid_rrf import reciprocal_rank_fusion
from retrieval.rerank import get_reranker
from retrieval.types import RetrievedChunk

log = get_logger(__name__)

VALID_MODES = {"bm25", "dense", "hybrid", "hybrid_rerank"}


@dataclass
class RetrievalDebug:
    mode: str
    candidates: int
    latency_ms: float
    index_size: int
    include_deprecated: bool

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "candidates": self.candidates,
            "latency_ms": round(self.latency_ms, 2),
            "index_size": self.index_size,
            "include_deprecated": self.include_deprecated,
        }


class RetrievalService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot: CorpusSnapshot | None = None
        self._bm25: BM25Index | None = None
        self._dense: DenseIndex | None = None
        self._dirty = True

    # ---------------------------------------------------------------- index
    def mark_dirty(self) -> None:
        with self._lock:
            self._dirty = True

    def ensure_index(self, session: Session | None = None) -> None:
        with self._lock:
            if not self._dirty and self._snapshot is not None:
                return
            start = time.perf_counter()
            if session is not None:
                snapshot = load_snapshot(session)
            else:
                with session_scope() as s:
                    snapshot = load_snapshot(s)
            self._snapshot = snapshot
            self._bm25 = BM25Index(snapshot)
            self._dense = DenseIndex(snapshot)
            self._dirty = False
            log.info(
                "Rebuilt retrieval index: %d chunks (%d active) in %.0f ms",
                len(snapshot),
                len(snapshot.active_positions),
                (time.perf_counter() - start) * 1000,
            )

    @property
    def size(self) -> int:
        return len(self._snapshot) if self._snapshot else 0

    def stats(self) -> dict:
        self.ensure_index()
        assert self._snapshot is not None
        docs = {c.doc_number for c in self._snapshot.chunks}
        active = len(self._snapshot.active_positions)
        return {
            "documents": len(docs),
            "chunks_total": len(self._snapshot),
            "chunks_active": active,
            "chunks_deprecated": len(self._snapshot) - active,
            "tables_indexed": sum(1 for c in self._snapshot.chunks if c.chunk_type == "table"),
        }

    # -------------------------------------------------------------- queries
    def search(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        candidate_k: int | None = None,
        include_deprecated: bool = False,
        session: Session | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        mode = (mode or settings.retrieval_mode).lower()
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Options: {sorted(VALID_MODES)}")
        top_k = top_k or settings.top_k
        candidate_k = candidate_k or max(settings.candidate_k, top_k)

        self.ensure_index(session)
        assert self._snapshot is not None and self._bm25 and self._dense
        started = time.perf_counter()

        allowed: set[int] | None = None
        if not include_deprecated:
            allowed = {
                i for i, c in enumerate(self._snapshot.chunks) if c.status == STATUS_ACTIVE
            }

        if not query.strip() or not self._snapshot.chunks:
            return [], RetrievalDebug(mode, 0, 0.0, len(self._snapshot), include_deprecated)

        if mode == "bm25":
            ranked = self._bm25.search(query, top_k, allowed)
            hits = [(i, s, {"bm25_score": s}) for i, s in ranked]
        elif mode == "dense":
            ranked = self._dense.search(query, top_k, allowed)
            hits = [(i, s, {"dense_score": s}) for i, s in ranked]
        else:
            rankings = {
                "bm25": self._bm25.search(query, candidate_k, allowed),
                "dense": self._dense.search(query, candidate_k, allowed),
            }
            fused = reciprocal_rank_fusion(rankings, k=settings.rrf_k, top_k=candidate_k)
            if mode == "hybrid":
                hits = fused[:top_k]
            else:
                reranker = get_reranker()
                passages = [self._snapshot.chunks[i].text for i, _, _ in fused]
                reranked = reranker.rerank(query, passages, top_k)
                hits = []
                for pos, score in reranked:
                    idx, rrf_score, comps = fused[pos]
                    comps = dict(comps)
                    comps["rrf_score"] = rrf_score
                    comps["rerank_score"] = score
                    hits.append((idx, score, comps))

        results = [self._to_result(i, score, comps, mode) for i, score, comps in hits]
        debug = RetrievalDebug(
            mode=mode,
            candidates=len(hits),
            latency_ms=(time.perf_counter() - started) * 1000,
            index_size=len(self._snapshot),
            include_deprecated=include_deprecated,
        )
        return results, debug

    def _to_result(self, idx: int, score: float, comps: dict, mode: str) -> RetrievedChunk:
        assert self._snapshot is not None
        c = self._snapshot.chunks[idx]
        return RetrievedChunk(
            chunk_id=c.chunk_id,
            text=c.text,
            score=score,
            doc_number=c.doc_number,
            doc_title=c.doc_title,
            section_ref=c.section_ref,
            chunk_type=c.chunk_type,
            status=c.status,
            version_number=c.version_number,
            published_date=c.published_date,
            source_url=c.source_url,
            retriever=mode,
            components=comps,
        )


_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global _service
    if _service is None:
        _service = RetrievalService()
    return _service


def reset_retrieval_service() -> None:
    """Used by tests and by CLI runs that rebuild the DB from scratch."""
    global _service
    _service = None
