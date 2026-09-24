"""In-memory snapshot of the indexed corpus.

Both the BM25 index and the dense matrix are built from this snapshot, so the
two retrievers always see exactly the same set of chunks (which is what makes
the BM25 vs dense vs hybrid comparison fair).

The snapshot is versioned by a monotonically-increasing revision so that the
retrieval service can rebuild lazily after an ingest/re-index, rather than
querying the DB per request.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.models import STATUS_ACTIVE, Chunk, Document, DocumentVersion
from ingestion.embed import unpack


@dataclass
class IndexedChunk:
    chunk_id: str
    text: str
    status: str
    chunk_type: str
    section_ref: str | None
    doc_number: str
    doc_title: str
    source_url: str | None
    version_number: int
    published_date: str | None


class CorpusSnapshot:
    def __init__(self, chunks: list[IndexedChunk], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings
        self.id_to_pos = {c.chunk_id: i for i, c in enumerate(chunks)}

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def active_positions(self) -> list[int]:
        return [i for i, c in enumerate(self.chunks) if c.status == STATUS_ACTIVE]


def load_snapshot(session: Session) -> CorpusSnapshot:
    rows = session.execute(
        select(Chunk, DocumentVersion, Document)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .order_by(Document.doc_number, DocumentVersion.version_number, Chunk.ordinal)
    ).all()

    chunks: list[IndexedChunk] = []
    vectors: list[np.ndarray] = []
    dim = 0
    for chunk, version, document in rows:
        chunks.append(
            IndexedChunk(
                chunk_id=chunk.id,
                text=chunk.chunk_text,
                status=chunk.status,
                chunk_type=chunk.chunk_type,
                section_ref=chunk.section_ref,
                doc_number=document.doc_number,
                doc_title=document.title,
                source_url=document.source_url,
                version_number=version.version_number,
                published_date=version.published_date.isoformat() if version.published_date else None,
            )
        )
        vec = unpack(chunk.embedding) if chunk.embedding else np.zeros(0, dtype=np.float32)
        dim = max(dim, len(vec))
        vectors.append(vec)

    if chunks:
        matrix = np.zeros((len(chunks), dim or 1), dtype=np.float32)
        for i, vec in enumerate(vectors):
            if len(vec) == matrix.shape[1]:
                matrix[i] = vec
            elif len(vec):  # dimension drift -> ignore, reembed_all fixes it
                n = min(len(vec), matrix.shape[1])
                matrix[i, :n] = vec[:n]
    else:
        matrix = np.zeros((0, 1), dtype=np.float32)

    return CorpusSnapshot(chunks, matrix)
