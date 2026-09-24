"""Cross-encoder reranking of the fused candidate list.

Two providers, same interface:

* ``cross-encoder`` - ``BAAI/bge-reranker-base`` via sentence-transformers
  (optional install). This is the configuration reported in the graded runs.
* ``lexical`` - a dependency-free fallback that scores each candidate by a
  blend of query-term coverage, IDF-weighted overlap and phrase proximity.
  It is weaker than a real cross-encoder but measurably better than no
  reranking at all, and it keeps the system runnable with no model downloads.
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from core.config import settings
from core.logging_conf import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BaseReranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]: ...

    def rerank(self, query: str, passages: list[str], top_k: int) -> list[tuple[int, float]]:
        if not passages:
            return []
        scores = self.score(query, passages)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class LexicalReranker(BaseReranker):
    name = "lexical"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q_tokens = _tokens(query)
        q_set = set(q_tokens)
        if not q_set:
            return [0.0] * len(passages)

        # document frequency over the candidate set -> cheap IDF
        df: dict[str, int] = {}
        tokenised = [_tokens(p) for p in passages]
        for toks in tokenised:
            for term in set(toks) & q_set:
                df[term] = df.get(term, 0) + 1
        n = len(passages)

        scores: list[float] = []
        for toks in tokenised:
            tok_set = set(toks)
            covered = q_set & tok_set
            if not covered:
                scores.append(0.0)
                continue
            idf_sum = sum(math.log(1 + n / (1 + df.get(t, 0))) for t in covered)
            coverage = len(covered) / len(q_set)
            # bigram/proximity bonus
            bigrams = {f"{a} {b}" for a, b in zip(toks, toks[1:])}
            q_bigrams = {f"{a} {b}" for a, b in zip(q_tokens, q_tokens[1:])}
            phrase = len(bigrams & q_bigrams) / max(len(q_bigrams), 1)
            length_penalty = 1.0 / (1.0 + math.log(1 + len(toks) / 200))
            scores.append((0.6 * coverage + 0.25 * phrase + 0.15 * min(idf_sum / 10, 1.0)) * length_penalty)
        return scores


class CrossEncoderReranker(BaseReranker):
    name = "cross-encoder"

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        pairs = [(query, p) for p in passages]
        raw = self._model.predict(pairs, show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


@lru_cache(maxsize=4)
def get_reranker(provider: str | None = None, model: str | None = None) -> BaseReranker:
    provider = (provider or settings.reranker_provider).lower()
    model = model or settings.reranker_model
    if provider in {"cross-encoder", "crossencoder", "neural"}:
        try:
            return CrossEncoderReranker(model)
        except Exception as exc:  # pragma: no cover - optional dep missing
            log.warning("cross-encoder reranker unavailable (%s); using lexical reranker.", exc)
    return LexicalReranker()
