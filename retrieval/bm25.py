"""Sparse lexical retrieval (BM25 Okapi) over the corpus snapshot."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from retrieval.corpus_index import CorpusSnapshot

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")

# Regulatory text is full of identifiers like "14.01.001" and "KYC"; we keep
# dotted identifiers as single tokens instead of shattering them.
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "by", "is",
    "are", "be", "as", "at", "with", "that", "this", "it", "shall", "may",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class BM25Index:
    name = "bm25"

    def __init__(self, snapshot: CorpusSnapshot):
        self.snapshot = snapshot
        self._docs = [tokenize(c.text) for c in snapshot.chunks]
        self._bm25 = BM25Okapi(self._docs or [["placeholder"]])

    def search(self, query: str, top_k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        if not self.snapshot.chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = [
            (i, float(s))
            for i, s in enumerate(scores)
            if (allowed is None or i in allowed) and s > 0
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
