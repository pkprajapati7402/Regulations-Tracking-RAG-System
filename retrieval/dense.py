"""Dense (embedding) retrieval.

Vectors are L2-normalised at write time, so cosine similarity is a dot product.
At this corpus size an exact NumPy matmul is both faster and more accurate than
an ANN index; the same interface is what a pgvector `<=>` query would return,
so swapping in a pgvector-backed search is a drop-in change here.
"""
from __future__ import annotations

import numpy as np

from ingestion.embed import get_embedder
from retrieval.corpus_index import CorpusSnapshot


class DenseIndex:
    name = "dense"

    def __init__(self, snapshot: CorpusSnapshot):
        self.snapshot = snapshot
        self.embedder = get_embedder()
        self.matrix = snapshot.embeddings

    def search(self, query: str, top_k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        if self.matrix.shape[0] == 0:
            return []
        qvec = self.embedder.encode_one(query, is_query=True)
        if qvec.shape[0] != self.matrix.shape[1]:
            n = min(qvec.shape[0], self.matrix.shape[1])
            padded = np.zeros(self.matrix.shape[1], dtype=np.float32)
            padded[:n] = qvec[:n]
            qvec = padded
        scores = self.matrix @ qvec
        order = np.argsort(-scores)
        out: list[tuple[int, float]] = []
        for idx in order:
            i = int(idx)
            if allowed is not None and i not in allowed:
                continue
            if scores[i] <= 0:
                continue
            out.append((i, float(scores[i])))
            if len(out) >= top_k:
                break
        return out
