"""Reciprocal Rank Fusion of the sparse and dense rankings.

    RRF(d) = sum_over_rankers 1 / (k + rank_r(d))

k defaults to 60 (the value from Cormack et al., 2009). RRF is rank-based, so
it needs no score normalisation between BM25 (unbounded) and cosine (0-1).
"""
from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[int, float]]], k: int = 60, top_k: int = 40
) -> list[tuple[int, float, dict[str, float]]]:
    fused: dict[int, float] = defaultdict(float)
    components: dict[int, dict[str, float]] = defaultdict(dict)

    for ranker_name, ranked in rankings.items():
        for rank, (idx, score) in enumerate(ranked, start=1):
            fused[idx] += 1.0 / (k + rank)
            components[idx][f"{ranker_name}_score"] = score
            components[idx][f"{ranker_name}_rank"] = float(rank)

    out = [(idx, score, components[idx]) for idx, score in fused.items()]
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]
