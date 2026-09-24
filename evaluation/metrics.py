"""IR metrics: Precision@k, Recall@k, nDCG@k, MRR.

``ranx``/``pytrec_eval`` are the reference implementations named in the spec;
they are optional here (they pull in heavy deps). When ``ranx`` is installed we
delegate to it and cross-check; otherwise these graded-relevance
implementations are used. ``tests/test_metrics.py`` pins them against
hand-computed values.

Conventions (identical to trec_eval):
  * qrels: {query_id: {doc_id: grade}} with grades 0 / 1 / 2
  * run:   {query_id: [doc_id, ...]} ordered by descending score
  * a document is "relevant" for binary metrics when grade >= 1
"""
from __future__ import annotations

import math
from statistics import mean


def precision_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    hits = sum(1 for d in top if relevant.get(d, 0) >= 1)
    return hits / k


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    total = sum(1 for g in relevant.values() if g >= 1)
    if total == 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if relevant.get(d, 0) >= 1)
    return hits / total


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    gains = [float(relevant.get(d, 0)) for d in ranked[:k]]
    ideal = sorted((float(g) for g in relevant.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranked: list[str], relevant: dict[str, int]) -> float:
    for i, d in enumerate(ranked, start=1):
        if relevant.get(d, 0) >= 1:
            return 1.0 / i
    return 0.0


def evaluate_run(
    qrels: dict[str, dict[str, int]], run: dict[str, list[str]], k: int = 10
) -> dict[str, float]:
    """Macro-averaged metrics over all judged queries."""
    p, r, n, rr = [], [], [], []
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])
        p.append(precision_at_k(ranked, relevant, k))
        r.append(recall_at_k(ranked, relevant, k))
        n.append(ndcg_at_k(ranked, relevant, 10))
        rr.append(reciprocal_rank(ranked, relevant))
    if not p:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0, "ndcg@10": 0.0, "mrr": 0.0, "queries": 0}
    return {
        f"precision@{k}": mean(p),
        f"recall@{k}": mean(r),
        "ndcg@10": mean(n),
        "mrr": mean(rr),
        "queries": len(p),
    }


def evaluate_with_ranx(qrels: dict, run: dict, k: int = 10) -> dict[str, float] | None:
    """Delegate to ranx when available (reference cross-check)."""
    try:
        from ranx import Qrels, Run, evaluate  # type: ignore
    except ImportError:
        return None
    scored = {q: {d: 1.0 / (i + 1) for i, d in enumerate(docs)} for q, docs in run.items()}
    metrics = [f"precision@{k}", f"recall@{k}", "ndcg@10", "mrr"]
    res = evaluate(Qrels(qrels), Run(scored), metrics)
    return {m: float(res[m]) for m in metrics}
