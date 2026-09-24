import math

from evaluation.metrics import (
    evaluate_run,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

QRELS = {"q1": {"a": 2, "b": 1, "c": 0}}


def test_precision_and_recall():
    ranked = ["a", "x", "b", "y", "z"]
    assert precision_at_k(ranked, QRELS["q1"], 5) == 2 / 5
    assert recall_at_k(ranked, QRELS["q1"], 5) == 1.0
    assert precision_at_k(ranked, QRELS["q1"], 1) == 1.0
    assert recall_at_k(ranked, QRELS["q1"], 1) == 0.5


def test_mrr():
    assert reciprocal_rank(["x", "b", "a"], QRELS["q1"]) == 0.5
    assert reciprocal_rank(["x", "y"], QRELS["q1"]) == 0.0


def test_ndcg_matches_hand_computation():
    ranked = ["b", "a"]            # gains 1, 2
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert abs(ndcg_at_k(ranked, QRELS["q1"], 10) - dcg / idcg) < 1e-9
    assert ndcg_at_k(["a", "b"], QRELS["q1"], 10) == 1.0


def test_evaluate_run_aggregates():
    out = evaluate_run(QRELS, {"q1": ["a", "b"]}, k=2)
    assert out["queries"] == 1
    assert out["precision@2"] == 1.0
    assert out["mrr"] == 1.0
