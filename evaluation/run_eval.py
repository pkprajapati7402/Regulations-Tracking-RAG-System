"""Evaluation harness.

Two experiments, both scored with Precision@k / Recall@k / nDCG@10 / MRR:

  1. retrieval sweep - bm25 vs dense vs hybrid vs hybrid+rerank, same chunking
  2. chunking sweep  - fixed vs recursive vs structure-aware, same retriever

Plus an end-to-end citation-faithfulness rate over the same query set.

Judgments are anchored to (doc_number, anchor phrase) pairs, so the labelled
set survives re-chunking - see evaluation/domain_track/queries.json.

    python -m evaluation.run_eval                     # everything
    python -m evaluation.run_eval --experiment retrieval
    python -m evaluation.run_eval --experiment chunking --k 5
    python -m evaluation.run_eval --experiment faithfulness
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sqlalchemy import select

from core.config import ROOT, settings
from core.db import init_db, session_scope
from core.logging_conf import get_logger
from core.models import STATUS_ACTIVE, Chunk, Document, DocumentVersion, EvalRun, EvalResult, EvalQuery
from evaluation.metrics import (
    evaluate_run,
    evaluate_with_ranx,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from generation.answer import generate_answer
from ingestion.pipeline import rechunk_all
from retrieval.service import get_retrieval_service

log = get_logger(__name__)

QUERY_FILE = ROOT / "evaluation" / "domain_track" / "queries.json"
RESULTS_DIR = ROOT / "evaluation" / "results"
RETRIEVAL_MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]
CHUNKERS = ["fixed", "recursive", "structure_aware"]


def load_queries() -> list[dict]:
    data = json.loads(QUERY_FILE.read_text())
    return data["queries"]


def build_qrels(session, queries: list[dict]) -> dict[str, dict[str, int]]:
    """Resolve (doc_number, anchor) judgments to the chunk ids of the *current* index."""
    rows = session.execute(
        select(Chunk.id, Chunk.chunk_text, Document.doc_number)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(Chunk.status == STATUS_ACTIVE)
    ).all()

    by_doc: dict[str, list[tuple[str, str]]] = {}
    for chunk_id, text, doc_number in rows:
        by_doc.setdefault(doc_number, []).append((chunk_id, text))

    qrels: dict[str, dict[str, int]] = {}
    missing: list[str] = []
    for q in queries:
        rel: dict[str, int] = {}
        for j in q["judgments"]:
            anchor = j["anchor"].lower()
            hits = [cid for cid, text in by_doc.get(j["doc"], []) if anchor in text.lower()]
            if not hits:
                missing.append(f"{q['id']}::{j['doc']}::{j['anchor'][:40]}")
            for cid in hits:
                rel[cid] = max(rel.get(cid, 0), int(j["grade"]))
        if rel:
            qrels[q["id"]] = rel
    if missing:
        log.warning("%d judgment anchors did not resolve to any chunk: %s", len(missing), missing[:5])
    return qrels


def run_retrieval(queries: list[dict], mode: str, k: int, session) -> dict[str, list[str]]:
    svc = get_retrieval_service()
    run: dict[str, list[str]] = {}
    for q in queries:
        chunks, _ = svc.search(q["text"], mode=mode, top_k=k, candidate_k=max(40, k * 4), session=session)
        run[q["id"]] = [c.chunk_id for c in chunks]
    return run


def per_query_rows(qrels, run, k) -> list[dict]:
    out = []
    for qid, rel in qrels.items():
        ranked = run.get(qid, [])
        out.append(
            {
                "query_id": qid,
                "precision_at_k": precision_at_k(ranked, rel, k),
                "recall_at_k": recall_at_k(ranked, rel, k),
                "ndcg_at_10": ndcg_at_k(ranked, rel, 10),
                "mrr": reciprocal_rank(ranked, rel),
            }
        )
    return out


def persist(session, config_name: str, rows: list[dict], notes: str = "") -> None:
    run_row = EvalRun(config_name=config_name, track="domain_track", notes=notes)
    session.add(run_row)
    session.flush()
    for r in rows:
        session.add(
            EvalResult(
                eval_run_id=run_row.id,
                query_id=r["query_id"],
                precision_at_k=r["precision_at_k"],
                recall_at_k=r["recall_at_k"],
                ndcg_at_10=r["ndcg_at_10"],
                mrr=r["mrr"],
            )
        )
    session.flush()


def markdown_table(rows: list[dict], first_col: str) -> str:
    header = f"| {first_col} | Precision@k | Recall@k | nDCG@10 | MRR | Latency (ms/query) |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['ndcg']:.3f} | {r['mrr']:.3f} | {r['latency_ms']:.0f} |"
        )
    return "\n".join(lines)


def experiment_retrieval(k: int) -> list[dict]:
    print(f"\n=== Experiment 1: retrieval strategies (chunker={settings.chunker}, k={k}) ===")
    rows = []
    with session_scope() as session:
        queries = load_queries()
        qrels = build_qrels(session, queries)
        print(f"{len(qrels)} judged queries, {sum(len(v) for v in qrels.values())} graded chunks\n")
        for mode in RETRIEVAL_MODES:
            start = time.perf_counter()
            run = run_retrieval(queries, mode, k, session)
            latency = (time.perf_counter() - start) * 1000 / max(len(queries), 1)
            metrics = evaluate_run(qrels, run, k)
            cross = evaluate_with_ranx(qrels, run, k)
            if cross:
                log.info("ranx cross-check for %s: %s", mode, cross)
            rows.append(
                {
                    "config": mode,
                    "precision": metrics[f"precision@{k}"],
                    "recall": metrics[f"recall@{k}"],
                    "ndcg": metrics["ndcg@10"],
                    "mrr": metrics["mrr"],
                    "latency_ms": latency,
                }
            )
            persist(session, f"retrieval={mode}", per_query_rows(qrels, run, k))
            print(
                f"  {mode:<14} P@{k}={metrics[f'precision@{k}']:.3f}  R@{k}={metrics[f'recall@{k}']:.3f}  "
                f"nDCG@10={metrics['ndcg@10']:.3f}  MRR={metrics['mrr']:.3f}  {latency:.0f} ms/q"
            )
    return rows


def experiment_chunking(k: int) -> list[dict]:
    print(f"\n=== Experiment 2: chunking strategies (retriever={settings.retrieval_mode}, k={k}) ===")
    rows = []
    original = settings.chunker
    svc = get_retrieval_service()
    for chunker in CHUNKERS:
        with session_scope() as session:
            rechunk_all(session, chunker)
        svc.mark_dirty()
        with session_scope() as session:
            queries = load_queries()
            qrels = build_qrels(session, queries)
            start = time.perf_counter()
            run = run_retrieval(queries, settings.retrieval_mode, k, session)
            latency = (time.perf_counter() - start) * 1000 / max(len(queries), 1)
            metrics = evaluate_run(qrels, run, k)
            n_chunks = session.scalar(select(Chunk.id).limit(1)) and len(
                list(session.scalars(select(Chunk.id)))
            )
            rows.append(
                {
                    "config": f"{chunker} ({n_chunks} chunks)",
                    "precision": metrics[f"precision@{k}"],
                    "recall": metrics[f"recall@{k}"],
                    "ndcg": metrics["ndcg@10"],
                    "mrr": metrics["mrr"],
                    "latency_ms": latency,
                }
            )
            persist(session, f"chunker={chunker}", per_query_rows(qrels, run, k))
            print(
                f"  {chunker:<16} P@{k}={metrics[f'precision@{k}']:.3f}  R@{k}={metrics[f'recall@{k}']:.3f}  "
                f"nDCG@10={metrics['ndcg@10']:.3f}  MRR={metrics['mrr']:.3f}  ({n_chunks} chunks)"
            )
    # restore the configured chunker so the DB is left in its default state
    with session_scope() as session:
        rechunk_all(session, original)
    svc.mark_dirty()
    return rows


def experiment_faithfulness(k: int) -> dict:
    print("\n=== Experiment 3: end-to-end citation faithfulness ===")
    svc = get_retrieval_service()
    rates, uncited, invalid = [], 0, 0
    with session_scope() as session:
        for q in load_queries():
            chunks, _ = svc.search(q["text"], top_k=k, session=session)
            result = generate_answer(q["text"], chunks)
            rates.append(result.faithfulness.rate)
            uncited += result.faithfulness.uncited_claims
            invalid += len(result.faithfulness.invalid_citations)
    summary = {
        "queries": len(rates),
        "mean_faithfulness": sum(rates) / max(len(rates), 1),
        "answers_fully_faithful": sum(1 for r in rates if r >= 0.999),
        "uncited_claims": uncited,
        "invalid_citations": invalid,
    }
    print(
        f"  mean faithfulness={summary['mean_faithfulness']:.3f}  "
        f"fully-faithful answers={summary['answers_fully_faithful']}/{summary['queries']}  "
        f"uncited claims={uncited}  invalid citations={invalid}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IR evaluation suite.")
    parser.add_argument(
        "--experiment", default="all", choices=["all", "retrieval", "chunking", "faithfulness"]
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args()

    init_db()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"k": args.k, "embedding": settings.embedding_provider, "reranker": settings.reranker_provider}
    md: list[str] = [f"# Evaluation results\n", f"_k={args.k}, embedding={settings.embedding_provider}, "
                     f"reranker={settings.reranker_provider}, chunker={settings.chunker}_\n"]

    if args.experiment in {"all", "retrieval"}:
        rows = experiment_retrieval(args.k)
        report["retrieval"] = rows
        md += ["\n## Retrieval strategy comparison\n", markdown_table(rows, "Configuration"), ""]
    if args.experiment in {"all", "chunking"}:
        rows = experiment_chunking(args.k)
        report["chunking"] = rows
        md += ["\n## Chunking strategy comparison\n", markdown_table(rows, "Chunker"), ""]
    if args.experiment in {"all", "faithfulness"}:
        summary = experiment_faithfulness(args.k)
        report["faithfulness"] = summary
        md += [
            "\n## Citation faithfulness\n",
            "| Queries | Mean faithfulness | Fully faithful answers | Uncited claims | Invalid citations |",
            "|---|---|---|---|---|",
            f"| {summary['queries']} | {summary['mean_faithfulness']:.3f} | "
            f"{summary['answers_fully_faithful']} | {summary['uncited_claims']} | {summary['invalid_citations']} |",
            "",
        ]

    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    (out_dir / "results.md").write_text("\n".join(md))
    print(f"\nWrote {out_dir/'results.json'} and {out_dir/'results.md'}")
    if args.experiment in {"all", "chunking"}:
        print(
            "Note: the chunking sweep re-chunked the corpus, which resets chunk-level "
            "deprecation flags. Run `python -m ingestion.reindex --reset --seed` to restore "
            "the amendment state before demoing the case study.\n"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
