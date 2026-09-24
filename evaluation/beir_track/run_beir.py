"""Track A - benchmark credibility on a BEIR finance subset (FiQA-2018).

Runs the *same* retrieval stack (BM25 / dense / hybrid / hybrid+rerank) against a
standard IR benchmark with pre-existing relevance judgments, so the numbers are
comparable to published baselines instead of being self-reported on a corpus we
built ourselves.

The BEIR corpus is downloaded on first use (~60 MB for FiQA) to
``evaluation/beir_track/data``; it is git-ignored. Requires network access.

    python -m evaluation.beir_track.run_beir --dataset fiqa --subset 2000 --k 10

Implementation note: instead of loading BEIR into Postgres, the benchmark
corpus is indexed in a throwaway in-memory snapshot using the same BM25Index,
DenseIndex, RRF and reranker classes the production path uses - so what is
measured here is genuinely the same code.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from core.logging_conf import get_logger
from evaluation.metrics import evaluate_run
from ingestion.embed import get_embedder
from retrieval.bm25 import BM25Index
from retrieval.corpus_index import CorpusSnapshot, IndexedChunk
from retrieval.dense import DenseIndex
from retrieval.hybrid_rrf import reciprocal_rank_fusion
from retrieval.rerank import get_reranker

log = get_logger(__name__)

DATA_DIR = Path(__file__).parent / "data"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds}.zip"


def download(dataset: str) -> Path:
    import httpx

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / dataset
    if target.exists():
        return target
    url = BEIR_URL.format(ds=dataset)
    zip_path = DATA_DIR / f"{dataset}.zip"
    log.info("Downloading %s ...", url)
    with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink(missing_ok=True)
    return target


def load_beir(path: Path, subset: int | None) -> tuple[dict, dict, dict]:
    corpus: dict[str, str] = {}
    with open(path / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            corpus[row["_id"]] = (row.get("title", "") + "\n" + row.get("text", "")).strip()

    queries: dict[str, str] = {}
    with open(path / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            queries[row["_id"]] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    qrels_file = path / "qrels" / "test.tsv"
    with open(qrels_file, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, did, score = line.strip().split("\t")
            qrels.setdefault(qid, {})[did] = int(score)

    if subset:
        qids = list(qrels)[:subset]
        qrels = {q: qrels[q] for q in qids}
        keep = {d for rel in qrels.values() for d in rel}
        # add distractors so the task stays non-trivial
        for did in list(corpus)[: subset * 10]:
            keep.add(did)
        corpus = {d: t for d, t in corpus.items() if d in keep}
        queries = {q: queries[q] for q in qids if q in queries}
    return corpus, queries, qrels


def build_snapshot(corpus: dict[str, str]) -> CorpusSnapshot:
    embedder = get_embedder()
    ids = list(corpus)
    texts = [corpus[i] for i in ids]
    chunks = [
        IndexedChunk(
            chunk_id=i, text=t, status="active", chunk_type="prose", section_ref=None,
            doc_number=i, doc_title="beir", source_url=None, version_number=1, published_date=None,
        )
        for i, t in zip(ids, texts)
    ]
    vectors = np.zeros((len(texts), embedder.dim), dtype=np.float32)
    for start in range(0, len(texts), 256):
        batch = texts[start : start + 256]
        vectors[start : start + len(batch)] = embedder.encode(batch)
    return CorpusSnapshot(chunks, vectors)


def evaluate(snapshot: CorpusSnapshot, queries: dict, qrels: dict, k: int) -> dict:
    bm25, dense = BM25Index(snapshot), DenseIndex(snapshot)
    reranker = get_reranker()
    results: dict[str, dict] = {}

    for mode in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
        run: dict[str, list[str]] = {}
        for qid, text in queries.items():
            if mode == "bm25":
                hits = [i for i, _ in bm25.search(text, k)]
            elif mode == "dense":
                hits = [i for i, _ in dense.search(text, k)]
            else:
                fused = reciprocal_rank_fusion(
                    {"bm25": bm25.search(text, 50), "dense": dense.search(text, 50)}, top_k=50
                )
                if mode == "hybrid":
                    hits = [i for i, _, _ in fused][:k]
                else:
                    passages = [snapshot.chunks[i].text for i, _, _ in fused]
                    hits = [fused[pos][0] for pos, _ in reranker.rerank(text, passages, k)]
            run[qid] = [snapshot.chunks[i].chunk_id for i in hits]
        results[mode] = evaluate_run(qrels, run, k)
        log.info("%s -> %s", mode, results[mode])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BEIR track.")
    parser.add_argument("--dataset", default="fiqa")
    parser.add_argument("--subset", type=int, default=200, help="Number of test queries (0 = all)")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    path = download(args.dataset)
    corpus, queries, qrels = load_beir(path, args.subset or None)
    log.info("Loaded %d documents / %d queries", len(corpus), len(queries))
    snapshot = build_snapshot(corpus)
    results = evaluate(snapshot, queries, qrels, args.k)

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
