# Evaluation results

_k=5, embedding=hashing, reranker=lexical, chunker=structure_aware_


## Retrieval strategy comparison

| Configuration | Precision@k | Recall@k | nDCG@10 | MRR | Latency (ms/query) |
|---|---|---|---|---|---|
| bm25 | 0.223 | 1.000 | 0.922 | 0.898 | 59 |
| dense | 0.206 | 0.929 | 0.928 | 0.943 | 0 |
| hybrid | 0.217 | 0.971 | 0.945 | 0.936 | 1 |
| hybrid_rerank | 0.223 | 1.000 | 0.920 | 0.896 | 4 |
