# Evaluation results

_k=5, embedding=hashing, reranker=lexical, chunker=structure_aware_


## Retrieval strategy comparison

| Configuration | Precision@k | Recall@k | nDCG@10 | MRR | Latency (ms/query) |
|---|---|---|---|---|---|
| bm25 | 0.223 | 1.000 | 0.936 | 0.914 | 24 |
| dense | 0.206 | 0.929 | 0.928 | 0.943 | 0 |
| hybrid | 0.217 | 0.971 | 0.942 | 0.936 | 1 |
| hybrid_rerank | 0.223 | 1.000 | 0.920 | 0.896 | 3 |


## Chunking strategy comparison

| Chunker | Precision@k | Recall@k | nDCG@10 | MRR | Latency (ms/query) |
|---|---|---|---|---|---|
| fixed (26 chunks) | 0.274 | 1.000 | 0.926 | 0.901 | 3 |
| recursive (31 chunks) | 0.229 | 1.000 | 0.946 | 0.930 | 3 |
| structure_aware (39 chunks) | 0.234 | 0.971 | 0.891 | 0.868 | 4 |


## Citation faithfulness

| Queries | Mean faithfulness | Fully faithful answers | Uncited claims | Invalid citations |
|---|---|---|---|---|
| 35 | 0.862 | 7 | 0 | 0 |
