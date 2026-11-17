# Search Evaluation Report (HSES)

Generated 2026-11-17T22:16:18 · 43,116 docs · 80 queries (56 test / 24 tuning) · 2006 judgments by assessor: legacy 143, claude 1863

## Hybrid Search Evaluation Score (deployed system)

**HSES: incomplete.** Partial score 75.5 / 100 over the 90% of weight that was measured. Missing: `usefulness.task_completion`, `usefulness.satisfaction`

| Component | Weight | Normalised score | Points |
|---|---|---|---|
| ndcg@10 | 20% | 0.525 | 10.50 |
| p@5 | 15% | 0.596 | 8.95 |
| map | 5% | 0.370 | 1.85 |
| mrr | 5% | 0.720 | 3.60 |
| robustness.abbreviation | 5% | 1.000 | 5.00 |
| robustness.synonym | 5% | 1.000 | 5.00 |
| robustness.hinglish_typo | 5% | 0.733 | 3.67 |
| robustness.numerical_comparison | 5% | 1.000 | 5.00 |
| efficiency.p95_latency | 10% | 1.000 | 10.00 |
| efficiency.index_memory | 5% | 0.874 | 4.37 |
| correctness.index | 5% | 1.000 | 5.00 |
| correctness.filters | 5% | 1.000 | 5.00 |
| usefulness.task_completion | 5% | – | – |
| usefulness.satisfaction | 5% | – | – |

## Model comparison and ablation (test split)

| System | NDCG@10 | P@5 | MAP | MRR | Recall@10 | Success@5 | Judged@10 | P95 ms | ΔNDCG vs baseline (p) | Partial HSES |
|---|---|---|---|---|---|---|---|---|---|---|
| BM25F, equal field weights (baseline) | 0.484 | 0.554 | 0.329 | 0.701 | 0.474 | 0.875 | 1.00 | 10.6 | +0.000 (–) | 73.4 |
| BM25F (title 2.5x) | 0.529 | 0.575 | 0.363 | 0.727 | 0.501 | 0.911 | 1.00 | 10.6 | +0.044 (0.039) | 75.9 |
| BM25F + engagement boost | 0.517 | 0.564 | 0.352 | 0.728 | 0.488 | 0.893 | 1.00 | 12.5 | +0.032 (0.126) | 74.7 |
| BM25F + engagement + dictionary expansion (Word2Vec + TF-IDF) | 0.523 | 0.579 | 0.356 | 0.723 | 0.495 | 0.911 | 1.00 | 12.6 | +0.038 (0.077) | 75.5 |
| Semantic (multilingual MiniLM) | 0.375 | 0.425 | 0.247 | 0.616 | 0.336 | 0.750 | 1.00 | 9.0 | -0.110 (0.018) | 64.3 |
| Hybrid RRF (deployed: search default + chatbot) | 0.525 | 0.596 | 0.370 | 0.720 | 0.499 | 0.911 | 1.00 | 37.0 | +0.040 (0.101) | 75.5 |

p = two-sided paired randomization test on per-query NDCG@10 against the baseline (10,000 permutations). Metrics use condensed rankings: unjudged documents are skipped, not counted as irrelevant; Judged@10 is the share of the raw top 10 that was judged.

## Query robustness (Success@5 by category)

| System | abbreviation | comparison | general | hinglish | numerical | synonym | typo |
|---|---|---|---|---|---|---|---|
| BM25F, equal field weights (baseline) | 1.00 | 1.00 | 0.92 | 0.78 | 0.83 | 1.00 | 0.50 |
| BM25F (title 2.5x) | 1.00 | 1.00 | 0.92 | 0.89 | 1.00 | 1.00 | 0.50 |
| BM25F + engagement boost | 0.88 | 1.00 | 0.92 | 0.89 | 1.00 | 1.00 | 0.50 |
| BM25F + engagement + dictionary expansion (Word2Vec + TF-IDF) | 0.88 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 0.50 |
| Semantic (multilingual MiniLM) | 1.00 | 0.50 | 0.83 | 0.44 | 0.67 | 0.89 | 0.83 |
| Hybrid RRF (deployed: search default + chatbot) | 1.00 | 1.00 | 0.92 | 0.89 | 1.00 | 1.00 | 0.50 |

## Word2Vec query expansion (system 3 vs 4)

Expansion fired on 17 of 56 test queries: NDCG@10 rose on 6 and fell on 6 (mean change +0.0188 where it fired). Query drift: expansion added 7 judged-irrelevant and 7 relevant documents to top-10s. Mean extra latency: +0.37 ms.

## Failed queries (deployed system, no relevant result in top 5)

- `mock test mein marks kam aa rahe hai` (hinglish; judged@10 1.00)
- `thermodynamcs doubts` (typo; judged@10 1.00)
- `integeration practice` (typo; judged@10 1.00)
- `electrostatcs book` (typo; judged@10 1.00)
- `attempt vs accuracy in jee mains` (general; judged@10 1.00)

## Performance (device: cuda:0)

| System | avg ms | P50 | P95 | P99 |
|---|---|---|---|---|
| BM25F, equal field weights (baseline) | 6.93 | 3.27 | 10.61 | 110.63 |
| BM25F (title 2.5x) | 6.89 | 3.25 | 10.61 | 108.6 |
| BM25F + engagement boost | 7.7 | 3.87 | 12.46 | 112.36 |
| BM25F + engagement + dictionary expansion (Word2Vec + TF-IDF) | 7.85 | 4.07 | 12.64 | 112.25 |
| Semantic (multilingual MiniLM) | 8.02 | 7.86 | 9.01 | 10.29 |
| Hybrid RRF (deployed: search default + chatbot) | 23.44 | 18.0 | 37.02 | 146.81 |

Index: lexical 6.17 MB, embeddings 63.16 MB, indexing time 5.32 s, vocabulary 28,255. Memory (RSS): {'start_mb': 142.5, 'after_bm25_mb': 264.7, 'after_semantic_mb': 1230.6}. Throughput: 1 worker(s) 41.4 q/s, 4 worker(s) 41.4 q/s.

## Reliability checks

**index_integrity: 14/14 passed**
**filters_api: 43/43 passed**

## User usefulness

2 feedback responses (2 solved votes, 1 ratings; 20 needed for a score). Task completion: –, satisfaction: – (mean rating 1.000 / 5).

## Method notes

- Judgments are pooled from every system's top 10 and graded 0/1/2 against the query intent. The assessor of each label is recorded in `qrels.json`; `claude` labels are LLM judgments and should be spot-checked by a human (`python evaluate.py annotate --review`). Human labels always take precedence.
- Tuning queries (the original 10, seen during development, plus a stratified sample) are excluded from every reported number; HSES uses the test split only.
- Ranking metrics are omitted from HSES for a system whose Judged@10 is below 0.8, rather than guessed.
