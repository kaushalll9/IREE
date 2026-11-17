# Search Evaluation Report (HSES)

Generated 2026-11-17T19:18:41 · 43,116 docs · 80 queries (56 test / 24 tuning) · 1974 judgments by assessor: legacy 143, claude 1831

## Hybrid Search Evaluation Score (deployed system)

**HSES: incomplete.** Partial score 75.5 / 100 over the 90% of weight that was measured. Missing: `usefulness.task_completion`, `usefulness.satisfaction`

| Component | Weight | Normalised score | Points |
|---|---|---|---|
| ndcg@10 | 20% | 0.526 | 10.52 |
| p@5 | 15% | 0.593 | 8.89 |
| map | 5% | 0.375 | 1.88 |
| mrr | 5% | 0.717 | 3.58 |
| robustness.abbreviation | 5% | 1.000 | 5.00 |
| robustness.synonym | 5% | 1.000 | 5.00 |
| robustness.hinglish_typo | 5% | 0.733 | 3.67 |
| robustness.numerical_comparison | 5% | 1.000 | 5.00 |
| efficiency.p95_latency | 10% | 1.000 | 10.00 |
| efficiency.index_memory | 5% | 0.875 | 4.37 |
| correctness.index | 5% | 1.000 | 5.00 |
| correctness.filters | 5% | 1.000 | 5.00 |
| usefulness.task_completion | 5% | – | – |
| usefulness.satisfaction | 5% | – | – |

## Model comparison and ablation (test split)

| System | NDCG@10 | P@5 | MAP | MRR | Recall@10 | Success@5 | Judged@10 | P95 ms | ΔNDCG vs baseline (p) | Partial HSES |
|---|---|---|---|---|---|---|---|---|---|---|
| BM25F, equal field weights (baseline) | 0.488 | 0.554 | 0.339 | 0.701 | 0.486 | 0.875 | 1.00 | 11.2 | +0.000 (–) | 73.5 |
| BM25F (title 2.5x) | 0.532 | 0.575 | 0.372 | 0.727 | 0.511 | 0.911 | 1.00 | 10.9 | +0.043 (0.045) | 76.0 |
| BM25F + engagement boost | 0.520 | 0.564 | 0.361 | 0.728 | 0.498 | 0.893 | 1.00 | 12.8 | +0.032 (0.138) | 74.8 |
| BM25F + engagement + Word2Vec expansion | 0.517 | 0.579 | 0.359 | 0.722 | 0.495 | 0.911 | 1.00 | 13.3 | +0.029 (0.179) | 75.3 |
| Semantic (multilingual MiniLM) | 0.376 | 0.425 | 0.251 | 0.616 | 0.342 | 0.750 | 1.00 | 15.7 | -0.112 (0.017) | 64.4 |
| Hybrid RRF (deployed: search default + chatbot) | 0.526 | 0.593 | 0.375 | 0.717 | 0.510 | 0.911 | 1.00 | 52.3 | +0.038 (0.121) | 75.5 |

p = two-sided paired randomization test on per-query NDCG@10 against the baseline (10,000 permutations). Metrics use condensed rankings: unjudged documents are skipped, not counted as irrelevant; Judged@10 is the share of the raw top 10 that was judged.

## Query robustness (Success@5 by category)

| System | abbreviation | comparison | general | hinglish | numerical | synonym | typo |
|---|---|---|---|---|---|---|---|
| BM25F, equal field weights (baseline) | 1.00 | 1.00 | 0.92 | 0.78 | 0.83 | 1.00 | 0.50 |
| BM25F (title 2.5x) | 1.00 | 1.00 | 0.92 | 0.89 | 1.00 | 1.00 | 0.50 |
| BM25F + engagement boost | 0.88 | 1.00 | 0.92 | 0.89 | 1.00 | 1.00 | 0.50 |
| BM25F + engagement + Word2Vec expansion | 0.88 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 0.50 |
| Semantic (multilingual MiniLM) | 1.00 | 0.50 | 0.83 | 0.44 | 0.67 | 0.89 | 0.83 |
| Hybrid RRF (deployed: search default + chatbot) | 1.00 | 1.00 | 0.92 | 0.89 | 1.00 | 1.00 | 0.50 |

## Word2Vec query expansion (system 3 vs 4)

Expansion fired on 19 of 56 test queries: NDCG@10 rose on 6 and fell on 9 (mean change -0.0091 where it fired). Query drift: expansion added 17 judged-irrelevant and 11 relevant documents to top-10s. Mean extra latency: +0.74 ms.

## Failed queries (deployed system, no relevant result in top 5)

- `mock test mein marks kam aa rahe hai` (hinglish; judged@10 1.00)
- `thermodynamcs doubts` (typo; judged@10 1.00)
- `integeration practice` (typo; judged@10 1.00)
- `electrostatcs book` (typo; judged@10 1.00)
- `attempt vs accuracy in jee mains` (general; judged@10 1.00)

## Performance (device: cuda:0)

| System | avg ms | P50 | P95 | P99 |
|---|---|---|---|---|
| BM25F, equal field weights (baseline) | 6.96 | 3.27 | 11.18 | 120.98 |
| BM25F (title 2.5x) | 7.29 | 3.21 | 10.91 | 122.47 |
| BM25F + engagement boost | 8.19 | 4.0 | 12.82 | 123.68 |
| BM25F + engagement + Word2Vec expansion | 8.39 | 4.38 | 13.31 | 121.7 |
| Semantic (multilingual MiniLM) | 11.09 | 11.07 | 15.65 | 16.83 |
| Hybrid RRF (deployed: search default + chatbot) | 32.47 | 24.96 | 52.32 | 177.38 |

Index: lexical 6.17 MB, embeddings 63.16 MB, indexing time 5.32 s, vocabulary 28,255. Memory (RSS): {'start_mb': 142.4, 'after_bm25_mb': 264.6, 'after_semantic_mb': 1228.1}. Throughput: 1 worker(s) 33.4 q/s, 4 worker(s) 32.3 q/s.

## Reliability checks

**index_integrity: 14/14 passed**
**filters_api: 43/43 passed**

## User usefulness

0 feedback responses (0 solved votes, 0 ratings; 20 needed for a score). Task completion: –, satisfaction: – (mean rating – / 5).

## Method notes

- Judgments are pooled from every system's top 10 and graded 0/1/2 against the query intent. The assessor of each label is recorded in `qrels.json`; `claude` labels are LLM judgments and should be spot-checked by a human (`python evaluate.py annotate --review`). Human labels always take precedence.
- Tuning queries (the original 10, seen during development, plus a stratified sample) are excluded from every reported number; HSES uses the test split only.
- Ranking metrics are omitted from HSES for a system whose Judged@10 is below 0.8, rather than guessed.
