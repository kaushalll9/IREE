# Information Retrieval Evaluation Report (Phase 4)

**Course:** Information Retrieval and Extraction (IRE) Project  
**Dataset:** `r/JEENEETards` Cleaned Corpus (43,116 documents)  
**Test Set:** 10 Representative Domain Information Needs  

---

## 1. Executive Summary & Model Comparison

| Ranking Model | P@1 | P@5 | P@10 | MRR | MAP | NDCG@10 | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard BM25 (Unfielded)** | 1.0 | 0.84 | 0.73 | 1.0 | 0.704 | 0.7726 | 2.62 ms |
| **BM25F (Field-Weighted: Title 2.5x)** | 0.8 | 0.84 | 0.74 | 0.9 | 0.6832 | 0.7693 | 2.36 ms |
| **BM25F + Engagement Boosting** | 0.8 | 0.84 | 0.76 | 0.9 | 0.7007 | 0.8051 | 2.88 ms |
| **BM25F + Engagement + Domain Expansion (Proposed Model)** | 0.8 | 0.82 | 0.76 | 0.9 | 0.6951 | 0.8068 | 6.38 ms |

### Key Observations (computed from this run):
- **Best MAP**: Standard BM25 (Unfielded) (0.704)
- **Best NDCG@10**: BM25F + Engagement + Domain Expansion (Proposed Model) (0.8068)
- **Best P@1**: Standard BM25 (Unfielded) (1.0)
- **Best P@5**: Standard BM25 (Unfielded) (0.84)
- **Latency**: 2.36-6.38 ms per query across all configurations.
- With only 10 queries, differences of a few points are within noise; treat them as indicative, not significant.

---

## 2. Per-Query Breakdown (Proposed Model: BM25F + Engagement + Domain Expansion)

| Query ID | Query | P@1 | P@5 | MRR | MAP | NDCG@10 | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | `physics wallah vs unacademy` | 1 | 1.0 | 1.0 | 0.6846 | 0.8414 | 4.91 ms |
| **Q2** | `best book for organic chemistry` | 1 | 1.0 | 1.0 | 0.7324 | 0.8764 | 32.62 ms |
| **Q3** | `marks vs percentile 2022` | 1 | 1.0 | 1.0 | 0.8333 | 0.9266 | 5.2 ms |
| **Q4** | `allen test series review` | 0 | 0.6 | 0.5 | 0.4872 | 0.6862 | 2.86 ms |
| **Q5** | `dc pandey vs hc verma` | 1 | 1.0 | 1.0 | 0.6396 | 0.8871 | 0.58 ms |
| **Q6** | `maths dropper strategy` | 1 | 1.0 | 1.0 | 0.9468 | 0.7658 | 2.55 ms |
| **Q7** | `neet 600 marks government medical college` | 1 | 0.8 | 1.0 | 0.7735 | 0.6883 | 6.09 ms |
| **Q8** | `mock test score stuck` | 1 | 0.6 | 1.0 | 0.6931 | 0.8112 | 3.97 ms |
| **Q9** | `hostel food kota life` | 1 | 0.8 | 1.0 | 0.6608 | 0.9005 | 1.51 ms |
| **Q10** | `iit bombay cse cutoff rank` | 0 | 0.4 | 0.5 | 0.5 | 0.685 | 3.46 ms |

---

## 3. Evaluation Methodology
- **Test Queries**: 10 hand-curated queries covering Coaching Comparisons, Book Suggestions, Score Cutoffs, Dropper Strategies, Test Series, and Mental Wellbeing.
- **Metrics Defined**: Precision@K ($K=1, 3, 5, 10$), Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), and Normalized Discounted Cumulative Gain (NDCG@10).
- **Relevance Assessments (Qrels)**: Graded 0 (Irrelevant), 1 (Partially Relevant), 2 (Highly Relevant).
- **Pooling**: judgments cover the union of every configuration's top-10 per query (`python evaluate.py --pool`), so each evaluated ranking is fully judged in its top 10. Unjudged documents count as irrelevant.