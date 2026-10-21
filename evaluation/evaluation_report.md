# Information Retrieval Evaluation Report (Phase 4)

**Course:** Information Retrieval and Extraction (IRE) Project  
**Dataset:** `r/JEENEETards` Cleaned Corpus (43,116 documents)  
**Test Set:** 10 Representative Domain Information Needs  

---

## 1. Executive Summary & Model Comparison

| Ranking Model | P@1 | P@5 | P@10 | MRR | MAP | NDCG@10 | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard BM25 (Unfielded)** | 0.8 | 0.44 | 0.26 | 0.9 | 0.3856 | 0.6855 | 4.73 ms |
| **BM25F (Field-Weighted: Title 2.5x)** | 1.0 | 0.54 | 0.29 | 1.0 | 0.515 | 0.8027 | 4.22 ms |
| **BM25F + Engagement Boosting (Proposed Model)** | 1.0 | 0.58 | 0.3 | 1.0 | 0.5387 | 0.8272 | 4.81 ms |

### Key Observations & Insights:
1. **Field-Weighted BM25F vs Standard BM25**: Giving higher weight to the `title` field ($w_{\text{title}}=2.5$) significantly improves early precision (P@1 and P@5) and NDCG@10, because Reddit post titles directly state the specific user question.
2. **Engagement Boosting**: Community upvotes and discussion activity act as an effective tiebreaker, lifting authoritative recommendations and comprehensive guides.
3. **Real-time Efficiency**: All models maintain sub-10ms query latencies across the entire corpus, meeting production search standards.

---

## 2. Per-Query Breakdown (Proposed BM25F + Engagement Model)

| Query ID | Query | P@1 | P@5 | MRR | MAP | NDCG@10 | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | `physics wallah vs unacademy` | 1 | 0.6 | 1.0 | 0.5 | 0.8449 | 4.83 ms |
| **Q2** | `best book for organic chemistry` | 1 | 0.6 | 1.0 | 0.5 | 0.8449 | 6.71 ms |
| **Q3** | `marks vs percentile 2022` | 1 | 0.6 | 1.0 | 0.5 | 0.8449 | 6.24 ms |
| **Q4** | `allen test series review` | 1 | 0.6 | 1.0 | 0.6 | 0.7073 | 4.67 ms |
| **Q5** | `dc pandey vs hc verma` | 1 | 0.6 | 1.0 | 0.6 | 0.8684 | 0.83 ms |
| **Q6** | `maths dropper strategy` | 1 | 0.6 | 1.0 | 0.6 | 0.8866 | 4.37 ms |
| **Q7** | `neet 600 marks government medical college` | 1 | 0.6 | 1.0 | 0.6 | 0.8866 | 8.69 ms |
| **Q8** | `mock test score stuck` | 1 | 0.4 | 1.0 | 0.4667 | 0.7301 | 4.99 ms |
| **Q9** | `hostel food kota life` | 1 | 0.6 | 1.0 | 0.6 | 0.8866 | 2.21 ms |
| **Q10** | `iit bombay cse cutoff rank` | 1 | 0.6 | 1.0 | 0.42 | 0.7715 | 4.57 ms |

---

## 3. Evaluation Methodology
- **Test Queries**: 10 hand-curated queries covering Coaching Comparisons, Book Suggestions, Score Cutoffs, Dropper Strategies, Test Series, and Mental Wellbeing.
- **Metrics Defined**: Precision@K ($K=1, 3, 5, 10$), Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), and Normalized Discounted Cumulative Gain (NDCG@10).
- **Relevance Assessments (Qrels)**: Graded 0 (Irrelevant), 1 (Partially Relevant), 2 (Highly Relevant).