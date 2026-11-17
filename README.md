# JEE/NEET Assistant — IRE Project

An Information Retrieval & Extraction project: a chatbot that answers JEE/NEET
students' questions using **official NTA notices** (for dates, syllabus, admit
cards…) and **r/JEENEETards discussions** (for books, coaching, strategy…),
always citing its sources.

```
                         ┌──────────────── question ────────────────┐
                         ▼                                          │
                   ┌───────────┐  names an exam + official topic?   │
                   │  router   │──────────── yes ───────┐           │
                   └───────────┘                        ▼           │
                         │ no                ┌────────────────────┐ │
                         ▼                   │ NTA notices        │ │
        ┌─────────────────────────────┐      │ (nta_crawler.py)   │ │
        │ query expansion             │      │ newest match +     │ │
        │ (domain_dictionary.py)      │      │ key date lines     │ │
        └─────────────────────────────┘      └────────────────────┘ │
                         ▼                              │           │
        ┌─────────────────────────────┐                 │           │
        │ BM25F + semantic search,    │                 │           │
        │ fused by RRF (43k posts)    │                 │           │
        └─────────────────────────────┘                 │           │
                         ▼                              │           │
        ┌─────────────────────────────┐                 │           │
        │ top-voted replies of the    │                 │           │
        │ best threads (Arctic Shift) │                 │           │
        └─────────────────────────────┘                 │           │
                         ▼                              ▼           │
               ┌───────────────────────────────────────────────┐    │
               │ answer + numbered sources                     │    │
               │ extractive, or written by Claude if a key set │◄───┘
               └───────────────────────────────────────────────┘
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data: either unzip the shared data.zip into data/, or download it yourself
python download_dataset.py --type posts     # r/JEENEETards posts (Arctic Shift API)
python clean_dataset.py                     # -> data/cleaned_corpus.jsonl

python indexer.py                           # -> index/postings.pkl.gz, metadata, stats
python domain_dictionary.py                 # -> index/domain_dictionary.json  (~10 s)
python semantic_search.py                   # -> index/doc_embeddings.npy (one-time; app also builds it if missing)
python nta_crawler.py                       # -> data/nta_updates.json         (~1 min)

python app.py                               # http://127.0.0.1:5001
```

The web app has three tabs: **Ask Assistant** (the chatbot), **Search Engine**
(raw ranked results) and **Evaluation** (benchmark results). The search tab
offers Hybrid (default), Semantic, and the three BM25 variants.

The embedding cache is rebuilt automatically when `cleaned_corpus.jsonl` or the
model changes. That takes a few minutes on a laptop CPU, and the Sentence
Transformer uses the GPU (Apple MPS on M-series Macs) when it can.

Optional: `export ANTHROPIC_API_KEY=...` before `python app.py` and Claude
writes each answer from the retrieved sources instead of quoting them. Without
a key everything still works.

Command-line versions: `python chatbot.py "hcv or irodov"`,
`python search.py "physics wallah vs unacademy"`, `python evaluate.py` (see Evaluation below).

## How it works (for IR newcomers)

| Step | File | What it does | IR concept |
|---|---|---|---|
| 1 | `download_dataset.py` | Downloads ~47k r/JEENEETards posts (2020–2022) | Corpus collection |
| 2 | `clean_dataset.py` | Strips markdown/URLs, drops deleted and bot posts → 43,116 docs | Preprocessing |
| 3 | `indexer.py` | Tokenises, removes stopwords, Porter-stems, and builds a map *term → list of (doc, title tf, body tf)* | Inverted index |
| 4 | `ranker.py` | Scores docs with BM25F: rare terms count more (IDF), repeated terms saturate (k1), long posts are normalised (b), titles weigh 2.5×; plus a small upvote boost | Ranking |
| 5 | `domain_dictionary.py` | Trains Word2Vec on the corpus and groups slang/abbreviations around seed terms (`hall ticket ≈ admit card`, `pw ≈ physics wallah`); search adds these at reduced weight | Embeddings, query expansion |
| 5b | `semantic_search.py` | Encodes every post (title + body) with the multilingual `paraphrase-multilingual-MiniLM-L12-v2` Sentence Transformer, caches the normalised vectors, and ranks by cosine similarity, so Hinglish queries like "JEE ki preparation kaise karu?" match posts with different wording | Dense retrieval |
| 5c | `rrf.py`, `hybrid_search.py` | Takes the top 100 from BM25F and from semantic search and fuses them with Reciprocal Rank Fusion, score = Σ 1/(60 + rank). This is the search tab's default mode | Rank fusion |
| 6 | `nta_crawler.py` | Crawls nta.ac.in (robots.txt respected), classifies notices by exam/type, extracts publish dates and the date lines from PDFs | Web crawling, information extraction |
| 7 | `chatbot.py` | Routes the question, retrieves with hybrid search, fetches top replies, builds a cited answer | Retrieval-augmented QA |
| 8 | `evaluate.py`, `eval_metrics.py`, `eval_perf.py`, `hses.py` | 80 graded benchmark queries, ranking metrics, latency/memory, integrity and API checks, user feedback, combined into the HSES score | Evaluation |

**Why fetch comments?** The dataset contains posts only, but on Reddit the
answers are in the replies. For the top threads, the chatbot fetches replies
from the Arctic Shift API (cached in `data/comments_cache/`). It keeps the
top-voted direct answers and drops chatter (counter-questions, "same here",
replies nobody upvoted).

## Evaluation (HSES)

`python evaluate.py` scores the system with the **Hybrid Search Evaluation Score**,
a weighted 0–100 score built from separately reported metrics. The weights and
normalisation targets live in `evaluation/hses_config.json`.

| Category | Metrics (weight) |
|---|---|
| Retrieval & ranking (45%) | NDCG@10 (20), P@5 (15), MAP (5), MRR (5) |
| Query understanding (20%) | Success@5 on abbreviations, synonyms, Hinglish & typos, numerical & comparison queries (5 each) |
| Performance (15%) | P95 latency (10), index size & memory (5) |
| Reliability (10%) | index integrity checks (5), filter & API checks (5) |
| User usefulness (10%) | "Did this answer your question?" (5), 1–5 rating (5), from the chat tab |

How it is measured:

- **Queries.** 80 queries in `evaluation/queries.json`, each tagged with a category
  and an intent. The 24 tuning queries (the original 10, which were seen during
  development, plus a stratified sample) are used only for tuning. All reported
  numbers come from the 56 test queries.
- **Judgments.** Graded 0/1/2 relevance labels in `evaluation/qrels.json`, pooled
  from every system's top 10 (1,974 labels, 100% of the pooled top 10s judged).
  Each label records its assessor. The 1,831 `claude` labels are LLM judgments and
  should be spot-checked by a person (`python evaluate.py annotate --review
  --assessor <name>`). Human labels always override them.
- **Unjudged documents** are skipped (condensed rankings), not counted as
  irrelevant. Judged@10 reports coverage.
- **Missing measurements** make the score *incomplete*. The report then shows a
  partial score over the measured weight instead of guessing.

### Results (test split, 56 queries)

| System | NDCG@10 | P@5 | MAP | MRR | Success@5 | P95 latency |
|---|---|---|---|---|---|---|
| BM25F, equal field weights (baseline) | 0.488 | 0.554 | 0.339 | 0.701 | 0.875 | 11 ms |
| BM25F (title 2.5×) | **0.532** | 0.575 | 0.372 | **0.727** | 0.911 | 11 ms |
| BM25F + engagement boost | 0.520 | 0.564 | 0.361 | 0.728 | 0.893 | 13 ms |
| + Word2Vec expansion | 0.517 | 0.579 | 0.359 | 0.722 | 0.911 | 13 ms |
| Semantic only (MiniLM) | 0.376 | 0.425 | 0.251 | 0.616 | 0.750 | 16 ms |
| **Hybrid RRF (deployed)** | 0.526 | **0.593** | **0.375** | 0.717 | 0.911 | 52 ms |

The deployed hybrid scores **HSES 75.5 / 100, marked incomplete** (90% of the
weight is measured). It becomes complete once the chat tab has collected 20
feedback responses.

- **Hybrid tuning.** Plain RRF with equal weights scored 0.481 NDCG@10, below
  BM25F. The semantic ranking was then down-weighted to 0.25 using the tuning
  split only (`python evaluate.py tune`), which gave the numbers above. Hybrid is
  now level with BM25F on NDCG (the difference is not significant) and best on
  P@5 and MAP.
- **Semantic search alone is clearly worse** (p = 0.02 vs baseline). It does
  recover misspelled queries (typo Success@5 0.83 vs 0.50), but most of that
  benefit is lost at weight 0.25.
- **Word2Vec expansion** fired on 19 test queries: it helped 6 and hurt 9, and
  added 17 judged-irrelevant documents to top-10s (query drift), so its net effect
  is slightly negative.
- **Failures.** The deployed system still fails on 5 test queries, mostly
  typos (`integeration practice`, `electrostatcs book`).

Full report with per-query analysis: `evaluation/evaluation_report.md`, or the
**Evaluation** tab in the app (it also links JSON and CSV exports).

### Commands

```bash
python evaluate.py                 # full suite: ranking, performance, reliability, usefulness + reports (~45 s with a GPU)
python evaluate.py ranking         # or: performance | reliability | usefulness | report
python evaluate.py tune            # RRF grid search on the tuning split (reports only)
python evaluate.py pool --show     # unjudged pooled documents after a ranking change
python evaluate.py annotate --assessor <name> [--review]   # interactive human judging
python evaluate.py import-judgments FILE --assessor <name> # bulk "query_id doc_id grade" lines
python -m unittest discover -s tests -v                    # unit tests (synthetic corpus, no data needed)
python test_search_modes.py        # integration checks for BM25 / semantic / hybrid search
```

## Limitations

- The Reddit data ends in Aug 2022, so community answers about cutoffs and
  coaching may be outdated. Official dates come from the live NTA crawl;
  re-run `nta_crawler.py` to refresh.
- `jeemain.nta.nic.in` and `neet.nta.nic.in` reject non-browser crawlers, so
  notices are collected through `nta.ac.in`.
- Without an `ANTHROPIC_API_KEY`, answers quote community replies. Replies are
  kept only if they are semantically close to the question, so memes and polls
  are dropped, but quoted replies are student opinions and can be wrong.
- Some Hinglish questions still retrieve chatty threads ("JEE ki preparation
  kaise karu" matches "Kaise karu face?"), and typo robustness is the weakest
  category.
- Most relevance labels come from one LLM assessor; a human review of a sample
  would make the numbers more trustworthy. User-usefulness needs real users.

## Project structure

```
app.py                 Flask web app: /api/chat, /api/search, /api/evaluation
chatbot.py             question routing, reply fetching, answer generation
ranker.py              BM25/BM25F scoring with query expansion
semantic_search.py     Sentence Transformer embeddings (cached) + cosine retrieval
rrf.py                 Reciprocal Rank Fusion
hybrid_search.py       BM25 + semantic fused with RRF
test_search_modes.py   checks BM25 / semantic / hybrid on English + Hinglish queries
indexer.py             inverted index builder + shared tokenizer
domain_dictionary.py   Word2Vec domain dictionary + expand_query()
nta_crawler.py         NTA notice crawler and PDF date extraction
evaluate.py            evaluation CLI: ranking, robustness, ablation, pooling, annotation, reports
eval_metrics.py        P@k, Recall, AP, RR, NDCG, Success@k on condensed rankings
eval_perf.py           latency percentiles, throughput, memory, index integrity + API checks
hses.py                HSES scoring (weights in evaluation/hses_config.json)
tests/                 unit tests on a synthetic corpus
clean_dataset.py, download_dataset.py
templates/, static/    web UI (chat.js = chatbot tab)
index/                 generated index + domain_dictionary.json
data/                  corpus, NTA crawl, comment cache (gitignored)
evaluation/            generated results and report
```
