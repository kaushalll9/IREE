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
        │ BM25F search over 43k posts │                 │           │
        │ (indexer.py + ranker.py)    │                 │           │
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
python nta_crawler.py                       # -> data/nta_updates.json         (~1 min)

python app.py                               # http://127.0.0.1:5001
```

The web app has three tabs: **Ask Assistant** (the chatbot), **Search Engine**
(raw ranked results) and **Evaluation** (benchmark results).

Optional: `export ANTHROPIC_API_KEY=...` before `python app.py` and Claude
writes each answer from the retrieved sources instead of quoting them. Without
a key everything still works.

Command-line versions: `python chatbot.py "hcv or irodov"`,
`python search.py "physics wallah vs unacademy"`, `python evaluate.py`.

## How it works (for IR newcomers)

| Step | File | What it does | IR concept |
|---|---|---|---|
| 1 | `download_dataset.py` | Downloads ~47k r/JEENEETards posts (2020–2022) | Corpus collection |
| 2 | `clean_dataset.py` | Strips markdown/URLs, drops deleted and bot posts → 43,116 docs | Preprocessing |
| 3 | `indexer.py` | Tokenises, removes stopwords, Porter-stems, and builds a map *term → list of (doc, title tf, body tf)* | Inverted index |
| 4 | `ranker.py` | Scores docs with BM25F: rare terms count more (IDF), repeated terms saturate (k1), long posts are normalised (b), titles weigh 2.5×; plus a small upvote boost | Ranking |
| 5 | `domain_dictionary.py` | Trains Word2Vec on the corpus and groups slang/abbreviations around seed terms (`hall ticket ≈ admit card`, `pw ≈ physics wallah`); search adds these at reduced weight | Embeddings, query expansion |
| 6 | `nta_crawler.py` | Crawls nta.ac.in (robots.txt respected), classifies notices by exam/type, extracts publish dates and the date lines from PDFs | Web crawling, information extraction |
| 7 | `chatbot.py` | Routes the question, retrieves, fetches top replies, builds a cited answer | Retrieval-augmented QA |
| 8 | `evaluate.py` | P@k, MAP, MRR, NDCG@10 over 10 queries with pooled relevance judgments | Evaluation |

**Why fetch comments?** The dataset contains posts only, but on Reddit the
answers are in the replies. For the top threads, the chatbot fetches replies
from the Arctic Shift API (cached in `data/comments_cache/`). It keeps the
top-voted direct answers and drops chatter (counter-questions, "same here",
replies nobody upvoted).

## Results

10 benchmark queries; judgments pooled from every configuration's top 10
(`python evaluate.py --pool`):

| Configuration | P@1 | P@5 | MAP | NDCG@10 |
|---|---|---|---|---|
| Standard BM25 | 1.0 | 0.84 | **0.704** | 0.773 |
| BM25F (title 2.5×) | 0.8 | 0.84 | 0.683 | 0.769 |
| BM25F + engagement boost | 0.8 | 0.84 | 0.701 | 0.805 |
| + domain expansion (used by the app) | 0.8 | 0.82 | 0.695 | **0.807** |

Title weighting improves the graded ordering (NDCG) but sometimes ranks a
bodiless meme post whose title matches first. Expansion is neutral here because
these queries already use the posts' own words; it helps with mismatches like
"hall ticket" vs "admit card". With 10 queries, gaps of a few points are noise.
See `evaluation/evaluation_report.md`.

## Limitations

- The Reddit data ends in Aug 2022, so community answers about cutoffs and
  coaching may be outdated. Official dates come from the live NTA crawl;
  re-run `nta_crawler.py` to refresh.
- `jeemain.nta.nic.in` and `neet.nta.nic.in` reject non-browser crawlers, so
  notices are collected through `nta.ac.in`.
- Reply filtering and routing are keyword heuristics, and many replies are in
  Hinglish.
- The benchmark has only 10 queries, judged by one assessor.

## Project structure

```
app.py                 Flask web app: /api/chat, /api/search, /api/evaluation
chatbot.py             question routing, reply fetching, answer generation
ranker.py              BM25/BM25F scoring with query expansion
indexer.py             inverted index builder + shared tokenizer
domain_dictionary.py   Word2Vec domain dictionary + expand_query()
nta_crawler.py         NTA notice crawler and PDF date extraction
evaluate.py            benchmark queries, qrels, metrics, pooling
clean_dataset.py, download_dataset.py
templates/, static/    web UI (chat.js = chatbot tab)
index/                 generated index + domain_dictionary.json
data/                  corpus, NTA crawl, comment cache (gitignored)
evaluation/            generated results and report
```
