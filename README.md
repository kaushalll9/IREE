# IRE Project — JEE/NEET Search Engine

An Information Retrieval and Extraction project for searching
JEENEETards posts using an inverted index and BM25 ranking.

## Features
- Dataset cleaning and preprocessing
- Inverted index construction
- BM25-based document ranking
- Search interface
- Search evaluation

## Tech Stack
- Python
- BM25
- Inverted Index
- HTML, CSS, JavaScript

## Setup
Install the dependencies required by the project, then run
the dataset preparation, indexing, and application scripts.

The dataset and generated index files are not included in Git.
Run the relevant preparation scripts to generate them locally.

## Domain Dictionary
`python domain_dictionary.py` trains Word2Vec (with bigram phrases) on the
cleaned corpus and expands seed terms per category (exam, subject, topic,
institute, coaching, resource, admission, student) into their nearest
embedding neighbours, writing `index/domain_dictionary.json`.
`python domain_dictionary.py --query "pcm cutoff for nit"` shows query expansion.
