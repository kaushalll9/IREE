"""
Domain Dictionary (embedding-based) for the JEE/NEET assistant.

Trains Word2Vec on the Reddit corpus (with bigram phrase detection, so
"jee_mains", "josaa_counselling" become single tokens), then expands a small
set of hand-picked seed terms per category into their nearest neighbours in
embedding space. The result is a dictionary of domain terms, their category
and their related/synonymous terms (slang, abbreviations, misspellings).

Output: index/domain_dictionary.json
    {
      "terms":   {term: {"category": str, "seed": str, "similarity": float}},
      "related": {seed: [[term, similarity], ...]}
    }

Usage:
    pip install gensim
    python domain_dictionary.py                 # build
    python domain_dictionary.py --query "pcm cutoff for nit"   # demo expansion
"""

import os
import re
import json
import argparse

CORPUS = "data/cleaned_corpus.jsonl"
MODEL_PATH = "index/word2vec.model"
OUT_PATH = "index/domain_dictionary.json"

# Seed vocabulary per category; embeddings discover the rest.
SEEDS = {
    "exam": ["jee", "mains", "advanced", "neet", "bitsat", "boards", "cuet", "kvpy",
             "olympiad", "mhtcet", "wbjee", "comedk", "viteee"],
    "subject": ["physics", "chemistry", "maths", "biology", "botany", "zoology",
                "organic", "inorganic", "pcm", "pcb"],
    "topic": ["rotational", "electrostatics", "thermodynamics", "kinematics", "optics",
              "calculus", "integration", "probability", "matrices", "genetics",
              "electrochemistry", "equilibrium", "mole", "modern_physics",
              "coordinate", "vectors", "magnetism"],
    "institute": ["iit", "nit", "iiit", "aiims", "bits", "mbbs", "btech",
                  "cse", "iisc", "iiser", "ece"],
    "coaching": ["allen", "aakash", "fiitjee", "pw", "unacademy", "vedantu",
                 "resonance", "narayana", "byjus"],
    "resource": ["ncert", "hcv", "irodov", "cengage", "pyq", "pyqs", "mock", "dpp",
                 "module", "lectures"],
    "admission": ["counselling", "josaa", "csab", "cutoff", "rank", "percentile",
                  "admit_card", "registration", "syllabus", "marks", "result",
                  "answer_key", "nta", "reservation", "category"],
    "student": ["dropper", "drop", "repeater", "aspirant", "11th", "12th", "backlog"],
}

TOPN = 15
MIN_SIM = 0.55   # ponytail: single global threshold; tune per category if noisy
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def sentences(path=CORPUS):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                toks = TOKEN_RE.findall(r.get("full_text", "").lower())
                if toks:
                    yield toks


def train(path=CORPUS):
    from gensim.models import Word2Vec
    from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

    sents = list(sentences(path))
    print(f"[+] {len(sents):,} posts loaded")
    bigram = Phrases(sents, min_count=10, threshold=15, connector_words=ENGLISH_CONNECTOR_WORDS)
    sents = [bigram[s] for s in sents]
    model = Word2Vec(sents, vector_size=150, window=5, min_count=5, sg=1,
                     epochs=10, workers=os.cpu_count() or 4, seed=42)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[+] Word2Vec trained: vocab {len(model.wv):,}")
    return model.wv


def build(wv):
    terms, related = {}, {}
    for cat, seeds in SEEDS.items():
        for seed in seeds:
            if seed not in wv:
                print(f"    [skip] seed '{seed}' not in vocabulary")
                continue
            terms.setdefault(seed, {"category": cat, "seed": seed, "similarity": 1.0})
            neigh = [(t, round(float(s), 3)) for t, s in wv.most_similar(seed, topn=TOPN)
                     if s >= MIN_SIM]
            related[seed] = neigh
            for t, s in neigh:
                # A term near several seeds keeps its closest one.
                if t not in terms or terms[t]["similarity"] < s:
                    terms[t] = {"category": cat, "seed": seed, "similarity": s}
    return {"terms": terms, "related": related}


def expand_query(query, dictionary, k=3):
    """Append up to k embedding neighbours for each dictionary seed in the query."""
    toks = TOKEN_RE.findall(query.lower())
    extra = []
    for t in toks:
        seed = dictionary["terms"].get(t, {}).get("seed", t)
        for r, _ in dictionary["related"].get(seed, [])[:k]:
            if r not in toks and r not in extra:
                extra.append(r)
    return toks + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="show expansion of a query using the built dictionary")
    args = ap.parse_args()

    if args.query:
        with open(OUT_PATH, encoding="utf-8") as f:
            print(expand_query(args.query, json.load(f)))
        return

    d = build(train())
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    print(f"[✓] {len(d['terms']):,} domain terms across {len(SEEDS)} categories -> {OUT_PATH}")
    for seed in ("jee", "neet", "josaa", "pw", "dropper"):
        print(f"    {seed}: {[t for t, _ in d['related'].get(seed, [])[:8]]}")

    # self-check on a toy dictionary
    toy = {"terms": {"mains": {"seed": "mains"}}, "related": {"mains": [["jee_mains", 0.9]]}}
    assert expand_query("Mains cutoff", toy) == ["mains", "cutoff", "jee_mains"]


if __name__ == "__main__":
    main()
