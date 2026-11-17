"""
Domain Dictionary (hybrid: TF-IDF + embeddings) for the JEE/NEET assistant.

1. Phrase detection: bigrams that co-occur far more than chance become single
   tokens ("jee_mains", "physics_wallah", "hall_ticket").
2. TF-IDF (statistical): every term gets a corpus salience rank, the sum of its
   tf-idf weights over all posts. This says which terms MATTER in the corpus.
3. Word2Vec (embeddings): a small set of hand-picked seed terms per category is
   expanded into its nearest neighbours. This says which terms MEAN THE SAME.
4. Hybrid:
   - TF-IDF candidates: salient terms close to a seed join that seed's category
     (finds "alakh_pandey", "mathongo", "competishun" that aren't top neighbours).
   - TF-IDF filter: a neighbour is only added to a query if it is also salient;
     rare neighbours ("hbcse", "aptitude_test") were the main source of drift.

Output: index/domain_dictionary.json
    {
      "terms":   {term: {"category", "seed", "similarity", "tfidf_rank", "source"}},
      "related": {seed: [[term, similarity], ...]}
    }

Usage:
    pip install gensim scikit-learn
    python domain_dictionary.py                 # build (reuses index/word2vec.model if present)
    python domain_dictionary.py --retrain       # retrain Word2Vec too
    python domain_dictionary.py --query "pcm cutoff for nit"   # demo expansion
"""

import os
import re
import json
import argparse

import numpy as np

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

SEED_TERMS = {s for seeds in SEEDS.values() for s in seeds}
TOPN = 15
MIN_SIM = 0.55   # ponytail: single global threshold; tune per category if noisy
# Categories whose neighbours are synonyms/variants. In subject/topic/institute
# the neighbours are siblings (chemistry -> physics, iit -> nit), which would drift the query.
EXPANDABLE = {"exam", "coaching", "resource", "admission", "student"}
MIN_WEIGHT = 0.6
# TF-IDF settings, chosen on the tuning split (python domain_dictionary.py --tune).
TFIDF_TOP = 3000         # expansion terms must rank in the top N by TF-IDF salience (None = no filter)
TFIDF_CANDIDATES = 0     # salient top-N terms near a seed are added to the dictionary (0 = off)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def sentences(path=CORPUS):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                toks = TOKEN_RE.findall(r.get("full_text", "").lower())
                if toks:
                    yield toks


def phrased_sentences(path=CORPUS):
    """Tokenised posts with frequent collocations merged ("physics wallah" -> "physics_wallah")."""
    from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
    sents = list(sentences(path))
    print(f"[+] {len(sents):,} posts loaded")
    bigram = Phrases(sents, min_count=10, threshold=15, connector_words=ENGLISH_CONNECTOR_WORDS)
    return [bigram[s] for s in sents]


def tfidf_ranks(sents):
    """Corpus-level TF-IDF salience: each term's L2-normalised, sublinear tf-idf summed over
    all posts, as a rank (0 = most salient). Terms in fewer than 5 posts get no rank."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer=lambda d: d, min_df=5, sublinear_tf=True)
    score = np.asarray(vec.fit_transform(sents).sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    return {str(terms[i]): r for r, i in enumerate(np.argsort(-score))}


def train(sents):
    from gensim.models import Word2Vec
    model = Word2Vec(sents, vector_size=150, window=5, min_count=5, sg=1,
                     epochs=10, workers=os.cpu_count() or 4, seed=42)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[+] Word2Vec trained: vocab {len(model.wv):,}")
    return model.wv


def load_or_train(sents, retrain=False):
    if not retrain and os.path.exists(MODEL_PATH):
        from gensim.models import Word2Vec
        print(f"[+] Reusing {MODEL_PATH} (--retrain to rebuild)")
        return Word2Vec.load(MODEL_PATH).wv
    return train(sents)


def build(wv, ranks=None, n_candidates=None):
    ranks = ranks or {}
    n_candidates = TFIDF_CANDIDATES if n_candidates is None else n_candidates
    # Seeds first, so a seed is never remapped as another seed's neighbour.
    terms = {seed: {"category": cat, "seed": seed, "similarity": 1.0, "source": "seed"}
             for cat, seeds in SEEDS.items() for seed in seeds if seed in wv}
    related = {}
    for cat, seeds in SEEDS.items():
        for seed in seeds:
            if seed not in wv:
                print(f"    [skip] seed '{seed}' not in vocabulary")
                continue
            neigh = [(t, round(float(s), 3)) for t, s in wv.most_similar(seed, topn=TOPN)
                     if s >= MIN_SIM]
            related[seed] = neigh
            for t, s in neigh:
                # A term near several seeds keeps its closest one.
                if t not in terms or terms[t]["similarity"] < s:
                    terms[t] = {"category": cat, "seed": seed, "similarity": s, "source": "embedding"}

    # TF-IDF candidates: salient terms that aren't a seed's top neighbour but are close to one.
    if n_candidates:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        seed_list = [(cat, seed) for cat, seeds in SEEDS.items() for seed in seeds if seed in wv]
        seed_vecs = np.stack([wv.get_vector(seed, norm=True) for _, seed in seed_list])
        for t, r in ranks.items():
            if (r >= n_candidates or t in terms or t in ENGLISH_STOP_WORDS or t not in wv
                    or any(c.isdigit() for c in t)):
                continue
            sims = seed_vecs @ wv.get_vector(t, norm=True)
            i = int(sims.argmax())
            if sims[i] >= MIN_SIM:
                cat, seed = seed_list[i]
                terms[t] = {"category": cat, "seed": seed, "similarity": round(float(sims[i]), 3), "source": "tfidf"}

    for t, e in terms.items():
        e["tfidf_rank"] = ranks.get(t)
    return {"terms": terms, "related": related}


def load_dictionary(path=OUT_PATH):
    """The built dictionary, or None if domain_dictionary.py hasn't been run."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_query(query, dictionary, k=3, tfidf_top=None):
    """
    Expansion terms for a query: {phrase: weight}, weight in (0, 1].
    Every query unigram/bigram found in the dictionary pulls in its seed and the
    seed's top-k embedding neighbours, weighted by cosine similarity.
    Phrases come back space-separated ("physics wallah"), ready for tokenize().
    With a TF-IDF filter, added terms must also rank in the top `tfidf_top` by salience.
    """
    tfidf_top = TFIDF_TOP if tfidf_top is None else tfidf_top
    toks = TOKEN_RE.findall(query.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    present = set(toks) | set(bigrams)
    # A matched bigram consumes its unigrams ("physics wallah" is not "physics").
    used = {t for bg in bigrams if bg in dictionary["terms"] for t in bg.split("_")}
    out = {}
    for g in bigrams + [t for t in toks if t not in used]:
        entry = dictionary["terms"].get(g)
        if not entry or entry["category"] not in EXPANDABLE:
            continue
        seed, w_seed = entry["seed"], entry["similarity"]
        for term, sim in [(seed, w_seed)] + [(t, s * w_seed) for t, s in dictionary["related"].get(seed, [])[:k]]:
            phrase = term.replace("_", " ")
            sibling = term in SEED_TERMS and term != seed   # allen -> aakash is drift
            noisy = any(c.isdigit() for c in term)            # 2024tard, 40_marks, 5k
            rank = dictionary["terms"].get(term, {}).get("tfidf_rank")
            rare = bool(tfidf_top) and (rank is None or rank >= tfidf_top)   # TF-IDF says: not salient
            if term in present or sibling or noisy or rare or sim < MIN_WEIGHT:
                continue
            out[phrase] = max(out.get(phrase, 0), round(sim, 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="show expansion of a query using the built dictionary")
    ap.add_argument("--retrain", action="store_true", help="retrain Word2Vec instead of reusing the saved model")
    args = ap.parse_args()

    if args.query:
        print(expand_query(args.query, load_dictionary()))
        return

    sents = phrased_sentences()
    ranks = tfidf_ranks(sents)
    print(f"[+] TF-IDF salience computed for {len(ranks):,} terms")
    d = build(load_or_train(sents, args.retrain), ranks)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    by_source = {}
    for e in d["terms"].values():
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1
    print(f"[✓] {len(d['terms']):,} domain terms across {len(SEEDS)} categories {by_source} -> {OUT_PATH}")
    for seed in ("jee", "neet", "josaa", "pw", "dropper"):
        print(f"    {seed}: {[t for t, _ in d['related'].get(seed, [])[:8]]}")


def _selfcheck():
    toy = {"terms": {"mains": {"category": "exam", "seed": "mains", "similarity": 1.0},
                     "physics_wallah": {"category": "coaching", "seed": "pw", "similarity": 0.8},
                     "physics": {"category": "subject", "seed": "physics", "similarity": 1.0}},
           "related": {"mains": [["jee_mains", 0.9], ["cutoff", 0.7]], "pw": [["physics_wallah", 0.8]]}}
    assert expand_query("Mains cutoff", toy, tfidf_top=0) == {"jee mains": 0.9}
    assert expand_query("physics wallah", toy, tfidf_top=0) == {"pw": 0.8}
    toy["related"]["mains"].append(["neet", 0.95])                 # sibling seed
    assert expand_query("mains", toy, tfidf_top=0) == {"jee mains": 0.9}
    toy["terms"]["jee_mains"] = {"category": "exam", "seed": "mains", "similarity": 0.9, "tfidf_rank": 5000}
    assert expand_query("mains", toy, tfidf_top=1000) == {}             # TF-IDF: neighbour too rare
    toy["terms"]["jee_mains"]["tfidf_rank"] = 10
    assert expand_query("mains", toy, tfidf_top=1000) == {"jee mains": 0.9}
    ranks = tfidf_ranks([["pw", "lecture"], ["pw", "notes"], ["pw"], ["pw"], ["pw", "lecture"]] * 2)
    assert ranks["pw"] == 0 and "notes" not in ranks                     # min_df=5


if __name__ == "__main__":
    _selfcheck()
    main()
