"""
Domain-dictionary ablation: which way of building the dictionary helps search most?

Variants (all use the same seeds, categories, expansion rules and BM25F ranker):
  none        no query expansion
  tfidf       TF-IDF only: a seed's related terms are the terms whose TF-IDF document
              vectors are most similar to it (they occur in the same posts)
  embedding   Word2Vec only: nearest neighbours in embedding space
  hybrid      Word2Vec neighbours, filtered by TF-IDF salience, plus salient TF-IDF
              candidates attached to their nearest seed (domain_dictionary.py)

Each variant's setting is chosen on the tuning split; all reported numbers are on the test split.
Intrinsic quality: graded judgments of (seed, added term) pairs in
evaluation/dictionary_pairs.json (2 = same meaning, 1 = related, 0 = unrelated).

Usage:
    python eval_dictionary.py            # -> evaluation/dictionary_eval.json
    python eval_dictionary.py --pairs    # print unjudged (seed, term) pairs as "seed<TAB>term"
"""

import os
import json
import argparse

import numpy as np

import domain_dictionary as dd
import evaluate as ev
from eval_metrics import evaluate_ranking, mean

OUT_PATH = "evaluation/dictionary_eval.json"
PAIRS_PATH = "evaluation/dictionary_pairs.json"
TFIDF_THRESHOLDS = (0.1, 0.15, 0.2, 0.3)
HYBRID_GRID = [(top, cand) for top in (1000, 2000, 3000, 5000) for cand in (0, 1000, 2000)]
BM25_EXP = next(s for s in ev.SYSTEMS if s["id"] == "bm25f_eng_exp")
DEPLOYED = next(s for s in ev.SYSTEMS if s["id"] == ev.DEPLOYED)


def tfidf_cooccurrence_dictionary(sents, ranks, threshold, pool=5000):
    """TF-IDF-only dictionary: related terms by cosine of TF-IDF document vectors.
    Cosines live on a different scale than Word2Vec similarities, so [threshold, 1]
    is mapped onto the same [MIN_WEIGHT, 1] weight range."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(analyzer=lambda d: d, min_df=5, sublinear_tf=True)
    cols = normalize(vec.fit_transform(sents).T.tocsr())       # one row per term, unit length
    vocab = {str(t): i for i, t in enumerate(vec.get_feature_names_out())}
    salient = np.array([vocab[t] for t, r in ranks.items() if r < pool])
    terms, related = {}, {}
    for cat, seeds in dd.SEEDS.items():
        for seed in seeds:
            if seed in vocab:
                terms[seed] = {"category": cat, "seed": seed, "similarity": 1.0, "source": "seed"}
    names = {i: t for t, i in vocab.items()}
    for seed, e in list(terms.items()):
        if e["source"] != "seed":
            continue
        sims = (cols[salient] @ cols[vocab[seed]].T).toarray().ravel()
        best = []
        for j in np.argsort(-sims)[:dd.TOPN + 1]:
            t, c = names[salient[j]], float(sims[j])
            if t != seed and c >= threshold:
                w = dd.MIN_WEIGHT + (1 - dd.MIN_WEIGHT) * (c - threshold) / (1 - threshold)
                best.append((t, round(w, 3)))
        related[seed] = best[:dd.TOPN]
        for t, w in related[seed]:
            if t not in terms or terms[t]["similarity"] < w:
                terms[t] = {"category": e["category"], "seed": seed, "similarity": w, "source": "tfidf"}
    return {"terms": terms, "related": related}


def run(runner, dictionary, tfidf_top, queries, qrels, system):
    """Per-query metrics for one dictionary variant (None = expansion off)."""
    runner.engine.domain_dict = dictionary
    dd.TFIDF_TOP = tfidf_top
    rows = []
    for q in queries:
        if dictionary is None:
            sysdef = {**system, "kwargs": {**system["kwargs"], "expand": False}}
        else:
            sysdef = system
        res = runner.search(sysdef, q["query"], top_k=ev.DEPTH)
        ids = [r["doc_id"] for r in res["results"]]
        m = evaluate_ranking(ids, qrels.get(q["query_id"], {}))
        m["expanded"] = sorted(res.get("expanded_terms", {}))
        m["query_id"], m["query"] = q["query_id"], q["query"]
        rows.append(m)
    return rows


def summary(rows, base_rows=None):
    out = {k: round(mean(r[k if k not in ("map", "mrr") else {"map": "ap", "mrr": "rr"}[k]] for r in rows), 4)
           for k in ("ndcg@10", "p@5", "map", "mrr", "success@5", "judged@10")}
    out["queries"] = len(rows)
    out["queries_expanded"] = sum(bool(r["expanded"]) for r in rows)
    if base_rows:
        d = [r["ndcg@10"] - b["ndcg@10"] for r, b in zip(rows, base_rows)]
        out["helped"] = sum(x > 1e-9 for x in d)
        out["hurt"] = sum(x < -1e-9 for x in d)
        out["p_vs_none"] = round(ev.paired_randomization_p([r["ndcg@10"] for r in rows],
                                                           [b["ndcg@10"] for b in base_rows]), 4)
    return out


def seed_pairs(dictionary, tfidf_top):
    """The (trigger seed, added term) pairs this dictionary would put into queries."""
    dd.TFIDF_TOP = tfidf_top
    pairs = set()
    for seed in dictionary["related"]:
        e = dictionary["terms"].get(seed)
        if e and e["category"] in dd.EXPANDABLE:
            for t in dd.expand_query(seed.replace("_", " "), dictionary):
                pairs.add((seed, t))
    return sorted(pairs)


def pair_quality(pairs, judged):
    grades = [judged.get(f"{s}\t{t}") for s, t in pairs]
    known = [g for g in grades if g is not None]
    return {"pairs": len(pairs), "judged": len(known),
            "same_meaning": round(sum(g == 2 for g in known) / len(known), 3) if known else None,
            "related_or_better": round(sum(g >= 1 for g in known) / len(known), 3) if known else None,
            "unrelated": round(sum(g == 0 for g in known) / len(known), 3) if known else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true", help="print unjudged (seed, term) pairs")
    args = ap.parse_args()

    sents = dd.phrased_sentences()
    ranks = dd.tfidf_ranks(sents)
    wv = dd.load_or_train(sents)
    emb = dd.build(wv, ranks, n_candidates=0)
    hybrids = {cand: dd.build(wv, ranks, n_candidates=cand) for cand in {c for _, c in HYBRID_GRID}}
    tfidf_dicts = {t: tfidf_cooccurrence_dictionary(sents, ranks, t) for t in TFIDF_THRESHOLDS}
    judged = ev.load_json(PAIRS_PATH, {})

    if args.pairs:
        allp = set(seed_pairs(emb, None))
        for top, cand in HYBRID_GRID:
            allp |= set(seed_pairs(hybrids[cand], top))
        for d in tfidf_dicts.values():
            allp |= set(seed_pairs(d, None))
        for s, t in sorted(allp):
            if f"{s}\t{t}" not in judged:
                print(f"{s}\t{t}")
        return

    runner = ev.Runner(need_semantic=True)
    qrels = ev.load_qrels()
    tune, test = ev.load_queries("tune"), ev.load_queries("test")

    # ---- tuning (deployed hybrid system's NDCG@10 on the tuning split)
    tuning = {"tfidf": {}, "hybrid": {}}
    for t, d in tfidf_dicts.items():
        tuning["tfidf"][str(t)] = summary(run(runner, d, None, tune, qrels, DEPLOYED))["ndcg@10"]
    for top, cand in HYBRID_GRID:
        tuning["hybrid"][f"top{top}_cand{cand}"] = summary(run(runner, hybrids[cand], top, tune, qrels, DEPLOYED))["ndcg@10"]
    best_t = max(TFIDF_THRESHOLDS, key=lambda t: tuning["tfidf"][str(t)])
    best_h = max(HYBRID_GRID, key=lambda h: tuning["hybrid"][f"top{h[0]}_cand{h[1]}"])
    print("tuning:", json.dumps(tuning, indent=1), "\nchosen tfidf threshold", best_t, "hybrid", best_h)

    variants = {
        "none": (None, None),
        "tfidf": (tfidf_dicts[best_t], None),
        "embedding": (emb, None),
        "hybrid": (hybrids[best_h[1]], best_h[0]),
    }
    names = {"none": "No expansion", "tfidf": "TF-IDF only (co-occurrence)",
             "embedding": "Embeddings only (Word2Vec)", "hybrid": "Hybrid (Word2Vec + TF-IDF)"}

    results = {}
    for sysname, system in (("bm25", BM25_EXP), ("deployed", DEPLOYED)):
        base_test = run(runner, None, None, test, qrels, system)
        base_tune = run(runner, None, None, tune, qrels, system)
        results[sysname] = {}
        for v, (d, top) in variants.items():
            rows = base_test if d is None else run(runner, d, top, test, qrels, system)
            tune_rows = base_tune if d is None else run(runner, d, top, tune, qrels, system)
            results[sysname][v] = {"test": summary(rows, None if d is None else base_test),
                                   "tune": summary(tune_rows, None if d is None else base_tune)}
            if sysname == "bm25" and d is not None:
                results[sysname][v]["examples"] = [
                    {"query": r["query"], "added": r["expanded"], "ndcg@10_delta": round(r["ndcg@10"] - b["ndcg@10"], 3)}
                    for r, b in zip(rows, base_test) if r["expanded"]]

    dictionaries = {}
    for v, (d, top) in variants.items():
        if d is None:
            continue
        src = {}
        for e in d["terms"].values():
            src[e.get("source", "?")] = src.get(e.get("source", "?"), 0) + 1
        dictionaries[v] = {"terms": len(d["terms"]), "by_source": src,
                           "quality": pair_quality(seed_pairs(d, top), judged)}

    out = {"chosen": {"tfidf_threshold": best_t, "hybrid_tfidf_top": best_h[0], "hybrid_candidates": best_h[1]},
           "names": names, "tuning_ndcg@10": tuning, "retrieval": results, "dictionaries": dictionaries,
           "tfidf_vocab": len(ranks), "word2vec_vocab": len(wv)}
    ev.save_json(OUT_PATH, out)
    for sysname in results:
        for v in variants:
            print(sysname, v, results[sysname][v]["test"])
    print(json.dumps(dictionaries, indent=1))


if __name__ == "__main__":
    main()
