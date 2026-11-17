"""
Search performance and system reliability measurements for the evaluation suite.
Everything here is measured live; nothing is estimated.
"""

import os
import sys
import json
import time
import random
import resource
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List

from eval_metrics import mean, percentile


# ------------------------------------------------------------------ performance
def measure_latency(search_fn: Callable[[str], dict], queries: List[str],
                    runs: int = 5, warmup: int = 1) -> dict:
    """Wall-clock latency per query over `runs` repetitions after `warmup` untimed passes."""
    for _ in range(warmup):
        for q in queries:
            search_fn(q)
    samples = []
    for _ in range(runs):
        for q in queries:
            t0 = time.perf_counter()
            search_fn(q)
            samples.append((time.perf_counter() - t0) * 1000)
    return {"n": len(samples), "runs": runs, "warmup_runs": warmup,
            "avg_ms": round(mean(samples), 2),
            **{f"p{p}_ms": round(percentile(samples, p), 2) for p in (50, 95, 99)},
            "max_ms": round(max(samples), 2)}


def measure_throughput(search_fn: Callable[[str], dict], queries: List[str],
                       workers=(1, 4), repeats: int = 2) -> List[dict]:
    """Queries/second when `w` threads drain the query list `repeats` times (in-process, GIL-bound)."""
    out = []
    for w in workers:
        work = queries * repeats
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=w) as pool:
            list(pool.map(search_fn, work))
        dt = time.perf_counter() - t0
        out.append({"workers": w, "queries": len(work), "seconds": round(dt, 2),
                    "qps": round(len(work) / dt, 1)})
    return out


def current_rss_mb() -> float:
    """Resident memory of this process (Linux /proc; peak RSS on macOS)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def index_footprint(index_dir: str = "index") -> dict:
    files = {}
    for name in sorted(os.listdir(index_dir)):
        p = os.path.join(index_dir, name)
        if os.path.isfile(p):
            files[name] = round(os.path.getsize(p) / 2**20, 2)
    stats = {}
    stats_path = os.path.join(index_dir, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
    lexical = ("postings.pkl.gz", "metadata.json.gz", "stats.json")
    return {
        "files_mb": files,
        "lexical_index_mb": round(sum(files.get(n, 0) for n in lexical), 2),
        "embeddings_mb": files.get("doc_embeddings.npy"),
        "total_mb": round(sum(files.values()), 2),
        "indexing_time_s": stats.get("build_time_seconds"),
        "total_docs": stats.get("total_docs"),
        "vocab_size": stats.get("vocab_size"),
    }


# ------------------------------------------------------------------ reliability
def _check(name, passed, detail=""):
    return {"check": name, "passed": bool(passed), "detail": detail}


def check_index_integrity(engine, semantic=None, corpus_path: str = "data/cleaned_corpus.jsonl",
                          sample: int = 300, seed: int = 0) -> List[dict]:
    """Structural checks of the inverted index (and embeddings, if given)."""
    from indexer import tokenize

    meta, postings, stats = engine.metadata, engine.postings, engine.stats
    n = len(meta)
    checks = [
        _check("stats.total_docs matches metadata", stats["total_docs"] == n, f"{stats['total_docs']} vs {n}"),
        _check("stats.vocab_size matches postings", stats["vocab_size"] == len(postings),
               f"{stats['vocab_size']} vs {len(postings)}"),
        _check("doc_int_id equals metadata position", all(m["doc_int_id"] == i for i, m in enumerate(meta))),
        _check("doc_ids unique and non-empty", len({m["doc_id"] for m in meta}) == n and all(m["doc_id"] for m in meta)),
        _check("every result has a reddit URL", all(m["url"].startswith("https://www.reddit.com/") for m in meta)),
    ]

    title_sum, body_sum = Counter(), Counter()
    bad_range = bad_tf = unsorted = empty = 0
    for plist in postings.values():
        if not plist:
            empty += 1
            continue
        prev = -1
        for doc, tft, tfb in plist:
            if not 0 <= doc < n:
                bad_range += 1
                continue
            if tft < 0 or tfb < 0 or tft + tfb == 0:
                bad_tf += 1
            if doc <= prev:
                unsorted += 1
            prev = doc
            title_sum[doc] += tft
            body_sum[doc] += tfb
    checks += [
        _check("no empty posting lists", empty == 0, f"{empty} empty"),
        _check("posting doc ids in range", bad_range == 0, f"{bad_range} out of range"),
        _check("term frequencies positive", bad_tf == 0, f"{bad_tf} bad"),
        _check("posting lists sorted by doc id", unsorted == 0, f"{unsorted} out of order"),
        _check("postings tf sums equal stored field lengths",
               all(title_sum[i] == m["len_title"] and body_sum[i] == m["len_body"] for i, m in enumerate(meta))),
    ]

    if os.path.exists(corpus_path):
        rng = random.Random(seed)
        picks = set(rng.sample(range(n), min(sample, n)))
        mismatched = []
        with open(corpus_path, encoding="utf-8") as f:
            i = 0
            for line in f:
                if not line.strip():
                    continue
                if i in picks:
                    rec = json.loads(line)
                    m = meta[i]
                    if (rec.get("doc_id") != m["doc_id"] or len(tokenize(rec.get("title", ""))) != m["len_title"]
                            or len(tokenize(rec.get("body", ""))) != m["len_body"]):
                        mismatched.append(m["doc_id"])
                i += 1
        checks.append(_check(f"re-tokenised sample of {len(picks)} docs matches index", not mismatched,
                             f"{len(mismatched)} mismatched"))

    if semantic is not None and semantic.embeddings is not None:
        import numpy as np
        emb = semantic.embeddings
        norms = np.linalg.norm(emb, axis=1)
        checks += [
            _check("one embedding per document", emb.shape[0] == n, f"{emb.shape[0]} vs {n}"),
            _check("embeddings L2-normalised", bool(np.allclose(norms, 1.0, atol=1e-3)),
                   f"norm range {norms.min():.4f}-{norms.max():.4f}"),
            _check("embeddings finite", bool(np.isfinite(emb).all())),
        ]
    return checks


def check_filters_and_api(client, metadata, modes=("bm25", "bm25f", "boosted", "semantic", "hybrid")) -> List[dict]:
    """Black-box checks through the Flask API: filters, result limits, expansion toggle, errors."""
    checks = []

    def get(**params):
        r = client.get("/api/search", query_string=params)
        return r.status_code, (r.get_json(silent=True) or {})

    flairs = Counter(m["flair"] for m in metadata).most_common(3)
    probe = "best book for physics"
    for mode in modes:
        for flair, _ in flairs[1:3]:
            code, data = get(q=probe, model=mode, flair=flair, top_k=10)
            res = data.get("results", [])
            checks.append(_check(f"[{mode}] flair filter '{flair}'",
                                 code == 200 and all(flair.lower() in r["flair"].lower() for r in res),
                                 f"{len(res)} results"))
        code, data = get(q=probe, model=mode, min_score=10, top_k=10)
        res = data.get("results", [])
        checks.append(_check(f"[{mode}] min-upvote filter (>=10)",
                             code == 200 and res and all(r["upvotes"] >= 10 for r in res), f"{len(res)} results"))
        for k in (5, 10, 20):
            code, data = get(q=probe, model=mode, top_k=k)
            res = data.get("results", [])
            ids = [r["doc_id"] for r in res]
            checks.append(_check(f"[{mode}] top_k={k} limit, unique, ranked",
                                 code == 200 and len(res) == k and len(set(ids)) == k
                                 and [r["rank"] for r in res] == list(range(1, k + 1))))
        code, data = get(q="", model=mode)
        checks.append(_check(f"[{mode}] empty query returns no results", code == 200 and data.get("results") == []))

    code_on, on = get(q="hall ticket", model="boosted", expand=1)
    code_off, off = get(q="hall ticket", model="boosted", expand=0)
    checks += [
        _check("expansion on adds dictionary terms", code_on == 200 and bool(on.get("expanded_terms")),
               str(on.get("expanded_terms"))),
        _check("expansion off adds no terms", code_off == 200 and not off.get("expanded_terms")),
        _check("invalid top_k is rejected cleanly (400)", get(q=probe, top_k="abc")[0] == 400),
        _check("invalid min_score is rejected cleanly (400)", get(q=probe, min_score="x")[0] == 400),
        _check("unknown model is rejected cleanly (400)", get(q=probe, model="nope")[0] == 400),
    ]
    r = client.post("/api/chat", json={})
    checks.append(_check("chat: missing question -> 400", r.status_code == 400))
    r = client.post("/api/chat", json={"question": "x" * 501})
    checks.append(_check("chat: over-long question -> 400", r.status_code == 400))
    r = client.post("/api/feedback", json={"solved": "maybe"})
    checks.append(_check("feedback: invalid payload -> 400", r.status_code == 400))
    return checks
