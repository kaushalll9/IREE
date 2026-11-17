"""
IR evaluation suite and Hybrid Search Evaluation Score (HSES) for the JEE/NEET search engine.

Data (evaluation/):
  queries.json      80 benchmark queries: category, intent, split (tune / test)
  qrels.json        graded judgments {query_id: {doc_id: {"grade": 0|1|2, "assessor": ...}}}
  hses_config.json  HSES weights and normalisation targets
Outputs (evaluation/): evaluation_results.json, model_comparison.csv, per_query.csv,
  evaluation_report.md. User feedback for the usefulness score is read from data/feedback.jsonl.

Commands (python evaluate.py <cmd>):
  all          every section below, then HSES + reports (default)
  ranking      ranking metrics + ablation + robustness for all systems
  performance  latency percentiles, throughput, index size, memory
  reliability  index integrity + filter/API correctness checks
  usefulness   task completion + satisfaction from chatbot feedback
  report       recompute HSES and rewrite reports from saved sections
  tune         RRF k / candidate depth grid on the tuning split only
  pool         list unjudged pooled docs per query
  annotate     interactive human relevance judging of the pool
  import-judgments FILE --assessor NAME   bulk-load "query_id doc_id grade" lines
"""

import os
import csv
import json
import random
import argparse
import datetime
import platform
from collections import defaultdict
from typing import Dict, List

from eval_metrics import evaluate_ranking, mean
from hses import CATEGORY_GROUPS, compute_hses, linear_score, load_config

EVAL_DIR = "evaluation"
QUERIES_PATH = os.path.join(EVAL_DIR, "queries.json")
QRELS_PATH = os.path.join(EVAL_DIR, "qrels.json")
RESULTS_PATH = os.path.join(EVAL_DIR, "evaluation_results.json")
FEEDBACK_PATH = os.path.join("data", "feedback.jsonl")
CORPUS_PATH = os.path.join("data", "cleaned_corpus.jsonl")
NON_HUMAN_ASSESSORS = {"claude", "legacy"}
DEPTH = 20          # retrieve this deep so condensed rankings still have 10 judged docs
POOL_DEPTH = 10

_BM25F = {"k1": 1.2, "w_title": 2.5, "w_body": 1.0}
# Ablation: expansion is off in the first three so each step adds exactly one change.
SYSTEMS = [
    {"id": "bm25_equal", "kind": "bm25", "name": "BM25F, equal field weights (baseline)",
     "kwargs": {"k1": 1.2, "w_title": 1.0, "w_body": 1.0, "use_engagement_boost": False, "expand": False}},
    {"id": "bm25f", "kind": "bm25", "name": "BM25F (title 2.5x)",
     "kwargs": {**_BM25F, "use_engagement_boost": False, "expand": False}},
    {"id": "bm25f_eng", "kind": "bm25", "name": "BM25F + engagement boost",
     "kwargs": {**_BM25F, "use_engagement_boost": True, "expand": False}},
    {"id": "bm25f_eng_exp", "kind": "bm25", "name": "BM25F + engagement + Word2Vec expansion",
     "kwargs": {**_BM25F, "use_engagement_boost": True, "expand": True}},
    {"id": "semantic", "kind": "semantic", "name": "Semantic (multilingual MiniLM)", "kwargs": {}},
    {"id": "hybrid", "kind": "hybrid", "name": "Hybrid RRF (deployed: search default + chatbot)",
     "kwargs": {"expand": True}},
]
DEPLOYED = "hybrid"
BASELINE = "bm25_equal"


# ------------------------------------------------------------------ data
def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def load_queries(split=None) -> List[dict]:
    qs = load_json(QUERIES_PATH)
    return [q for q in qs if split in (None, "all", q["split"])]


def load_qrels() -> Dict[str, Dict[str, int]]:
    """Grades only. Human judgments override claude/legacy ones on save, so grades are final here."""
    raw = load_json(QRELS_PATH, {})
    return {qid: {d: j["grade"] for d, j in docs.items()} for qid, docs in raw.items()}


def add_judgment(raw, qid, doc_id, grade, assessor):
    if grade not in (0, 1, 2):
        raise ValueError(f"grade must be 0, 1 or 2, got {grade}")
    prev = raw.setdefault(qid, {}).get(doc_id)
    if prev and prev["assessor"] not in NON_HUMAN_ASSESSORS and assessor in NON_HUMAN_ASSESSORS:
        return False  # never let a machine label overwrite a human one
    raw[qid][doc_id] = {"grade": grade, "assessor": assessor}
    return True


# ------------------------------------------------------------------ engines
class Runner:
    """Loads the engines once and runs any SYSTEMS entry."""

    def __init__(self, need_semantic=True):
        from eval_perf import current_rss_mb as rss
        from ranker import BM25SearchEngine
        self.rss = {"start_mb": rss()}
        self.engine = BM25SearchEngine(index_dir="index")
        self.engine.load_index()
        self.rss["after_bm25_mb"] = rss()
        self.semantic = None
        if need_semantic:
            from semantic_search import SemanticSearchEngine
            self.semantic = SemanticSearchEngine(self.engine)
            self.semantic.load()
            self.rss["after_semantic_mb"] = rss()
            self.device = str(self.semantic.model.device)

    def search(self, system, query, top_k=DEPTH, **filters):
        if system["kind"] == "bm25":
            return self.engine.search(query=query, top_k=top_k, **system["kwargs"], **filters)
        if system["kind"] == "semantic":
            return self.semantic.search(query, top_k=top_k, **filters)
        from hybrid_search import hybrid_search
        return hybrid_search(self.engine, self.semantic, query, top_k=top_k, **system["kwargs"], **filters)


# ------------------------------------------------------------------ ranking
METRIC_KEYS = ["ndcg@10", "ndcg@5", "p@1", "p@5", "p@10", "recall@10", "success@5", "map", "mrr", "judged@10"]


def aggregate(per_query: List[dict]) -> dict:
    out = {k: mean(q[k if k not in ("map", "mrr") else {"map": "ap", "mrr": "rr"}[k]] for q in per_query)
           for k in METRIC_KEYS}
    out = {k: (round(v, 4) if v is not None else None) for k, v in out.items()}
    out["queries"] = len(per_query)
    out["zero_result_rate"] = round(mean(float(q["num_results"] == 0) for q in per_query) or 0, 4)
    out["failed_queries"] = [q["query_id"] for q in per_query if q["success@5"] == 0]
    out["avg_latency_ms"] = round(mean(q["latency_ms"] for q in per_query), 2)
    return out


def paired_randomization_p(a: List[float], b: List[float], trials: int = 10000, seed: int = 0):
    """Two-sided paired randomization test on the mean difference (Smucker et al. 2007)."""
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return None
    obs = abs(sum(diffs))
    rng = random.Random(seed)
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12 for _ in range(trials))
    return round((hits + 1) / (trials + 1), 4)


def run_ranking(runner: Runner, cfg: dict) -> dict:
    queries, qrels = load_queries(), load_qrels()
    per_system = {}
    for system in SYSTEMS:
        print(f"[+] Ranking: {system['name']}")
        rows = []
        for q in queries:
            res = runner.search(system, q["query"])
            ids = [r["doc_id"] for r in res["results"]]
            m = evaluate_ranking(ids, qrels.get(q["query_id"], {}))
            rows.append({"query_id": q["query_id"], "query": q["query"], "category": q["category"],
                         "split": q["split"], "latency_ms": res["latency_ms"], "top10": ids[:10],
                         "expanded_terms": res.get("expanded_terms", {}), **m})
        per_system[system["id"]] = rows

    summary = {}
    for system in SYSTEMS:
        rows = per_system[system["id"]]
        by_split = {s: aggregate([r for r in rows if r["split"] == s]) for s in ("tune", "test")}
        test = [r for r in rows if r["split"] == cfg["split"]]
        cats = defaultdict(list)
        for r in test:
            cats[r["category"]].append(r)
        groups = defaultdict(list)
        for r in test:
            if r["category"] in CATEGORY_GROUPS:
                groups[CATEGORY_GROUPS[r["category"]]].append(r)
        base = {r["query_id"]: r["ndcg@10"] for r in per_system[BASELINE] if r["split"] == cfg["split"]}
        mine = {r["query_id"]: r["ndcg@10"] for r in test}
        common = sorted(base)
        summary[system["id"]] = {
            "name": system["name"], "kind": system["kind"], "config": system["kwargs"],
            **by_split,
            "categories": {c: aggregate(rs) for c, rs in sorted(cats.items())},
            "robustness": {g: round(mean(r["success@5"] for r in rs), 4) for g, rs in sorted(groups.items())},
            "vs_baseline": {
                "wins": sum(mine[q] > base[q] + 1e-9 for q in common),
                "losses": sum(mine[q] < base[q] - 1e-9 for q in common),
                "ties": sum(abs(mine[q] - base[q]) <= 1e-9 for q in common),
                "ndcg@10_delta": round(mean(mine[q] - base[q] for q in common), 4),
                "p_value": None if system["id"] == BASELINE
                else paired_randomization_p([mine[q] for q in common], [base[q] for q in common]),
            },
        }
    return {"systems": summary, "per_query": per_system, "expansion": expansion_analysis(per_system, qrels, cfg)}


def expansion_analysis(per_system, qrels, cfg) -> dict:
    """Does Word2Vec expansion help? Compares systems 3 (off) and 4 (on) query by query."""
    off = {r["query_id"]: r for r in per_system["bm25f_eng"]}
    on = {r["query_id"]: r for r in per_system["bm25f_eng_exp"]}
    rows = []
    for qid, a in on.items():
        b = off[qid]
        if a["split"] != cfg["split"]:
            continue
        new_docs = [d for d in a["top10"] if d not in b["top10"]]
        q = qrels.get(qid, {})
        rows.append({
            "query_id": qid, "query": a["query"], "category": a["category"],
            "expanded": bool(a["expanded_terms"]), "terms": a["expanded_terms"],
            "ndcg@10_delta": round(a["ndcg@10"] - b["ndcg@10"], 4),
            "success@5_delta": a["success@5"] - b["success@5"],
            "overlap@10": round(len(set(a["top10"]) & set(b["top10"])) / max(len(b["top10"]), 1), 3),
            "new_docs": len(new_docs),
            "new_irrelevant": sum(q.get(d) == 0 for d in new_docs),   # drift: judged-irrelevant docs it pulled in
            "new_relevant": sum((q.get(d) or 0) > 0 for d in new_docs),
            "latency_delta_ms": round(a["latency_ms"] - b["latency_ms"], 2),
        })
    touched = [r for r in rows if r["expanded"]]
    return {
        "queries_expanded": len(touched), "queries_total": len(rows),
        "helped": sum(r["ndcg@10_delta"] > 0 for r in touched),
        "hurt": sum(r["ndcg@10_delta"] < 0 for r in touched),
        "mean_ndcg@10_delta_when_expanded": round(mean(r["ndcg@10_delta"] for r in touched) or 0, 4),
        "drift_new_irrelevant": sum(r["new_irrelevant"] for r in touched),
        "drift_new_relevant": sum(r["new_relevant"] for r in touched),
        "mean_latency_delta_ms": round(mean(r["latency_delta_ms"] for r in rows) or 0, 2),
        "per_query": rows,
    }


# ------------------------------------------------------------------ performance / reliability / usefulness
def run_performance(runner: Runner, cfg: dict) -> dict:
    from eval_perf import index_footprint, measure_latency, measure_throughput
    queries = [q["query"] for q in load_queries(cfg["split"])]
    latency = {}
    for system in SYSTEMS:
        print(f"[+] Latency: {system['name']}")
        latency[system["id"]] = measure_latency(lambda q, s=system: runner.search(s, q, top_k=10), queries,
                                                runs=cfg["latency_runs"], warmup=cfg["warmup_runs"])
    deployed = next(s for s in SYSTEMS if s["id"] == DEPLOYED)
    print("[+] Throughput (deployed system)")
    throughput = measure_throughput(lambda q: runner.search(deployed, q, top_k=10), queries)
    return {"latency": latency, "throughput": throughput, "index": index_footprint("index"),
            "memory_rss_mb": {k: round(v, 1) for k, v in runner.rss.items()},
            "device": getattr(runner, "device", "n/a"), "workload": {
                "queries": len(queries), "split": cfg["split"], "top_k": 10,
                "latency_runs": cfg["latency_runs"], "warmup_runs": cfg["warmup_runs"],
                "throughput": "each query list drained twice by N threads, in-process"}}


def run_reliability(runner: Runner) -> dict:
    from eval_perf import check_filters_and_api, check_index_integrity
    import app as webapp  # shares nothing with runner: exercises the real Flask app end to end
    integrity = check_index_integrity(runner.engine, runner.semantic, CORPUS_PATH)
    api = check_filters_and_api(webapp.app.test_client(), webapp.engine.metadata)
    return {"index_integrity": integrity, "filters_api": api}


def run_usefulness(cfg: dict) -> dict:
    rows = []
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    solved = [r["solved"] for r in rows if r.get("solved") is not None]
    ratings = [r["rating"] for r in rows if r.get("rating") is not None]
    enough = cfg["min_feedback"]
    return {
        "responses": len(rows), "min_required": enough,
        "task_completion": round(mean(float(s) for s in solved), 4) if len(solved) >= enough else None,
        "satisfaction": round((mean(ratings) - 1) / 4, 4) if len(ratings) >= enough else None,  # 1-5 -> 0-1
        "solved_votes": len(solved), "ratings": len(ratings),
        "mean_rating_1_5": round(mean(ratings), 2) if ratings else None,
        "recent_unsolved": [r.get("question", "") for r in rows if r.get("solved") is False][-10:],
    }


# ------------------------------------------------------------------ HSES
def hses_inputs(results: dict, system_id: str, cfg: dict) -> dict:
    """Normalised [0,1] inputs for one system; None where not measured (or not trustworthy)."""
    s = {m: None for m in cfg["weights"]}
    rk = results.get("ranking", {}).get("systems", {}).get(system_id)
    if rk:
        test = rk[cfg["split"]]
        if (test.get("judged@10") or 0) >= cfg["min_judged_at_10"]:
            s.update({"ndcg@10": test["ndcg@10"], "p@5": test["p@5"], "map": test["map"], "mrr": test["mrr"]})
            for g, v in rk["robustness"].items():
                s[f"robustness.{g}"] = v
    perf = results.get("performance")
    if perf and system_id in perf["latency"]:
        t = cfg["targets"]
        s["efficiency.p95_latency"] = linear_score(perf["latency"][system_id]["p95_ms"], **t["p95_latency_ms"])
        kind = next(x["kind"] for x in SYSTEMS if x["id"] == system_id)
        idx = perf["index"]
        disk = idx["lexical_index_mb"] + ((idx["embeddings_mb"] or 0) if kind != "bm25" else 0)
        rss = perf["memory_rss_mb"]
        mem = rss.get("after_bm25_mb") if kind == "bm25" else rss.get("after_semantic_mb")
        if mem is not None:
            s["efficiency.index_memory"] = (linear_score(disk, **t["index_disk_mb"])
                                            + linear_score(mem, **t["memory_mb"])) / 2
    rel = results.get("reliability")
    if rel:
        for key, name in (("correctness.index", "index_integrity"), ("correctness.filters", "filters_api")):
            checks = rel[name]
            s[key] = sum(c["passed"] for c in checks) / len(checks) if checks else None
    use = results.get("usefulness")
    if use and system_id == DEPLOYED:  # feedback is collected on the deployed chatbot only
        s["usefulness.task_completion"] = use["task_completion"]
        s["usefulness.satisfaction"] = use["satisfaction"]
    return s


def finalize(results: dict, cfg: dict) -> dict:
    results["hses"] = {sys_["id"]: compute_hses(hses_inputs(results, sys_["id"], cfg), cfg["weights"])
                       for sys_ in SYSTEMS}
    qrels_raw = load_json(QRELS_PATH, {})
    assessors = defaultdict(int)
    for docs in qrels_raw.values():
        for j in docs.values():
            assessors[j["assessor"]] += 1
    queries = load_queries()
    results["meta"] = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "deployed_system": DEPLOYED, "baseline_system": BASELINE,
        "dataset": {"name": "r/JEENEETards cleaned corpus", "docs": load_json("index/stats.json", {}).get("total_docs")},
        "queries": {"total": len(queries), **{s: sum(q["split"] == s for q in queries) for s in ("tune", "test")}},
        "judgments": {"total": sum(assessors.values()), "by_assessor": dict(assessors),
                      "queries_judged": sum(1 for q in queries if qrels_raw.get(q["query_id"]))},
        "config": cfg,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    return results


# ------------------------------------------------------------------ reports
def write_reports(results: dict):
    save_json(RESULTS_PATH, results)
    rk = results.get("ranking")
    split = results["meta"]["config"]["split"]
    if rk:
        with open(os.path.join(EVAL_DIR, "model_comparison.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["system", "split", *METRIC_KEYS, "zero_result_rate", "p95_latency_ms", "hses", "partial_hses"])
            for sid, s in rk["systems"].items():
                lat = results.get("performance", {}).get("latency", {}).get(sid, {})
                h = results["hses"][sid]
                for sp in ("tune", "test"):
                    w.writerow([s["name"], sp, *[s[sp][k] for k in METRIC_KEYS], s[sp]["zero_result_rate"],
                                lat.get("p95_ms"), h["hses"], h["partial_hses"]])
        with open(os.path.join(EVAL_DIR, "per_query.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            cols = ["query_id", "query", "category", "split", "ndcg@10", "p@5", "recall@10", "success@5",
                    "ap", "rr", "judged@10", "num_results", "latency_ms"]
            w.writerow(["system", *cols])
            for sid, rows in rk["per_query"].items():
                for r in rows:
                    w.writerow([sid, *[r[c] for c in cols]])
    with open(os.path.join(EVAL_DIR, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write(markdown_report(results, split))


def _fmt(v, nd=3):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def markdown_report(r: dict, split: str) -> str:
    m = r["meta"]
    dep = r["hses"][DEPLOYED]
    L = ["# Search Evaluation Report (HSES)", "",
         f"Generated {m['generated_at']} · {m['dataset']['docs']:,} docs · {m['queries']['total']} queries "
         f"({m['queries']['test']} test / {m['queries']['tune']} tuning) · "
         f"{m['judgments']['total']} judgments by assessor: "
         + ", ".join(f"{k} {v}" for k, v in m["judgments"]["by_assessor"].items()), "",
         "## Hybrid Search Evaluation Score (deployed system)", ""]
    if dep["complete"]:
        L.append(f"**HSES = {dep['hses']} / 100**")
    else:
        L.append(f"**HSES: incomplete.** Partial score {_fmt(dep['partial_hses'], 1)} / 100 over the "
                 f"{dep['measured_weight'] * 100:.0f}% of weight that was measured. Missing: "
                 + ", ".join(f"`{x}`" for x in dep["missing"]))
    L += ["", "| Component | Weight | Normalised score | Points |", "|---|---|---|---|"]
    w = m["config"]["weights"]
    for k in w:
        L.append(f"| {k} | {w[k] * 100:.0f}% | {_fmt(dep['scores'][k])} | {_fmt(dep['contributions'][k], 2)} |")

    rk = r.get("ranking")
    if rk:
        L += ["", f"## Model comparison and ablation ({split} split)", "",
              "| System | NDCG@10 | P@5 | MAP | MRR | Recall@10 | Success@5 | Judged@10 | P95 ms | ΔNDCG vs baseline (p) | Partial HSES |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for sid, s in rk["systems"].items():
            t, vb = s[split], s["vs_baseline"]
            lat = r.get("performance", {}).get("latency", {}).get(sid, {}).get("p95_ms")
            L.append(f"| {s['name']} | {_fmt(t['ndcg@10'])} | {_fmt(t['p@5'])} | {_fmt(t['map'])} | {_fmt(t['mrr'])} | "
                     f"{_fmt(t['recall@10'])} | {_fmt(t['success@5'])} | {_fmt(t['judged@10'], 2)} | {_fmt(lat, 1)} | "
                     f"{vb['ndcg@10_delta']:+.3f} ({_fmt(vb['p_value'])}) | {_fmt(r['hses'][sid]['partial_hses'], 1)} |")
        L += ["", "p = two-sided paired randomization test on per-query NDCG@10 against the baseline "
              "(10,000 permutations). Metrics use condensed rankings: unjudged documents are skipped, not "
              "counted as irrelevant; Judged@10 is the share of the raw top 10 that was judged.", "",
              "## Query robustness (Success@5 by category)", "",
              "| System | " + " | ".join(sorted(rk["systems"][DEPLOYED]["categories"])) + " |",
              "|---|" + "---|" * len(rk["systems"][DEPLOYED]["categories"])]
        for sid, s in rk["systems"].items():
            L.append(f"| {s['name']} | " + " | ".join(_fmt(s["categories"][c]["success@5"], 2)
                                                     for c in sorted(s["categories"])) + " |")
        e = rk["expansion"]
        L += ["", "## Word2Vec query expansion (system 3 vs 4)", "",
              f"Expansion fired on {e['queries_expanded']} of {e['queries_total']} {split} queries: "
              f"NDCG@10 rose on {e['helped']} and fell on {e['hurt']} (mean change "
              f"{e['mean_ndcg@10_delta_when_expanded']:+.4f} where it fired). Query drift: expansion added "
              f"{e['drift_new_irrelevant']} judged-irrelevant and {e['drift_new_relevant']} relevant documents "
              f"to top-10s. Mean extra latency: {e['mean_latency_delta_ms']:+.2f} ms.", ""]
        fails = [q for q in rk["per_query"][DEPLOYED] if q["split"] == split and q["success@5"] == 0]
        L += ["## Failed queries (deployed system, no relevant result in top 5)", ""]
        L += [f"- `{q['query']}` ({q['category']}; judged@10 {_fmt(q['judged@10'], 2)})" for q in fails] or ["- none"]

    perf = r.get("performance")
    if perf:
        L += ["", f"## Performance (device: {perf['device']})", "",
              "| System | avg ms | P50 | P95 | P99 |", "|---|---|---|---|---|"]
        names = {s["id"]: s["name"] for s in SYSTEMS}
        for sid, lat in perf["latency"].items():
            L.append(f"| {names[sid]} | {lat['avg_ms']} | {lat['p50_ms']} | {lat['p95_ms']} | {lat['p99_ms']} |")
        idx = perf["index"]
        L += ["", f"Index: lexical {idx['lexical_index_mb']} MB, embeddings {idx['embeddings_mb']} MB, "
              f"indexing time {idx['indexing_time_s']} s, vocabulary {idx['vocab_size']:,}. "
              f"Memory (RSS): {perf['memory_rss_mb']}. Throughput: "
              + ", ".join(f"{t['workers']} worker(s) {t['qps']} q/s" for t in perf["throughput"]) + "."]

    rel = r.get("reliability")
    if rel:
        L += ["", "## Reliability checks", ""]
        for name, checks in rel.items():
            ok = sum(c["passed"] for c in checks)
            L.append(f"**{name}: {ok}/{len(checks)} passed**")
            L += [f"- FAIL {c['check']} {c['detail']}" for c in checks if not c["passed"]]
    use = r.get("usefulness")
    if use:
        L += ["", "## User usefulness", "",
              f"{use['responses']} feedback responses ({use['solved_votes']} solved votes, {use['ratings']} ratings; "
              f"{use['min_required']} needed for a score). Task completion: {_fmt(use['task_completion'])}, "
              f"satisfaction: {_fmt(use['satisfaction'])} (mean rating {_fmt(use['mean_rating_1_5'])} / 5)."]
    L += ["", "## Method notes", "",
          "- Judgments are pooled from every system's top 10 and graded 0/1/2 against the query intent. The "
          "assessor of each label is recorded in `qrels.json`; `claude` labels are LLM judgments and should be "
          "spot-checked by a human (`python evaluate.py annotate --review`). Human labels always take precedence.",
          "- Tuning queries (the original 10, seen during development, plus a stratified sample) are excluded "
          "from every reported number; HSES uses the test split only.",
          "- Ranking metrics are omitted from HSES for a system whose Judged@10 is below "
          f"{m['config']['min_judged_at_10']}, rather than guessed."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ pooling / annotation / tuning
def build_pool(runner: Runner, depth: int = POOL_DEPTH) -> Dict[str, List[str]]:
    pool = {}
    for q in load_queries():
        docs = []
        for system in SYSTEMS:
            for r in runner.search(system, q["query"], top_k=depth)["results"]:
                if r["doc_id"] not in docs:
                    docs.append(r["doc_id"])
        pool[q["query_id"]] = docs
    return pool


def corpus_text() -> Dict[str, str]:
    out = {}
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["doc_id"]] = (rec.get("title", ""), rec.get("body", ""))
    return out


def cmd_pool(runner, args):
    pool, qrels = build_pool(runner, args.depth), load_qrels()
    texts = corpus_text() if args.show else {}
    todo = 0
    for q in load_queries():
        unj = [d for d in pool[q["query_id"]] if d not in qrels.get(q["query_id"], {})]
        todo += len(unj)
        if args.show and unj:
            print(f"\n## {q['query_id']} [{q['category']}] {q['query']!r} :: {q['intent']}")
            for d in unj:
                t, b = texts[d]
                print(f"  {d} | {t[:100]} | {' '.join(b.split())[:args.chars]}")
    print(f"\n{todo} unjudged pooled documents (depth {args.depth}).")


def cmd_annotate(runner, args):
    if not args.assessor or args.assessor in NON_HUMAN_ASSESSORS:
        raise SystemExit("--assessor <your name> is required (and must not be 'claude' or 'legacy')")
    raw, texts = load_json(QRELS_PATH, {}), corpus_text()
    pool = build_pool(runner, args.depth)
    for q in load_queries(args.split):
        for d in pool[q["query_id"]]:
            j = raw.get(q["query_id"], {}).get(d)
            if j and not (args.review and j["assessor"] in NON_HUMAN_ASSESSORS):
                continue
            t, b = texts[d]
            print("\n" + "=" * 70 + f"\nQuery: {q['query']}\nIntent: {q['intent']}\n" + "-" * 70)
            print(f"{t}\n\n{' '.join(b.split())[:700]}")
            if j:
                print(f"\n(current: {j['grade']} by {j['assessor']})")
            ans = input("Relevance [0 no / 1 partly / 2 highly / s skip / q quit]: ").strip().lower()
            if ans == "q":
                save_json(QRELS_PATH, raw)
                return
            if ans in ("0", "1", "2"):
                add_judgment(raw, q["query_id"], d, int(ans), args.assessor)
                save_json(QRELS_PATH, raw)


def cmd_import(args):
    raw, n = load_json(QRELS_PATH, {}), 0
    known = {q["query_id"] for q in load_queries()}
    with open(args.file, encoding="utf-8") as f:
        for line in f:
            parts = line.split("#")[0].split()
            if not parts:
                continue
            qid, doc, grade = parts[0], parts[1], int(parts[2])
            if qid not in known:
                raise ValueError(f"unknown query id {qid}")
            n += add_judgment(raw, qid, doc, grade, args.assessor)
    save_json(QRELS_PATH, raw)
    print(f"[✓] imported {n} judgments as '{args.assessor}'")


def cmd_tune(runner):
    """Grid over RRF k, candidate depth and semantic weight on the tuning split only. Reports; changes nothing."""
    from hybrid_search import hybrid_search
    queries, qrels = load_queries("tune"), load_qrels()
    rows = []
    for k in (10, 30, 60):
        for cand in (20, 50, 100):
            for w in (0.0, 0.25, 0.5, 0.75, 1.0):
                ms = [evaluate_ranking([r["doc_id"] for r in hybrid_search(
                    runner.engine, runner.semantic, q["query"], top_k=DEPTH, candidate_k=cand, rrf_k=k,
                    semantic_weight=w)["results"]], qrels.get(q["query_id"], {})) for q in queries]
                rows.append((mean(m["ndcg@10"] for m in ms), mean(m["p@5"] for m in ms),
                             mean(m["judged@10"] for m in ms), k, cand, w))
    print(f"{'rrf_k':>6} {'cand':>5} {'sem_w':>6} {'NDCG@10':>8} {'P@5':>6} {'judged@10':>9}  (tuning split, best first)")
    for ndcg, p5, judged, k, cand, w in sorted(rows, reverse=True)[:15]:
        print(f"{k:>6} {cand:>5} {w:>6} {ndcg:>8.4f} {p5:>6.3f} {judged:>9.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["all", "ranking", "performance", "reliability", "usefulness", "report",
                             "tune", "pool", "annotate", "import-judgments"])
    ap.add_argument("file", nargs="?", help="judgments file for import-judgments")
    ap.add_argument("--assessor", help="name recorded with judgments")
    ap.add_argument("--depth", type=int, default=POOL_DEPTH, help="pool depth per system")
    ap.add_argument("--split", default="all", choices=["all", "tune", "test"], help="queries to annotate")
    ap.add_argument("--review", action="store_true", help="annotate: also re-judge claude/legacy labels")
    ap.add_argument("--show", action="store_true", help="pool: print unjudged docs")
    ap.add_argument("--chars", type=int, default=200, help="pool --show: body characters")
    args = ap.parse_args()
    cfg = load_config()

    if args.cmd == "import-judgments":
        if not args.file or not args.assessor:
            raise SystemExit("usage: evaluate.py import-judgments FILE --assessor NAME")
        return cmd_import(args)

    results = load_json(RESULTS_PATH, {}) if args.cmd != "all" else {}
    results = {k: v for k, v in results.items() if k in ("ranking", "performance", "reliability", "usefulness")}
    runner = None if args.cmd in ("usefulness", "report") else Runner()

    if args.cmd == "pool":
        return cmd_pool(runner, args)
    if args.cmd == "annotate":
        return cmd_annotate(runner, args)
    if args.cmd == "tune":
        return cmd_tune(runner)
    if args.cmd in ("all", "ranking"):
        results["ranking"] = run_ranking(runner, cfg)
    if args.cmd in ("all", "performance"):
        results["performance"] = run_performance(runner, cfg)
    if args.cmd in ("all", "reliability"):
        results["reliability"] = run_reliability(runner)
    if args.cmd in ("all", "usefulness"):
        results["usefulness"] = run_usefulness(cfg)

    write_reports(finalize(results, cfg))
    h = results["hses"][DEPLOYED]
    print(f"\nHSES (deployed hybrid): {h['hses'] if h['complete'] else 'INCOMPLETE'}"
          f" | partial {h['partial_hses']} over {h['measured_weight'] * 100:.0f}% of weight")
    if h["missing"]:
        print("  missing:", ", ".join(h["missing"]))
    print(f"[✓] Reports written to {EVAL_DIR}/ (evaluation_results.json, model_comparison.csv, per_query.csv, "
          "evaluation_report.md)")


if __name__ == "__main__":
    main()
