"""
Phase 4: Flask Search Application & Evaluation Dashboard for IRE Project
Provides high-performance search API, flair filtering, and live IR evaluation metrics.
"""

import os
import json
import time
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
import evaluate
from hses import compute_hses, load_config
from ranker import BM25SearchEngine
from chatbot import Chatbot
from semantic_search import SemanticSearchEngine
from hybrid_search import hybrid_search

app = Flask(__name__)

# Initialize search engine in memory
engine = BM25SearchEngine(index_dir="index")
engine.load_index()
semantic = SemanticSearchEngine(engine, index_dir="index")
semantic.load()  # loads cached embeddings; rebuilds only if corpus or model changed
bot = Chatbot(engine=engine, semantic=semantic)

# Pre-extract available flairs for the filter dropdown
FLAIR_COUNTS = {}
with open("data/corpus_summary.json", "r", encoding="utf-8") as f:
    corpus_summary = json.load(f)
    top_flairs = corpus_summary.get("top_flairs", {})
    FLAIR_COUNTS = top_flairs

SEARCH_MODELS = {"hybrid", "semantic", "boosted", "bm25f", "bm25"}
FEEDBACK_LOCK = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    model = request.args.get("model", "hybrid")
    flair = request.args.get("flair", "").strip() or None
    min_score_str = request.args.get("min_score", "").strip()
    expand = request.args.get("expand", "1") != "0"   # domain-dictionary query expansion
    try:
        top_k = int(request.args.get("top_k", 10))
        min_score = int(min_score_str) if min_score_str else None
    except ValueError:
        return jsonify({"error": "top_k and min_score must be integers"}), 400
    if not 1 <= top_k <= 100:
        return jsonify({"error": "top_k must be between 1 and 100"}), 400
    if model not in SEARCH_MODELS:
        return jsonify({"error": f"model must be one of {sorted(SEARCH_MODELS)}"}), 400

    if not query:
        return jsonify({
            "query": "",
            "stems": [],
            "expanded_terms": {},
            "total_matches": 0,
            "latency_ms": 0.0,
            "results": []
        })

    # Model configuration switch
    if model == "semantic":
        res = semantic.search(query, top_k=top_k, filter_flair=flair, min_score=min_score)
    elif model == "hybrid":
        # Default: existing BM25F + engagement config fused with semantic ranks via RRF
        res = hybrid_search(engine, semantic, query, top_k=top_k, filter_flair=flair,
                            min_score=min_score, expand=expand)
    elif model == "bm25":
        # Standard unfielded BM25
        res = engine.search(
            query=query,
            top_k=top_k,
            k1=1.2,
            w_title=1.0,
            w_body=1.0,
            use_engagement_boost=False,
            filter_flair=flair,
            min_score=min_score,
            expand=expand
        )
    elif model == "bm25f":
        # Field-weighted BM25F
        res = engine.search(
            query=query,
            top_k=top_k,
            k1=1.2,
            w_title=2.5,
            w_body=1.0,
            use_engagement_boost=False,
            filter_flair=flair,
            min_score=min_score,
            expand=expand
        )
    else:
        # Default: Proposed BM25F + Engagement Boost
        res = engine.search(
            query=query,
            top_k=top_k,
            k1=1.2,
            w_title=2.5,
            w_body=1.0,
            use_engagement_boost=True,
            filter_flair=flair,
            min_score=min_score,
            expand=expand
        )

    for r in res["results"]:
        r.setdefault("method", model)
    res["active_model"] = model
    return jsonify(res)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    question = ((request.get_json(silent=True) or {}).get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    if len(question) > 500:
        return jsonify({"error": "question too long (max 500 characters)"}), 400
    return jsonify(bot.ask(question))

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Chatbot answer feedback: solved (bool) and/or rating (1-5). Feeds the HSES usefulness score."""
    data = request.get_json(silent=True) or {}
    solved, rating = data.get("solved"), data.get("rating")
    if solved is not None and not isinstance(solved, bool):
        return jsonify({"error": "solved must be true or false"}), 400
    if rating is not None and (not isinstance(rating, int) or isinstance(rating, bool) or not 1 <= rating <= 5):
        return jsonify({"error": "rating must be an integer 1-5"}), 400
    if solved is None and rating is None:
        return jsonify({"error": "send solved and/or rating"}), 400
    row = {"ts": time.time(), "question": str(data.get("question", ""))[:500],
           "mode": str(data.get("mode", ""))[:20], "solved": solved, "rating": rating}
    with FEEDBACK_LOCK, open(evaluate.FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return jsonify({"ok": True})

@app.route("/api/evaluation")
def api_evaluation():
    data = evaluate.load_json(evaluate.RESULTS_PATH)
    if data is None:
        return jsonify({"error": "Evaluation results not found. Run: python evaluate.py"}), 404
    # Usefulness comes from live feedback, so refresh it (and the HSES it feeds) on every request.
    cfg = load_config()
    data["usefulness"] = evaluate.run_usefulness(cfg)
    data["hses"] = {s["id"]: compute_hses(evaluate.hses_inputs(data, s["id"], cfg), cfg["weights"])
                    for s in evaluate.SYSTEMS}
    rk = data.get("ranking")
    if rk:  # keep only the deployed system's per-query rows; the full set is in the CSV export
        rk["per_query"] = [{k: v for k, v in r.items() if k != "top10"}
                           for r in rk["per_query"][evaluate.DEPLOYED]]
    return jsonify(data)

EXPORTS = {"evaluation_results.json", "model_comparison.csv", "per_query.csv", "evaluation_report.md"}

@app.route("/api/evaluation/export/<name>")
def api_evaluation_export(name):
    if name not in EXPORTS or not os.path.exists(os.path.join(evaluate.EVAL_DIR, name)):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(os.path.abspath(evaluate.EVAL_DIR), name, as_attachment=True)

@app.route("/api/meta")
def api_meta():
    stats_path = os.path.join("index", "stats.json")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
            
    return jsonify({
        "stats": stats,
        "flairs": FLAIR_COUNTS
    })

if __name__ == "__main__":
    print("[+] Starting IRE Search Web App on http://127.0.0.1:5001 ...")
    app.run(host="127.0.0.1", port=5001, debug=False)
