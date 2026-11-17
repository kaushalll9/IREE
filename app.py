"""
Phase 4: Flask Search Application & Evaluation Dashboard for IRE Project
Provides high-performance search API, flair filtering, and live IR evaluation metrics.
"""

import os
import json
from flask import Flask, render_template, request, jsonify
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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", 10))
    model = request.args.get("model", "hybrid")
    flair = request.args.get("flair", "").strip() or None
    min_score_str = request.args.get("min_score", "").strip()
    min_score = int(min_score_str) if min_score_str and min_score_str.isdigit() else None
    expand = request.args.get("expand", "1") != "0"   # domain-dictionary query expansion

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

@app.route("/api/evaluation")
def api_evaluation():
    eval_path = os.path.join("evaluation", "evaluation_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Evaluation results not found. Run evaluate.py first."}), 404

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
