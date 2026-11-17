"""
Runs the same queries through BM25, Semantic and Hybrid (via the Flask API) and checks:
BM25 output is identical to calling the existing engine directly, semantic/hybrid doc ids
are valid and unique, metadata/URLs match the index, and hybrid scores equal weighted RRF of the ranks.
Usage: python test_search_modes.py   (needs the index and embedding cache)
"""

from app import app, engine

QUERIES = [
    ("english", "How can I improve my JEE score?"),
    ("romanized hindi", "JEE ki preparation kaise karu?"),
    ("romanized hindi", "Chemistry ke liye best resources kya hain?"),
    ("code-mixed", "Maths ke important chapters batao"),
    ("code-mixed", "Physics mein numericals kaise solve kare?"),
    ("code-mixed", "JEE mains ke liye kitne mock tests dene chahiye?"),
    ("spelling variation", "orgainc chemsitry best book"),
    ("spelling variation", "physics wala vs unacadmy"),
    ("paraphrase", "my mock test marks are not increasing"),
    ("paraphrase", "score stuck in practice exams"),
]
TOP_K, RRF_K, SEM_W = 10, 60, 0.25  # must match hybrid_search defaults
BY_ID = {m["doc_id"]: m for m in engine.metadata}


def api(q, model):
    r = client.get("/api/search", query_string={"q": q, "model": model, "top_k": TOP_K})
    assert r.status_code == 200, r.status_code
    return r.get_json()["results"]


def check_metadata(results):
    ids = [r["doc_id"] for r in results]
    assert len(ids) == len(set(ids)), "duplicate documents"
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
    for r in results:
        m = BY_ID[r["doc_id"]]  # KeyError == invalid doc id
        assert (r["title"], r["url"], r["snippet"]) == (m["title"], m["url"], m["snippet"])


client = app.test_client()
for kind, q in QUERIES:
    bm25 = api(q, "boosted")
    direct = engine.search(query=q, top_k=TOP_K, expand=True)["results"]
    assert [(r["doc_id"], r["score"]) for r in bm25] == [(r["doc_id"], r["score"]) for r in direct], \
        "BM25 via API differs from the existing engine"
    sem, hyb = api(q, "semantic"), api(q, "hybrid")
    for res in (bm25, sem, hyb):
        check_metadata(res)
    assert len(sem) == TOP_K and all(r["method"] == "semantic" for r in sem)
    assert all(-1.0 <= r["score"] <= 1.0 for r in sem)
    assert [r["score"] for r in sem] == sorted((r["score"] for r in sem), reverse=True)
    for r in hyb:
        expect = sum(w / (RRF_K + rk) for w, rk in ((1.0, r["bm25_rank"]), (SEM_W, r["semantic_rank"])) if rk)
        assert abs(r["score"] - expect) < 1e-6, (r["score"], expect)
    assert [r["score"] for r in hyb] == sorted((r["score"] for r in hyb), reverse=True)

    print(f"\n[{kind}] {q}")
    for name, res in (("BM25", bm25), ("Semantic", sem), ("Hybrid", hyb)):
        print(f"  {name}:")
        for r in res[:3]:
            extra = f" (bm25 #{r['bm25_rank']}, sem #{r['semantic_rank']})" if name == "Hybrid" else ""
            print(f"    {r['rank']}. {r['score']:<9} {r['title'][:70]}{extra}")

print(f"\nAll checks passed for {len(QUERIES)} queries x 3 modes.")
