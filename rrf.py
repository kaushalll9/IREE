"""Reciprocal Rank Fusion: RRF(d) = sum_i w_i / (k + rank_i(d)), ranks one-based (w_i = 1 is standard RRF)."""

from typing import Dict, List, Optional


def rrf_fuse(rankings: Dict[str, List[dict]], k: int = 60, weights: Optional[Dict[str, float]] = None) -> List[dict]:
    """
    rankings: {method_name: results ordered best-first}, each result a dict with "doc_id".
    Returns one dict per unique doc, sorted by RRF score desc, carrying the doc's original
    metadata plus "score" (RRF), "<method>_rank" (None if absent) and "<method>_score".
    """
    fused: Dict[str, dict] = {}
    for method, results in rankings.items():
        for rank, res in enumerate(results, start=1):
            doc = fused.get(res["doc_id"])
            if doc is None:
                doc = fused[res["doc_id"]] = {**res, "score": 0.0,
                                              **{f"{m}_rank": None for m in rankings}}
            doc["score"] += (weights or {}).get(method, 1.0) / (k + rank)
            doc[f"{method}_rank"] = rank
            doc[f"{method}_score"] = res.get("score")
    # Tie-break on best single rank so equal RRF scores order deterministically.
    return sorted(fused.values(), key=lambda d: (-d["score"], min(
        r for r in (d[f"{m}_rank"] for m in rankings) if r is not None)))


if __name__ == "__main__":
    bm25 = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]
    sem = [{"doc_id": "c"}, {"doc_id": "d"}, {"doc_id": "a"}]
    out = rrf_fuse({"bm25": bm25, "semantic": sem}, k=60)
    s = {d["doc_id"]: d["score"] for d in out}
    assert abs(s["a"] - (1 / 61 + 1 / 63)) < 1e-12   # in both lists: both contributions
    assert abs(s["c"] - (1 / 63 + 1 / 61)) < 1e-12
    assert abs(s["b"] - 1 / 62) < 1e-12 and abs(s["d"] - 1 / 62) < 1e-12  # single-list docs
    assert len(out) == len({d["doc_id"] for d in out}) == 4          # no duplicates
    assert [d["doc_id"] for d in out][:2] == ["a", "c"]               # tie broken by best rank
    assert out[0]["bm25_rank"] == 1 and out[0]["semantic_rank"] == 3
    assert next(d for d in out if d["doc_id"] == "d")["bm25_rank"] is None
    w = {d["doc_id"]: d["score"] for d in rrf_fuse({"bm25": bm25, "semantic": sem}, k=60, weights={"semantic": 0.5})}
    assert abs(w["a"] - (1 / 61 + 0.5 / 63)) < 1e-12 and abs(w["d"] - 0.5 / 62) < 1e-12  # weighted variant
    print("rrf ok:", [(d["doc_id"], round(d["score"], 5)) for d in out])
