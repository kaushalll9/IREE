"""
Hybrid retrieval: the existing BM25 engine + semantic search, fused with RRF.
BM25 is called as-is (ranker.BM25SearchEngine.search); nothing about it is changed here.
"""

import time
from typing import Any, Dict, Optional

from rrf import rrf_fuse


def hybrid_search(bm25_engine, semantic_engine, query: str, top_k: int = 10,
                  candidate_k: int = 100, rrf_k: int = 60, semantic_weight: float = 0.25,
                  filter_flair: Optional[str] = None,
                  min_score: Optional[int] = None, expand: bool = True,
                  bm25_kwargs: Optional[dict] = None) -> Dict[str, Any]:
    # semantic_weight 0.25 = best NDCG@10 on the tuning split (python evaluate.py tune); 1.0 is plain RRF.
    start = time.time()
    depth = max(candidate_k, top_k)
    bm25_res = bm25_engine.search(query=query, top_k=depth, filter_flair=filter_flair,
                                  min_score=min_score, expand=expand, **(bm25_kwargs or {}))
    sem_res = semantic_engine.search(query, top_k=depth, filter_flair=filter_flair,
                                     min_score=min_score)

    fused = rrf_fuse({"bm25": bm25_res["results"], "semantic": sem_res["results"]}, k=rrf_k,
                     weights={"bm25": 1.0, "semantic": semantic_weight})
    results = []
    for rank, doc in enumerate(fused[:top_k], start=1):
        doc.pop("matched_terms", None)
        doc.update(rank=rank, score=round(doc["score"], 6), method="hybrid")
        results.append(doc)

    return {
        "query": query,
        "stems": bm25_res["stems"],
        "expanded_terms": bm25_res.get("expanded_terms", {}),
        "total_matches": len(fused),
        "latency_ms": round((time.time() - start) * 1000, 2),
        "results": results,
    }
