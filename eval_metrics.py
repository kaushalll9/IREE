"""
Ranking metrics over graded qrels (0 = irrelevant, 1 = partially, 2 = highly relevant).

Unjudged documents are never treated as irrelevant: every metric is computed on the
*condensed* ranking (unjudged docs removed, Sakai 2007), and judged@k reports how much
of the raw top-k was judged so low-coverage numbers can be spotted.
"""

import math
from typing import Dict, List, Optional


def compute_dcg(relevances: List[int], k: int) -> float:
    """DCG@k with exponential gain (2^rel - 1) / log2(rank + 1)."""
    return sum((2 ** rel - 1) / math.log2(i + 1)
               for i, rel in enumerate(relevances[:k], start=1) if rel > 0)


def compute_ndcg(relevances: List[int], ideal_relevances: List[int], k: int) -> float:
    idcg = compute_dcg(sorted(ideal_relevances, reverse=True), k)
    return compute_dcg(relevances, k) / idcg if idcg else 0.0


def condense(ranked_ids: List[str], qrels: Dict[str, int]) -> List[str]:
    """Drop unjudged docs, keeping order."""
    return [d for d in ranked_ids if d in qrels]


def judged_at(ranked_ids: List[str], qrels: Dict[str, int], k: int) -> Optional[float]:
    top = ranked_ids[:k]
    return sum(d in qrels for d in top) / len(top) if top else None


def evaluate_ranking(ranked_ids: List[str], qrels: Dict[str, int], k: int = 10) -> Dict[str, Optional[float]]:
    """
    All per-query metrics for one ranking. Relevant = grade >= 1.
    Returns None for metrics that are undefined (no relevant docs judged for this query).
    """
    cond = condense(ranked_ids, qrels)
    graded = [qrels[d] for d in cond]
    binary = [1 if g > 0 else 0 for g in graded]
    n_rel = sum(1 for g in qrels.values() if g > 0)

    def p_at(n):
        return sum(binary[:n]) / n

    hits, ap_sum, rr = 0, 0.0, 0.0
    for rank, rel in enumerate(binary[:k], start=1):
        if rel:
            hits += 1
            ap_sum += hits / rank
            rr = rr or 1.0 / rank

    defined = n_rel > 0
    return {
        "p@1": p_at(1), "p@3": p_at(3), "p@5": p_at(5), "p@10": p_at(10),
        "recall@10": sum(binary[:10]) / n_rel if defined else None,
        "success@5": float(any(binary[:5])),
        "rr": rr,
        "ap": ap_sum / n_rel if defined else None,   # AP@k normalised by all known relevant docs
        "ndcg@5": compute_ndcg(graded, list(qrels.values()), 5),
        "ndcg@10": compute_ndcg(graded, list(qrels.values()), 10),
        "judged@10": judged_at(ranked_ids, qrels, 10),
        "num_results": len(ranked_ids),
    }


def mean(values) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(values: List[float], pct: float) -> Optional[float]:
    """Linear-interpolated percentile (same as numpy's default)."""
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * pct / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
