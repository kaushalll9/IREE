"""
Hybrid Search Evaluation Score (HSES), 0-100.

HSES = 100 * sum(weight_m * score_m) over the metrics in evaluation/hses_config.json,
each score_m already normalised to [0, 1]. A metric that was not measured is None:
the score is then reported as incomplete, with a partial score over the measured weight,
never with a guessed value.
"""

import json
import os
from typing import Dict, Optional

CONFIG_PATH = os.path.join("evaluation", "hses_config.json")

CATEGORY_GROUPS = {          # query category -> robustness component
    "abbreviation": "abbreviation",
    "synonym": "synonym",
    "hinglish": "hinglish_typo",
    "typo": "hinglish_typo",
    "numerical": "numerical_comparison",
    "comparison": "numerical_comparison",
}


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    total = sum(cfg["weights"].values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"HSES weights must sum to 1.0, got {total}")
    return cfg


def linear_score(value: Optional[float], best: float, worst: float) -> Optional[float]:
    """1.0 at or better than `best`, 0.0 at or worse than `worst`, linear in between."""
    if value is None:
        return None
    if value <= best:
        return 1.0
    if value >= worst:
        return 0.0
    return (worst - value) / (worst - best)


def compute_hses(scores: Dict[str, Optional[float]], weights: Dict[str, float]) -> dict:
    """
    scores: metric -> normalised [0,1] value or None (not measured).
    Returns the score (None unless complete), the partial score over measured weight,
    each metric's weighted contribution, and what is missing.
    """
    contributions, missing, measured_weight = {}, [], 0.0
    for metric, w in weights.items():
        v = scores.get(metric)
        if v is None:
            missing.append(metric)
            contributions[metric] = None
            continue
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{metric} must be normalised to [0,1], got {v}")
        contributions[metric] = round(100 * w * v, 3)
        measured_weight += w
    earned = sum(c for c in contributions.values() if c is not None)
    complete = not missing
    return {
        "hses": round(earned, 2) if complete else None,
        "complete": complete,
        "partial_hses": round(earned / measured_weight, 2) if measured_weight else None,
        "measured_weight": round(measured_weight, 4),
        "missing": missing,
        "contributions": contributions,
        "scores": {m: (round(scores[m], 4) if scores.get(m) is not None else None) for m in weights},
    }
