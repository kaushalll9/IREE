"""
Unit tests for the evaluation framework on a tiny synthetic corpus (no real data, model or GPU needed).
Run: python -m unittest discover -s tests -v
"""

import os
import sys
import json
import math
import tempfile
import unittest
import contextlib
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_metrics import evaluate_ranking, percentile, compute_ndcg  # noqa: E402
from hses import compute_hses, linear_score, load_config  # noqa: E402
import evaluate  # noqa: E402

DOCS = [
    ("d1", "Best book for organic chemistry", "MS Chouhan is great for organic chemistry practice"),
    ("d2", "HC Verma physics", "HCV theory is enough for mains physics"),
    ("d3", "Organic chemistry lecture", "Which teacher explains organic reactions well"),
    ("d4", "Mock test score stuck", "My mock test marks are not improving, any tips"),
    ("d5", "Kota hostel food", "Hostel food in Kota is bad"),
    ("d6", "Physics wallah vs unacademy", "Which platform is better for physics"),
]


def build_synthetic_index(tmp):
    from indexer import build_index
    corpus = os.path.join(tmp, "corpus.jsonl")
    with open(corpus, "w", encoding="utf-8") as f:
        for i, (doc_id, title, body) in enumerate(DOCS):
            f.write(json.dumps({"doc_id": doc_id, "title": title, "body": body, "score": i, "num_comments": 1,
                                "flair": "Discussion" if i % 2 else "Help", "date": "2022-01-01",
                                "author": "a", "url": f"https://i.redd.it/{doc_id}.jpg",
                                "permalink": f"https://www.reddit.com/r/JEENEETards/comments/{doc_id}/x/"}) + "\n")
    index_dir = os.path.join(tmp, "index")
    with contextlib.redirect_stdout(io.StringIO()):
        build_index(corpus, index_dir)
    return corpus, index_dir


class MetricTests(unittest.TestCase):
    def test_hand_computed_metrics(self):
        qrels = {"a": 2, "b": 0, "c": 1, "z": 1}           # z relevant but never retrieved
        m = evaluate_ranking(["a", "b", "c", "d"], qrels)  # d unjudged -> skipped
        self.assertEqual(m["p@1"], 1.0)
        self.assertAlmostEqual(m["p@3"], 2 / 3)            # condensed: a, b, c
        self.assertEqual(m["rr"], 1.0)
        self.assertAlmostEqual(m["ap"], (1 / 1 + 2 / 3) / 3)
        self.assertAlmostEqual(m["recall@10"], 2 / 3)
        self.assertEqual(m["success@5"], 1.0)
        self.assertAlmostEqual(m["judged@10"], 3 / 4)
        dcg = 3 / math.log2(2) + 1 / math.log2(4)
        idcg = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(m["ndcg@10"], dcg / idcg)

    def test_unjudged_docs_are_not_counted_irrelevant(self):
        qrels = {"rel": 2}
        self.assertEqual(evaluate_ranking(["x", "y", "rel"], qrels)["rr"], 1.0)  # x, y unjudged

    def test_undefined_without_relevant_judgments(self):
        m = evaluate_ranking(["a"], {"a": 0})
        self.assertIsNone(m["ap"])
        self.assertIsNone(m["recall@10"])
        self.assertEqual(m["success@5"], 0.0)

    def test_percentile_and_ndcg_edges(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(percentile([5], 95), 5)
        self.assertIsNone(percentile([], 95))
        self.assertEqual(compute_ndcg([0, 0], [0], 10), 0.0)


class HsesTests(unittest.TestCase):
    W = {"a": 0.5, "b": 0.3, "c": 0.2}

    def test_complete_score(self):
        r = compute_hses({"a": 1.0, "b": 0.5, "c": 0.0}, self.W)
        self.assertTrue(r["complete"])
        self.assertAlmostEqual(r["hses"], 65.0)

    def test_missing_metric_marks_incomplete_never_guesses(self):
        r = compute_hses({"a": 1.0, "b": 0.5, "c": None}, self.W)
        self.assertFalse(r["complete"])
        self.assertIsNone(r["hses"])
        self.assertEqual(r["missing"], ["c"])
        self.assertAlmostEqual(r["partial_hses"], 65.0 / 0.8)

    def test_rejects_unnormalised_scores(self):
        with self.assertRaises(ValueError):
            compute_hses({"a": 2.0, "b": 0, "c": 0}, self.W)

    def test_linear_score(self):
        self.assertEqual(linear_score(50, 100, 1000), 1.0)
        self.assertEqual(linear_score(2000, 100, 1000), 0.0)
        self.assertAlmostEqual(linear_score(550, 100, 1000), 0.5)
        self.assertIsNone(linear_score(None, 100, 1000))

    def test_shipped_config_weights_sum_to_one(self):
        cfg = load_config(os.path.join(os.path.dirname(evaluate.__file__), "evaluation", "hses_config.json"))
        self.assertAlmostEqual(sum(cfg["weights"].values()), 1.0)


class SyntheticIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ranker import BM25SearchEngine
        cls.tmp = tempfile.TemporaryDirectory()
        cls.corpus, index_dir = build_synthetic_index(cls.tmp.name)
        cls.engine = BM25SearchEngine(index_dir=index_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.engine.load_index()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_search_and_metrics_end_to_end(self):
        ids = [r["doc_id"] for r in self.engine.search("organic chemistry book", top_k=5)["results"]]
        self.assertEqual(ids[0], "d1")
        m = evaluate_ranking(ids, {"d1": 2, "d3": 1, "d5": 0})
        self.assertEqual(m["p@1"], 1.0)
        self.assertEqual(m["success@5"], 1.0)

    def test_permalink_used_as_result_url(self):
        self.assertTrue(all(m["url"].startswith("https://www.reddit.com/") for m in self.engine.metadata))

    def test_integrity_checks_pass_on_clean_index(self):
        from eval_perf import check_index_integrity
        failed = [c for c in check_index_integrity(self.engine, None, self.corpus) if not c["passed"]]
        self.assertEqual(failed, [])

    def test_integrity_checks_catch_corruption(self):
        from eval_perf import check_index_integrity
        term = next(iter(self.engine.postings))
        saved = self.engine.postings[term]
        self.engine.postings[term] = saved + [(999, 1, 0)]  # doc id out of range
        try:
            failed = {c["check"] for c in check_index_integrity(self.engine, None, self.corpus) if not c["passed"]}
        finally:
            self.engine.postings[term] = saved
        self.assertIn("posting doc ids in range", failed)

    def test_filters(self):
        res = self.engine.search("physics", top_k=10, filter_flair="discussion", min_score=1)["results"]
        self.assertTrue(res)
        self.assertTrue(all("discussion" in r["flair"].lower() and r["upvotes"] >= 1 for r in res))


class JudgmentAndFeedbackTests(unittest.TestCase):
    def test_human_labels_take_precedence(self):
        raw = {}
        self.assertTrue(evaluate.add_judgment(raw, "Q1", "d1", 0, "claude"))
        self.assertTrue(evaluate.add_judgment(raw, "Q1", "d1", 2, "alice"))
        self.assertFalse(evaluate.add_judgment(raw, "Q1", "d1", 0, "claude"))
        self.assertEqual(raw["Q1"]["d1"], {"grade": 2, "assessor": "alice"})
        with self.assertRaises(ValueError):
            evaluate.add_judgment(raw, "Q1", "d2", 3, "alice")

    def test_randomization_test(self):
        self.assertEqual(evaluate.paired_randomization_p([0.5] * 10, [0.5] * 10), 1.0)
        self.assertLess(evaluate.paired_randomization_p([1.0] * 20, [0.0] * 20), 0.01)

    def test_usefulness_requires_minimum_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "feedback.jsonl")
            old = evaluate.FEEDBACK_PATH
            evaluate.FEEDBACK_PATH = path
            try:
                rows = [{"solved": i % 4 != 0, "rating": 5 if i % 2 else 3} for i in range(8)]
                with open(path, "w") as f:
                    f.write("\n".join(json.dumps(r) for r in rows))
                few = evaluate.run_usefulness({"min_feedback": 20})
                self.assertIsNone(few["task_completion"])     # too few: not measured, not guessed
                enough = evaluate.run_usefulness({"min_feedback": 8})
                self.assertAlmostEqual(enough["task_completion"], 6 / 8)
                self.assertAlmostEqual(enough["satisfaction"], (4.0 - 1) / 4)
            finally:
                evaluate.FEEDBACK_PATH = old


if __name__ == "__main__":
    unittest.main()
