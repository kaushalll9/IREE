"""
JEE/NEET assistant chatbot.

Routes each question to one of two answer sources:
  - official:  exam dates, syllabus, admit cards, results, registration...
               -> latest NTA notices crawled by nta_crawler.py (data/nta_updates.json)
  - community: everything else (books, coaching, strategy, colleges...)
               -> BM25F search (with domain-dictionary expansion) over r/JEENEETards

Usage:
    python chatbot.py "when is jee mains 2027 registration last date"
    python chatbot.py            # interactive
"""

import os
import re
import json
import math
from concurrent.futures import ThreadPoolExecutor

import requests

from nta_crawler import CATEGORIES, EXAMS, classify, OUT_PATH as NTA_PATH

# A question is "official" when it names an NTA exam AND an official topic.
OFFICIAL_TOPIC_RE = re.compile(
    r"\b(syllabus|admit ?card|hall ?ticket|exam city|answer ?key|results?|score ?card|"
    r"registration|register|application|apply|last date|deadline|bulletin|notification|"
    r"exam (?:date|pattern|schedule)|dates?|schedule|when (?:is|will|does|are))\b", re.I)
OFFICIAL_EXAM_RE = re.compile(r"\b(jee|neet|mains?|nta)\b", re.I)
# Query words that mean a crawler category but aren't in its title patterns.
CATEGORY_SYNONYMS = [("admit_card", r"hall ?ticket"), ("application", r"last date|deadline|register"),
                     ("dates", r"when (?:is|will|does|are)|exam date")]
# PDF footer/boilerplate lines that contain dates but say nothing.
BOILERPLATE_RE = re.compile(r"generated from eoffice|page \d+ of \d+|^dated?\s*:", re.I)


COMMENTS_API = "https://arctic-shift.photon-reddit.com/api/comments/search"
COMMENTS_CACHE = "data/comments_cache"
SKIP_AUTHORS = {"AutoModerator", "[deleted]"}


def fetch_comments(post_id, timeout=10):
    """
    Comments of one post (Arctic Shift API), cached on disk as JSON.
    The dataset only has posts; on Reddit the answers live in the comments.
    Returns [] when the API is unreachable, so the chatbot degrades to post titles.
    """
    path = os.path.join(COMMENTS_CACHE, f"{post_id}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    try:
        r = requests.get(COMMENTS_API, timeout=timeout, params={
            "link_id": f"t3_{post_id}", "limit": 100,
            "fields": "id,body,score,author,parent_id"})
        r.raise_for_status()
        comments = r.json().get("data") or []
    except (requests.RequestException, ValueError):
        return []
    os.makedirs(COMMENTS_CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False)
    return comments


MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# Replies that aren't answers: counter-questions, thanks, "same here".
# ponytail: keyword heuristic; a trained answer classifier would do better.
CHATTER_RE = re.compile(r"\?\s*$|^\s*(thanks|thank you|thx)\b|\bsame (case|here|problem)\b", re.I)


def best_replies(comments, n=2, max_chars=400, min_score=2, min_chars=30):
    """Top-voted direct replies to the post: the community's answers.
    min_score/min_chars drop chatter ("same here bro") that nobody upvoted."""
    good = [c for c in comments
            if c.get("parent_id", "").startswith("t3_")          # direct reply, not a sub-thread
            and c.get("author") not in SKIP_AUTHORS
            and c.get("body") not in (None, "[deleted]", "[removed]")
            and c.get("score", 0) >= min_score
            and len(c["body"]) >= min_chars
            and not CHATTER_RE.search(c["body"])]
    good.sort(key=lambda c: c.get("score", 0), reverse=True)
    out = []
    for c in good[:n]:
        body = " ".join(MD_LINK_RE.sub(r"\1", c["body"]).split())
        out.append({"text": body[:max_chars] + ("..." if len(body) > max_chars else ""),
                    "score": c.get("score", 0), "author": c.get("author", "")})
    return out


def route(question):
    """'official' or 'community'."""
    if OFFICIAL_TOPIC_RE.search(question) and OFFICIAL_EXAM_RE.search(question):
        return "official"
    return "community"


class Chatbot:
    def __init__(self, engine=None, nta_path=NTA_PATH):
        if engine is None:
            from ranker import BM25SearchEngine
            engine = BM25SearchEngine(index_dir="index")
            engine.load_index()
        self.engine = engine
        self.nta = None
        if os.path.exists(nta_path):
            with open(nta_path, encoding="utf-8") as f:
                self.nta = json.load(f)

    def ask(self, question):
        question = question.strip()
        if not question:
            return {"mode": "none", "answer": "Ask me anything about JEE or NEET.", "sources": []}
        if route(question) == "official":
            res = self.official(question)
            if res:
                return res
        return self.community(question)

    # ---------------------------------------------------------------- official
    def official(self, question):
        """Newest matching NTA notices; None if there is no crawl data or no match."""
        if not self.nta:
            return None
        exam = classify(question, EXAMS) or ("JEE" if re.search(r"\bmains?\b", question, re.I) else None)
        category = classify(question, CATEGORIES + CATEGORY_SYNONYMS)
        notices = [n for n in self.nta["notices"]
                   if (not exam or n["exam"] == exam) and (not category or n["category"] == category)]
        if not notices:
            return None
        top = notices[:3]
        sources = [{
            "type": "nta",
            "title": n["title"],
            "url": n["url"],
            "date": n["published"],
            "exam": n["exam"],
            "category": n["category"],
            "snippet": "\n".join(n.get("important_dates", [])[:6]),
        } for n in top]

        lead = top[0]
        lines = [f"Latest official NTA notice{' for ' + exam if exam else ''}"
                 f"{' (' + category.replace('_', ' ') + ')' if category else ''}:",
                 f"**{lead['title']}** (published {lead['published']}) [1]"]
        dates = [d for d in lead.get("important_dates", []) if not BOILERPLATE_RE.search(d)]
        if dates:
            lines.append("Key lines from the notice:")
            lines += [f"- {d}" for d in dates[:6]]
        if len(top) > 1:
            lines.append("Other recent notices: " + "; ".join(
                f"{n['title']} [{i}]" for i, n in enumerate(top[1:], start=2)))
        lines.append(f"(NTA data crawled {self.nta['crawled_at'][:10]}; always confirm on the official site.)")
        return {"mode": "official", "answer": "\n".join(lines), "sources": sources,
                "expanded_terms": {}, "generator": "extractive"}

    # --------------------------------------------------------------- community
    def community(self, question, k=5):
        res = self.engine.search(question, top_k=k, expand=True)
        sources = [{
            "type": "reddit",
            "doc_id": r["doc_id"],
            "title": r["title"],
            "url": r["url"],
            "date": r["date"][:10],
            "upvotes": r["upvotes"],
            "comments": r["num_comments"],
            "snippet": r["snippet"],
        } for r in res["results"]]
        # Replies for the top threads that have any, fetched in parallel.
        with_comments = [s for s in sources if s["comments"] > 0][:3]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for s, comments in zip(with_comments, pool.map(fetch_comments, (s["doc_id"] for s in with_comments))):
                s["replies"] = best_replies(comments)
        answer = self.extractive_answer(sources)
        return {"mode": "community", "answer": answer, "sources": sources,
                "expanded_terms": res.get("expanded_terms", {}), "generator": "extractive"}

    @staticmethod
    def extractive_answer(sources):
        """
        Extractive answer: the highest-voted replies from the most relevant threads,
        ordered by retrieval rank and reply score, each cited to its thread.
        """
        if not sources:
            return "I couldn't find any r/JEENEETards discussion about that. Try rephrasing."
        replies = [(i, r) for i, s in enumerate(sources, start=1) for r in s.get("replies", [])]
        if not replies:
            return "Most relevant r/JEENEETards discussions:\n" + "\n".join(
                f"- {s['title']} [{i}]" for i, s in enumerate(sources, start=1))
        # Earlier (more relevant) threads first, then upvotes within the pool.
        replies.sort(key=lambda x: (x[0] - 1) - math.log10(1 + max(x[1]["score"], 0)))
        n = len({i for i, _ in replies})
        lines = [f"Here's what r/JEENEETards says ({n} discussion{'s' if n > 1 else ''}):"]
        lines += [f"- \"{r['text']}\" ({r['score']} upvotes) [{i}]" for i, r in replies[:4]]
        lines.append("These are student opinions, not official advice.")
        return "\n".join(lines)


def _selfcheck():
    cs = [{"parent_id": "t3_x", "author": "a", "body": "Buy MS Chouhan for [practice](http://x.y)", "score": 9},
          {"parent_id": "t3_x", "author": "e", "body": "same here bro, no idea what to do", "score": 1},
          {"parent_id": "t1_y", "author": "b", "body": "nested reply should be skipped", "score": 50},
          {"parent_id": "t3_x", "author": "AutoModerator", "body": "I am a bot, beep boop", "score": 99},
          {"parent_id": "t3_x", "author": "c", "body": "[deleted]", "score": 5},
          {"parent_id": "t3_x", "author": "d", "body": "Solomons is great for theory, really", "score": 3}]
    assert [r["author"] for r in best_replies(cs)] == ["a", "d"]
    assert best_replies(cs)[0]["text"] == "Buy MS Chouhan for practice"
    assert CHATTER_RE.search("Ye Advanced ke liye hai ya Mains ke liye??")
    assert CHATTER_RE.search("Yaar same case hai mere sath bhi")
    assert not CHATTER_RE.search("Dc pandey for theory and questions")
    assert route("When is JEE Mains 2027 registration last date?") == "official"
    assert route("neet syllabus 2026") == "official"
    assert route("jee admit card kab aayega") == "official"
    assert route("best book for organic chemistry") == "community"
    assert route("iit bombay cse cutoff rank") == "community"
    assert route("when should I start revision") == "community"   # no exam named
    assert BOILERPLATE_RE.search("Generated from eOffice by X on 22/12/2025")
    assert classify("jee hall ticket", CATEGORIES + CATEGORY_SYNONYMS) == "admit_card"


def main():
    import sys
    _selfcheck()
    bot = Chatbot()
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else iter(lambda: input("\nYou > "), "exit")
    for q in questions:
        r = bot.ask(q)
        print(f"\n[{r['mode']} / {r.get('generator')}]\n{r['answer']}\n")
        for i, s in enumerate(r["sources"], start=1):
            print(f"  [{i}] {s['title'][:80]} - {s['url']}")


if __name__ == "__main__":
    main()
