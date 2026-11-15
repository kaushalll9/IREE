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
        if not sources:
            answer = "I couldn't find any r/JEENEETards discussion about that. Try rephrasing."
        else:
            answer = "Most relevant r/JEENEETards discussions:\n" + "\n".join(
                f"- {s['title']} [{i}]" for i, s in enumerate(sources, start=1))
        return {"mode": "community", "answer": answer, "sources": sources,
                "expanded_terms": res.get("expanded_terms", {}), "generator": "extractive"}


def _selfcheck():
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
