"""
JEE/NEET assistant chatbot.

Routes each question to one of two answer sources:
  - official:  exam dates, syllabus, admit cards, results, registration...
               -> latest NTA notices crawled by nta_crawler.py (data/nta_updates.json)
  - community: everything else (books, coaching, strategy, colleges...)
               -> hybrid BM25F + semantic search (RRF) over r/JEENEETards

Answers are extractive by default (no API key needed). If ANTHROPIC_API_KEY is
set, Claude writes the final answer from the same retrieved sources (RAG), with
the extractive answer as the fallback on any API error.

Usage:
    python chatbot.py "when is jee mains 2027 registration last date"
    python chatbot.py            # interactive
"""

import os
import re
import html
import json
import math
from concurrent.futures import ThreadPoolExecutor

import requests

from hybrid_search import hybrid_search
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
# "Date of Examination 20,21 ... January 2027" rows in bulletins / application notices.
EXAM_DATE_RE = re.compile(r"date (?:of|for) (?:the )?exam|exam(?:ination)? dates?\b", re.I)
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
MIN_REPLY_RELEVANCE = 0.20  # cosine floor: memes/polls scored <= 0.15, real answers >= 0.22 on sampled questions
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
        body = " ".join(html.unescape(MD_LINK_RE.sub(r"\1", c["body"])).split())  # Reddit sends &gt; etc.
        out.append({"text": body[:max_chars] + ("..." if len(body) > max_chars else ""),
                    "score": c.get("score", 0), "author": c.get("author", "")})
    return out


CLAUDE_MODEL = "claude-opus-5-5"
SYSTEM_PROMPT = """You help Indian students preparing for JEE and NEET.
Answer the student's question using only the numbered sources you are given, and cite them inline like [1] or [2][3].
NTA sources are official notices; treat their dates as authoritative and quote them exactly.
Reddit sources are other students' opinions; present them as such ("students on r/JEENEETards suggest..."), and mention real disagreement between them.
Many Reddit replies are in Hinglish; convey their meaning in clear English.
If the sources don't answer the question, say so briefly and say what they do cover. Never invent dates, cutoffs, or facts that aren't in the sources.
Keep the answer under 150 words; use short bullets when listing options."""


def format_sources(sources):
    """Numbered source block for the model, matching the [n] citations shown in the UI."""
    blocks = []
    for i, s in enumerate(sources, start=1):
        if s["type"] == "nta":
            head = f"[{i}] NTA official notice ({s['exam']}, published {s['date']}): {s['title']}"
        else:
            head = f"[{i}] r/JEENEETards thread ({s['date']}, {s['upvotes']} upvotes): {s['title']}"
        body = [s["snippet"]] if s.get("snippet") else []
        body += [f"Reply ({r['score']} upvotes): {r['text']}" for r in s.get("replies", [])]
        blocks.append("\n".join([head] + body))
    return "\n\n".join(blocks)


def claude_answer(question, sources):
    """Claude-written answer grounded in `sources`, or None (no key, API error, refusal)."""
    if not os.environ.get("ANTHROPIC_API_KEY") or not sources:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        resp = anthropic.Anthropic().beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",              # retry declined requests on a fallback model
            output_config={"effort": "low"},  # short grounded answers; keeps chat latency down
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content":
                       f"<sources>\n{format_sources(sources)}\n</sources>\n\nQuestion: {question}"}],
        )
    except anthropic.APIError as e:
        print(f"[!] Claude unavailable, using extractive answer: {e}")
        return None
    if resp.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text or None


def route(question):
    """'official' or 'community'."""
    if OFFICIAL_TOPIC_RE.search(question) and OFFICIAL_EXAM_RE.search(question):
        return "official"
    return "community"


class Chatbot:
    def __init__(self, engine=None, semantic=None, nta_path=NTA_PATH):
        if engine is None:
            from ranker import BM25SearchEngine
            engine = BM25SearchEngine(index_dir="index")
            engine.load_index()
        if semantic is None:
            from semantic_search import SemanticSearchEngine
            semantic = SemanticSearchEngine(engine)
            semantic.load()
        self.engine = engine
        self.semantic = semantic
        self.nta = None
        if os.path.exists(nta_path):
            with open(nta_path, encoding="utf-8") as f:
                self.nta = json.load(f)

    def ask(self, question):
        question = question.strip()
        if not question:
            return {"mode": "none", "answer": "Ask me anything about JEE or NEET.", "sources": []}
        res = (self.official(question) if route(question) == "official" else None) or self.community(question)
        text = claude_answer(question, res["sources"])
        if text:
            res["answer"], res["generator"] = text, "claude"
        return res

    # ---------------------------------------------------------------- official
    def official(self, question):
        """Newest matching NTA notices; None if there is no crawl data or no match."""
        if not self.nta:
            return None
        exam = classify(question, EXAMS) or ("JEE" if re.search(r"\bmains?\b", question, re.I) else None)
        category = classify(question, CATEGORIES + CATEGORY_SYNONYMS)
        pool = [n for n in self.nta["notices"] if not exam or n["exam"] == exam]
        # "JEE Main 2027": only notices about that exam year (title or extracted date lines).
        years = set(re.findall(r"\b20\d\d\b", question))
        missing_year = None
        if years:
            for_year = [n for n in pool if any(y in n["title"] or y in " ".join(n.get("important_dates", []))
                                               for y in years)]
            if for_year:
                pool = for_year
            else:
                missing_year = ", ".join(sorted(years))
        # Category narrows the pool, but a year match beats a category match
        # ("when is jee 2027" -> the 2027 application notice holds the dates).
        def has_exam_date(n):
            return any(EXAM_DATE_RE.search(d) for d in n.get("important_dates", []))
        # A "when is the exam" question is also answered by any notice that states the exam date
        # (usually the application notice), not only by notices titled "schedule"/"dates".
        notices = [n for n in pool if not category or n["category"] == category
                   or (category == "dates" and has_exam_date(n))]
        if not notices and years and not missing_year:
            notices = pool
        if category == "dates":
            notices.sort(key=lambda n: not has_exam_date(n))   # stable: newest first within each group
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
        lines = [f"No NTA notice mentioning {missing_year} has been crawled yet." if missing_year else ""]
        label = category if category and lead["category"] == category else None
        lines += [f"Latest official NTA notice{' for ' + exam if exam else ''}"
                  f"{' ' + '/'.join(sorted(years)) if years and not missing_year else ''}"
                  f"{' (' + label.replace('_', ' ') + ')' if label else ''}:",
                 f"**{lead['title']}** (published {lead['published']}) [1]"]
        dates = [d for d in lead.get("important_dates", []) if not BOILERPLATE_RE.search(d)]
        dates.sort(key=lambda d: not EXAM_DATE_RE.search(d))  # exam date first
        if dates:
            lines.append("Key lines from the notice:")
            lines += [f"- {d}" for d in dates[:6]]
        if len(top) > 1:
            lines.append("Other recent notices: " + "; ".join(
                f"{n['title']} [{i}]" for i, n in enumerate(top[1:], start=2)))
        lines.append(f"(NTA data crawled {self.nta['crawled_at'][:10]}; always confirm on the official site.)")
        return {"mode": "official", "answer": "\n".join(l for l in lines if l), "sources": sources,
                "expanded_terms": {}, "generator": "extractive"}

    # --------------------------------------------------------------- community
    def community(self, question, k=5):
        # Hybrid (BM25 + semantic, RRF) so Hinglish / reworded questions still find threads.
        res = hybrid_search(self.engine, self.semantic, question, top_k=k, expand=True)
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
                s["replies"] = best_replies(comments, n=6)
        self.rank_replies(question, sources)
        answer = self.extractive_answer(sources)
        return {"mode": "community", "answer": answer, "sources": sources,
                "expanded_terms": res.get("expanded_terms", {}), "generator": "extractive"}

    def rank_replies(self, question, sources, per_thread=2):
        """Keep the replies that actually answer the question: cosine similarity to it with the
        same multilingual model as search, so Hinglish replies to English questions still match."""
        replies = [r for s in sources for r in s.get("replies", [])]
        if not replies:
            return
        emb = self.semantic.model.encode([question] + [r["text"] for r in replies],
                                         normalize_embeddings=True, convert_to_numpy=True)
        for r, sim in zip(replies, emb[1:] @ emb[0]):
            r["relevance"] = round(float(sim), 3)
        for s in sources:
            if "replies" in s:
                # ponytail: fixed floor picked by inspecting ~30 replies; tune on judged replies if reply judgments are added
                kept = [r for r in s["replies"] if r["relevance"] >= MIN_REPLY_RELEVANCE]
                s["replies"] = sorted(kept, key=lambda r: r["relevance"], reverse=True)[:per_thread]

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
        # Most on-topic replies first (semantic relevance to the question), upvotes as a small tie-breaker;
        # without relevance scores fall back to thread rank + upvotes.
        if all("relevance" in r for _, r in replies):
            replies.sort(key=lambda x: -(x[1]["relevance"] + 0.05 * math.log10(1 + max(x[1]["score"], 0))))
        else:
            replies.sort(key=lambda x: (x[0] - 1) - math.log10(1 + max(x[1]["score"], 0)))
        n = len({i for i, _ in replies})
        lines = [f"Here's what r/JEENEETards says ({n} discussion{'s' if n > 1 else ''}):"]
        lines += [f"- \"{r['text']}\" ({r['score']} upvotes) [{i}]" for i, r in replies[:4]]
        lines.append("These are student opinions, not official advice.")
        return "\n".join(lines)


def _selfcheck_official():
    def notice(title, cat, pub, dates=()):
        return {"exam": "JEE", "category": cat, "title": title, "url": "u", "published": pub,
                "important_dates": list(dates)}
    bot = Chatbot.__new__(Chatbot)
    bot.nta = {"crawled_at": "2026-11-15", "notices": [
        notice("Inviting Online Application form for JEE-2027 (Session 1)", "application", "2026-10-24",
               ["Exam: January 2027"]),
        notice("Examination Schedule for JEE (Main)-2025 Session-2", "dates", "2025-03-10")]}
    assert "2027" in bot.official("when is jee main 2027?")["sources"][0]["title"]
    assert "2025" in bot.official("when is jee main exam date")["sources"][0]["title"]   # no year: category
    bot.nta["notices"][0]["important_dates"] = ["Date of Examination 20,21 and 28 January 2027"]
    r = bot.official("when is jee main exam")
    assert "2027" in r["sources"][0]["title"] and "Date of Examination" in r["answer"]
    r = bot.official("jee main 2029 dates")
    assert r["answer"].startswith("No NTA notice mentioning 2029") and "2027" in r["sources"][0]["title"]


def _selfcheck():
    _selfcheck_official()
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
    src = [{"type": "reddit", "date": "2022-06-01", "upvotes": 3, "title": "HCV or DC?", "snippet": "",
            "replies": [{"score": 5, "text": "DC Pandey"}]}]
    assert format_sources(src) == "[1] r/JEENEETards thread (2022-06-01, 3 upvotes): HCV or DC?\nReply (5 upvotes): DC Pandey"
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
