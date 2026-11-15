"""
NTA Web Crawler: latest official JEE Main / NEET information.

Crawls the official NTA exam sites (same-host BFS, robots.txt respected,
polite delay), collects every notice link (mostly PDFs), and extracts:
  - exam (JEE / NEET) and category (dates, syllabus, bulletin, admit_card,
    answer_key, result, application, other) from the notice title
  - published date (from the NTA upload filename / path)
  - dates mentioned in the title
  - for bulletins / syllabus / schedule PDFs: the full text and the lines that
    mention a date (the "important dates" table of the bulletin)

Output: data/nta_updates.json (newest notices first)

Usage:
    pip install requests beautifulsoup4 pypdf
    python nta_crawler.py                          # crawl
    python nta_crawler.py --latest neet            # show latest NEET notices
    python nta_crawler.py --latest jee --category dates
"""

import io
import os
import re
import json
import logging
import time
import argparse
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

SEEDS = {
    "JEE": "https://jeemain.nta.nic.in/",
    "NEET": "https://neet.nta.nic.in/",
    "NTA": "https://nta.ac.in/",   # agency-wide news; only JEE/NEET notices kept
}
OUT_PATH = "data/nta_updates.json"
UA = "IRE-JEE-NEET-Assistant/1.0 (academic project crawler)"
DELAY = 1.0          # seconds between requests to the same site
MAX_PAGES = 30       # HTML pages per site
MAX_DEPTH = 2
MAX_PDFS = 20        # PDFs downloaded for full-text extraction

CATEGORIES = [  # first match wins, order matters
    ("syllabus", r"syllabus"),
    ("bulletin", r"information bulletin|brochure|prospectus"),
    ("admit_card", r"admit card|exam(?:ination)? city|city intimation"),
    ("answer_key", r"answer key|recorded response|omr"),
    ("result", r"result|score ?card|topper|cut-?off|percentile|rank"),
    ("application", r"application|registration|correction|apply"),
    ("dates", r"\bdates?\b|schedule|calendar|postpone"),
]
EXAMS = [("JEE", r"(?<!nchm-)(?<!nchm )\bjee\b"), ("NEET", r"\bneet\b|national eligibility")]
FULLTEXT_CATS = {"bulletin", "syllabus", "dates", "application"}

MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
DATE_RE = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?[\s\-.]*{MONTHS}[\s,\-.]*\d{{4}}\b"   # 21st June 2026
    rf"|\b{MONTHS}\s+\d{{1,2}},?\s+\d{{4}}\b"                          # June 21, 2026
    r"|\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b",                            # 21.06.2026
    re.I)
STAMP_RE = re.compile(r"(?:/uploads/\d{4}/\d{2}/|Notice_)(20\d{6})")
MONTH_RE = re.compile(r"/uploads/(\d{4})/(\d{2})/")
GENERIC_RE = re.compile(r"^(read more|click here|download|view|more|link)\W*$", re.I)


def classify(title, table):
    t = title.lower()
    return next((name for name, pat in table if re.search(pat, t)), None)


def published_date(url):
    """NTA file names start with YYYYMMDD; fall back to the /uploads/YYYY/MM/ path."""
    m = STAMP_RE.search(url)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").date().isoformat()
        except ValueError:
            pass
    m = MONTH_RE.search(url)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def link_title(a):
    """Anchor text, or the preceding sibling's text for 'Read More'-style links."""
    title = " ".join(a.get_text().split())
    if GENERIC_RE.match(title) and (prev := a.find_previous_sibling()):
        title = " ".join(prev.get_text().split())
    return title


def date_lines(text, limit=60):
    seen, out = set(), []
    for line in text.splitlines():
        line = " ".join(line.split())
        if 10 < len(line) < 300 and DATE_RE.search(line) and line not in seen:
            seen.add(line)
            out.append(line)
    return out[:limit]


class Crawler:
    def __init__(self, ua=UA):
        self.ua = ua
        self.s = requests.Session()
        self.s.headers["User-Agent"] = ua
        self.robots = {}

    def allowed(self, url):
        host = urlparse(url).netloc
        if host not in self.robots:
            rp = RobotFileParser(f"https://{host}/robots.txt")
            try:
                rp.read()
            except Exception:
                rp.allow_all = True
            self.robots[host] = rp
        return self.robots[host].can_fetch(self.ua, url)

    def get(self, url):
        time.sleep(DELAY)
        r = self.s.get(url, timeout=30)
        r.raise_for_status()
        return r

    def crawl_site(self, source, seed):
        host = urlparse(seed).netloc
        queue, seen, notices, pages = deque([(seed, 0)]), {seed}, {}, 0
        while queue and pages < MAX_PAGES:
            url, depth = queue.popleft()
            if not self.allowed(url):
                continue
            try:
                r = self.get(url)
            except requests.RequestException as e:
                print(f"    [!] {url}: {e}")
                if depth == 0 and "403" in str(e):
                    print("        site rejected the User-Agent; see --user-agent")
                continue
            if "html" not in r.headers.get("Content-Type", ""):
                continue
            pages += 1
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                link = urldefrag(urljoin(url, a["href"]))[0]
                title = link_title(a)
                if not link.startswith("http") or len(title) < 12:
                    continue
                exam = classify(title, EXAMS) or (source if source != "NTA" else None)
                is_notice = link.lower().endswith(".pdf") or "/Download/" in link or "/uploads/" in link
                if is_notice and exam and link not in notices:
                    notices[link] = {
                        "exam": exam,
                        "source": source,
                        "title": title,
                        "url": link,
                        "category": classify(title, CATEGORIES) or "other",
                        "published": published_date(link),
                        "dates_mentioned": DATE_RE.findall(title),
                        "found_on": url,
                    }
                elif (urlparse(link).netloc == host and depth < MAX_DEPTH
                      and link not in seen and not is_notice):
                    seen.add(link)
                    queue.append((link, depth + 1))
        print(f"[+] {source}: {pages} pages crawled, {len(notices)} notices")
        return list(notices.values())

    def pdf_text(self, url):
        from pypdf import PdfReader
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        reader = PdfReader(io.BytesIO(self.get(url).content))
        return "\n".join(p.extract_text() or "" for p in reader.pages)


def crawl(ua=UA):
    c = Crawler(ua)
    notices = {}
    for source, seed in SEEDS.items():
        for n in c.crawl_site(source, seed):
            notices.setdefault(n["url"], n)   # exam-site copy wins over NTA duplicate
    notices = sorted(notices.values(), key=lambda n: n["published"] or "", reverse=True)

    # Full text for the newest bulletins / syllabus / schedule notices.
    fetched = 0
    priority = {"bulletin": 0, "syllabus": 0, "dates": 1, "application": 1}
    for n in sorted(notices, key=lambda n: priority.get(n["category"], 9)):
        if fetched >= MAX_PDFS:
            break
        if n["category"] in FULLTEXT_CATS and n["url"].lower().endswith(".pdf"):
            try:
                text = c.pdf_text(n["url"])
            except Exception as e:
                print(f"    [!] pdf {n['url']}: {e}")
                continue
            n["text"] = text
            n["important_dates"] = date_lines(text)
            fetched += 1
            print(f"    [pdf] {n['exam']} {n['category']}: {n['title'][:70]}")

    out = {"crawled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "notices": notices}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[✓] {len(notices)} notices ({fetched} with full text) -> {OUT_PATH}")


def show_latest(exam, category, k=10):
    with open(OUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Crawled at {data['crawled_at']}")
    hits = [n for n in data["notices"]
            if n["exam"] == exam.upper() and (not category or n["category"] == category)]
    for n in hits[:k]:
        print(f"[{n['published']}] ({n['category']}) {n['title']}\n    {n['url']}")
        for line in n.get("important_dates", [])[:8]:
            print(f"      - {line}")


def _selfcheck():
    assert classify("Information Bulletin for NEET(UG)-2026", CATEGORIES) == "bulletin"
    assert classify("Provisional Answer Keys for NEET (UG) – 2026", CATEGORIES) == "answer_key"
    assert classify("Final Answer Keys for NEET (UG) – 2026", EXAMS) == "NEET"
    assert published_date("https://x/uploads/2026/04/202604092096865379.pdf") == "2026-04-09"
    assert published_date("https://x/uploads/2024/10/2024102824.pdf") == "2024-10-28"
    assert published_date("https://x/uploads/2025/01/99999999.pdf") == "2025-01"
    assert published_date("https://nta.ac.in/Download/Notice/Notice_20261114192445.pdf") == "2026-11-14"
    assert classify("Hotel Management (NCHM-JEE) 2027", EXAMS) is None
    from bs4 import BeautifulSoup as BS
    a = BS("<p><content>Exam Calendar JEE</content><a href='x.pdf'>Read More</a></p>", "html.parser").a
    assert link_title(a) == "Exam Calendar JEE"
    assert DATE_RE.findall("held on 21st June, 2026 and 21.06.2026") == ["21st June, 2026", "21.06.2026"]


if __name__ == "__main__":
    _selfcheck()
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", choices=["jee", "neet"], help="show latest notices from saved crawl")
    ap.add_argument("--category", choices=[c for c, _ in CATEGORIES] + ["other"])
    ap.add_argument("--user-agent", default=UA,
                    help="User-Agent header (some NTA sites reject non-browser agents)")
    args = ap.parse_args()
    if args.latest:
        show_latest(args.latest, args.category)
    else:
        crawl(args.user_agent)
