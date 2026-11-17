"""
NTA crawler evaluation: crawl statistics, classification accuracy, publish-date accuracy,
PDF date-line extraction (pypdf vs pdfplumber) and end-to-end official-answer accuracy.

Ground truth:
  evaluation/nta_labels.json          exam + category of every crawled notice, labelled by hand from its title
  evaluation/official_questions.json  student questions with the expected route and expected notice

Usage:
    python eval_crawler.py              # -> evaluation/crawler_eval.json (downloads the fetched PDFs again)
"""

import io
import re
import json
import time
import logging
from collections import Counter
from datetime import date

import requests

import nta_crawler as nc
import chatbot
from evaluate import load_json, save_json

OUT_PATH = "evaluation/crawler_eval.json"
LABELS_PATH = "evaluation/nta_labels.json"
QUESTIONS_PATH = "evaluation/official_questions.json"
DATED_RE = re.compile(r"Dated?\s*:?\s*(\d{1,2})[./-](\d{1,2})[./-](\d{4})", re.I)


# The classifier as first evaluated. Its errors on the hand labels were then fixed, so the
# current classifier's accuracy on those same labels is optimistic; this one is not.
CATEGORIES_BEFORE_FIX = [
    ("syllabus", r"syllabus"),
    ("bulletin", r"information bulletin|brochure|prospectus"),
    ("admit_card", r"admit card|exam(?:ination)? city|city intimation"),
    ("answer_key", r"answer key|recorded response|omr"),
    ("result", r"result|score ?card|topper|cut-?off|percentile|rank"),
    ("application", r"application|registration|correction|apply"),
    ("dates", r"\bdates?\b|schedule|calendar|postpone"),
]


def classification(notices, labels, table=None):
    if table:
        notices = [{**n, "category": nc.classify(n["title"], table) or "other"} for n in notices]
    rows = [(n, labels[n["url"]]) for n in notices if n["url"] in labels]
    cats = sorted({g["category"] for _, g in rows} | {n["category"] for n, _ in rows})
    per_cat = {}
    for c in cats:
        tp = sum(n["category"] == c and g["category"] == c for n, g in rows)
        pred = sum(n["category"] == c for n, _ in rows)
        gold = sum(g["category"] == c for _, g in rows)
        per_cat[c] = {"gold": gold, "predicted": pred,
                      "precision": round(tp / pred, 3) if pred else None,
                      "recall": round(tp / gold, 3) if gold else None}
    errors = [{"title": n["title"][:110], "predicted": n["category"], "gold": g["category"]}
              for n, g in rows if n["category"] != g["category"]]
    return {"labelled": len(rows),
            "exam_accuracy": round(sum(n["exam"] == g["exam"] for n, g in rows) / len(rows), 3),
            "category_accuracy": round(sum(n["category"] == g["category"] for n, g in rows) / len(rows), 3),
            "per_category": per_cat,
            "confusions": Counter(f"{e['gold']} -> {e['predicted']}" for e in errors).most_common(),
            "errors": errors}


def publish_dates(notices):
    """Date parsed from the file URL vs the 'Dated: dd.mm.yyyy' line inside the PDF."""
    checked, exact, within3, rows = 0, 0, 0, []
    for n in notices:
        m = DATED_RE.search(n.get("text", ""))
        if not m or not n.get("published") or len(n["published"]) != 10:
            continue
        try:
            inside = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            continue
        url_date = date.fromisoformat(n["published"])
        gap = abs((url_date - inside).days)
        checked += 1
        exact += gap == 0
        within3 += gap <= 3
        rows.append({"title": n["title"][:80], "from_url": n["published"], "in_pdf": inside.isoformat(), "days_apart": gap})
    with_date = sum(bool(n.get("published")) for n in notices)
    return {"notices_with_published_date": with_date, "notices_total": len(notices),
            "day_precision": sum(len(n.get("published") or "") == 10 for n in notices),
            "checked_against_pdf": checked,
            "exact_match": round(exact / checked, 3) if checked else None,
            "within_3_days": round(within3 / checked, 3) if checked else None, "rows": rows}


def extraction(notices):
    """Re-download the full-text PDFs and compare pypdf with pdfplumber on the same files."""
    from pypdf import PdfReader
    import pdfplumber
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    s = requests.Session()
    s.headers["User-Agent"] = nc.UA
    errors = 0
    out = {"pypdf": {"date_lines": 0, "docs_with_date_lines": 0, "exam_date_rows": 0, "words_per_line": []},
           "pdfplumber": {"date_lines": 0, "docs_with_date_lines": 0, "exam_date_rows": 0, "words_per_line": []}}
    docs = [n for n in notices if "text" in n]
    for n in docs:
        time.sleep(nc.DELAY)
        try:
            raw = s.get(n["url"], timeout=30).content
            texts = {"pypdf": "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)}
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                texts["pdfplumber"] = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception as e:
            errors += 1
            print(f"    [!] {n['url']}: {e}")
            continue
        for lib, text in texts.items():
            lines = nc.date_lines(text)
            o = out[lib]
            o["date_lines"] += len(lines)
            o["docs_with_date_lines"] += bool(lines)
            o["exam_date_rows"] += any(chatbot.EXAM_DATE_RE.search(l) for l in lines)
            nonempty = [l for l in text.splitlines() if l.strip()]
            if nonempty:
                o["words_per_line"].append(sum(len(l.split()) for l in nonempty) / len(nonempty))
    for o in out.values():
        wpl = o.pop("words_per_line")
        o["mean_words_per_line"] = round(sum(wpl) / len(wpl), 1) if wpl else None
    out["pdfs"], out["download_errors"] = len(docs), errors
    return out


def official_answers(nta, questions, labels):
    bot = chatbot.Chatbot.__new__(chatbot.Chatbot)
    bot.nta = nta
    rows = []
    for q in questions:
        route = chatbot.route(q["question"])
        ok_route = route == q["route"]
        ok_answer = None
        detail = ""
        if q["route"] == "official":
            r = bot.official(q["question"]) if route == "official" else None
            if r is None:
                ok_answer, detail = False, "no answer"
            else:
                lead = r["sources"][0]
                checks = [lead["exam"] == q["exam"]]
                if q.get("missing_year"):
                    checks.append(r["answer"].startswith(f"No NTA notice mentioning {q['missing_year']}"))
                elif q.get("year"):
                    checks.append(q["year"] in lead["title"] or q["year"] in lead["snippet"])
                gold_exists = q.get("year") and q.get("category") not in (None, "exam_date") and any(
                    g["exam"] == q["exam"] and g["category"] == q["category"] and q["year"] in g["title"]
                    for k, g in labels.items() if k != "_meta")
                if q.get("year") and q.get("category") not in (None, "exam_date") and not gold_exists:
                    # no such notice in the crawl (per the hand labels): the right answer is to say so
                    checks = [r["answer"].startswith(f"No {q['exam']} {q['year']} {q['category'].replace('_', ' ')} notice")]
                elif q.get("category") == "exam_date":
                    checks.append(bool(chatbot.EXAM_DATE_RE.search(r["answer"])))
                elif q.get("category"):
                    checks.append(lead["category"] == q["category"])
                ok_answer = all(checks)
                first = r["answer"].splitlines()[0]
                detail = (f"says: {first[:95]}" if first.startswith("No ")
                          else f"{lead['exam']} | {lead['category']} | {lead['title'][:90]}")
        rows.append({"question": q["question"], "expected_route": q["route"], "route": route,
                     "route_ok": ok_route, "answer_ok": ok_answer, "answer": detail})
    off = [r for r in rows if r["expected_route"] == "official"]
    return {"questions": len(rows), "routing_accuracy": round(sum(r["route_ok"] for r in rows) / len(rows), 3),
            "official_questions": len(off),
            "official_answer_accuracy": round(sum(bool(r["answer_ok"]) for r in off) / len(off), 3),
            "rows": rows}


def main():
    data = load_json(nc.OUT_PATH)
    notices = data["notices"]
    labels = load_json(LABELS_PATH)
    stats = {
        "crawled_at": data["crawled_at"],
        "run": data.get("stats"),
        "notices": len(notices),
        "by_exam": Counter(n["exam"] for n in notices),
        "by_category": Counter(n["category"] for n in notices),
        "by_year_published": Counter((n.get("published") or "unknown")[:4] for n in notices),
        "pdf_full_text": sum("text" in n for n in notices),
        "pdf_no_text_layer": sum("text" in n and len(n["text"].strip()) < 50 for n in notices),   # scanned images
        "with_important_dates": sum(bool(n.get("important_dates")) for n in notices),
        "important_date_lines": sum(len(n.get("important_dates", [])) for n in notices),
        "newest": notices[0]["published"], "oldest": notices[-1]["published"],
    }
    print("[+] extraction comparison (downloads PDFs)...")
    out = {"stats": stats,
           "classification_before_fix": classification(notices, labels, CATEGORIES_BEFORE_FIX),
           "classification": classification(notices, labels),
           "publish_dates": publish_dates(notices),
           "extraction": extraction(notices),
           "official_answers": official_answers(data, load_json(QUESTIONS_PATH)["questions"], labels)}
    save_json(OUT_PATH, out)
    c, o = out["classification"], out["official_answers"]
    print(f"[✓] exam acc {c['exam_accuracy']}, category acc {c['category_accuracy']} "
          f"(before fix {out['classification_before_fix']['category_accuracy']}), "
          f"routing {o['routing_accuracy']}, official answers {o['official_answer_accuracy']} -> {OUT_PATH}")
    print(json.dumps({k: out[k] for k in ("publish_dates", "extraction")}, indent=1, default=str)[:1500])
    for r in o["rows"]:
        if not r["route_ok"] or r["answer_ok"] is False:
            print("  FAIL", r)


if __name__ == "__main__":
    main()
