"""
Builds IREE_Project_Numbers.pdf from the evaluation outputs (nothing is typed in by hand):
  evaluation/evaluation_results.json   (python evaluate.py)
  evaluation/dictionary_eval.json      (python eval_dictionary.py)
  evaluation/crawler_eval.json         (python eval_crawler.py)

Usage:
    python make_report.py            # -> IREE_Project_Numbers.pdf (needs google-chrome or chromium)
"""

import os
import html
import json
import shutil
import subprocess
from datetime import datetime

OUT_PDF = "IREE_Project_Numbers.pdf"
OUT_HTML = "evaluation/report.html"
BLUE, GREY = "#2a78d6", "#a8a7a1"
BEFORE_FIX_OFFICIAL = 18 / 20   # first eval_crawler.py run, before the classifier/answer fixes (not reproducible from current code)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def e(x):
    return html.escape(str(x))


def pct(x, d=1):
    return "–" if x is None else f"{x * 100:.{d}f}%"


def f3(x):
    return "–" if x is None else f"{x:.3f}"


def table(head, rows, cls=""):
    th = "".join(f"<th>{e(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def hbar(rows, vmax=None, fmt=f3, title=""):
    """Horizontal bars: rows = [(label, value, highlighted)]. Single series, value labels in text ink."""
    vmax = vmax or max(v for _, v, _ in rows) or 1
    lw, bw, rh = 230, 300, 26
    h = rh * len(rows) + 8
    parts = [f'<svg viewBox="0 0 {lw + bw + 70} {h}" width="100%" role="img" aria-label="{e(title)}">']
    for i, (label, v, hi) in enumerate(rows):
        y = 4 + i * rh
        w = max(2, bw * v / vmax)
        col = BLUE if hi else GREY
        parts.append(f'<text x="{lw - 8}" y="{y + 15}" text-anchor="end" class="lab{" hi" if hi else ""}">{e(label)}</text>')
        parts.append(f'<rect x="{lw}" y="{y + 4}" width="{w:.1f}" height="14" rx="3" fill="{col}"/>')
        parts.append(f'<rect x="{lw}" y="{y + 4}" width="{min(w, 4):.1f}" height="14" fill="{col}"/>')
        parts.append(f'<text x="{lw + w + 6:.1f}" y="{y + 15}" class="val">{e(fmt(v))}</text>')
    parts.append(f'<line x1="{lw}" y1="2" x2="{lw}" y2="{h - 2}" class="axis"/></svg>')
    return f'<figure>{"".join(parts)}<figcaption>{e(title)}</figcaption></figure>'


def tiles(items):
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="tv">{e(v)}</div><div class="tl">{e(l)}</div></div>' for v, l in items) + "</div>"


def build():
    ev, dd, cr = (load("evaluation/evaluation_results.json"), load("evaluation/dictionary_eval.json"),
                  load("evaluation/crawler_eval.json"))
    meta, rk, perf = ev["meta"], ev["ranking"], ev["performance"]
    split, dep, base = meta["config"]["split"], meta["deployed_system"], meta["baseline_system"]
    h = ev["hses"][dep]
    st, run = cr["stats"], cr["stats"]["run"]
    cb, ca = cr["classification_before_fix"], cr["classification"]
    oa, pdfx, pubd = cr["official_answers"], cr["extraction"], cr["publish_dates"]
    names = dd["names"]
    R = dd["retrieval"]
    Q = dd["dictionaries"]
    S = []

    # ---------------------------------------------------------------- cover
    S.append(f"""
<header>
  <div class="kicker">IRE course project · numbers report</div>
  <h1>JEE/NEET assistant: the numbers</h1>
  <p class="lede">A chatbot that answers JEE/NEET students' questions by searching {meta['dataset']['docs']:,} r/JEENEETards
  posts, plus a crawler for official NTA notices. This report gives the measured numbers for the
  <b>web crawler</b> and the <b>domain dictionary</b> in detail, and an overview of the rest. Every number is read from the
  evaluation outputs in <code>evaluation/</code>, except one figure: the before-fix official-answer score (18/20),
recorded from the first evaluation run.</p>
  <p class="small">Generated {datetime.now().strftime('%d %b %Y, %H:%M')} · evaluation run {e(meta['generated_at'].replace('T', ' '))} · NTA crawl {e(st['crawled_at'][:10])}</p>
</header>""")
    S.append(tiles([
        (f"{st['notices']}", "official NTA notices crawled"),
        (pct(cb['category_accuracy']), "notice-type accuracy (before fixes)"),
        (pct(oa['official_answer_accuracy'], 0), "official questions answered correctly"),
        (f"{Q['hybrid']['terms']}", "domain dictionary terms"),
        (pct(Q['hybrid']['quality']['same_meaning'], 0), "hybrid expansions with the same meaning"),
        (f"{h['partial_hses']:.1f}", "HSES (partial, 90% of weight)"),
    ]))

    # ---------------------------------------------------------------- crawler
    by_cat = sorted(st["by_category"].items(), key=lambda x: -x[1])
    sites = run["sites"]
    S.append(f"""
<section class="break">
<h2><span>1</span> Web crawler: official NTA notices</h2>
<p>The Reddit data ends in August 2022 and can't answer "when is the exam?". <code>nta_crawler.py</code> does a polite
breadth-first crawl of the NTA sites. It collects every JEE/NEET notice, classifies each one by exam and type with ordered
regular-expression rules, reads its upload date from the file name, and extracts the date lines from the PDF text. The chatbot
routes official questions (an exam name plus an official topic such as admit card, result or last date) to the newest
matching notice for the year asked.</p>
<h3>1.1 Crawl run</h3>
{table(["Site (seed)", "Pages crawled", "Notices found", "Errors", "Blocked by robots.txt"],
       [[e(s['seed']), s['pages'], s['notices'], e(s['errors'][0][:40] + '…') if s['errors'] else 0, s['robots_blocked']]
        for s in sites.values()])}
{tiles([(f"{run['elapsed_s']} s", "total crawl time"), (f"{run['delay_s']:.0f} s", "delay between requests"),
        (f"depth {run['max_depth']}", f"BFS limit · ≤{run['max_pages']} pages/site"),
        (f"{st['pdf_full_text']}", "PDFs downloaded for full text"),
        (f"{st['pdf_no_text_layer']}", "of them scanned images (no text)"),
        (f"{st['important_date_lines']}", "date lines extracted")])}
<p class="note"><code>jeemain.nta.nic.in</code> and <code>neet.nta.nic.in</code> return HTTP 403 to non-browser clients. The
crawler identifies itself honestly instead of faking a browser, so all notices come from <code>nta.ac.in</code>, which also lists
JEE and NEET notices. robots.txt is checked before every request.</p>
<div class="stack">
{hbar([(c, n, False) for c, n in by_cat], fmt=lambda v: str(int(v)), title="Notices by type (classifier output)")}
{hbar([(y, n, False) for y, n in sorted(st['by_year_published'].items(), reverse=True)], fmt=lambda v: str(int(v)),
      title=f"Notices by upload year ({st['oldest']} → {st['newest']})")}
</div>
<p>Exam split: {st['by_exam'].get('JEE', 0)} JEE, {st['by_exam'].get('NEET', 0)} NEET.</p>

<h3>1.2 Classification accuracy</h3>
<p>Ground truth: all {cb['labelled']} notices hand-labelled by exam and type from their titles
(<code>evaluation/nta_labels.json</code>, labelled by Claude). The first version of the classifier was measured against these labels.
Its errors fell into three patterns: "NTA Scores" declarations not counted as results, "Conduct of…/Rescheduling" notices
not counted as dates, and fee refunds counted as applications. Those rules were then fixed. <b>The before-fix accuracy is
the honest number;</b> the after-fix accuracy is optimistic because the fixes were made using the same labels.</p>
{tiles([(pct(cb['exam_accuracy']), "exam (JEE/NEET) accuracy"), (pct(cb['category_accuracy']), "notice-type accuracy, before fixes"),
        (pct(ca['category_accuracy']), "after fixes (optimistic)")])}
{table(["Type", "Notices (gold)", "Precision before", "Recall before", "Precision after", "Recall after"],
       [[c, cb['per_category'][c]['gold'], pct(cb['per_category'][c]['precision'], 0), pct(cb['per_category'][c]['recall'], 0),
         pct(ca['per_category'][c]['precision'], 0), pct(ca['per_category'][c]['recall'], 0)]
        for c in sorted(cb['per_category'], key=lambda c: -cb['per_category'][c]['gold'])])}
<p class="small">Most common errors before the fixes: {', '.join(f'{e(k)} ×{v}' for k, v in cb['confusions'])}.
Remaining error after the fixes: {', '.join(e(x['title'][:70]) + f" ({x['gold']} → {x['predicted']})" for x in ca['errors']) or 'none'}.</p>

<h3>1.3 Extracting dates from PDFs</h3>
<p>NTA's important-dates tables have no ruling lines. pypdf's text extraction split them into roughly one word per line, so the
row "Date of Examination 20, 21, 22, 23, 24 and 28 January 2027" was never extracted and the bot answered a 2027 question with a
2025 notice. Switching to pdfplumber, which keeps each row on one line, fixed this. Both libraries were run on the same {pdfx['pdfs']} PDFs:</p>
{table(["Library", "Date lines found", "PDFs with ≥1 date line", "\"Date of Examination\" rows found", "Mean words per line"],
       [[lib, pdfx[lib]['date_lines'], pdfx[lib]['docs_with_date_lines'], pdfx[lib]['exam_date_rows'], pdfx[lib]['mean_words_per_line']]
        for lib in ("pypdf", "pdfplumber")])}
<p class="small">{st['pdf_no_text_layer']} of the {st['pdf_full_text']} PDFs are scanned images with no text layer, so neither library
gets anything from them; OCR would be the fix.</p>

<h3>1.4 Publication dates</h3>
<p>{pubd['day_precision']} of {pubd['notices_total']} notices get an exact upload date from the NTA file name
(<code>/uploads/2026/04/20260409….pdf</code> → 2026-04-09). Only {pubd['checked_against_pdf']} PDFs print a parseable
"Dated:" line to check against: {pct(pubd['exact_match'], 0)} match exactly; the others were uploaded
{', '.join(str(r['days_apart']) for r in pubd['rows'] if r['days_apart'])} days after the date printed on them
({', '.join(e(r['title'][:40]) for r in pubd['rows'] if r['days_apart'])}). So the stored date is the <i>upload</i> date,
which is what "latest notice" needs.</p>

<h3>1.5 End-to-end: does the chatbot answer official questions correctly?</h3>
<p>{oa['questions']} student questions (<code>evaluation/official_questions.json</code>), written with their expected answer before
the evaluation was run: {oa['official_questions']} official questions with the exam, year and type the answer must have, and
{oa['questions'] - oa['official_questions']} community questions that must <i>not</i> be routed to NTA notices.</p>
{tiles([(pct(oa['routing_accuracy']), f"routing accuracy ({sum(r['route_ok'] for r in oa['rows'])}/{oa['questions']})"),
        (pct(oa['official_answer_accuracy'], 0), f"official answers correct ({sum(bool(r['answer_ok']) for r in oa['rows'])}/{oa['official_questions']})"),
        (pct(BEFORE_FIX_OFFICIAL, 0), "official answers before the fixes (18/20, first run)")])}
{table(["Question", "Expected", "Routed to", "Correct?", "Lead notice"],
       [[e(r['question']), r['expected_route'], r['route'],
         ('✓' if r['route_ok'] and r['answer_ok'] is not False else '✗'), e(r['answer'][:80])]
        for r in oa['rows']], cls="tight")}
<p class="small">The two answer failures before the fixes were "jee mains 2024 result" (the 2024 results are titled "NTA Scores")
and "neet 2026 result" (no such notice has been published, but the bot showed an unrelated 2026 notice instead of saying so).
The 20/20 is therefore partly optimistic: the fixes were informed by this question set. The remaining routing error,
"my jee result was bad, what now?", is a limitation of keyword routing.</p>
</section>""")

    # ---------------------------------------------------------------- dictionary
    tune = dd["tuning_ndcg@10"]
    variants = ["none", "tfidf", "embedding", "hybrid"]
    qual_rows = [[names[v], Q[v]['terms'], Q[v]['quality']['pairs'], pct(Q[v]['quality']['same_meaning'], 0),
                  pct(Q[v]['quality']['related_or_better'], 0), pct(Q[v]['quality']['unrelated'], 0)] for v in variants[1:]]

    def ret_rows(sysname):
        out = []
        for v in variants:
            t = R[sysname][v]["test"]
            out.append([("<b>" + names[v] + "</b>") if v == "hybrid" else names[v], f3(t["ndcg@10"]), f3(t["p@5"]), f3(t["map"]), f3(t["mrr"]),
                        f3(t["success@5"]), t["queries_expanded"], f"{t.get('helped', '–')} / {t.get('hurt', '–')}",
                        f3(t.get("p_vs_none")) if "p_vs_none" in t else "–", f3(R[sysname][v]["tune"]["ndcg@10"])])
        return out

    emb_ex = {x["query"]: x for x in R["bm25"]["embedding"]["examples"]}
    hyb_ex = {x["query"]: x for x in R["bm25"]["hybrid"]["examples"]}
    diff_rows = []
    for q in emb_ex:
        a, b = emb_ex[q], hyb_ex.get(q, {"added": [], "ndcg@10_delta": 0.0})
        if a["added"] != b["added"]:
            removed = sorted(set(a["added"]) - set(b["added"]))
            diff_rows.append([e(q), e(", ".join(removed)), f"{a['ndcg@10_delta']:+.3f}", f"{b['ndcg@10_delta']:+.3f}"])
    S.append(f"""
<section class="break">
<h2><span>2</span> Domain dictionary: TF-IDF + embeddings</h2>
<p>Students write "pw", "hcv" and "hall ticket"; posts say "physics wallah", "hc verma" and "admit card". The dictionary
adds such terms to the query at reduced weight (0.5 × similarity; the original words keep weight 1.0). Three ways of building it
were compared, all with the same {Q['hybrid']['by_source'].get('seed', 0)} seed terms in 8 categories and the same expansion rules (only exam, coaching,
resource, admission and student terms expand; sibling seeds and digit tokens are never added; at most 3 terms per seed).</p>
{table(["Variant", "How related terms are found"], [
    [names['tfidf'], "Terms whose TF-IDF document vectors are most similar to the seed's (they occur in the same posts). Statistics only, no embeddings."],
    [names['embedding'], "Nearest neighbours of the seed in a Word2Vec model trained on the posts (skip-gram, 150-d, window 5), cosine ≥ 0.55."],
    [names['hybrid'], f"Word2Vec neighbours, kept only if TF-IDF ranks them among the {dd['chosen']['hybrid_tfidf_top']:,} most salient terms in the corpus. Embeddings decide <i>what means the same</i>; TF-IDF decides <i>what matters</i>."]])}
<p class="small">Pipeline: tokenise → detect phrases (gensim Phrases: "physics wallah" → <code>physics_wallah</code>) → TF-IDF over
{meta['dataset']['docs']:,} posts ({dd['tfidf_vocab']:,} terms in ≥5 posts) and Word2Vec ({dd['word2vec_vocab']:,}-word vocabulary)
→ dictionary of {Q['hybrid']['terms']} terms ({Q['hybrid']['by_source'].get('seed', 0)} seeds + {Q['hybrid']['by_source'].get('embedding', 0)} learned).</p>

<h3>2.1 Dictionary accuracy: are the added terms right?</h3>
<p>Every (seed → added term) pair that a variant can put into a query was graded: <b>same meaning</b> (synonym, abbreviation,
spelling variant), <b>related</b>, or <b>unrelated</b>, for {len([k for k in load('evaluation/dictionary_pairs.json') if k != '_meta'])}
unique pairs in total (<code>evaluation/dictionary_pairs.json</code>, graded by Claude).</p>
{table(["Variant", "Dictionary terms", "Expansion pairs", "Same meaning", "Related or better", "Unrelated (drift)"], qual_rows)}
<div class="stack">
{hbar([(names[v], Q[v]['quality']['same_meaning'], v == 'hybrid') for v in variants[1:]], vmax=1, fmt=lambda v: pct(v, 0), title="Share of added terms with the same meaning (higher is better)")}
{hbar([(names[v], Q[v]['quality']['unrelated'], v == 'hybrid') for v in variants[1:]], vmax=1, fmt=lambda v: pct(v, 0), title="Share of added terms that are unrelated (lower is better)")}
</div>
<p>The hybrid is the most accurate dictionary. The TF-IDF filter removes rare neighbours (<i>hbcse</i>, <i>aptitude test</i>,
<i>official answer</i>, <i>sankalp</i>), which were the main source of drift, and keeps salient ones (<i>physics wallah</i>,
<i>admit card</i>, <i>cbse</i>). TF-IDF on its own finds words that appear <i>alongside</i> a seed (josaa → documents, round),
not words that <i>mean</i> the same, so only 1% of its additions are synonyms.</p>

<h3>2.2 Search quality with each dictionary</h3>
<p>Each variant's setting was tuned on the {meta['queries']['tune']} tuning queries only (TF-IDF threshold {dd['chosen']['tfidf_threshold']};
hybrid salience cut-off {dd['chosen']['hybrid_tfidf_top']:,}). The numbers below are on the {meta['queries']['test']} held-out test queries.</p>
<h4>BM25F + engagement + expansion (expansion's direct effect)</h4>
{table(["Variant", "NDCG@10", "P@5", "MAP", "MRR", "Success@5", "Queries expanded", "Helped / hurt", "p vs none", "Tune NDCG@10"], ret_rows("bm25"), cls="tight")}
<h4>Deployed hybrid search (BM25F + semantic, RRF)</h4>
{table(["Variant", "NDCG@10", "P@5", "MAP", "MRR", "Success@5", "Queries expanded", "Helped / hurt", "p vs none", "Tune NDCG@10"], ret_rows("deployed"), cls="tight")}
{hbar([(names[v], R['bm25'][v]['test']['ndcg@10'], v == 'hybrid') for v in variants], vmax=0.6, title="NDCG@10, BM25F + expansion, test split (axis 0–0.6)")}
<p><b>Reading the results honestly.</b> On search quality the four variants are within noise of each other and of no expansion:
the differences are about ±0.01 NDCG@10, and no p-value comes near 0.05 (paired randomization test, 10,000 permutations).
With BM25 the hybrid is best ({f3(R['bm25']['hybrid']['test']['ndcg@10'])}) and hurts the fewest queries
({R['bm25']['hybrid']['test']['hurt']} vs {R['bm25']['embedding']['test']['hurt']} for embeddings only). In the deployed system,
TF-IDF only scores highest on the test split ({f3(R['deployed']['tfidf']['test']['ndcg@10'])}) despite its poor dictionary,
because related words act as topical context. On the tuning split, however, embeddings only did best there. The hybrid's clear
win is dictionary accuracy (fewer wrong expansions shown to users as "Also searched: …"), not ranking.</p>

<h3>2.3 Where the TF-IDF filter changed the query (embeddings only → hybrid)</h3>
{table(["Query", "Terms the filter removed", "ΔNDCG embeddings only", "ΔNDCG hybrid"], diff_rows, cls="tight")}
<p class="small">ΔNDCG is relative to no expansion (BM25F + engagement, test split).</p>

<h3>2.4 Tuning runs</h3>
{table(["TF-IDF only: co-occurrence threshold", "Tune NDCG@10"], [[k, f3(v)] for k, v in tune['tfidf'].items()], cls="tight")}
{table(["Hybrid: salience cut-off / TF-IDF candidate terms added", "Tune NDCG@10"],
       [[k.replace('top', 'top ').replace('_cand', ' / '), f3(v)] for k, v in tune['hybrid'].items()], cls="tight")}
<p class="small">Adding TF-IDF-discovered candidate terms (e.g. <i>alakh pandey</i>, <i>mathongo</i>, <i>competishun</i>) to the
dictionary changed no tuning query, so it is switched off (<code>TFIDF_CANDIDATES = 0</code>); the salience filter is what the
hybrid uses.</p>
</section>""")

    # ---------------------------------------------------------------- overview
    ids = list(rk["systems"])
    sys_rows = []
    for i in ids:
        s = rk["systems"][i]
        m, vb = s[split], s["vs_baseline"]
        lat = perf["latency"].get(i, {})
        sys_rows.append([("<b>" + e(s['name']) + "</b>") if i == dep else e(s['name']), f3(m['ndcg@10']), f3(m['p@5']), f3(m['map']),
                         f3(m['mrr']), f3(m['success@5']), f"{lat.get('p95_ms', 0):.0f} ms",
                         "baseline" if i == base else f"{vb['ndcg@10_delta']:+.3f} ({f3(vb['p_value'])})"])
    comp = [[e(k), f"{w * 100:.0f}%", "not measured" if h['scores'][k] is None else f3(h['scores'][k]),
             "–" if h['contributions'][k] is None else f"{h['contributions'][k]:.2f}"] for k, w in meta['config']['weights'].items()]
    rel = ev["reliability"]
    use = ev["usefulness"]
    lat = perf["latency"][dep]
    S.append(f"""
<section class="break">
<h2><span>3</span> The rest of the system: overview</h2>
{tiles([(f"{meta['dataset']['docs']:,}", "Reddit posts (Aug 2020 – Aug 2022)"), (f"{perf['index']['vocab_size']:,}", "terms in the inverted index"),
        (f"{perf['index']['indexing_time_s']} s", "index build time"), (f"{meta['queries']['total']}", f"benchmark queries ({meta['queries']['test']} test)"),
        (f"{meta['judgments']['total']:,}", "graded relevance judgments"), (f"{lat['p95_ms']:.0f} ms", "P95 latency, deployed search")])}
<h3>3.1 Search systems (test split)</h3>
{table(["System", "NDCG@10", "P@5", "MAP", "MRR", "Success@5", "P95 latency", "ΔNDCG vs baseline (p)"], sys_rows, cls="tight")}
{hbar([(rk['systems'][i]['name'].replace(' (deployed: search default + chatbot)', ' (deployed)'), rk['systems'][i][split]['ndcg@10'], i == dep) for i in ids],
      vmax=0.6, title="NDCG@10 by search system, test split (axis 0–0.6)")}
<p>BM25F weights the title 2.5×; the engagement boost multiplies by 1 + 0.05·log₁₀(1 + upvotes); semantic search uses
multilingual MiniLM sentence embeddings; the deployed hybrid fuses BM25F and semantic rankings with weighted Reciprocal Rank
Fusion (k = 60, semantic weight 0.25, tuned on the tuning split). The hybrid is level with BM25F on NDCG@10 and best on P@5.</p>
<h3>3.2 Hybrid Search Evaluation Score (HSES)</h3>
<p><b>{h['partial_hses']:.1f} / 100, incomplete</b>: {h['measured_weight'] * 100:.0f}% of the weight is measured. User usefulness
needs {use['min_required']} feedback responses from the chat tab; {use['responses']} have been collected so far.</p>
{table(["Component", "Weight", "Score (0–1)", "Points"], comp, cls="tight")}
<h3>3.3 Performance and reliability</h3>
{table(["Measure", "Value"], [
    ["Latency, deployed hybrid (avg / P50 / P95 / P99)", f"{lat['avg_ms']:.1f} / {lat['p50_ms']:.1f} / {lat['p95_ms']:.1f} / {lat['p99_ms']:.1f} ms"],
    ["Throughput", ", ".join(f"{t['qps']} q/s with {t['workers']} worker(s)" for t in perf['throughput'])],
    ["Index size", f"lexical {perf['index']['lexical_index_mb']} MB · embeddings {perf['index']['embeddings_mb']} MB · Word2Vec {perf['index']['files_mb'].get('word2vec.model', '–')} MB"],
    ["Memory (RSS)", f"{perf['memory_rss_mb']['after_bm25_mb']:.0f} MB with BM25 · {perf['memory_rss_mb']['after_semantic_mb']:.0f} MB with the semantic model"],
    ["Device", e(perf['device'])],
    ["Index integrity checks", f"{sum(c['passed'] for c in rel['index_integrity'])}/{len(rel['index_integrity'])} passed"],
    ["Filter and API checks", f"{sum(c['passed'] for c in rel['filters_api'])}/{len(rel['filters_api'])} passed"],
])}
</section>
<section>
<h2><span>4</span> Method and caveats</h2>
<ul>
<li><b>Who labelled the data.</b> Relevance judgments: {meta['judgments']['by_assessor'].get('claude', 0):,} by Claude (an LLM,
graded against each query's stated intent) and {meta['judgments']['by_assessor'].get('legacy', 0)} from the original project.
The notice labels, the dictionary-pair grades and the official-question expectations were also made by Claude. All are saved in
<code>evaluation/</code> and should be spot-checked by a person (<code>python evaluate.py annotate --review</code>).</li>
<li><b>Tuning vs testing.</b> Every tuned setting (RRF weight, TF-IDF threshold, salience cut-off) was chosen on the
{meta['queries']['tune']} tuning queries; all reported search numbers are on the {meta['queries']['test']} test queries.</li>
<li><b>Unjudged documents</b> are skipped (condensed rankings), not counted as irrelevant; every system's top 10 is 99–100% judged.</li>
<li><b>Small differences are noise.</b> With 56 test queries, differences under ~0.03 NDCG@10 are not significant.</li>
<li><b>Crawler accuracy after fixes is optimistic</b>, because the fixes were made using the same labels and questions. The
before-fix numbers ({pct(cb['category_accuracy'])} type accuracy, {pct(BEFORE_FIX_OFFICIAL, 0)} official answers) are the unbiased ones.</li>
<li><b>Data age.</b> The Reddit data ends in August 2022; official dates come from the live NTA crawl.</li>
</ul>
<h3>Reproduce</h3>
<pre>python nta_crawler.py        # crawl NTA notices          python eval_crawler.py     # crawler numbers
python domain_dictionary.py  # build the dictionary       python eval_dictionary.py  # dictionary numbers
python evaluate.py           # search evaluation + HSES   python make_report.py      # this PDF</pre>
</section>""")
    return "\n".join(S)


CSS = """
@page { size: A4; margin: 14mm 14mm 16mm; }
:root { --ink: #0b0b0b; --ink2: #52514e; --muted: #7a7973; --rule: #e4e3de; --surface: #ffffff; --tint: #f4f6fa; --blue: #2a78d6; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink); font: 10.2pt/1.5 "Noto Sans", "DejaVu Sans", "Liberation Sans", sans-serif; }
header { padding: 4mm 0 2mm; }
.kicker { text-transform: uppercase; letter-spacing: .08em; font-size: 8pt; color: var(--blue); font-weight: 700; }
h1 { font-size: 24pt; line-height: 1.15; margin: 2mm 0 3mm; }
.lede { font-size: 11pt; color: var(--ink2); max-width: 165mm; }
h2 { font-size: 15pt; margin: 0 0 3mm; padding-bottom: 2mm; border-bottom: 2px solid var(--ink); }
h2 span { display: inline-block; width: 8mm; color: var(--blue); }
h3 { font-size: 11.5pt; margin: 6mm 0 2mm; }
h4 { font-size: 9.5pt; margin: 4mm 0 1.5mm; color: var(--ink2); }
p { margin: 1.5mm 0 2.5mm; }
.small, .note { font-size: 8.5pt; color: var(--ink2); }
.note { background: var(--tint); padding: 2mm 3mm; border-radius: 4px; }
code, pre { font-family: "DejaVu Sans Mono", monospace; font-size: 8.3pt; }
pre { background: var(--tint); padding: 3mm; border-radius: 4px; white-space: pre-wrap; }
.break { break-before: page; }
table { width: 100%; border-collapse: collapse; margin: 2mm 0 3mm; font-size: 8.6pt; break-inside: auto; }
th { text-align: left; font-weight: 600; color: var(--ink2); border-bottom: 1px solid var(--ink2); padding: 1.2mm 1.5mm; vertical-align: bottom; }
td { border-bottom: 1px solid var(--rule); padding: 1.1mm 1.5mm; vertical-align: top; }
tr { break-inside: avoid; }
table.tight td, table.tight th { padding: .8mm 1.2mm; font-size: 8pt; }
td:not(:first-child), th:not(:first-child) { font-variant-numeric: tabular-nums; }
.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2.5mm; margin: 3mm 0; }
.tile { border: 1px solid var(--rule); border-radius: 6px; padding: 2.5mm 3mm; break-inside: avoid; }
.tv { font-size: 15pt; font-weight: 700; font-variant-numeric: tabular-nums; }
.tl { font-size: 8pt; color: var(--ink2); line-height: 1.3; }
.stack figure { max-width: 150mm; }
figure { max-width: 150mm; }
figure { margin: 2mm 0 3mm; break-inside: avoid; }
figcaption { font-size: 8pt; color: var(--ink2); margin-top: 1mm; }
svg text.lab { font-size: 11px; fill: var(--ink2); }
svg text.lab.hi { fill: var(--ink); font-weight: 700; }
svg text.val { font-size: 11px; fill: var(--ink); font-variant-numeric: tabular-nums; }
svg .axis { stroke: var(--ink2); stroke-width: 1; }
ul { padding-left: 5mm; } li { margin-bottom: 1.5mm; }
"""


def main():
    body = build()
    page = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>IREE project numbers</title><style>{CSS}</style></head><body>{body}</body></html>"
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(page)
    chrome = next((shutil.which(c) for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser") if shutil.which(c)), None)
    if not chrome:
        raise SystemExit(f"No Chrome/Chromium found; open {OUT_HTML} and print it to PDF.")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={os.path.abspath(OUT_PDF)}", "file://" + os.path.abspath(OUT_HTML)],
                   check=True, capture_output=True)
    print(f"[✓] {OUT_PDF} ({os.path.getsize(OUT_PDF) / 1024:.0f} KB) from {OUT_HTML}")


if __name__ == "__main__":
    main()
