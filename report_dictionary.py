"""
Chapter 2 of IREE_Project_Numbers.pdf: the domain dictionary, explained end to end.
Everything quoted (neighbours, ranks, weights, rankings, grades) is computed here from
index/domain_dictionary.json, the BM25 index, the qrels and evaluation/dictionary_eval.json.
"""

import io
import contextlib

import domain_dictionary as dd
from indexer import tokenize
from eval_metrics import evaluate_ranking

EXAMPLE_SEEDS = ["pw", "admit_card", "hcv", "josaa", "kvpy", "answer_key", "cutoff", "boards", "mock", "chemistry"]
WORKED = ["hall ticket kab aayega", "PW vs Unacademy", "boards aur jee dono kaise manage kare", "answer key challenge"]


def neighbour_fate(d, seed, term, sim):
    """Why expand_query does or doesn't add `term` when `seed` is in the query (mirrors its rules)."""
    if d["terms"][seed]["category"] not in dd.EXPANDABLE:
        return "not expanded (category)"
    if term in dd.SEED_TERMS and term != seed:
        return "blocked: sibling seed"
    if any(c.isdigit() for c in term):
        return "blocked: contains digits"
    rank = d["terms"].get(term, {}).get("tfidf_rank")
    if rank is None or rank >= dd.TFIDF_TOP:
        return "filtered: rare (TF-IDF)"
    if sim < dd.MIN_WEIGHT:
        return "dropped: similarity < 0.6"
    return "added"


def section(h, ctx):
    """h: html helpers (table, hbar, tiles, e, pct, f3). ctx: loaded results + engine + qrels."""
    table, hbar, tiles, e, pct, f3 = h["table"], h["hbar"], h["tiles"], h["e"], h["pct"], h["f3"]
    dde, meta, engine, qrels, queries = ctx["dd"], ctx["meta"], ctx["engine"], ctx["qrels"], ctx["queries"]
    d = engine.domain_dict
    names, R, Q, tune = dde["names"], dde["retrieval"], dde["dictionaries"], dde["tuning_ndcg@10"]
    variants = ["none", "tfidf", "embedding", "hybrid"]

    with contextlib.redirect_stdout(io.StringIO()):
        ranks = dd.tfidf_ranks(dd.phrased_sentences())
    from gensim.models import Word2Vec
    w2v = Word2Vec.load(dd.MODEL_PATH)
    n_phrases = sum("_" in w for w in w2v.wv.index_to_key)
    phrase_examples = sorted((t for t in d["terms"] if "_" in t), key=lambda t: d["terms"][t].get("tfidf_rank") or 1e9)[:14]
    top_overall = sorted(ranks, key=ranks.get)[:16]
    top_domain = sorted((t for t in d["terms"] if d["terms"][t].get("tfidf_rank") is not None),
                        key=lambda t: d["terms"][t]["tfidf_rank"])[:16]
    src = Q["hybrid"]["by_source"]
    S = []

    # ------------------------------------------------------------ 2.1 problem
    S.append(f"""
<section class="break">
<h2><span>2</span> Domain dictionary</h2>
<h3>2.1 The problem: vocabulary mismatch</h3>
<p>The search engine (BM25F) finds posts that contain the <i>same words</i> as the question, after stemming. Students and
posts often use different words for the same thing:</p>
{table(["A student types…", "…but many relevant posts say", "Kind of variation"], [
    ["pw", "physics wallah, physicswallah", "abbreviation / brand spelling"],
    ["hall ticket", "admit card", "synonym"],
    ["hcv", "hc verma", "abbreviation of a book"],
    ["boards", "board, cbse", "inflection / specific instance"],
    ["unacademy", "unacdemy", "misspelling"],
    ["josaa", "jossa, counselling", "misspelling / process name"]], cls="tight")}
<p>A general thesaurus such as WordNet knows none of this slang, abbreviations or Hinglish. The domain dictionary
<b>learns these equivalences from the {meta['dataset']['docs']:,} posts themselves</b> and adds them to the query. This
is called <b>query expansion</b>.</p>

<h3>2.2 Where it is used</h3>
<div class="flow">
  <div class="box">Student question<br><small>chat tab or search tab</small></div><div class="arr">→</div>
  <div class="box hi"><b>expand_query()</b><br><small>adds dictionary terms</small></div><div class="arr">→</div>
  <div class="box">BM25F ranking<br><small>added terms at reduced weight</small></div><div class="arr">→</div>
  <div class="box">RRF fusion<br><small>with semantic ranking</small></div><div class="arr">→</div>
  <div class="box">Answer / results<br><small>"Also searched: …"</small></div>
</div>
{table(["Place", "What happens"], [
    ["<b>Chat tab</b> (community answers)", "<code>chatbot.py</code> calls <code>hybrid_search(…, expand=True)</code>. The expanded query goes to the BM25F half; the semantic half uses the original question; RRF fuses both. The answer shows \"Also searched: …\"."],
    ["<b>Search tab</b>", "<code>/api/search?expand=1</code> (default on; the <i>expand</i> toggle turns it off). The added terms appear as <code>+term</code> tags, and matches are highlighted in snippets."],
    ["<b>Ranker</b>", "<code>ranker.py</code> loads <code>index/domain_dictionary.json</code> at start-up and calls <code>expand_query()</code> when <code>expand=True</code>."],
    ["<b>Not used</b>", "Official NTA questions (they are matched by the crawler's classifier) and the semantic model (it already handles meaning through sentence embeddings)."],
])}
</section>""")

    # ------------------------------------------------------------ 2.3 building
    seed_rows = [[c, ", ".join(e(s) for s in seeds), "yes" if c in dd.EXPANDABLE else "no (siblings, not synonyms)"]
                 for c, seeds in dd.SEEDS.items()]
    nb_rows = []
    for seed in EXAMPLE_SEEDS:
        for t, sim in d["related"].get(seed, [])[:3]:
            rank = d["terms"].get(t, {}).get("tfidf_rank")
            fate = neighbour_fate(d, seed, t, sim)
            nb_rows.append([e(seed), e(t.replace("_", " ")), f"{sim:.3f}", "–" if rank is None else f"{rank + 1:,}",
                            f'<span class="{"ok" if fate == "added" else "bad"}">{e(fate)}</span>'])
    S.append(f"""
<section class="break">
<h3>2.3 How the dictionary is built (<code>python domain_dictionary.py</code>)</h3>
<div class="flow small-flow">
  <div class="box">{meta['dataset']['docs']:,} posts</div><div class="arr">→</div>
  <div class="box">1 · tokenise</div><div class="arr">→</div>
  <div class="box">2 · phrases</div><div class="arr">→</div>
  <div class="box">3 · TF-IDF ranks<br>4 · Word2Vec</div><div class="arr">→</div>
  <div class="box">5 · seeds → 6 · neighbours</div><div class="arr">→</div>
  <div class="box hi">domain_dictionary.json</div>
</div>

<h4>Step 1 · Tokenisation</h4>
<p>Each post's title and body are lower-cased and split into word tokens with the pattern <code>[a-z0-9]+('[a-z]+)?</code>.
Hinglish is kept as is ("kaise", "karu"). No stop words are removed, because Word2Vec needs the full context.</p>

<h4>Step 2 · Phrase detection (collocations)</h4>
<p>Many domain terms are multi-word ("physics wallah", "hall ticket", "jee mains"). gensim's <code>Phrases</code> joins two
adjacent words into one token when they occur together far more often than chance:</p>
<div class="formula">score(a, b) = (count(a b) − min_count) × |V| ÷ (count(a) × count(b)) &nbsp;&nbsp;→ join if score &gt; 15
<br><small>min_count = 10, |V| = vocabulary size; this is a PMI-style association score</small></div>
<p>The final vocabulary has <b>{n_phrases:,} phrase tokens</b>. Examples in the dictionary: {', '.join('<code>' + e(t) + '</code>' for t in phrase_examples)}.</p>

<h4>Step 3 · TF-IDF salience (the statistical part)</h4>
<p>TF-IDF scores how characteristic a term is of a document: frequent in the post (TF), rare across posts (IDF). We use it to
rank <b>every term in the corpus by importance</b>:</p>
<div class="formula">tfidf(t, d) = (1 + ln tf(t, d)) × (ln((1 + N) ÷ (1 + df(t))) + 1), each post vector L2-normalised
<br>salience(t) = Σ<sub>d</sub> tfidf(t, d) &nbsp;&nbsp;→ rank 1 = most salient
<br><small>N = {meta['dataset']['docs']:,} posts; terms in fewer than 5 posts are dropped ({dde['tfidf_vocab']:,} terms ranked); scikit-learn TfidfVectorizer, sublinear tf</small></div>
<p>On its own, TF-IDF salience can't tell domain terms from chatter. The most salient terms overall are
{', '.join('<code>' + e(t) + '</code>' for t in top_overall)}. Restricted to dictionary terms, it ranks
{', '.join('<code>' + e(t) + '</code>' for t in top_domain)} highest. So TF-IDF knows what <i>matters</i>, but not what
<i>means the same</i>.</p>

<h4>Step 4 · Word2Vec embeddings (the semantic part)</h4>
<p>Word2Vec learns a vector for every word so that <b>words used in similar contexts get similar vectors</b> (the
<i>distributional hypothesis</i>). "pw" and "physics wallah" appear around the same words ("lectures", "batch", "alakh sir"), so
their vectors end up close together. We train the <b>skip-gram</b> variant: for each word, predict the words within ±5 positions,
learning with <b>negative sampling</b> (5 random "wrong" words per real one). Similarity is the <b>cosine</b> of two vectors:</p>
<div class="formula">cos(u, v) = (u · v) ÷ (‖u‖ ‖v‖) &nbsp;&nbsp;∈ [−1, 1]
<br><small>{w2v.vector_size} dimensions · window {w2v.window} · min_count {w2v.min_count} · {w2v.epochs} epochs · seed 42 · vocabulary {len(w2v.wv):,} words (incl. phrases)</small></div>

<h4>Step 5 · Seed terms and categories</h4>
<p>{sum(len(v) for v in dd.SEEDS.values())} hand-picked seed terms in 8 categories anchor the dictionary. The embeddings then
discover the rest automatically. Only 5 categories are <i>expanded</i>: in subject, topic and institute, the nearest neighbours
are siblings (chemistry → physics, iit → nit), not synonyms, and adding them would change the question's meaning.</p>
{table(["Category", "Seed terms", "Expanded in queries?"], seed_rows, cls="tight")}

<h4>Step 6 · Neighbours + the TF-IDF filter</h4>
<p>For each seed, the 15 nearest words with cosine ≥ {dd.MIN_SIM} become dictionary terms ({src.get('embedding', 0)} learned + {src.get('seed', 0)} seeds
= {Q['hybrid']['terms']} terms). Each term records its category, its closest seed, the similarity and its TF-IDF rank. At query
time a neighbour is only used if it ranks in the top {dd.TFIDF_TOP:,} by TF-IDF salience. Real neighbours and what happens to them:</p>
{table(["Seed", "Word2Vec neighbour", "Cosine", "TF-IDF rank", "In a query"], nb_rows, cls="tight")}
<p class="small">Output: <code>index/domain_dictionary.json</code> = {{"terms": {{term: {{category, seed, similarity, tfidf_rank, source}}}},
"related": {{seed: [[neighbour, cosine], …]}}}}. Build time: about 10 s (Word2Vec is reused from <code>index/word2vec.model</code>;
<code>--retrain</code> rebuilds it).</p>
</section>""")

    # ------------------------------------------------------------ 2.4 query time
    worked_html = []
    qid_by_text = {q["query"]: q["query_id"] for q in queries}
    for query in WORKED:
        exp = dd.expand_query(query, d)
        weights = {s: 1.0 for s in tokenize(query)}
        for phrase, sim in exp.items():
            for s in tokenize(phrase):
                weights[s] = max(weights.get(s, 0.0), 0.5 * sim)
        qr = qrels.get(qid_by_text.get(query, ""), {})
        res = {}
        for flag in (False, True):
            r = engine.search(query, top_k=20, expand=flag)
            ids = [x["doc_id"] for x in r["results"]]
            res[flag] = (r["results"][:5], evaluate_ranking(ids, qr)["ndcg@10"] if qr else None)
        grade = lambda doc: {2: "highly relevant", 1: "relevant", 0: "not relevant"}.get(qr.get(doc), "not judged") if qr else ""
        rows = []
        for i in range(5):
            cells = []
            for flag in (False, True):
                lst = res[flag][0]
                cells.append(e(lst[i]["title"][:58]) + (f' <span class="g">({grade(lst[i]["doc_id"])})</span>' if qr else "") if i < len(lst) else "")
            rows.append([i + 1] + cells)
        nd = (f"NDCG@10 without expansion {f3(res[False][1])} → with {f3(res[True][1])}" if qr else
              "Not a benchmark query, so no relevance judgments.")
        worked_html.append(f"""
<div class="worked">
<p><b>Query:</b> <code>{e(query)}</code></p>
<p class="small">Stems: {', '.join(e(s) for s in tokenize(query))} · Dictionary adds: {', '.join(f'<b>{e(t)}</b> (cosine {s:.3f})' for t, s in exp.items()) or 'nothing'}
· Term weights: {', '.join(f'{e(s)} {w:.2f}' for s, w in weights.items())}</p>
{table(["#", "Top 5 without expansion", "Top 5 with expansion"], rows, cls="tight")}
<p class="small">{nd}</p>
</div>""")

    S.append(f"""
<section class="break">
<h3>2.4 How a query is expanded (<code>expand_query()</code>)</h3>
<ol>
<li>Split the query into words and adjacent word pairs ("hall ticket" → <code>hall_ticket</code>). A matched pair uses up
its words, so "physics wallah" is not also treated as "physics".</li>
<li>Look each one up in the dictionary. A hit pulls in its <b>seed</b> and the seed's <b>top 3 neighbours</b>.</li>
<li>Drop a candidate term if it is already in the query, is another seed (allen → aakash would change the question), contains
digits ("2024tard"), ranks outside the TF-IDF top {dd.TFIDF_TOP:,}, or has similarity below {dd.MIN_WEIGHT}.</li>
<li>Give each added term weight <b>0.5 × similarity</b>. The student's own words keep weight 1.0, so expansion can nudge the
ranking but never overpower the question.</li>
</ol>
<p>Inside BM25F, the weight multiplies the term's score:</p>
<div class="formula">score(D, Q) = Σ<sub>t ∈ Q</sub> <b>w(t)</b> × IDF(t) × tf̃(t, D) × (k₁ + 1) ÷ (tf̃(t, D) + k₁)
<br><small>tf̃ = 2.5 × normalised title tf + 1.0 × normalised body tf; k₁ = 1.2, b = 0.75; w(t) = 1 for query words, 0.5 × cosine for added words</small></div>
<h4>Worked examples (BM25F + engagement, real index)</h4>
{''.join(worked_html)}
</section>""")

    # ------------------------------------------------------------ 2.5 concepts + params
    S.append(f"""
<section>
<h3>2.5 Key concepts</h3>
{table(["Concept", "What it means here"], [
    ["Query expansion", "Adding related terms to a query to find relevant posts that use different words. It improves <i>recall</i>, at the risk of drift."],
    ["Vocabulary mismatch", "The question and a relevant post describe the same thing with different words (pw / physics wallah)."],
    ["Query drift", "Expansion adds a term that changes the question's meaning (\"answer key challenge\" + \"official answer\" pulls in unrelated posts). Precision drops."],
    ["Distributional hypothesis", "Words that occur in similar contexts have similar meanings. This is the basis of Word2Vec."],
    ["Word embedding (Word2Vec, skip-gram)", "A dense vector per word, learned by predicting nearby words, so that similar words are near each other."],
    ["Negative sampling", "Training trick: instead of scoring every word in the vocabulary, push the real context word up and 5 random words down."],
    ["Cosine similarity", "The angle between two vectors (1 = same direction). It measures how similar two words are."],
    ["Collocation / phrase detection", "Finding word pairs that occur together more than chance (PMI-style score) and treating them as one term."],
    ["TF-IDF", "Term frequency × inverse document frequency: a term matters if it is frequent in a post but rare across posts."],
    ["Sublinear TF, smoothed IDF", "1 + ln(tf) stops very repetitive posts from dominating; +1 smoothing avoids division by zero."],
    ["Salience", "A term's total TF-IDF weight over the corpus: how important it is to this corpus. Used here as a filter."],
    ["Paradigmatic vs syntagmatic relations", "Word2Vec finds words that can <i>replace</i> each other (pw ↔ physics wallah); TF-IDF co-occurrence finds words that appear <i>together</i> (josaa ↔ documents). This is why TF-IDF alone gives few synonyms."],
    ["Seed-based lexicon induction", "Starting from a few hand-picked terms per category and growing the dictionary automatically from the data (weak supervision)."],
    ["Intrinsic vs extrinsic evaluation", "Intrinsic: are the dictionary entries correct? Extrinsic: does search get better with them? Both are reported below."],
    ["Tuning / test split", "Settings are chosen on 24 tuning queries; results are reported on 56 unseen test queries, so they are not overfitted."],
], cls="tight")}

<h3>2.6 All parameters</h3>
{table(["Parameter", "Value", "Where / why"], [
    ["Phrase min_count / threshold", "10 / 15", "gensim Phrases: a pair must occur ≥10 times and score >15"],
    ["Word2Vec", f"skip-gram, {w2v.vector_size}-d, window {w2v.window}, min_count {w2v.min_count}, {w2v.epochs} epochs, negative 5, seed 42", "word vectors"],
    ["TF-IDF", "min_df 5, sublinear tf, smoothed idf, L2 norm", "salience ranks for every term"],
    ["Neighbours per seed (TOPN)", str(dd.TOPN), "candidate dictionary terms"],
    ["MIN_SIM", str(dd.MIN_SIM), "minimum cosine to enter the dictionary"],
    ["Neighbours used per hit (k)", "3", "how many neighbours a query word pulls in"],
    ["MIN_WEIGHT", str(dd.MIN_WEIGHT), "minimum (seed similarity × neighbour similarity) to be added"],
    ["TFIDF_TOP", f"{dd.TFIDF_TOP:,}", "neighbour must be among the most salient terms (tuned on the tuning split)"],
    ["TFIDF_CANDIDATES", str(dd.TFIDF_CANDIDATES), "TF-IDF-discovered terms added to the dictionary (off: no effect on tuning queries)"],
    ["expansion_weight", "0.5", "added term's weight = 0.5 × similarity (ranker.py)"],
    ["Expanded categories", ", ".join(sorted(dd.EXPANDABLE)), "subject, topic and institute are not expanded"],
], cls="tight")}
</section>""")

    # ------------------------------------------------------------ 2.7+ evaluation (existing numbers)
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
            diff_rows.append([e(q), e(", ".join(sorted(set(a["added"]) - set(b["added"])))),
                              f"{a['ndcg@10_delta']:+.3f}", f"{b['ndcg@10_delta']:+.3f}"])
    eb, hb = R["bm25"]["embedding"]["test"], R["bm25"]["hybrid"]["test"]
    S.append(f"""
<section class="break">
<h3>2.7 How it was evaluated</h3>
<p>Three ways of building the dictionary were compared. All use the same seeds, categories and expansion rules:</p>
{table(["Variant", "How related terms are found"], [
    [names['tfidf'], "Terms whose TF-IDF document vectors are most similar to the seed's (they occur in the same posts). Statistics only. Cosines are rescaled onto the same 0.6–1.0 weight range."],
    [names['embedding'], "Word2Vec nearest neighbours (the original dictionary)."],
    [names['hybrid'], f"Word2Vec neighbours kept only if TF-IDF ranks them in the top {dde['chosen']['hybrid_tfidf_top']:,} (the current dictionary)."]])}
<ul>
<li><b>Intrinsic:</b> every (seed → added term) pair a variant can put into a query was graded <i>same meaning</i> (synonym,
abbreviation, spelling variant), <i>related</i> or <i>unrelated</i>, for {sum(Q[v]['quality']['pairs'] for v in variants[1:])} pairs
across the variants (183 unique; <code>evaluation/dictionary_pairs.json</code>, graded by Claude).</li>
<li><b>Extrinsic:</b> search quality on the {meta['queries']['test']} test queries with graded relevance judgments: NDCG@10, P@5,
MAP, MRR and Success@5, plus a paired randomization test against no expansion. Settings (TF-IDF threshold
{dde['chosen']['tfidf_threshold']}, salience cut-off {dde['chosen']['hybrid_tfidf_top']:,}) were tuned on the
{meta['queries']['tune']} tuning queries only.</li>
</ul>

<h3>2.8 Results: dictionary accuracy</h3>
{table(["Variant", "Dictionary terms", "Expansion pairs", "Same meaning", "Related or better", "Unrelated (drift)"], qual_rows)}
<div class="stack">
{hbar([(names[v], Q[v]['quality']['same_meaning'], v == 'hybrid') for v in variants[1:]], vmax=1, fmt=lambda v: pct(v, 0), title="Share of added terms with the same meaning (higher is better)")}
{hbar([(names[v], Q[v]['quality']['unrelated'], v == 'hybrid') for v in variants[1:]], vmax=1, fmt=lambda v: pct(v, 0), title="Share of added terms that are unrelated (lower is better)")}
</div>

<h3>2.9 Results: search quality (test split)</h3>
<h4>BM25F + engagement + expansion (expansion's direct effect)</h4>
{table(["Variant", "NDCG@10", "P@5", "MAP", "MRR", "Success@5", "Queries expanded", "Helped / hurt", "p vs none", "Tune NDCG@10"], ret_rows("bm25"), cls="tight")}
<h4>Deployed hybrid search (BM25F + semantic, RRF)</h4>
{table(["Variant", "NDCG@10", "P@5", "MAP", "MRR", "Success@5", "Queries expanded", "Helped / hurt", "p vs none", "Tune NDCG@10"], ret_rows("deployed"), cls="tight")}
{hbar([(names[v], R['bm25'][v]['test']['ndcg@10'], v == 'hybrid') for v in variants], vmax=0.6, title="NDCG@10, BM25F + expansion, test split (axis 0–0.6)")}
</section>

<section class="break">
<h3>2.10 How the hybrid improved on the original Word2Vec dictionary</h3>
{table(["Measure", "Word2Vec only (before)", "Hybrid (now)"], [
    ["Expansions with the same meaning", pct(Q['embedding']['quality']['same_meaning'], 0), pct(Q['hybrid']['quality']['same_meaning'], 0)],
    ["Unrelated expansions (drift)", pct(Q['embedding']['quality']['unrelated'], 0), pct(Q['hybrid']['quality']['unrelated'], 0)],
    ["Distinct expansion pairs", Q['embedding']['quality']['pairs'], Q['hybrid']['quality']['pairs']],
    ["Test queries expanded", eb['queries_expanded'], hb['queries_expanded']],
    ["Queries helped / hurt (vs no expansion)", f"{eb['helped']} / {eb['hurt']}", f"{hb['helped']} / {hb['hurt']}"],
    ["NDCG@10, BM25F + expansion", f3(eb['ndcg@10']), f3(hb['ndcg@10'])],
    ["NDCG@10, deployed hybrid search", f3(R['deployed']['embedding']['test']['ndcg@10']), f3(R['deployed']['hybrid']['test']['ndcg@10'])],
])}
<p>Queries where the TF-IDF filter changed what was added (ΔNDCG@10 relative to no expansion):</p>
{table(["Query", "Terms the filter removed", "Word2Vec only", "Hybrid"], diff_rows, cls="tight")}

<h3>2.11 What the numbers mean</h3>
<ul>
<li><b>The hybrid builds the most accurate dictionary.</b> {pct(Q['hybrid']['quality']['same_meaning'], 0)} of its additions mean
the same as the seed, against {pct(Q['embedding']['quality']['same_meaning'], 0)} for Word2Vec alone; unrelated additions fall from
{pct(Q['embedding']['quality']['unrelated'], 0)} to {pct(Q['hybrid']['quality']['unrelated'], 0)}. Rare neighbours are where Word2Vec is
least reliable (few training examples), and TF-IDF salience removes exactly those.</li>
<li><b>Search quality differences are within noise.</b> All variants are within about ±0.01 NDCG@10 of each other and of no
expansion, and no p-value is near 0.05. Expansion fires on only {hb['queries_expanded']} of {hb['queries']} test queries, and most of them already
match their posts word for word. The hybrid is best with BM25 and hurts the fewest queries; in the deployed system TF-IDF only
happens to score highest on the test split, while Word2Vec only scored highest on the tuning split.</li>
<li><b>TF-IDF alone finds context, not synonyms.</b> Only {pct(Q['tfidf']['quality']['same_meaning'], 0)} of its additions are
synonyms, but topical neighbours (josaa → documents, round) still help some queries, like pseudo-relevance feedback.</li>
<li><b>Where expansion helps:</b> abbreviations and brand names (PW → physics wallah), synonyms (hall ticket → admit card) and
Hinglish questions with English posts (boards aur jee → board, cbse).</li>
</ul>

<h3>2.12 Tuning runs</h3>
{table(["TF-IDF only: co-occurrence threshold", "Tune NDCG@10"], [[k, f3(v)] for k, v in tune['tfidf'].items()], cls="tight")}
{table(["Hybrid: salience cut-off / TF-IDF candidate terms added", "Tune NDCG@10"],
       [[k.replace('top', 'top ').replace('_cand', ' / '), f3(v)] for k, v in tune['hybrid'].items()], cls="tight")}
<p class="small">Adding TF-IDF-discovered candidate terms (<i>alakh pandey</i>, <i>mathongo</i>, <i>competishun</i>) changed no tuning
query, so it is off.</p>

<h3>2.13 Limitations and next steps</h3>
<ul>
<li>The pair grades and relevance labels were made by Claude (an LLM); a human review of a sample would make them more reliable.</li>
<li>Some kept expansions are still debatable: <i>ncert → hc verma, rd sharma</i> (other books), <i>percentile → score</i>.</li>
<li>Expansion is query-independent: "boards" always adds "cbse", even when a student means a state board.</li>
<li>Seeds are hand-picked; new slang (e.g. new coaching batches) needs a new seed or a retrain.</li>
<li>The data ends in August 2022, so newer terms are missing until the corpus is updated.</li>
<li>Next steps: per-category thresholds, judging more expansions by hand, and letting the semantic model veto expansions that move
the query's sentence embedding too far.</li>
</ul>
</section>""")
    return "\n".join(S)
