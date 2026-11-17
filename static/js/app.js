/**
 * IRE Search Engine - Phase 4 Frontend Logic
 * Manages search interactions, keyword highlighting, and evaluation dashboard rendering.
 */

let currentModel = 'hybrid';
let searchDebounceTimeout = null;
let evaluationDataLoaded = false;

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    // 1. Load metadata & flairs
    try {
        const resp = await fetch('/api/meta');
        const data = await resp.json();
        populateFlairs(data.flairs);
        if (data.stats && data.stats.total_docs) {
            document.getElementById('corpus-stats-badge').innerText = 
                `${data.stats.total_docs.toLocaleString()} Docs • BM25F + Semantic Hybrid`;
        }
    } catch (e) {
        console.error('Failed to load metadata:', e);
    }

    // 2. Setup input event listeners
    const input = document.getElementById('search-query-input');
    const clearBtn = document.getElementById('search-clear-btn');

    input.addEventListener('input', () => {
        clearBtn.classList.toggle('hidden', !input.value.trim());
        clearTimeout(searchDebounceTimeout);
        // Live search after 350ms of inactivity
        if (input.value.trim().length >= 3) {
            searchDebounceTimeout = setTimeout(() => {
                triggerSearch();
            }, 350);
        }
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            clearTimeout(searchDebounceTimeout);
            triggerSearch();
        }
    });

    // Pre-fill the search tab without leaving the (default) chat tab
    input.value = 'physics wallah vs unacademy';
    clearBtn.classList.remove('hidden');
    triggerSearch();
}

function switchTab(tab) {
    ['chat', 'search', 'eval'].forEach(name => {
        document.getElementById(`${name}-view`).classList.toggle('active', name === tab);
        document.getElementById(`tab-btn-${name}`).classList.toggle('active', name === tab);
    });
    if (tab === 'eval') loadEvaluationData();
}

function populateFlairs(flairs) {
    const select = document.getElementById('flair-filter-select');
    if (!flairs) return;
    
    // Sort flairs by post volume
    const sortedFlairs = Object.entries(flairs).sort((a, b) => b[1] - a[1]);
    sortedFlairs.slice(0, 15).forEach(([flair, count]) => {
        if (!flair || flair === 'None') return;
        const opt = document.createElement('option');
        opt.value = flair;
        opt.textContent = `${flair} (${count})`;
        select.appendChild(opt);
    });
}

function setModel(model) {
    currentModel = model;
    document.querySelectorAll('#ranking-model-pills .pill').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`model-pill-${model}`).classList.add('active');
    triggerSearch();
}

function setQuery(query) {
    const input = document.getElementById('search-query-input');
    input.value = query;
    document.getElementById('search-clear-btn').classList.remove('hidden');
    switchTab('search');
    triggerSearch();
}

function clearSearch() {
    const input = document.getElementById('search-query-input');
    input.value = '';
    document.getElementById('search-clear-btn').classList.add('hidden');
    document.getElementById('results-meta-bar').classList.add('hidden');
    document.getElementById('results-list').innerHTML = `
        <div class="empty-state" id="empty-state-prompt">
            <div class="empty-icon">💡</div>
            <h3>Start exploring student queries</h3>
            <p>Enter any topic or select a suggestion above to retrieve ranked Reddit posts with millisecond latency.</p>
        </div>
    `;
    input.focus();
}

async function triggerSearch() {
    const query = document.getElementById('search-query-input').value.trim();
    if (!query) return;

    const flair = document.getElementById('flair-filter-select').value;
    const minScore = document.getElementById('min-score-input').value;
    const topK = document.getElementById('top-k-select').value;
    const resultsContainer = document.getElementById('results-list');
    const metaBar = document.getElementById('results-meta-bar');

    resultsContainer.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">⚡</div>
            <h3>Retrieving documents...</h3>
        </div>
    `;

    try {
        const params = new URLSearchParams({
            q: query,
            model: currentModel,
            top_k: topK,
            expand: document.getElementById('expand-toggle').checked ? '1' : '0'
        });
        if (flair) params.append('flair', flair);
        if (minScore) params.append('min_score', minScore);

        const resp = await fetch(`/api/search?${params.toString()}`);
        const data = await resp.json();

        renderResults(data);
    } catch (err) {
        console.error('Search error:', err);
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">⚠️</div>
                <h3>Error executing search</h3>
                <p>${err.message}</p>
            </div>
        `;
    }
}

function renderResults(data) {
    const resultsContainer = document.getElementById('results-list');
    const metaBar = document.getElementById('results-meta-bar');
    const countText = document.getElementById('results-count-text');
    const latencyBadge = document.getElementById('results-latency-badge');
    const stemsList = document.getElementById('query-stems-list');

    metaBar.classList.remove('hidden');
    countText.innerText = `Found ${data.total_matches.toLocaleString()} results (${data.results.length} shown)`;
    latencyBadge.innerText = `⚡ ${data.latency_ms} ms`;

    // Render query stems
    stemsList.innerHTML = (data.stems || []).map(s => `<span class="stem-tag">${escapeHtml(s)}</span>`).join('');
    const expanded = Object.entries(data.expanded_terms || {});
    document.getElementById('query-expanded-list').innerHTML = expanded.map(([t, w]) =>
        `<span class="stem-tag expanded" title="similarity ${w}">+${escapeHtml(t)}</span>`).join('');

    if (!data.results || data.results.length === 0) {
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">🔍</div>
                <h3>No matching documents found</h3>
                <p>Try broadening your query, removing filters, or searching for other JEE/NEET keywords.</p>
            </div>
        `;
        return;
    }

    // Highlight expansion words too, so it's visible why a post matched.
    const stems = (data.stems || []).concat(expanded.flatMap(([t]) => t.split(' ')));
    resultsContainer.innerHTML = data.results.map(item => {
        const titleHighlighted = highlightQueryTerms(item.title, stems);
        const snippetHighlighted = item.snippet ? highlightQueryTerms(item.snippet, stems) : '';

        return `
            <article class="result-card" id="result-doc-${item.doc_id}">
                <div class="card-top">
                    <span class="rank-pill">#${item.rank}</span>
                    <span class="flair-pill">${escapeHtml(item.flair || 'General')}</span>
                    <span class="score-badge" title="${escapeHtml(scoreTooltip(item))}">${SCORE_LABELS[item.method] || 'BM25 Score'}: ${item.score}</span>
                    ${item.method === 'hybrid' ? `<span class="flair-pill">BM25 ${item.bm25_rank ? '#' + item.bm25_rank : '–'} · Semantic ${item.semantic_rank ? '#' + item.semantic_rank : '–'}</span>` : ''}
                </div>
                <h2 class="result-title">
                    <a href="${item.url}" target="_blank" rel="noopener noreferrer">${titleHighlighted}</a>
                </h2>
                ${snippetHighlighted ? `<p class="result-snippet">${snippetHighlighted}</p>` : ''}
                <div class="card-meta">
                    <span class="meta-item">▲ <strong>${item.upvotes}</strong> upvotes</span>
                    <span class="meta-item">💬 <strong>${item.num_comments}</strong> comments</span>
                    <span class="meta-item">📅 ${escapeHtml(item.date)}</span>
                    <span class="meta-item">👤 u/${escapeHtml(item.author || '[deleted]')}</span>
                    <a href="${item.url}" target="_blank" rel="noopener noreferrer" class="meta-link">
                        View on Reddit ↗
                    </a>
                </div>
            </article>
        `;
    }).join('');
}

const SCORE_LABELS = {hybrid: 'Hybrid RRF Score', semantic: 'Semantic Cosine'};

function scoreTooltip(item) {
    if (item.method !== 'hybrid') return `Method: ${item.method || 'bm25'}`;
    const part = (name, r) => `${name} rank: ${r == null ? 'not in top results' : '#' + r}`;
    return `Method: hybrid (RRF) | ${part('BM25', item.bm25_rank)} | ${part('Semantic', item.semantic_rank)}`;
}

function highlightQueryTerms(text, stems) {
    if (!text || !stems || stems.length === 0) return escapeHtml(text);
    
    // Simple word boundary regex matcher for query stems
    let safeText = escapeHtml(text);
    const pattern = new RegExp(`\\b(${stems.join('|')})\\w*\\b`, 'gi');
    return safeText.replace(pattern, match => `<mark class="highlight">${match}</mark>`);
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

/* ==========================================================================
   Evaluation Dashboard Data Loading
   ========================================================================== */

async function loadEvaluationData() {
    if (evaluationDataLoaded) return;
    try {
        const resp = await fetch('/api/evaluation');
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        renderEvaluation(data);
        evaluationDataLoaded = true;
    } catch (err) {
        console.error('Failed to load evaluation data:', err);
        document.getElementById('eval-subtitle').innerText = `Evaluation results unavailable: ${err.message}`;
    }
}

const fmt = (v, d = 3) => (v === null || v === undefined) ? '–' : Number(v).toFixed(d);
const COMPONENT_LABELS = {
    'ndcg@10': 'NDCG@10', 'p@5': 'Precision@5', 'map': 'MAP', 'mrr': 'MRR',
    'robustness.abbreviation': 'Abbreviations', 'robustness.synonym': 'Synonyms',
    'robustness.hinglish_typo': 'Hinglish & typos', 'robustness.numerical_comparison': 'Numerical & comparison',
    'efficiency.p95_latency': 'P95 latency', 'efficiency.index_memory': 'Index & memory',
    'correctness.index': 'Index correctness', 'correctness.filters': 'Filter & API correctness',
    'usefulness.task_completion': 'Task completion', 'usefulness.satisfaction': 'Satisfaction',
};

function bar(v) {
    const pct = v === null || v === undefined ? 0 : Math.max(0, Math.min(1, v)) * 100;
    return `<div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(1)}%"></div></div>`;
}

function renderEvaluation(data) {
    const meta = data.meta, cfg = meta.config, split = cfg.split;
    const dep = meta.deployed_system, base = meta.baseline_system;
    const h = data.hses[dep];
    const rk = data.ranking || null, perf = data.performance || null;
    const sysName = id => rk ? rk.systems[id].name : id;

    document.getElementById('eval-subtitle').innerText =
        `${meta.dataset.docs.toLocaleString()} posts · ${meta.queries.total} queries (${meta.queries.test} test / ` +
        `${meta.queries.tune} tuning) · ${meta.judgments.total.toLocaleString()} graded judgments · generated ${meta.generated_at.replace('T', ' ')}`;

    // HSES headline + components
    document.getElementById('hses-score').innerText = h.complete ? fmt(h.hses, 1) : `${fmt(h.partial_hses, 1)}*`;
    document.getElementById('hses-status').innerText = h.complete
        ? 'Complete: every component measured.'
        : `* Incomplete: partial score over the ${(h.measured_weight * 100).toFixed(0)}% of weight that was measured. ` +
          `Missing: ${h.missing.map(m => COMPONENT_LABELS[m] || m).join(', ')}.`;
    document.getElementById('hses-components').innerHTML = Object.entries(cfg.weights).map(([k, w]) => `
        <tr class="${h.scores[k] === null ? 'muted-row' : ''}">
            <td>${escapeHtml(COMPONENT_LABELS[k] || k)}</td><td>${(w * 100).toFixed(0)}%</td>
            <td>${h.scores[k] === null ? 'not measured' : fmt(h.scores[k])}</td><td>${bar(h.scores[k])}</td>
            <td>${fmt(h.contributions[k], 2)}</td>
        </tr>`).join('');

    // Metric cards for the deployed system
    if (rk) {
        const t = rk.systems[dep][split], b = rk.systems[base][split];
        const cards = [['NDCG@10', 'ndcg@10'], ['Precision@5', 'p@5'], ['MAP', 'map'], ['MRR', 'mrr'], ['Success@5', 'success@5']];
        document.getElementById('eval-metric-cards').innerHTML = cards.map(([label, k]) => `
            <div class="metric-card highlight">
                <span class="metric-title">${label}</span>
                <span class="metric-val">${fmt(t[k])}</span>
                <span class="metric-desc">baseline ${fmt(b[k])}</span>
            </div>`).join('') + (perf ? `
            <div class="metric-card">
                <span class="metric-title">P95 latency</span>
                <span class="metric-val">${fmt(perf.latency[dep].p95_ms, 1)} ms</span>
                <span class="metric-desc">target ≤ ${cfg.targets.p95_latency_ms.best} ms</span>
            </div>` : '');

        document.getElementById('split-tag').innerText = `${split} split · ${t.queries} queries`;
        const ids = Object.keys(rk.systems);
        document.getElementById('models-table-body').innerHTML = ids.map(id => {
            const s = rk.systems[id], m = s[split], vb = s.vs_baseline;
            const lat = perf && perf.latency[id] ? `${fmt(perf.latency[id].p95_ms, 1)} ms` : '–';
            return `<tr class="${id === dep ? 'highlight-row' : ''}">
                <td><strong>${escapeHtml(s.name)}</strong></td>
                <td>${fmt(m['ndcg@10'])}</td><td>${fmt(m['p@5'])}</td><td>${fmt(m.map)}</td><td>${fmt(m.mrr)}</td>
                <td>${fmt(m['recall@10'])}</td><td>${fmt(m['success@5'])}</td><td>${fmt(m['judged@10'], 2)}</td><td>${lat}</td>
                <td>${id === base ? 'baseline' : `${vb['ndcg@10_delta'] >= 0 ? '+' : ''}${fmt(vb['ndcg@10_delta'])} (${fmt(vb.p_value)})`}</td>
                <td>${fmt(data.hses[id].partial_hses, 1)}</td></tr>`;
        }).join('');
        const maxN = Math.max(...ids.map(id => rk.systems[id][split]['ndcg@10'] || 0));
        document.getElementById('ndcg-chart').innerHTML = ids.map(id => {
            const v = rk.systems[id][split]['ndcg@10'] || 0;
            return `<div class="bar-row ${id === dep ? 'is-deployed' : ''}">
                <span class="bar-label">${escapeHtml(rk.systems[id].name)}</span>
                <div class="bar-track"><div class="bar-fill" style="width:${(v / maxN * 100).toFixed(1)}%"></div></div>
                <span class="bar-value">${fmt(v)}</span></div>`;
        }).join('');

        // Robustness
        const cats = Object.keys(rk.systems[dep].categories).sort();
        document.getElementById('robust-head').innerHTML = `<tr><th>System</th>${cats.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr>`;
        document.getElementById('robust-body').innerHTML = ids.map(id => `<tr class="${id === dep ? 'highlight-row' : ''}">
            <td><strong>${escapeHtml(rk.systems[id].name)}</strong></td>
            ${cats.map(c => `<td>${fmt(rk.systems[id].categories[c]['success@5'], 2)}</td>`).join('')}</tr>`).join('');
        const failed = (rk.per_query || []).filter(q => q.split === split && q['success@5'] === 0);
        document.getElementById('failed-queries').innerHTML = failed.length
            ? failed.map(q => `<li><code>${escapeHtml(q.query)}</code> <span class="muted">(${escapeHtml(q.category)})</span></li>`).join('')
            : '<li>None</li>';

        // Expansion
        const e = rk.expansion;
        document.getElementById('expansion-summary').innerText =
            `Expansion fired on ${e.queries_expanded} of ${e.queries_total} ${split} queries. NDCG@10 rose on ${e.helped} and fell on ${e.hurt} ` +
            `(mean ${e['mean_ndcg@10_delta_when_expanded'] >= 0 ? '+' : ''}${fmt(e['mean_ndcg@10_delta_when_expanded'], 4)} where it fired). ` +
            `Drift: it pulled ${e.drift_new_irrelevant} judged-irrelevant and ${e.drift_new_relevant} relevant posts into top-10s. ` +
            `Extra latency: ${fmt(e.mean_latency_delta_ms, 2)} ms.`;
        document.getElementById('expansion-body').innerHTML = e.per_query.filter(r => r.expanded).map(r => `<tr>
            <td><code>${escapeHtml(r.query)}</code></td>
            <td>${Object.keys(r.terms).map(escapeHtml).join(', ')}</td>
            <td class="${r['ndcg@10_delta'] > 0 ? 'pos' : r['ndcg@10_delta'] < 0 ? 'neg' : ''}">${r['ndcg@10_delta'] >= 0 ? '+' : ''}${fmt(r['ndcg@10_delta'])}</td>
            <td>${r.new_relevant}</td><td>${r.new_irrelevant}</td><td>${fmt(r['overlap@10'], 2)}</td></tr>`).join('');

        // Per-query
        document.getElementById('queries-table-body').innerHTML = (rk.per_query || []).map(q => `
            <tr class="${q['success@5'] === 0 && q.split === split ? 'fail-row' : ''}">
                <td>${escapeHtml(q.query_id)}</td><td><code>${escapeHtml(q.query)}</code></td>
                <td>${escapeHtml(q.category)}</td><td>${escapeHtml(q.split)}</td>
                <td>${fmt(q['ndcg@10'])}</td><td>${fmt(q['p@5'], 2)}</td><td>${q['success@5'] ? '✓' : '✗'}</td>
                <td>${fmt(q['judged@10'], 2)}</td></tr>`).join('');
    }

    // Performance
    if (perf) {
        document.getElementById('perf-device').innerText = `device: ${perf.device}`;
        document.getElementById('latency-body').innerHTML = Object.entries(perf.latency).map(([id, l]) => `
            <tr class="${id === dep ? 'highlight-row' : ''}"><td>${escapeHtml(sysName(id))}</td>
            <td>${fmt(l.avg_ms, 1)} ms</td><td>${fmt(l.p50_ms, 1)} ms</td><td>${fmt(l.p95_ms, 1)} ms</td><td>${fmt(l.p99_ms, 1)} ms</td></tr>`).join('');
        const idx = perf.index, rss = perf.memory_rss_mb;
        document.getElementById('index-stats').innerText =
            `Inverted index ${idx.lexical_index_mb} MB (built in ${idx.indexing_time_s} s, ${idx.vocab_size.toLocaleString()} terms) · ` +
            `embeddings ${idx.embeddings_mb} MB · memory ${fmt(rss.after_bm25_mb, 0)} MB with BM25, ${fmt(rss.after_semantic_mb, 0)} MB with the semantic model · ` +
            `throughput ${perf.throughput.map(t => `${t.qps} q/s with ${t.workers} worker${t.workers > 1 ? 's' : ''}`).join(', ')} ` +
            `(${perf.workload.latency_runs} timed runs after ${perf.workload.warmup_runs} warm-up over ${perf.workload.queries} queries).`;
    }

    // Reliability
    const rel = data.reliability;
    document.getElementById('reliability').innerHTML = rel ? Object.entries(rel).map(([name, checks]) => {
        const ok = checks.filter(c => c.passed).length, bad = checks.filter(c => !c.passed);
        return `<div class="rel-group"><strong>${name === 'index_integrity' ? 'Index integrity' : 'Filters & API'}:</strong>
            <span class="${bad.length ? 'neg' : 'pos'}">${ok}/${checks.length} passed</span>
            ${bad.length ? `<ul class="failed-list">${bad.map(c => `<li>${escapeHtml(c.check)} <span class="muted">${escapeHtml(c.detail)}</span></li>`).join('')}</ul>` : ''}</div>`;
    }).join('') : '<p class="eval-note">Not measured. Run <code>python evaluate.py reliability</code>.</p>';

    // Usefulness
    const u = data.usefulness;
    document.getElementById('usefulness-summary').innerText = u.responses
        ? `${u.responses} responses (${u.solved_votes} “solved?” votes, ${u.ratings} ratings; ${u.min_required} needed per score). ` +
          `Task completion: ${u.task_completion === null ? 'not enough votes yet' : (u.task_completion * 100).toFixed(0) + '%'}. ` +
          `Satisfaction: ${u.mean_rating_1_5 === null ? '–' : u.mean_rating_1_5 + ' / 5'}${u.satisfaction === null ? ' (not enough ratings yet)' : ''}.`
        : `No feedback yet. Answers in the chat tab have “Did this answer your question?” buttons; ${u.min_required} responses complete the score.`;
    document.getElementById('unsolved-list').innerHTML = (u.recent_unsolved || []).map(q => `<li>Unsolved: <code>${escapeHtml(q)}</code></li>`).join('');

    // Completeness
    const assessors = Object.entries(meta.judgments.by_assessor).map(([a, n]) => `${a}: ${n}`).join(', ');
    document.getElementById('completeness').innerText =
        `Judgments by assessor (${assessors}). “claude” labels are LLM judgments made against each query's stated intent and should be ` +
        `spot-checked by a human (python evaluate.py annotate --review); human labels always take precedence. Tuning queries are excluded ` +
        `from every reported number. ${h.complete ? 'All HSES components are measured.' : `HSES is incomplete until ${h.missing.length} component(s) are measured.`}`;
}
