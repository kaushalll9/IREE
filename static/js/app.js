/**
 * IRE Search Engine - Phase 4 Frontend Logic
 * Manages search interactions, keyword highlighting, and evaluation dashboard rendering.
 */

let currentModel = 'boosted';
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
                `${data.stats.total_docs.toLocaleString()} Docs • BM25F Engine`;
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

    // Automatically trigger initial search with default query for immediate visual feedback
    setQuery('physics wallah vs unacademy');
}

function switchTab(tab) {
    const searchView = document.getElementById('search-view');
    const evalView = document.getElementById('eval-view');
    const btnSearch = document.getElementById('tab-btn-search');
    const btnEval = document.getElementById('tab-btn-eval');

    if (tab === 'search') {
        searchView.classList.add('active');
        evalView.classList.remove('active');
        btnSearch.classList.add('active');
        btnEval.classList.remove('active');
    } else {
        searchView.classList.remove('active');
        evalView.classList.add('active');
        btnSearch.classList.remove('active');
        btnEval.classList.add('active');
        loadEvaluationData();
    }
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
                    <span class="score-badge">BM25 Score: ${item.score}</span>
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
        
        // 1. Models comparison table
        const modelsTbody = document.getElementById('models-table-body');
        const bestIdx = data.models_evaluated.reduce((best, m, i, all) => m.map > all[best].map ? i : best, 0);
        modelsTbody.innerHTML = data.models_evaluated.map((m, idx) => {
            const isBest = idx === bestIdx;
            return `
                <tr class="${isBest ? 'highlight-row' : ''}">
                    <td><strong>${escapeHtml(m.config_name)}</strong> ${isBest ? '🏆' : ''}</td>
                    <td>${m['mean_p@1'].toFixed(4)}</td>
                    <td>${m['mean_p@5'].toFixed(4)}</td>
                    <td>${m['mean_p@10'].toFixed(4)}</td>
                    <td>${m.mrr.toFixed(4)}</td>
                    <td>${m.map.toFixed(4)}</td>
                    <td>${m['ndcg@10'].toFixed(4)}</td>
                    <td>${m.avg_latency_ms} ms</td>
                </tr>
            `;
        }).join('');

        // 2. Per-Query breakdown (using the proposed model)
        const proposedModel = data.models_evaluated[data.models_evaluated.length - 1];
        const queriesTbody = document.getElementById('queries-table-body');
        queriesTbody.innerHTML = proposedModel.per_query.map(q => `
            <tr>
                <td><strong>${escapeHtml(q.query_id)}</strong></td>
                <td><code>${escapeHtml(q.query)}</code></td>
                <td>${q['p@1'].toFixed(4)}</td>
                <td>${q['p@5'].toFixed(4)}</td>
                <td>${q.rr.toFixed(4)}</td>
                <td>${q.ap.toFixed(4)}</td>
                <td>${q['ndcg@10'].toFixed(4)}</td>
                <td>${q.latency_ms} ms</td>
            </tr>
        `).join('');

        evaluationDataLoaded = true;
    } catch (err) {
        console.error('Failed to load evaluation data:', err);
    }
}
