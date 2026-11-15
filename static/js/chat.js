/**
 * Chat assistant tab: sends questions to /api/chat and renders answers with
 * numbered, clickable source citations. Uses escapeHtml() from app.js.
 */

let chatMsgCount = 0;

const MODE_LABELS = {
    official: { text: 'Official NTA notices', cls: 'official' },
    community: { text: 'r/JEENEETards community', cls: 'community' },
};

function safeUrl(url) {
    return /^https?:\/\//i.test(url || '') ? url : '#';
}

/** Minimal, escape-first markdown: **bold**, "- " bullets, and [n] citations. */
function renderAnswer(text, msgId) {
    const inline = line => escapeHtml(line)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\[(\d+)\]/g, `<a class="cite" href="#m${msgId}-s$1">[$1]</a>`);
    let html = '';
    let inList = false;
    for (const line of text.split('\n')) {
        const bullet = line.match(/^\s*[-*•]\s+(.*)/);
        if (bullet && !inList) { html += '<ul>'; inList = true; }
        if (!bullet && inList) { html += '</ul>'; inList = false; }
        if (bullet) html += `<li>${inline(bullet[1])}</li>`;
        else if (line.trim()) html += `<p>${inline(line)}</p>`;
    }
    return html + (inList ? '</ul>' : '');
}

function renderSource(s, i, msgId) {
    const meta = s.type === 'nta'
        ? `${escapeHtml(s.exam)} · ${escapeHtml((s.category || '').replace('_', ' '))} · published ${escapeHtml(s.date)}`
        : `${escapeHtml(s.date)} · ▲ ${s.upvotes} · 💬 ${s.comments}`;
    const replies = (s.replies || []).map(r =>
        `<blockquote>${escapeHtml(r.text)} <span class="reply-score">▲ ${r.score}</span></blockquote>`).join('');
    const extra = replies || (s.snippet ? `<blockquote>${escapeHtml(s.snippet)}</blockquote>` : '');
    return `
        <li id="m${msgId}-s${i}">
            <a href="${escapeHtml(safeUrl(s.url))}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.title)}</a>
            <span class="source-meta">${meta}</span>
            ${extra ? `<details><summary>${replies ? 'Top replies' : 'Excerpt'}</summary>${extra}</details>` : ''}
        </li>`;
}

function appendBubble(role, innerHtml) {
    const log = document.getElementById('chat-log');
    const welcome = document.getElementById('chat-welcome');
    if (welcome) welcome.remove();
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    div.innerHTML = innerHtml;
    log.appendChild(div);
    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
    return div;
}

async function askChat(preset) {
    const input = document.getElementById('chat-input');
    const btn = document.getElementById('chat-send-btn');
    const question = (preset || input.value).trim();
    if (!question || btn.disabled) return;

    switchTab('chat');
    input.value = '';
    input.disabled = btn.disabled = true;
    appendBubble('user', `<p>${escapeHtml(question)}</p>`);
    const pending = appendBubble('assistant pending',
        '<div class="typing" aria-label="Thinking"><span></span><span></span><span></span></div>');

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);

        const msgId = ++chatMsgCount;
        const mode = MODE_LABELS[data.mode] || { text: data.mode, cls: '' };
        const gen = data.generator === 'claude' ? 'AI summary of sources' : 'Extracted from sources';
        const expanded = Object.keys(data.expanded_terms || {});
        pending.classList.remove('pending');
        pending.innerHTML = `
            <div class="chat-meta">
                <span class="mode-badge ${mode.cls}">${escapeHtml(mode.text)}</span>
                <span class="gen-label">${gen}</span>
            </div>
            <div class="chat-answer">${renderAnswer(data.answer || '', msgId)}</div>
            ${expanded.length ? `<p class="expanded-note">Also searched: ${expanded.map(escapeHtml).join(', ')}</p>` : ''}
            ${(data.sources || []).length ? `
                <div class="sources-title">Sources</div>
                <ol class="chat-sources">${data.sources.map((s, i) => renderSource(s, i + 1, msgId)).join('')}</ol>` : ''}`;
    } catch (err) {
        pending.classList.remove('pending');
        pending.classList.add('error');
        pending.innerHTML = `<p>Sorry, something went wrong: ${escapeHtml(err.message)}</p>`;
    } finally {
        input.disabled = btn.disabled = false;
        input.focus();
    }
}
