'use strict';
// Report panel for the Zovo Pro tools: a read-only webview with the facts, notes and the result table, a filter,
// and Save as CSV, Markdown or JSON. File text is put on the page with textContent only. Shared by the Zovo viewers.
const crypto = require('crypto');
const { cellText } = require('./table');

const MAX_ROWS_SHOWN = 5000;

/** The panel HTML. The table data travels as JSON in a non-executed script tag, with "<" escaped. */
function panelHtml(desc, cspSource, nonce) {
  const n = nonce || crypto.randomBytes(24).toString('base64url');
  const rows = desc.rows.slice(0, MAX_ROWS_SHOWN).map(r => ({ m: r._mark || '', c: desc.columns.map(c => cellText(r[c.key], c.type)) }));
  const data = JSON.stringify({ title: desc.title, subtitle: desc.subtitle || '', facts: desc.facts || [], notes: desc.notes || [],
    columns: desc.columns.map(c => ({ label: c.label, num: !!(c.type && c.type !== 'text') })), rows, total: desc.rows.length, more: Math.max(0, desc.rows.length - MAX_ROWS_SHOWN),
    marked: desc.rows.filter(r => r._mark).length, filterLabel: desc.filterLabel || 'Only rows to check' }).replace(/</g, '\\u003c');
  const csp = ["default-src 'none'", `style-src 'nonce-${n}'`, `script-src 'nonce-${n}'`].join('; ');
  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta http-equiv="Content-Security-Policy" content="${csp}">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Report</title>
<style nonce="${n}">
body{font-family:var(--vscode-font-family);font-size:var(--vscode-font-size);color:var(--vscode-foreground);padding:12px 16px}
h1{font-size:1.25em;margin:0 0 4px} .sub{opacity:.8;margin-bottom:10px;word-break:break-all}
.facts{display:flex;flex-wrap:wrap;gap:6px 18px;margin:8px 0} .fact{display:flex;flex-direction:column} .fl{font-size:.85em;opacity:.7} .fv{font-weight:600}
.note{border-left:3px solid var(--vscode-textBlockQuote-border,#888);padding:2px 8px;margin:6px 0;opacity:.9}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
button{background:var(--vscode-button-secondaryBackground);color:var(--vscode-button-secondaryForeground);border:0;padding:4px 10px;cursor:pointer;border-radius:2px}
button:hover{background:var(--vscode-button-secondaryHoverBackground)}
input[type=search]{background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border,transparent);padding:3px 6px;min-width:220px}
.wrap{overflow:auto;max-height:calc(100vh - 220px);border:1px solid var(--vscode-panel-border,#8884)}
table{border-collapse:collapse;width:100%} th,td{padding:3px 8px;border-bottom:1px solid var(--vscode-panel-border,#8883);white-space:nowrap;text-align:left}
th{position:sticky;top:0;background:var(--vscode-editor-background);font-weight:600} td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.bad td{color:var(--vscode-errorForeground)} tr.warn td{color:var(--vscode-editorWarning-foreground)}
td.wide{white-space:normal;min-width:260px} .count{margin-top:6px;opacity:.75}
</style></head>
<body><div id="app"></div>
<script id="data" type="application/json">${data}</script>
<script nonce="${n}">
(function(){
  const vscode = acquireVsCodeApi();
  const D = JSON.parse(document.getElementById('data').textContent);
  const app = document.getElementById('app');
  const el = (t, cls, text) => { const e = document.createElement(t); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  let q = '', only = false;
  app.append(el('h1', null, D.title));
  if (D.subtitle) app.append(el('div', 'sub', D.subtitle));
  const facts = el('div', 'facts');
  for (const f of D.facts) { const d = el('div', 'fact'); d.append(el('span', 'fl', f[0]), el('span', 'fv', f[1])); facts.append(d); }
  app.append(facts);
  for (const n of D.notes) app.append(el('div', 'note', n));
  const bar = el('div', 'bar');
  const search = el('input'); search.type = 'search'; search.placeholder = 'Filter by name'; search.setAttribute('aria-label', 'Filter rows');
  search.addEventListener('input', () => { q = search.value.toLowerCase(); draw(); });
  bar.append(search);
  if (D.marked) {
    const lab = el('label'); const cb = el('input'); cb.type = 'checkbox'; cb.id = 'only';
    cb.addEventListener('change', () => { only = cb.checked; draw(); });
    lab.append(cb, document.createTextNode(' ' + D.filterLabel + ' (' + D.marked + ')')); bar.append(lab);
  }
  for (const [f, label] of [['csv', 'Save CSV...'], ['md', 'Save Markdown...'], ['json', 'Save JSON...']]) {
    const b = el('button', null, label); b.id = 'save-' + f; b.addEventListener('click', () => vscode.postMessage({ type: 'save', format: f })); bar.append(b);
  }
  app.append(bar);
  const wrap = el('div', 'wrap'); app.append(wrap);
  const count = el('div', 'count'); app.append(count);
  function draw() {
    const table = el('table'); const head = el('tr');
    D.columns.forEach(c => head.append(el('th', c.num ? 'num' : null, c.label)));
    const thead = el('thead'); thead.append(head); table.append(thead);
    const body = el('tbody'); let shown = 0;
    for (const r of D.rows) {
      if (only && !r.m) continue;
      if (q && !String(r.c[0]).toLowerCase().includes(q)) continue;
      const tr = el('tr', r.m || null);
      r.c.forEach((v, i) => tr.append(el('td', D.columns[i].num ? 'num' : (i === r.c.length - 1 ? 'wide' : null), v)));
      body.append(tr); shown++;
    }
    table.append(body); wrap.replaceChildren(table);
    count.textContent = shown + ' of ' + D.total + ' rows' + (D.more ? ' (the first ' + D.rows.length + ' are shown here; the saved report has all of them)' : '');
  }
  draw();
})();
</script></body></html>`;
}

/**
 * Opens a report panel. onSave(format) is called when the user clicks a Save button ('csv' | 'md' | 'json').
 * @returns the WebviewPanel
 */
function openReportPanel({ vscode, viewType, title, desc, onSave }) {
  const panel = vscode.window.createWebviewPanel(viewType, title, { viewColumn: vscode.ViewColumn.Active, preserveFocus: false }, { enableScripts: true, localResourceRoots: [] });
  panel.webview.html = panelHtml(desc, panel.webview.cspSource);
  panel.webview.onDidReceiveMessage(m => {
    if (!m || m.type !== 'save' || !['csv', 'md', 'json'].includes(m.format)) return;
    Promise.resolve(onSave(m.format)).catch(e => vscode.window.showErrorMessage(`Could not save the report. ${e.message || e}`));
  });
  return panel;
}

module.exports = { panelHtml, openReportPanel, MAX_ROWS_SHOWN };
