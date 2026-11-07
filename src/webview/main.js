'use strict';
// Tensor Viewer webview. All file content is put on the page with textContent or canvas pixels,
// never as HTML, so a tensor name or metadata string cannot inject markup or script.
const { displayText, displayComplex, exactText, complexText, PRECISIONS } = require('../core/numtext');

(function () {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById('app');
  const saved = vscode.getState() || {};
  const S = { model: null, filter: '', sort: { key: null, dir: 1 }, selected: -1, limit: 500, tab: 'tensors', sample: null, sampleFor: -1, mdFilter: '', error: null,
    slices: {}, sliceStats: false, vpage: { row: 0, col: 0 }, vals: null, valsErr: null, prec: PRECISIONS.includes(saved.prec) ? saved.prec : 'auto',
    img: {}, imgRes: null, errorAction: null };
  const PAGE_ROWS = 32;

  // ---------- helpers ----------
  function el(tag, props, kids) {
    const e = document.createElement(tag);
    if (props) for (const k of Object.keys(props)) {
      const v = props[k];
      if (v === undefined || v === null) continue;
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = String(v);
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else if (k === 'dataset') Object.assign(e.dataset, v);
      else e.setAttribute(k, String(v));
    }
    if (kids) for (const c of kids) if (c !== null && c !== undefined && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
    return e;
  }
  const nf = new Intl.NumberFormat('en-US');
  function compact(n) {
    if (n === null || n === undefined) return '';
    const a = Math.abs(n);
    if (a >= 1e12) return (n / 1e12).toFixed(2) + 'T';
    if (a >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (a >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (a >= 1e4) return (n / 1e3).toFixed(1) + 'K';
    return nf.format(n);
  }
  function bytes(n) {
    if (n === null || n === undefined) return '';
    if (n < 1024) return n + ' B';
    const u = ['KB', 'MB', 'GB', 'TB'];
    let v = n / 1024, i = 0;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return v.toFixed(v < 10 ? 2 : 1) + ' ' + u[i];
  }
  function num(x) {
    if (x === null || x === undefined) return '-';
    if (!Number.isFinite(x)) return String(x);
    if (x === 0) return '0';
    const a = Math.abs(x);
    if (Number.isInteger(x) && a < 1e15) return nf.format(x);
    if (a >= 1e6 || a < 1e-4) return x.toExponential(4);
    return String(Number(x.toPrecision(6)));
  }
  const shapeStr = s => (s.length ? '[' + s.join(', ') + ']' : '[] scalar');
  function f32(b64) {
    const bin = atob(b64), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Float32Array(u.buffer);
  }
  function u8(b64) {
    const bin = atob(b64), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  const dark = () => !document.body.classList.contains('vscode-light') && !document.body.classList.contains('vscode-high-contrast-light');
  const cssVar = (n, d) => (getComputedStyle(document.body).getPropertyValue(n).trim() || d);

  // ---------- colors (reference palette: blue ramp, blue/red diverging with a gray midpoint) ----------
  const hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  function ramp(stops) {
    const cs = stops.map(([p, h]) => [p, hex(h)]);
    return t => {
      if (t <= cs[0][0]) return cs[0][1];
      for (let i = 1; i < cs.length; i++) if (t <= cs[i][0]) {
        const [p0, c0] = cs[i - 1], [p1, c1] = cs[i], f = (t - p0) / (p1 - p0);
        return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
      }
      return cs[cs.length - 1][1];
    };
  }
  function scales() {
    const mid = dark() ? '#383835' : '#f0efec';
    return {
      diverging: ramp([[0, '#0d366b'], [0.25, '#3987e5'], [0.5, mid], [0.75, '#e34948'], [1, '#7a1c1c']]),
      sequential: dark() ? ramp([[0, '#0d366b'], [0.5, '#3987e5'], [1, '#cde2fb']]) : ramp([[0, '#cde2fb'], [0.5, '#3987e5'], [1, '#0d366b']]),
      series: dark() ? '#3987e5' : '#2a78d6',
    };
  }

  // ---------- tooltip ----------
  const tip = el('div', { class: 'tip', role: 'tooltip' });
  document.body.append(tip);
  function showTip(ev, lines) {
    tip.replaceChildren(...lines.map(l => el('div', { text: l })));
    tip.style.display = 'block';
    const r = tip.getBoundingClientRect();
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + r.width > window.innerWidth - 4) x = ev.clientX - r.width - 14;
    if (y + r.height > window.innerHeight - 4) y = ev.clientY - r.height - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  const hideTip = () => { tip.style.display = 'none'; };

  function canvas(w, h) {
    const c = el('canvas', { width: Math.round(w * devicePixelRatio), height: Math.round(h * devicePixelRatio) });
    c.style.width = w + 'px'; c.style.height = h + 'px';
    const g = c.getContext('2d');
    g.scale(devicePixelRatio, devicePixelRatio);
    return { c, g };
  }

  // ---------- messages ----------
  window.addEventListener('message', ev => {
    const m = ev.data;
    if (!m || typeof m !== 'object') return;
    if (m.type === 'model') {
      S.model = m; S.error = null; S.errorAction = null;
      S.selected = -1; S.slices = {}; S.img = {}; S.imgRes = null; S.sample = null; S.sampleFor = -1; S.vals = null; S.valsErr = null;
      const first = m.tensors.findIndex(t => t.readable);
      render();
      if (first >= 0) select(first);
    } else if (m.type === 'sample' || m.type === 'sampleError') {
      if (m.index !== S.selected || !sameSlice(m.slice || null, statsSlice())) return;
      S.sample = m; S.sampleFor = m.index;
      renderDetail();
    } else if (m.type === 'values' || m.type === 'valuesError') {
      if (m.index !== S.selected || !sameSlice(m.slice, curSlice()) || m.row !== S.vpage.row || m.col !== S.vpage.col) return;
      if (m.type === 'values') { S.vals = m; S.valsErr = null; } else { S.vals = null; S.valsErr = m.message; }
      renderValues();
    } else if (m.type === 'exported') {
      flash(m.text);
    } else if (m.type === 'image' || m.type === 'imageError') {
      if (m.index !== S.selected || !sameImg(m.req, imgReq())) return;
      S.imgRes = m;
      renderImage();
    } else if (m.type === 'error') {
      S.error = m.message; S.errorAction = m.action === 'rawLayout' ? 'rawLayout' : null; render();
    } else if (m.type === 'copied') {
      flash(`Copied ${m.text.length > 60 ? m.text.slice(0, 60) + '...' : m.text}`);
    }
  });

  function flash(text) {
    const f = el('div', { class: 'flash', text });
    document.body.append(f);
    setTimeout(() => f.remove(), 1800);
  }

  // ---------- slices ----------
  // Tensors with 3 or more dimensions show one 2-D slice at a time in the values table: one index
  // per leading dimension. The stats and heatmap use it too when "Use this slice" is on.
  const leadDims = t => (t && t.shape.length >= 3 ? t.shape.slice(0, -2) : []);
  function curSlice() {
    const t = S.model && S.model.tensors[S.selected];
    if (!t) return [];
    const lead = leadDims(t);
    const s = S.slices[S.selected];
    return s && s.length === lead.length ? s : lead.map(() => 0);
  }
  const statsSlice = () => (S.sliceStats && leadDims(S.model && S.model.tensors[S.selected]).length ? curSlice() : null);
  const sameSlice = (a, b) => (a === null || b === null ? a === b : a.length === b.length && a.every((v, k) => v === b[k]));
  const sliceText = s => '[' + s.concat([':', ':']).join(', ') + ']';
  function requestSample() {
    const t = S.model.tensors[S.selected];
    S.sample = null; S.sampleFor = -1;
    if (!t || !t.readable) return;
    const sl = statsSlice();
    vscode.postMessage(sl ? { type: 'select', index: S.selected, slice: sl } : { type: 'select', index: S.selected });
  }

  function select(i) {
    S.selected = i; S.sample = null; S.sampleFor = -1; S.vpage = { row: 0, col: 0 }; S.vals = null; S.valsErr = null; S.imgRes = null; imgAsked = null;
    for (const r of document.querySelectorAll('tr.sel')) r.classList.remove('sel');
    const row = document.querySelector(`tr[data-i="${i}"]`);
    if (row) { row.classList.add('sel'); row.scrollIntoView({ block: 'nearest' }); }
    const t = S.model.tensors[i];
    renderDetail();
    requestSample();
  }

  // ---------- layout ----------
  function render() {
    if (S.error) {
      const kids = [el('h2', { text: S.errorAction === 'rawLayout' ? 'Read this file as a raw array' : 'Tensor Viewer could not open this file' }), el('p', { text: S.error })];
      if (S.errorAction === 'rawLayout') kids.push(el('div', { class: 'btns' }, [el('button', { class: 'btn primary', id: 'raw-layout', text: 'Choose type and shape...', onclick: () => vscode.postMessage({ type: 'rawLayout' }) })]));
      else kids.push(el('p', { class: 'note', text: 'Nothing in the file was run or changed. If this is an array written by ndarray.tofile(), right-click it in the Explorer and pick "Open as Raw Array...".' }));
      app.replaceChildren(el('div', { class: 'fatal' }, kids));
      return;
    }
    const m = S.model;
    if (!m) return;
    const readableBytes = m.tensors.reduce((a, t) => a + (t.nbytes || 0), 0);
    const head = el('header', { class: 'head' }, [
      el('div', { class: 'title' }, [el('span', { class: 'fname', text: m.fileName }), el('span', { class: 'badge', text: m.format === 'raw' ? 'RAW ARRAY' : m.format.toUpperCase() }),
        m.format === 'raw' ? el('button', { class: 'btn', id: 'raw-layout', text: 'Change type and shape...', onclick: () => vscode.postMessage({ type: 'rawLayout' }) }) : null]),
      el('div', { class: 'facts' }, [
        fact('Parameters', compact(m.totalParams), nf.format(m.totalParams)),
        fact('Tensors', nf.format(m.tensors.length)),
        fact('File size', bytes(m.fileSize), nf.format(m.fileSize) + ' bytes'),
        fact('Tensor data', bytes(readableBytes)),
        fact('Opened in', m.openMs + ' ms'),
      ]),
    ]);
    const info = m.info && m.info.length ? el('div', { class: 'info' }, m.info.map(x => el('div', { class: 'kv' }, [el('span', { class: 'k', text: x.label }), el('span', { class: 'v', text: x.value, title: x.value })]))) : null;
    const warns = m.warnings.length ? el('div', { class: 'warn' }, m.warnings.map(w => el('div', { text: w }))) : null;
    const tabs = el('div', { class: 'tabs', role: 'tablist' }, [
      tabBtn('tensors', `Tensors (${nf.format(m.tensors.length)})`),
      tabBtn('metadata', `Metadata (${nf.format(m.metadata.length)})`),
    ]);
    const body = S.tab === 'tensors'
      ? el('div', { class: 'split' }, [el('section', { class: 'list' }, [listControls(), el('div', { class: 'tablewrap', id: 'tablewrap' }, [tensorTable()])]), el('section', { class: 'detail', id: 'detail' })])
      : metadataView();
    app.replaceChildren(...[head, info, warns, tabs, body].filter(Boolean));
    if (S.tab === 'tensors') renderDetail();
  }
  function fact(label, value, title) { return el('div', { class: 'fact', title }, [el('span', { class: 'fl', text: label }), el('span', { class: 'fv', text: value })]); }
  function tabBtn(id, label) {
    return el('button', { class: 'tab' + (S.tab === id ? ' on' : ''), role: 'tab', 'aria-selected': S.tab === id, text: label, onclick: () => { S.tab = id; render(); } });
  }

  function listControls() {
    const input = el('input', { type: 'search', class: 'search', placeholder: 'Filter by name or type, e.g. attn_q or Q4_K', value: S.filter, 'aria-label': 'Filter tensors' });
    input.addEventListener('input', () => { S.filter = input.value; S.limit = 500; document.getElementById('tablewrap').replaceChildren(tensorTable()); });
    input.addEventListener('keydown', e => { if (e.key === 'ArrowDown') { e.preventDefault(); move(1); } else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); } });
    return el('div', { class: 'controls' }, [input]);
  }

  function visibleIndexes() {
    const m = S.model, q = S.filter.trim().toLowerCase();
    let idx = m.tensors.map((_, i) => i);
    if (q) idx = idx.filter(i => { const t = m.tensors[i]; return t.name.toLowerCase().includes(q) || String(t.dtype).toLowerCase() === q || shapeStr(t.shape).includes(q); });
    const k = S.sort.key;
    if (k) {
      const val = i => { const t = m.tensors[i]; return k === 'name' ? t.name : k === 'dtype' ? String(t.dtype) : k === 'shape' ? t.shape.length * 1e15 + t.numel : (t[k] === null ? -1 : t[k]); };
      idx.sort((a, b) => { const x = val(a), y = val(b); return (x < y ? -1 : x > y ? 1 : 0) * S.sort.dir; });
    }
    return idx;
  }

  function move(d) {
    const idx = visibleIndexes();
    if (!idx.length) return;
    const p = idx.indexOf(S.selected);
    const n = idx[Math.max(0, Math.min(idx.length - 1, p < 0 ? 0 : p + d))];
    if (n !== S.selected) select(n);
  }

  function tensorTable() {
    const m = S.model;
    const idx = visibleIndexes();
    const cols = [['name', 'Name'], ['dtype', 'Type'], ['shape', 'Shape'], ['numel', 'Params'], ['nbytes', 'Size'], ['offset', 'Offset']];
    const thead = el('thead', null, [el('tr', null, cols.map(([k, label]) => el('th', {
      class: 'c-' + k, tabindex: 0, 'aria-sort': S.sort.key === k ? (S.sort.dir > 0 ? 'ascending' : 'descending') : 'none',
      text: label + (S.sort.key === k ? (S.sort.dir > 0 ? ' ▲' : ' ▼') : ''),
      onclick: () => { if (S.sort.key === k) { if (S.sort.dir > 0) S.sort.dir = -1; else S.sort.key = null; } else { S.sort.key = k; S.sort.dir = k === 'name' || k === 'dtype' ? 1 : -1; } document.getElementById('tablewrap').replaceChildren(tensorTable()); },
    })))]);
    const rows = idx.slice(0, S.limit).map(i => {
      const t = m.tensors[i];
      return el('tr', { dataset: { i }, class: (i === S.selected ? 'sel ' : '') + (t.readable ? '' : 'muted'), onclick: () => select(i) }, [
        el('td', { class: 'c-name', text: t.name, title: t.name }),
        el('td', { class: 'c-dtype' }, [el('span', { class: 'dt dt-' + String(t.dtype).replace(/[^A-Za-z0-9]/g, '_'), text: t.dtype })]),
        el('td', { class: 'c-shape', text: shapeStr(t.shape) }),
        el('td', { class: 'c-numel', text: compact(t.numel), title: nf.format(t.numel) }),
        el('td', { class: 'c-nbytes', text: bytes(t.nbytes) }),
        el('td', { class: 'c-offset', text: t.offset === null || t.offset === undefined ? '' : nf.format(t.offset) }),
      ]);
    });
    const table = el('table', { class: 'tensors' }, [thead, el('tbody', null, rows)]);
    const more = idx.length > S.limit ? el('button', { class: 'more', text: `Show ${nf.format(Math.min(500, idx.length - S.limit))} more of ${nf.format(idx.length - S.limit)}`, onclick: () => { S.limit += 500; document.getElementById('tablewrap').replaceChildren(tensorTable()); } }) : null;
    const count = el('div', { class: 'count', text: idx.length === m.tensors.length ? `${nf.format(idx.length)} tensors` : `${nf.format(idx.length)} of ${nf.format(m.tensors.length)} tensors match` });
    const frag = document.createDocumentFragment();
    frag.append(table);
    if (more) frag.append(more);
    frag.append(count);
    return frag;
  }

  function metadataView() {
    const m = S.model;
    const input = el('input', { type: 'search', class: 'search', placeholder: 'Filter keys and values', value: S.mdFilter, 'aria-label': 'Filter metadata' });
    const wrap = el('div', { class: 'mdwrap' });
    const draw = () => {
      const q = S.mdFilter.toLowerCase();
      const rows = m.metadata.filter(x => !q || x.key.toLowerCase().includes(q) || String(x.value).toLowerCase().includes(q));
      wrap.replaceChildren(rows.length ? el('table', { class: 'md' }, [el('tbody', null, rows.map(x => el('tr', null, [el('td', { class: 'mk', text: x.key }), el('td', { class: 'mv' }, [el('div', { class: 'mvt', text: x.value })])])))]) : el('p', { class: 'empty', text: m.metadata.length ? 'No metadata matches.' : 'This file has no metadata.' }));
    };
    input.addEventListener('input', () => { S.mdFilter = input.value; draw(); });
    draw();
    return el('section', { class: 'meta' }, [el('div', { class: 'controls' }, [input]), wrap]);
  }

  // ---------- detail ----------
  function renderDetail() {
    const box = document.getElementById('detail');
    if (!box || !S.model) return;
    const t = S.model.tensors[S.selected];
    if (!t) { box.replaceChildren(el('p', { class: 'empty', text: 'Pick a tensor on the left.' })); return; }
    const copyBtn = (what, label) => el('button', { class: 'btn', text: label, onclick: () => vscode.postMessage({ type: 'copy', index: S.selected, what }) });
    const parts = [
      el('h2', { class: 'tname', text: t.name }),
      el('div', { class: 'tfacts' }, [
        fact('Type', t.rawDtype && t.rawDtype !== t.dtype ? `${t.dtype} (${t.rawDtype})` : t.dtype),
        fact('Shape', shapeStr(t.shape)),
        fact('Values', nf.format(t.numel)),
        fact('Size', bytes(t.nbytes)),
      ]),
      el('div', { class: 'btns' }, [copyBtn('shape', 'Copy shape'), copyBtn('torch', 'Copy torch.Size'), copyBtn('name', 'Copy name')]),
    ];
    if (t.note) parts.push(el('p', { class: 'note', text: t.note }));
    if (t.readable) {
      if (leadDims(t).length) parts.push(sliceBar(t));
      const save = fmt => el('button', { class: 'btn', id: 'save-' + fmt, text: fmt === 'npy' ? 'Save as .npy...' : 'Save as CSV...', title: (fmt === 'npy' ? 'Write this tensor to a NumPy .npy file' : 'Write the values to a CSV file, one row per line') + (statsSlice() ? ' (this slice only)' : ''),
        onclick: () => vscode.postMessage({ type: 'export', index: S.selected, format: fmt, slice: statsSlice() }) });
      parts.push(el('div', { class: 'btns' }, [save('npy'), save('csv')]));
    }
    const s = S.sampleFor === S.selected ? S.sample : null;
    if (!t.readable) { /* note says why */ } else if (!s) parts.push(el('p', { class: 'loading', text: 'Reading values...' }));
    else if (s.type === 'sampleError') parts.push(el('p', { class: 'note err', text: s.message }));
    else parts.push(...sampleView(t, s.result));
    if (t.readable && t.image) {
      // The picture goes right under the stats line, above the heatmap.
      const how = parts.findIndex(p => p && p.classList && p.classList.contains('how'));
      const block = [el('h3', { text: 'Image' }), el('div', { id: 'imgbox', class: 'imgbox' })];
      if (how >= 0) parts.splice(how + 1, 0, ...block); else parts.push(...block);
    }
    if (t.readable) {
      const hist = parts.findIndex(p => p && p.classList && p.classList.contains('hist-h'));
      const vbox = el('div', { id: 'values', class: 'values' });
      const block = [el('h3', { text: 'Values' }), vbox];
      if (hist >= 0) parts.splice(hist, 0, ...block); else parts.push(...block);
    }
    box.replaceChildren(...parts);
    if (t.readable) renderValues();
    if (t.readable && t.image) renderImage();
  }

  // ---------- image ----------
  // Any array with a picture layout (H x W, H x W x C, C x H x W, plus frame dimensions) can be drawn.
  // RGB layouts and uint8 arrays show at once; other arrays wait for "Show as image".
  const RANGE_LABELS = [['auto', 'Auto'], ['minmax', 'Min to max'], ['01', '0 to 1'], ['0255', '0 to 255'], ['sym', 'Symmetric around 0']];
  function imgState() {
    const t = S.model.tensors[S.selected];
    let st = S.img[S.selected];
    if (!st) st = S.img[S.selected] = { layout: t.image.layouts[0].id, frame: 0, range: 'auto', show: !!t.image.suggest };
    return st;
  }
  function imgReq() {
    const st = imgState();
    return { layout: st.layout, frame: st.frame, range: st.range };
  }
  const sameImg = (a, b) => !!a && !!b && a.layout === b.layout && a.frame === b.frame && a.range === b.range;
  function requestImage() {
    S.imgRes = null;
    requestImageOnce();
  }
  function renderImage() {
    const box = document.getElementById('imgbox');
    const t = S.model && S.model.tensors[S.selected];
    if (!box || !t || !t.image) return;
    const st = imgState();
    const L = t.image.layouts.find(l => l.id === st.layout) || t.image.layouts[0];
    if (!st.show) {
      const desc = `${L.H} x ${L.W}${L.C > 1 ? ' x ' + L.C : ''}${L.frames > 1 ? `, ${nf.format(L.frames)} frames` : ''}`;
      box.replaceChildren(el('div', { class: 'btns' }, [el('button', { class: 'btn', id: 'show-image', text: 'Show as image', title: `Draw this array as a picture (${desc})`, onclick: () => { st.show = true; requestImage(); renderImage(); } }),
        el('span', { class: 'cap', text: ` ${desc}` })]));
      return;
    }
    const set = patch => { Object.assign(st, patch); requestImage(); renderImage(); };
    const controls = [];
    if (t.image.layouts.length > 1) {
      const sel = el('select', { id: 'img-layout', 'aria-label': 'Picture layout' }, t.image.layouts.map(l => el('option', { value: l.id, text: l.label })));
      sel.value = L.id;
      sel.addEventListener('change', () => set({ layout: sel.value, frame: 0 }));
      controls.push(el('label', { class: 'ctl' }, [el('span', { class: 'fl', text: 'Layout' }), sel]));
    }
    if (L.frames > 1) {
      const inp = el('input', { type: 'number', class: 'num', id: 'img-frame', min: 0, max: L.frames - 1, step: 1, value: st.frame, 'aria-label': `Frame (0 to ${L.frames - 1})` });
      const go = v => { v = Math.max(0, Math.min(L.frames - 1, Math.round(Number(v)) || 0)); if (v !== st.frame) set({ frame: v }); else inp.value = v; };
      inp.addEventListener('change', () => go(inp.value));
      inp.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); go(inp.value); } });
      controls.push(el('span', { class: 'ctl' }, [el('span', { class: 'fl', text: 'Frame' }),
        el('button', { class: 'btn', id: 'img-prev', text: '<', title: 'Previous frame', 'aria-label': 'Previous frame', onclick: () => go(st.frame - 1) }), inp,
        el('button', { class: 'btn', id: 'img-next', text: '>', title: 'Next frame', 'aria-label': 'Next frame', onclick: () => go(st.frame + 1) }),
        el('span', { class: 'cap', text: ` of ${nf.format(L.frames)}` })]));
    }
    const rsel = el('select', { id: 'img-range', 'aria-label': 'Value range' }, RANGE_LABELS.map(([v, l]) => el('option', { value: v, text: l })));
    rsel.value = st.range;
    rsel.addEventListener('change', () => set({ range: rsel.value }));
    controls.push(el('label', { class: 'ctl' }, [el('span', { class: 'fl', text: 'Range' }), rsel]));
    const kids = [el('div', { class: 'imgctl' }, controls)];
    const r = S.imgRes && sameImg(S.imgRes.req, imgReq()) && S.imgRes.index === S.selected ? S.imgRes : null;
    if (!r) { kids.push(el('p', { class: 'loading', text: 'Drawing...' })); requestImageOnce(); }
    else if (r.type === 'imageError') kids.push(el('p', { class: 'note err', text: r.message }));
    else kids.push(imageView(r.result, Math.max(240, Math.min(640, (document.getElementById('detail').clientWidth || 480) - 44))));
    box.replaceChildren(...kids);
  }
  let imgAsked = null;
  function requestImageOnce() {
    const key = JSON.stringify([S.selected, imgReq()]);
    if (imgAsked === key) return;
    imgAsked = key;
    vscode.postMessage({ type: 'image', index: S.selected, ...imgReq() });
  }

  function sliceBar(t) {
    const lead = leadDims(t), cur = curSlice();
    const inputs = lead.map((n, k) => {
      const inp = el('input', { type: 'number', class: 'num slice-in', min: 0, max: n - 1, step: 1, value: cur[k], 'aria-label': `Index for dimension ${k} (0 to ${n - 1})`, title: `0 to ${n - 1}` });
      const apply = () => {
        let v = Math.round(Number(inp.value));
        if (!Number.isFinite(v)) v = 0;
        v = Math.max(0, Math.min(n - 1, v));
        inp.value = v;
        if (curSlice()[k] === v) return;
        const next = curSlice().slice(); next[k] = v;
        S.slices[S.selected] = next;
        S.vpage = { row: 0, col: 0 }; S.vals = null; S.valsErr = null;
        if (S.sliceStats) { requestSample(); renderDetail(); } else renderValues();
      };
      // A number field fires "change" only on blur, so Enter applies it too.
      inp.addEventListener('change', apply);
      inp.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); apply(); } });
      return inp;
    });
    const kids = [el('span', { class: 'fl', text: 'Slice' }), '['];
    inputs.forEach(inp => { kids.push(inp, ', '); });
    kids.push(':, :]');
    const box = el('input', { type: 'checkbox', id: 'slice-stats' });
    box.checked = S.sliceStats;
    box.addEventListener('change', () => { S.sliceStats = box.checked; requestSample(); renderDetail(); });
    return el('div', { class: 'slicebar' }, [el('div', { class: 'slicepick' }, kids),
      el('label', { class: 'slicestats', for: 'slice-stats' }, [box, ' Use only this slice for the stats, the heatmap and Save'])]);
  }

  // ---------- values table ----------
  function pageCols(total) {
    const w = (document.getElementById('detail') || {}).clientWidth || 480;
    const per = S.prec === '8' || S.prec === 'exact' ? 112 : 88;
    // At least 6 columns; the table scrolls sideways when the panel is narrow.
    return Math.max(1, Math.min(total, Math.max(6, Math.min(16, Math.floor((w - 90) / per)))));
  }
  function requestValues() {
    const t = S.model.tensors[S.selected];
    const g = t.shape.length === 0 ? [1, 1] : t.shape.length === 1 ? [t.shape[0], 1] : t.shape.slice(-2);
    vscode.postMessage({ type: 'values', index: S.selected, slice: curSlice(), row: S.vpage.row, col: S.vpage.col, rows: Math.min(PAGE_ROWS, g[0] - S.vpage.row), cols: pageCols(g[1] - S.vpage.col) });
  }
  function renderValues() {
    const box = document.getElementById('values');
    const t = S.model && S.model.tensors[S.selected];
    if (!box || !t) return;
    if (S.valsErr) { box.replaceChildren(el('p', { class: 'note err', text: S.valsErr })); return; }
    const v = S.vals;
    if (!v || v.index !== S.selected || !sameSlice(v.slice, curSlice()) || v.row !== S.vpage.row || v.col !== S.vpage.col) {
      if (!box.firstChild) box.append(el('p', { class: 'loading', text: 'Reading values...' }));
      requestValues();
      return;
    }
    const re = f64(v.re), im = v.im ? f64(v.im) : null;
    const go = (row, col) => {
      S.vpage = { row: Math.max(0, Math.min(row, v.totalRows - 1)), col: Math.max(0, Math.min(col, v.totalCols - 1)) };
      requestValues();
    };
    const range = (a, n, total, what) => `${what} ${nf.format(a)}-${nf.format(a + n - 1)} of ${nf.format(total)}`;
    const jump = (label, val, max, fn) => {
      const inp = el('input', { type: 'number', class: 'num', min: 0, max, step: 1, value: val, 'aria-label': label, title: label });
      const apply = () => { const x = Math.round(Number(inp.value)); if (Number.isFinite(x) && x !== val) fn(Math.max(0, Math.min(max, x))); };
      inp.addEventListener('change', apply);
      inp.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); apply(); } });
      return inp;
    };
    const nav = (txt, title, disabled, fn) => el('button', { class: 'btn nav', text: txt, title, 'aria-label': title, disabled: disabled ? '' : null, onclick: fn });
    const prec = el('select', { class: 'prec', 'aria-label': 'Number format', title: 'Number format' }, PRECISIONS.map(p => el('option', { value: p, text: { auto: 'Auto', exact: 'Exact', 2: '2 decimals', 4: '4 decimals', 8: '8 decimals', sci: 'Scientific' }[p] })));
    prec.value = S.prec;
    prec.addEventListener('change', () => { S.prec = prec.value; vscode.setState({ ...(vscode.getState() || {}), prec: S.prec }); renderValues(); });
    const bar = el('div', { class: 'vbar' }, [
      el('span', { class: 'vgroup' }, [nav('◀', 'Previous rows', v.row === 0, () => go(v.row - PAGE_ROWS, v.col)), nav('▶', 'Next rows', v.row + v.rows >= v.totalRows, () => go(v.row + PAGE_ROWS, v.col)),
        el('span', { class: 'vr', text: range(v.row, v.rows, v.totalRows, 'Rows') }), v.totalRows > PAGE_ROWS ? jump('Go to row', v.row, v.totalRows - 1, r => go(r, v.col)) : null]),
      v.totalCols > 1 ? el('span', { class: 'vgroup' }, [nav('◀', 'Previous columns', v.col === 0, () => go(v.row, v.col - v.cols)), nav('▶', 'Next columns', v.col + v.cols >= v.totalCols, () => go(v.row, v.col + v.cols)),
        el('span', { class: 'vr', text: range(v.col, v.cols, v.totalCols, 'Columns') }), v.totalCols > v.cols ? jump('Go to column', v.col, v.totalCols - 1, c => go(v.row, c)) : null]) : null,
      el('span', { class: 'vgroup' }, [prec, el('button', { class: 'btn', id: 'copy-values', text: 'Copy values', title: 'Copy this page as tab-separated text, with every digit', onclick: () => vscode.postMessage({ type: 'copyValues', index: S.selected, slice: curSlice(), row: v.row, col: v.col, rows: v.rows, cols: v.cols }) })]),
    ]);
    const cell = i => (im ? displayComplex(re[i], im[i], v.kind, S.prec) : displayText(re[i], v.kind, S.prec));
    const exact = i => (im ? complexText(re[i], im[i], v.kind) : exactText(re[i], v.kind));
    const lead = curSlice(), nd = t.shape.length;
    const where = (r, c) => (nd === 0 ? '[]' : nd === 1 ? `[${r}]` : '[' + lead.concat([r, c]).join(', ') + ']');
    const head = el('tr', null, [el('th', { class: 'corner', text: nd >= 3 ? sliceText(lead) : '' })].concat(nd >= 2 ? Array.from({ length: v.cols }, (_, c) => el('th', { text: nf.format(v.col + c) })) : [el('th', { text: 'value' })]));
    const body = [];
    for (let r = 0; r < v.rows; r++) {
      const tds = [el('th', { text: nf.format(v.row + r) })];
      for (let c = 0; c < v.cols; c++) {
        const i = r * v.cols + c, x = re[i];
        tds.push(el('td', { class: !Number.isFinite(x) ? 'nf' : x < 0 ? 'neg' : null, text: cell(i), title: `${where(v.row + r, v.col + c)} = ${exact(i)}` }));
      }
      body.push(el('tr', null, tds));
    }
    const cap = [];
    if (S.model.format === 'gguf' && /^Q/.test(String(v.kind))) cap.push(`${t.dtype} values after dequantizing, as llama.cpp computes them.`);
    else if (/^(f16|bf16|f8e4m3|f8e5m2)$/.test(String(v.kind))) cap.push(`${t.dtype} values, shown as their exact float32 value.`);
    if (v.kind === 'i64' || v.kind === 'u64') cap.push('64-bit integers above 9,007,199,254,740,991 lose their last digits here.');
    if (nd >= 3) cap.push(`Showing slice ${sliceText(lead)}.`);
    cap.push('Hover a cell for its exact value.');
    box.replaceChildren(bar, el('div', { class: 'vwrap' }, [el('table', { class: 'vals' }, [el('thead', null, [head]), el('tbody', null, body)])]), el('p', { class: 'cap', text: cap.join(' ') }));
  }
  function f64(b64) {
    const bin = atob(b64), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Float64Array(u.buffer);
  }

  function sampleView(t, r) {
    const st = r.stats;
    const out = [];
    out.push(el('div', { class: 'stats' }, [
      fact('Min', num(st.min)), fact('Max', num(st.max)), fact('Mean', num(st.mean)), fact('Std', num(st.std)),
      fact('NaN', nf.format(st.nan)), fact('+Inf', nf.format(st.posinf)), fact('-Inf', nf.format(st.neginf)), fact('Zeros', nf.format(st.zeros)),
    ]));
    const sl = statsSlice();
    out.push(el('p', { class: 'how', text: (sl ? `Slice ${sliceText(sl)}. ` : '') + (r.exact ? 'Exact, from ' : 'Sample of ') + r.how + `. Read in ${r.ms} ms.` }));
    const width = Math.max(240, Math.min(640, (document.getElementById('detail').clientWidth || 480) - 44));
    if (r.view && r.view.kind === 'heatmap') out.push(el('h3', { text: 'Heatmap' }), heatmap(r.view, width));
    if (r.view && r.view.kind === 'line') out.push(el('h3', { text: 'Values by index' }), linePlot(r.view, width));
    if (r.view && r.view.kind === 'image' && !t.image) out.push(el('h3', { text: 'Image' }), imageView(r.view, width));
    if (r.hist) out.push(el('h3', { class: 'hist-h', text: 'Histogram' }), histogram(r.hist, width));
    return out;
  }

  function histogram(h, W) {
    const H = 120, pad = { l: 8, r: 8, t: 6, b: 20 };
    const { c, g } = canvas(W, H);
    const col = scales().series, fg = cssVar('--vscode-descriptionForeground', '#999');
    const n = h.counts.length, maxC = Math.max(...h.counts);
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b, bw = pw / n;
    g.fillStyle = col;
    h.counts.forEach((v, i) => { const bh = v ? Math.max(1, (v / maxC) * ph) : 0; g.fillRect(pad.l + i * bw + 0.5, pad.t + ph - bh, Math.max(1, bw - 1), bh); });
    g.fillStyle = fg; g.font = '11px ' + cssVar('--vscode-font-family', 'sans-serif');
    g.textAlign = 'left'; g.fillText(num(h.min), pad.l, H - 5);
    g.textAlign = 'right'; g.fillText(num(h.max), W - pad.r, H - 5);
    if (h.min < 0 && h.max > 0) { const x = pad.l + (-h.min / (h.max - h.min)) * pw; g.textAlign = 'center'; g.fillText('0', x, H - 5); g.fillRect(x, pad.t + ph, 1, 3); }
    const total = h.counts.reduce((a, b) => a + b, 0);
    c.addEventListener('mousemove', ev => {
      const b = Math.floor((ev.offsetX - pad.l) / bw);
      if (b < 0 || b >= n) return hideTip();
      const w = (h.max - h.min) / n;
      showTip(ev, [`${num(h.min + b * w)} to ${num(h.min + (b + 1) * w)}`, `${nf.format(h.counts[b])} values (${((h.counts[b] / total) * 100).toFixed(2)}%)`]);
    });
    c.addEventListener('mouseleave', hideTip);
    c.setAttribute('role', 'img');
    c.setAttribute('aria-label', `Histogram of ${nf.format(total)} values from ${num(h.min)} to ${num(h.max)}`);
    return c;
  }

  function percentileAbs(data, p, signed) {
    const vals = [];
    for (const v of data) if (Number.isFinite(v)) vals.push(signed ? Math.abs(v) : v);
    if (!vals.length) return null;
    vals.sort((a, b) => a - b);
    const at = q => vals[Math.min(vals.length - 1, Math.max(0, Math.round(q * (vals.length - 1))))];
    return signed ? { hi: at(p) } : { lo: at(1 - p), hi: at(p) };
  }

  function heatmap(v, W) {
    const data = f32(v.data);
    let lo = Infinity, hi = -Infinity;
    for (const x of data) if (Number.isFinite(x)) { if (x < lo) lo = x; if (x > hi) hi = x; }
    const sc = scales();
    const diverging = lo < 0 && hi > 0;
    let a, b, clipped = false;
    if (diverging) { const p = percentileAbs(data, 0.995, true); const m = p.hi || Math.max(-lo, hi); a = -m; b = m; clipped = m < Math.max(-lo, hi); } else { const p = percentileAbs(data, 0.995, false) || { lo, hi }; a = p.lo; b = p.hi; clipped = a > lo || b < hi; if (a === b) { a = lo; b = hi + (hi === lo ? 1 : 0); } }
    const color = diverging ? sc.diverging : sc.sequential;
    const img = new ImageData(v.cols, v.rows);
    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      if (!Number.isFinite(x)) { img.data[i * 4 + 3] = 0; continue; }
      const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
      const c = color(t);
      img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
    }
    const off = el('canvas', { width: v.cols, height: v.rows });
    off.getContext('2d').putImageData(img, 0, 0);
    const aspect = v.srcRows / v.srcCols;
    const H = Math.round(Math.max(48, Math.min(300, W * aspect)));
    const { c, g } = canvas(W, H);
    g.imageSmoothingEnabled = false;
    g.drawImage(off, 0, 0, W, H);
    c.className = 'heat';
    c.addEventListener('mousemove', ev => {
      const col = Math.min(v.cols - 1, Math.floor((ev.offsetX / W) * v.cols)), row = Math.min(v.rows - 1, Math.floor((ev.offsetY / H) * v.rows));
      const r0 = Math.floor((row * v.srcRows) / v.rows), r1 = Math.floor(((row + 1) * v.srcRows) / v.rows) - 1;
      const c0 = Math.floor((col * v.srcCols) / v.cols), c1 = Math.floor(((col + 1) * v.srcCols) / v.cols) - 1;
      const val = data[row * v.cols + col];
      showTip(ev, [r0 === r1 ? `row ${r0}` : `rows ${r0} to ${r1}`, c0 === c1 ? `column ${c0}` : `columns ${c0} to ${c1}`, Number.isFinite(val) ? (r0 === r1 && c0 === c1 ? 'value ' : 'mean ') + num(val) : 'not sampled']);
    });
    c.addEventListener('mouseleave', hideTip);
    c.setAttribute('role', 'img');
    c.setAttribute('aria-label', `Heatmap of ${v.srcRows} by ${v.srcCols} values`);
    const bar = canvas(W, 10);
    for (let x = 0; x < W; x++) { const cc = color(x / (W - 1)); bar.g.fillStyle = `rgb(${cc[0]|0},${cc[1]|0},${cc[2]|0})`; bar.g.fillRect(x, 0, 1.5, 10); }
    const cap = [];
    cap.push(v.rows === v.srcRows && v.cols === v.srcCols ? `${nf.format(v.srcRows)} x ${nf.format(v.srcCols)}, one cell per value.` : `${nf.format(v.srcRows)} x ${nf.format(v.srcCols)} shown as ${v.rows} x ${v.cols} cells, each the mean of its block.`);
    if (v.reshaped) cap.push('Leading dimensions are flattened into rows.');
    if (clipped) cap.push('Colors are clipped at the 99.5th percentile so outliers do not wash out the rest.');
    return el('div', { class: 'viz' }, [c, bar.c, el('div', { class: 'scale' }, [el('span', { text: num(a) }), diverging ? el('span', { text: '0' }) : null, el('span', { text: num(b) })]), el('p', { class: 'cap', text: cap.join(' ') })]);
  }

  function linePlot(v, W) {
    const H = 160, pad = { l: 8, r: 8, t: 8, b: 20 };
    const mn = f32(v.min), mx = f32(v.max), me = f32(v.mean);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < v.points; i++) { if (Number.isFinite(mn[i]) && mn[i] < lo) lo = mn[i]; if (Number.isFinite(mx[i]) && mx[i] > hi) hi = mx[i]; }
    if (!(hi >= lo)) return el('p', { class: 'empty', text: 'No finite values to plot.' });
    if (hi === lo) { hi += 1; lo -= 1; }
    const { c, g } = canvas(W, H);
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const X = i => pad.l + (v.points > 1 ? (i / (v.points - 1)) * pw : pw / 2), Y = y => pad.t + (1 - (y - lo) / (hi - lo)) * ph;
    const col = scales().series, fg = cssVar('--vscode-descriptionForeground', '#999'), grid = cssVar('--vscode-panel-border', '#444');
    g.fillStyle = grid; if (lo < 0 && hi > 0) g.fillRect(pad.l, Math.round(Y(0)), pw, 1);
    const env = v.length > v.points;
    if (env) {
      g.globalAlpha = 0.3; g.fillStyle = col; g.beginPath();
      let started = false;
      for (let i = 0; i < v.points; i++) if (Number.isFinite(mx[i])) { if (!started) { g.moveTo(X(i), Y(mx[i])); started = true; } else g.lineTo(X(i), Y(mx[i])); }
      for (let i = v.points - 1; i >= 0; i--) if (Number.isFinite(mn[i])) g.lineTo(X(i), Y(mn[i]));
      g.closePath(); g.fill(); g.globalAlpha = 1;
    }
    g.strokeStyle = col; g.lineWidth = 2; g.lineJoin = 'round'; g.beginPath();
    let pen = false;
    for (let i = 0; i < v.points; i++) { if (!Number.isFinite(me[i])) { pen = false; continue; } if (!pen) { g.moveTo(X(i), Y(me[i])); pen = true; } else g.lineTo(X(i), Y(me[i])); }
    g.stroke();
    g.fillStyle = fg; g.font = '11px ' + cssVar('--vscode-font-family', 'sans-serif');
    g.textAlign = 'left'; g.fillText(num(hi), pad.l + 2, pad.t + 10); g.fillText('0', pad.l, H - 5);
    g.fillText(num(lo), pad.l + 2, pad.t + ph - 2);
    g.textAlign = 'right'; g.fillText(nf.format(v.length - 1), W - pad.r, H - 5);
    const hair = el('div', { class: 'hair' });
    c.addEventListener('mousemove', ev => {
      const i = Math.max(0, Math.min(v.points - 1, Math.round(((ev.offsetX - pad.l) / pw) * (v.points - 1))));
      hair.style.display = 'block'; hair.style.left = X(i) + 'px';
      const s0 = Math.floor((i * v.length) / v.points), s1 = Math.floor(((i + 1) * v.length) / v.points) - 1;
      showTip(ev, [s0 === s1 ? `index ${nf.format(s0)}` : `index ${nf.format(s0)} to ${nf.format(s1)}`, Number.isFinite(me[i]) ? (s0 === s1 ? `value ${num(me[i])}` : `mean ${num(me[i])}, min ${num(mn[i])}, max ${num(mx[i])}`) : 'not sampled']);
    });
    c.addEventListener('mouseleave', () => { hideTip(); hair.style.display = 'none'; });
    c.setAttribute('role', 'img');
    c.setAttribute('aria-label', `Line plot of ${nf.format(v.length)} values`);
    const cap = env ? `${nf.format(v.length)} values in ${v.points} buckets. The line is the bucket mean and the band spans its min and max.` : `${nf.format(v.length)} values.`;
    return el('div', { class: 'viz rel' }, [c, hair, el('p', { class: 'cap', text: cap })]);
  }

  function imageView(v, W) {
    const img = new ImageData(new Uint8ClampedArray(u8(v.data).buffer), v.width, v.height);
    const off = el('canvas', { width: v.width, height: v.height });
    off.getContext('2d').putImageData(img, 0, 0);
    const scale = Math.min(W / v.width, 300 / v.height, 4);
    const w = Math.round(v.width * scale), h = Math.round(v.height * scale);
    const { c, g } = canvas(w, h);
    g.imageSmoothingEnabled = scale < 1;
    g.drawImage(off, 0, 0, w, h);
    c.setAttribute('role', 'img');
    c.setAttribute('aria-label', `Image ${v.srcWidth} by ${v.srcHeight} pixels`);
    const parts = [`${v.srcHeight} x ${v.srcWidth} pixels, ${v.channels} channel${v.channels > 1 ? 's' : ''}, ${v.dtype || 'uint8'}.`];
    if (v.width !== v.srcWidth) parts.push(`Shown at ${v.width} x ${v.height}.`);
    if (v.frames > 1) parts.push(`Frame ${nf.format(v.frame)} of ${nf.format(v.frames)}, index [${v.frameIndex.join(', ')}].`);
    if (v.lo !== undefined && !(v.lo === 0 && v.hi === 255 && v.dtype === 'uint8')) parts.push(`Black is ${num(v.lo)} and white is ${num(v.hi)}${v.channels >= 3 ? ' in each channel' : ''}.`);
    if (v.nonFinite) parts.push(`${nf.format(v.nonFinite)} NaN or Inf value${v.nonFinite > 1 ? 's' : ''}${v.channels === 1 ? ', drawn in magenta' : ''}.`);
    if (v.lo !== undefined) {
      c.addEventListener('mousemove', ev => {
        const rect = c.getBoundingClientRect();
        const x = Math.floor(((ev.clientX - rect.left) / rect.width) * v.srcWidth), y = Math.floor(((ev.clientY - rect.top) / rect.height) * v.srcHeight);
        if (x >= 0 && y >= 0 && x < v.srcWidth && y < v.srcHeight) showTip(ev, [`row ${y}, column ${x}`]);
      });
      c.addEventListener('mouseleave', hideTip);
    }
    return el('div', { class: 'viz' }, [c, el('p', { class: 'cap', text: parts.join(' ') })]);
  }

  let rz = 0;
  window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { S.vals = null; renderDetail(); }, 150); });
  vscode.postMessage({ type: 'ready' });
})();
