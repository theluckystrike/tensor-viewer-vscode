'use strict';
const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const { buildHtml } = require('./html');
const { FileSource, BufferSource } = require('./core/source');
const { openTensorFile } = require('./core/formats');
const { sampleTensor } = require('./core/sample');
const { validateWebviewMessage, publicModel, shapeText } = require('./core/protocol');
const { readValues, sliceTensor, pageText, planExport, writeExport, exportName } = require('./core/values');
const { readImage } = require('./core/image');
const { RAW_TYPES, parseShape, resolveRawLayout, shapeHint, cleanLayout } = require('./core/raw');
const { PLAIN } = require('./core/dtypes');
const { createRatingPrompt } = require('./rate');
const { registerSupport, commandUris, showLinks } = require('./support');
const { registerPro } = require('./proHost');

let rating = { success() {}, failure() {} };
let proHost = null; // Zovo Pro tools (compare, scan); set in activate()
const proState = () => (proHost ? proHost.state() : { show: false, licensed: false, trials: {} });

const VIEW_TYPE = 'tensorViewer.view';
const RAW_VIEW_TYPE = 'tensorViewer.raw';
const RAW_KEY = 'tensorViewer.rawLayouts';
const rawPanels = new Map(); // uri string -> Set of reload functions for open raw editors

/** The layout last used for a raw file, from workspace state. */
function storedLayout(context, uri) {
  const all = context.workspaceState.get(RAW_KEY, {}) || {};
  return cleanLayout(all[uri.toString()]);
}
async function storeLayout(context, uri, layout) {
  const all = { ...(context.workspaceState.get(RAW_KEY, {}) || {}) };
  delete all[uri.toString()];
  all[uri.toString()] = layout;
  const keys = Object.keys(all);
  for (const k of keys.slice(0, Math.max(0, keys.length - 200))) delete all[k]; // keep the 200 most recent
  await context.workspaceState.update(RAW_KEY, all);
}

/**
 * Ask for the value type, shape, header size and byte order of a raw file.
 * @returns {Promise<{kind:string, shape:number[], offset:number, littleEndian:boolean}|null>} null when cancelled
 */
async function askRawLayout(uri, size, prev) {
  const name = path.posix.basename(uri.path);
  const items = RAW_TYPES.map(t => ({ label: t.np, description: `${PLAIN[t.kind].size} byte${PLAIN[t.kind].size > 1 ? 's' : ''} per value`, detail: shapeHint(t.kind, size, prev ? prev.offset : 0), kind: t.kind }));
  if (prev) { const i = items.findIndex(x => x.kind === prev.kind); if (i > 0) items.unshift(...items.splice(i, 1)); }
  const pick = await vscode.window.showQuickPick(items, { title: `Open ${name} as a raw array (1/4): value type`, placeHolder: 'The dtype the file was written with, for example float32 for np.float32(...).tofile()', matchOnDescription: true });
  if (!pick) return null;
  const kind = pick.kind;
  const offText = await vscode.window.showInputBox({
    title: `Open ${name} as a raw array (2/4): header bytes to skip`, value: String(prev ? prev.offset : 0), prompt: 'Bytes before the first value. Use 0 for files from ndarray.tofile().',
    validateInput: v => (/^\s*\d+\s*$/.test(v) && Number(v) <= size ? null : `A whole number from 0 to ${size}.`),
  });
  if (offText === undefined) return null;
  const offset = Number(offText.trim());
  const shapeText = await vscode.window.showInputBox({
    title: `Open ${name} as a raw array (3/4): shape`, value: prev && prev.kind === kind ? prev.shape.join(', ') : '',
    placeHolder: 'For example 480, 640, 3 or -1, 3. Leave empty for one long 1-D array.',
    prompt: `${shapeHint(kind, size, offset)}. Use -1 once to fill in from the file size.`,
    validateInput: v => {
      const p = parseShape(v);
      if (!p.ok) return p.error;
      const r = resolveRawLayout({ kind, shape: p.shape, offset }, size);
      return r.ok ? null : r.error;
    },
  });
  if (shapeText === undefined) return null;
  let littleEndian = true;
  if (PLAIN[kind].size > 1) {
    const order = await vscode.window.showQuickPick([
      { label: 'Little-endian', description: 'x86, ARM, Apple silicon: almost every file', le: true },
      { label: 'Big-endian', description: 'some old instruments, network formats and Java tools', le: false },
    ], { title: `Open ${name} as a raw array (4/4): byte order` });
    if (!order) return null;
    littleEndian = order.le;
  }
  return { kind, shape: parseShape(shapeText).shape, offset, littleEndian };
}

/** Command: open any file as a raw array. Asks for the layout, then opens or reloads the raw viewer. */
async function openRawCommand(context, arg) {
  const isUri = u => !!u && typeof u === 'object' && typeof u.scheme === 'string' && typeof u.path === 'string';
  let uri = isUri(arg) ? arg : null;
  if (!uri) {
    const tab = vscode.window.tabGroups.activeTabGroup.activeTab;
    const input = tab && tab.input;
    if (input && isUri(input.uri)) uri = input.uri;
  }
  if (!uri) {
    const picked = await vscode.window.showOpenDialog({ canSelectMany: false, openLabel: 'Open as raw array', title: 'Pick a raw binary file' });
    if (!picked || !picked.length) return;
    uri = picked[0];
  }
  let size;
  try { size = (await vscode.workspace.fs.stat(uri)).size; } catch (e) { vscode.window.showErrorMessage(`Could not read ${uri.path}: ${e.message || e}`); return; }
  const layout = await askRawLayout(uri, size, storedLayout(context, uri));
  if (!layout) return;
  await storeLayout(context, uri, layout);
  const open = rawPanels.get(uri.toString());
  if (open && open.size) { for (const reload of open) reload(); }
  await vscode.commands.executeCommand('vscode.openWith', uri, RAW_VIEW_TYPE);
}

function settings() {
  const c = vscode.workspace.getConfiguration('tensorViewer');
  return { sampleValues: c.get('sampleValues', 2000000), maxRemoteFileMB: c.get('maxRemoteFileMB', 512) };
}

/** True when a file starts like plain text: no pickle or zip header, no NUL byte, valid UTF-8 in the first 4 KB. */
async function looksLikeText(uri) {
  let src;
  try {
    src = await openSource(uri);
    const head = await src.read(0, Math.min(4096, src.size));
    if (head.length && (head[0] === 0x80 || (head[0] === 0x50 && head[1] === 0x4b))) return false;
    if (head.includes(0)) return false;
    new TextDecoder('utf-8', { fatal: true }).decode(head, { stream: true });
    return true;
  } catch { return false; } finally { if (src) await src.close(); }
}

async function openSource(uri) {
  if (uri.scheme === 'file') return FileSource.open(uri.fsPath);
  // Files that are not on disk (remote or virtual) can only be read whole.
  const limit = settings().maxRemoteFileMB;
  const stat = await vscode.workspace.fs.stat(uri);
  if (stat.size > limit * 1048576) throw new Error(`This file is not on local disk and is ${(stat.size / 1048576).toFixed(0)} MB. Such files are read whole, and the limit is ${limit} MB (tensorViewer.maxRemoteFileMB).`);
  return new BufferSource(await vscode.workspace.fs.readFile(uri));
}

function b64(typed) { return Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength).toString('base64'); }

function sizeText(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1073741824) return `${(n / 1048576).toFixed(1)} MB`;
  return `${(n / 1073741824).toFixed(2)} GB`;
}

/**
 * Save a tensor (or one slice) as .npy or CSV. The user picks the file; the source file is never written.
 * File targets are written to a temporary name and renamed at the end, so a cancel leaves nothing behind.
 * @returns {Promise<{ok:boolean, text:string}>}
 */
async function exportTensor(srcUri, t, format, slice) {
  let plan;
  try { plan = planExport(t, format, slice); } catch (e) {
    vscode.window.showErrorMessage(e.message || String(e));
    return { ok: false, text: e.message || String(e) };
  }
  const name = exportName(t.name, plan.label, format);
  const dir = vscode.Uri.joinPath(srcUri, '..');
  const target = await vscode.window.showSaveDialog({
    defaultUri: vscode.Uri.joinPath(dir, name),
    filters: format === 'npy' ? { 'NumPy array': ['npy'] } : { CSV: ['csv'] },
    saveLabel: format === 'npy' ? 'Save .npy' : 'Save CSV',
    title: `Save ${t.name}${plan.label ? ' ' + plan.label : ''} as ${format === 'npy' ? '.npy' : 'CSV'}`,
  });
  if (!target) return { ok: false, text: 'Not saved.' };
  if (target.toString() === srcUri.toString() || (target.scheme === 'file' && srcUri.scheme === 'file' && path.resolve(target.fsPath) === path.resolve(srcUri.fsPath))) {
    const text = 'Pick another file name. Tensor Viewer never writes over the file it is showing.';
    vscode.window.showErrorMessage(text);
    return { ok: false, text };
  }
  const label = path.posix.basename(target.path);
  try {
    const result = await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: `Saving ${label}`, cancellable: true }, async (progress, token) => {
      let last = 0, written = 0;
      const report = f => { const pct = Math.floor(f * 100); if (pct > last) { progress.report({ increment: pct - last }); last = pct; } };
      if (target.scheme === 'file') {
        const tmp = `${target.fsPath}.part-${process.pid}-${Date.now()}`;
        const fh = await fs.promises.open(tmp, 'wx');
        let done = false;
        try {
          done = await writeExport(plan, async buf => { await fh.write(buf); written += buf.length; }, { cancelled: () => token.isCancellationRequested, progress: report });
          await fh.close();
          if (done) await fs.promises.rename(tmp, target.fsPath);
        } finally {
          await fh.close().catch(() => {});
          if (!done) await fs.promises.unlink(tmp).catch(() => {});
        }
        return { done, written };
      }
      const chunks = [];
      const done = await writeExport(plan, async buf => {
        written += buf.length;
        if (written > 512 * 1048576) throw new Error('Files outside local disk are written in one piece, up to 512 MB. Save to a local folder instead.');
        chunks.push(buf);
      }, { cancelled: () => token.isCancellationRequested, progress: report });
      if (done) await vscode.workspace.fs.writeFile(target, Buffer.concat(chunks));
      return { done, written };
    });
    if (!result.done) return { ok: false, text: 'Saving was cancelled. Nothing was written.' };
    const text = `Saved ${label} (${sizeText(result.written)}).${plan.note ? ' ' + plan.note : ''}`;
    vscode.window.showInformationMessage(text);
    return { ok: true, text };
  } catch (e) {
    const text = `Could not save ${label}. ${e.message || String(e)}`;
    vscode.window.showErrorMessage(text);
    return { ok: false, text };
  }
}

class TensorEditorProvider {
  constructor(context, raw = false) { this.context = context; this.raw = raw; }

  openCustomDocument(uri) { return { uri, dispose() {} }; }

  async resolveCustomEditor(document, panel) {
    const ctx = this.context;
    const dist = vscode.Uri.joinPath(ctx.extensionUri, 'dist');
    const media = vscode.Uri.joinPath(ctx.extensionUri, 'media');
    // Command links are limited to the two support commands and the settings page (footer "Hide" link).
    panel.webview.options = { enableScripts: true, localResourceRoots: [dist, media], enableCommandUris: commandUris('tensorViewer') };
    const uri = document.uri;
    const name = path.posix.basename(uri.path);
    panel.webview.html = buildHtml({
      cspSource: panel.webview.cspSource,
      scriptUri: panel.webview.asWebviewUri(vscode.Uri.joinPath(dist, 'webview.js')).toString(),
      styleUri: panel.webview.asWebviewUri(vscode.Uri.joinPath(media, 'webview.css')).toString(),
      title: name,
      showSupportLinks: showLinks(vscode, 'tensorViewer'),
    });

    let disposed = false;
    let source = null;
    let seq = 0; // newest select wins; older samples are dropped
    let counted = false; // one success per opened tab, for the rating prompt
    const post = m => { if (!disposed) panel.webview.postMessage(m); };
    const isRaw = this.raw;
    let modelP, openMsP;
    // Raw files need a type and shape first: from the last time, or asked now.
    const load = async () => {
      const t0 = Date.now();
      modelP = (async () => {
        let layout = null;
        if (isRaw) {
          layout = storedLayout(ctx, uri);
          if (!layout) {
            const size = (await vscode.workspace.fs.stat(uri)).size;
            layout = await askRawLayout(uri, size, null);
            if (!layout) { const e = new Error(`Pick the value type and shape to read ${name} as a raw array.`); e.needsLayout = true; e.noLayoutYet = true; throw e; }
            await storeLayout(ctx, uri, layout);
          }
        }
        // The old file stays open until the new read works, so a reload that fails leaves the view usable.
        const next = await openSource(uri);
        if (disposed) { await next.close(); throw new Error('closed'); }
        let model;
        try { model = await openTensorFile(next, name, layout ? { raw: layout } : {}); } catch (e) { await next.close(); if (isRaw) e.needsLayout = true; throw e; }
        const old = source; source = next;
        if (old) old.close();
        return model;
      })();
      modelP.catch(() => {});
      // Time to read the header, measured once per load. The webview asks again when a hidden tab comes back.
      openMsP = modelP.then(() => Date.now() - t0, () => 0);
    };
    const sendModel = async (reloaded = false) => {
      try {
        const model = await modelP;
        post({ ...publicModel(model, name), openMs: await openMsP, reloaded, pro: proState() });
        if (!counted) { counted = true; rating.success(uri); }
      } catch (e) {
        post({ type: 'error', message: e.message || String(e), action: e.needsLayout ? 'rawLayout' : undefined });
        // Rating: a file that did not open blocks the ask. Not counted: a raw file whose type and shape were not
        // picked yet, a closed tab, and a plain-text .pth path file, which moves to the text editor (see below).
        if (!e.noLayoutYet && e.message !== 'closed' && !(!isRaw && /\.pth$/i.test(name) && await looksLikeText(uri).catch(() => false))) rating.failure();
      }
    };
    load();
    // Python's site-packages also use .pth, for plain-text path files. When a .pth file is not a PyTorch file and
    // reads as text, show it in the text editor instead of an error, as VS Code did before this viewer opened .pth.
    if (!isRaw && /\.pth$/i.test(name)) {
      modelP.then(() => {}, async () => {
        if (disposed || !(await looksLikeText(uri)) || disposed) return;
        try { await vscode.commands.executeCommand('vscode.openWith', uri, 'default', panel.viewColumn); panel.dispose(); } catch { /* the error view stays */ }
      });
    }
    // Auto-reload: a training loop that saves a checkpoint, or a script that rewrites the file, updates the view.
    // Writers often truncate first and write for a while, so wait until the size and time stop changing,
    // and retry a few times when the file does not parse yet.
    let watchTimer = null, tries = 0, lastStat = null;
    const scheduleReload = ms => { clearTimeout(watchTimer); watchTimer = setTimeout(reloadIfStable, ms); };
    const reloadIfStable = async () => {
      if (disposed) return;
      let st;
      try { st = await vscode.workspace.fs.stat(uri); } catch { if (++tries < 10) scheduleReload(1000); return; }
      const key = `${st.size}:${st.mtime}`;
      if (key !== lastStat) { lastStat = key; scheduleReload(400); return; }
      load();
      try { await modelP; } catch (e) {
        if (e && e.message === 'closed') return;
        if (++tries < 6) { lastStat = null; scheduleReload(1000); return; }
      }
      tries = 0; lastStat = null;
      await sendModel(true);
    };
    let watcher = null;
    if (vscode.workspace.getConfiguration('tensorViewer').get('autoReload', true)) {
      try {
        watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(vscode.Uri.joinPath(uri, '..'), path.posix.basename(uri.path)));
        const changed = () => { tries = 0; lastStat = null; scheduleReload(300); };
        // A rewrite in place fires a change; tools that write a temp file and rename it fire a create.
        watcher.onDidChange(changed);
        watcher.onDidCreate(changed);
      } catch { watcher = null; }
    }
    let reloadFn = null;
    if (isRaw) {
      reloadFn = () => { load(); sendModel(); };
      const key = uri.toString();
      if (!rawPanels.has(key)) rawPanels.set(key, new Set());
      rawPanels.get(key).add(reloadFn);
    }

    panel.webview.onDidReceiveMessage(async raw => {
      if (raw && raw.type === 'rawLayout') {
        if (!isRaw) return;
        let size;
        try { size = (await vscode.workspace.fs.stat(uri)).size; } catch { return; }
        const layout = await askRawLayout(uri, size, storedLayout(ctx, uri));
        if (!layout) return;
        await storeLayout(ctx, uri, layout);
        for (const reload of rawPanels.get(uri.toString()) || []) reload();
        return;
      }
      if (raw && raw.type === 'ready') { await sendModel(); return; }
      let model;
      try { model = await modelP; } catch { return; }
      const v = validateWebviewMessage(raw, model.tensors.length, model.tensors);
      if (!v.ok) return;
      const msg = v.msg;
      if (msg.type === 'pro') {
        if (!proHost) return;
        if (msg.action === 'compare') await proHost.runCompare(model, name, uri);
        else await proHost.runScan(model, name, uri);
        return;
      }
      if (msg.type === 'image') {
        const req = { layout: msg.layout, frame: msg.frame, range: msg.range };
        try {
          const result = await readImage(model.tensors[msg.index], req);
          post({ type: 'image', index: msg.index, req, result });
        } catch (e) {
          post({ type: 'imageError', index: msg.index, req, message: e.message || String(e) });
        }
        return;
      }
      if (msg.type === 'select') {
        const my = ++seq;
        const slice = msg.slice || null;
        try {
          const t = slice ? sliceTensor(model.tensors[msg.index], slice) : model.tensors[msg.index];
          const result = await sampleTensor(t, { sampleValues: settings().sampleValues });
          if (my === seq) post({ type: 'sample', index: msg.index, slice, result });
        } catch (e) {
          if (my === seq) post({ type: 'sampleError', index: msg.index, slice, message: e.message || String(e) });
        }
        return;
      }
      if (msg.type === 'values') {
        try {
          const r = await readValues(model.tensors[msg.index], msg);
          post({ type: 'values', index: msg.index, slice: r.slice, row: r.row, col: r.col, rows: r.rows, cols: r.cols, totalRows: r.totalRows, totalCols: r.totalCols, kind: r.kind, re: b64(r.re), im: r.im ? b64(r.im) : null });
        } catch (e) {
          post({ type: 'valuesError', index: msg.index, slice: msg.slice, row: msg.row, col: msg.col, message: e.message || String(e) });
        }
        return;
      }
      if (msg.type === 'copyValues') {
        try {
          const text = pageText(await readValues(model.tensors[msg.index], msg));
          if (text.length > 1000000) throw new Error('This page is too long to copy.');
          await vscode.env.clipboard.writeText(text);
          post({ type: 'copied', what: 'values', text: `${msg.rows} x ${msg.cols} values` });
        } catch (e) {
          post({ type: 'copied', what: 'error', text: e.message || String(e) });
        }
        return;
      }
      if (msg.type === 'export') {
        const r = await exportTensor(uri, model.tensors[msg.index], msg.format, msg.slice);
        post({ type: 'exported', ok: r.ok, text: r.text });
        return;
      }
      if (msg.type === 'copy') {
        const text = shapeText(model.tensors[msg.index], msg.what);
        await vscode.env.clipboard.writeText(text);
        post({ type: 'copied', what: msg.what, text });
      }
    });
    const proSubs = proHost ? [
      proHost.onDidChange(() => post({ type: 'pro', ...proState() })),
      vscode.workspace.onDidChangeConfiguration(e => { if (e.affectsConfiguration('tensorViewer.showProFeatures')) post({ type: 'pro', ...proState() }); }),
    ] : [];
    panel.onDidDispose(() => {
      disposed = true;
      for (const d of proSubs) d.dispose();
      clearTimeout(watchTimer);
      if (watcher) watcher.dispose();
      if (source) source.close();
      if (reloadFn) { const set = rawPanels.get(uri.toString()); if (set) { set.delete(reloadFn); if (!set.size) rawPanels.delete(uri.toString()); } }
    });
  }
}

function activate(context) {
  rating = createRatingPrompt({ vscode, context, extensionId: 'zovo.tensor-viewer', section: 'tensorViewer', name: 'NPZ Viewer & NPY Viewer' });
  registerSupport({ vscode, context, prefix: 'tensorViewer', selfId: 'zovo.tensor-viewer' });
  // The Pro tools must never stop the free viewer: if they cannot start, the viewer works without them.
  try { proHost = registerPro({ vscode, context, openSource, openTensorFile }); } catch (e) { proHost = null; console.error('Zovo Pro tools did not start:', e && e.message); }
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(VIEW_TYPE, new TensorEditorProvider(context), {
      supportsMultipleEditorsPerDocument: true,
      webviewOptions: { retainContextWhenHidden: false },
    }),
    vscode.window.registerCustomEditorProvider(RAW_VIEW_TYPE, new TensorEditorProvider(context, true), {
      supportsMultipleEditorsPerDocument: true,
      webviewOptions: { retainContextWhenHidden: false },
    }),
    vscode.commands.registerCommand('tensorViewer.openRaw', arg => openRawCommand(context, arg)),
    vscode.commands.registerCommand('tensorViewer.openWalkthrough', () =>
      vscode.commands.executeCommand('workbench.action.openWalkthrough', 'zovo.tensor-viewer#tensorViewer.start', false)),
  );
}

function deactivate() {}

module.exports = { activate, deactivate, TensorEditorProvider, VIEW_TYPE, RAW_VIEW_TYPE, exportTensor, askRawLayout };
