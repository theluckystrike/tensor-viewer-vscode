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
const { createRatingPrompt } = require('./rate');
const { registerSupport, commandUris, showLinks } = require('./support');

let rating = { success() {} };

const VIEW_TYPE = 'tensorViewer.view';

function settings() {
  const c = vscode.workspace.getConfiguration('tensorViewer');
  return { sampleValues: c.get('sampleValues', 2000000), maxRemoteFileMB: c.get('maxRemoteFileMB', 512) };
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
  constructor(context) { this.context = context; }

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
    const t0 = Date.now();
    const modelP = (async () => {
      source = await openSource(uri);
      if (disposed) { await source.close(); throw new Error('closed'); }
      return openTensorFile(source, name);
    })();
    modelP.catch(() => {});
    // Time to read the header, measured once. The webview asks again when a hidden tab comes back.
    const openMsP = modelP.then(() => Date.now() - t0, () => 0);

    panel.webview.onDidReceiveMessage(async raw => {
      let model;
      try { model = await modelP; } catch (e) { if (raw && raw.type === 'ready') post({ type: 'error', message: e.message || String(e) }); return; }
      const v = validateWebviewMessage(raw, model.tensors.length, model.tensors);
      if (!v.ok) return;
      const msg = v.msg;
      if (msg.type === 'ready') {
        post({ ...publicModel(model, name), openMs: await openMsP });
        if (!counted) { counted = true; rating.success(); }
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
    panel.onDidDispose(() => { disposed = true; if (source) source.close(); });
  }
}

function activate(context) {
  rating = createRatingPrompt({ vscode, context, extensionId: 'zovo.tensor-viewer', section: 'tensorViewer', message: (n) => `Tensor Viewer has opened ${n} files for you. A rating helps other people find it.` });
  registerSupport({ vscode, context, prefix: 'tensorViewer', selfId: 'zovo.tensor-viewer' });
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(VIEW_TYPE, new TensorEditorProvider(context), {
      supportsMultipleEditorsPerDocument: true,
      webviewOptions: { retainContextWhenHidden: false },
    }),
    vscode.commands.registerCommand('tensorViewer.openWalkthrough', () =>
      vscode.commands.executeCommand('workbench.action.openWalkthrough', 'zovo.tensor-viewer#tensorViewer.start', false)),
  );
}

function deactivate() {}

module.exports = { activate, deactivate, TensorEditorProvider, VIEW_TYPE, exportTensor };
