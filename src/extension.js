'use strict';
const vscode = require('vscode');
const path = require('path');
const { buildHtml } = require('./html');
const { FileSource, BufferSource } = require('./core/source');
const { openTensorFile } = require('./core/formats');
const { sampleTensor } = require('./core/sample');
const { validateWebviewMessage, publicModel, shapeText } = require('./core/protocol');
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

    panel.webview.onDidReceiveMessage(async raw => {
      let model;
      try { model = await modelP; } catch (e) { if (raw && raw.type === 'ready') post({ type: 'error', message: e.message || String(e) }); return; }
      const v = validateWebviewMessage(raw, model.tensors.length);
      if (!v.ok) return;
      const msg = v.msg;
      if (msg.type === 'ready') {
        post({ ...publicModel(model, name), openMs: Date.now() - t0 });
        if (!counted) { counted = true; rating.success(); }
        return;
      }
      if (msg.type === 'select') {
        const my = ++seq;
        try {
          const result = await sampleTensor(model.tensors[msg.index], { sampleValues: settings().sampleValues });
          if (my === seq) post({ type: 'sample', index: msg.index, result });
        } catch (e) {
          if (my === seq) post({ type: 'sampleError', index: msg.index, message: e.message || String(e) });
        }
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

module.exports = { activate, deactivate, TensorEditorProvider, VIEW_TYPE };
