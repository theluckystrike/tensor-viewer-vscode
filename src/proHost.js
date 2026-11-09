'use strict';
// Host side of the Pro tools: the commands, the free tries, the progress notification and the report panel.
// The tools read the files and write only a report file the user picks in a save dialog.
const path = require('path');
const { createPro } = require('./pro');
const { compareModels } = require('./pro/compare');
const { scanModel } = require('./pro/scan');
const { describe } = require('./pro/describe');
const { render } = require('./pro/table');
const { openReportPanel } = require('./pro/panel');

const TENSOR_EXT = ['npy', 'npz', 'safetensors', 'gguf', 'pt', 'pth', 'ckpt', 'bin'];
// Gate builds only (esbuild define); release builds trust only the live key inside src/license.
const TEST_KEY = typeof __ZOVO_TEST_PUBKEY__ === 'string' ? __ZOVO_TEST_PUBKEY__ : ''; // eslint-disable-line no-undef

const isUri = u => !!u && typeof u === 'object' && typeof u.scheme === 'string' && typeof u.path === 'string';
const baseName = uri => path.posix.basename(uri.path);

/**
 * deps: { vscode, context, openSource(uri), openTensorFile(source, name) }
 * Returns { pro, state(), runCompare(modelA, nameA, uriA), runScan(model, name), onDidChange }
 */
function registerPro({ vscode, context, openSource, openTensorFile, prefix = 'tensorViewer' }) {
  const pro = createPro({ vscode, context, prefix, testKey: TEST_KEY || undefined });
  const show = () => vscode.workspace.getConfiguration(prefix).get('showProFeatures', true) !== false;
  const state = () => ({ show: show() || pro.licensed, licensed: pro.licensed, trials: pro.trials() });

  function activeUri() {
    const tab = vscode.window.tabGroups.activeTabGroup.activeTab;
    const input = tab && tab.input;
    if (input && isUri(input.uri)) return input.uri;
    const ed = vscode.window.activeTextEditor;
    return ed ? ed.document.uri : null;
  }

  /** The uri when it names a tensor file, else null (then the command asks for a file). */
  const tensorUri = u => (u && TENSOR_EXT.includes(path.posix.extname(u.path).slice(1).toLowerCase()) ? u : null);

  async function pickFile(title, openLabel, near) {
    const r = await vscode.window.showOpenDialog({ canSelectMany: false, openLabel, title, defaultUri: near ? vscode.Uri.joinPath(near, '..') : undefined,
      filters: { 'Tensor files': TENSOR_EXT, 'All files': ['*'] } });
    return r && r[0] ? r[0] : null;
  }

  async function openModel(uri) {
    const source = await openSource(uri);
    try { return { source, model: await openTensorFile(source, baseName(uri)) }; } catch (e) { await source.close(); throw e; }
  }

  async function withProgress(title, fn) {
    return vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title, cancellable: true }, async (progress, token) => {
      let last = 0;
      const report = (f, text) => { const pct = Math.floor(f * 100); if (pct > last) { progress.report({ increment: pct - last, message: text }); last = pct; } };
      return fn({ cancelled: () => token.isCancellationRequested, progress: report });
    });
  }

  function showReport(report, near) {
    const desc = describe(report);
    const title = report.kind === 'compare' ? `Compare: ${report.fileA} vs ${report.fileB}` : `Scan: ${report.file}`;
    const stem = (report.kind === 'compare' ? `compare-${report.fileA}-vs-${report.fileB}` : `scan-${report.file}`).replace(/[^A-Za-z0-9_.-]+/g, '_').slice(0, 120);
    const sources = [near].concat(report._uris || []).filter(Boolean).map(u => u.toString());
    return openReportPanel({ vscode, viewType: `${prefix}.proReport`, title, desc, onSave: async format => {
      const ext = format === 'md' ? 'md' : format;
      const target = await vscode.window.showSaveDialog({
        defaultUri: near ? vscode.Uri.joinPath(near, '..', `${stem}.${ext}`) : undefined,
        filters: format === 'csv' ? { CSV: ['csv'] } : format === 'md' ? { Markdown: ['md'] } : { JSON: ['json'] },
        saveLabel: 'Save report', title: 'Save the report',
      });
      if (!target) return;
      if (sources.includes(target.toString())) { vscode.window.showErrorMessage('Pick another file name. The report never writes over a tensor file.'); return; }
      await vscode.workspace.fs.writeFile(target, Buffer.from(render(format, desc, report), 'utf8'));
      const a = await vscode.window.showInformationMessage(`Saved ${baseName(target)}.`, 'Open');
      if (a === 'Open') await vscode.commands.executeCommand('vscode.open', target);
    } });
  }

  /** Compare an open model (A) with a file the user picks (B), or two files. */
  async function runCompare(modelA, nameA, uriA, uriB) {
    if (!(await pro.allowed('compare'))) return;
    let a = modelA ? { model: modelA, source: null } : null;
    if (!a) {
      if (!uriA) uriA = await pickFile('Compare: pick the first file (A)', 'Pick A');
      if (!uriA) return;
    }
    if (!uriB) uriB = await pickFile(`Compare ${nameA || baseName(uriA)} with...`, 'Compare', uriA);
    if (!uriB) return;
    let b = null;
    try {
      if (!a) a = await openModel(uriA);
      b = await openModel(uriB);
    } catch (e) {
      if (a && a.source) await a.source.close();
      vscode.window.showErrorMessage(`Could not open the file for the compare. ${e.message || e}`);
      return;
    }
    await pro.useTrial('compare');
    const nA = nameA || baseName(uriA), nB = baseName(uriB);
    try {
      const report = await withProgress(`Comparing ${nA} with ${nB}`, o => compareModels(a.model, b.model, nA, nB, o));
      report._uris = [uriA, uriB];
      showReport(report, uriA);
      return report;
    } catch (e) {
      vscode.window.showErrorMessage(`The compare stopped. ${e.message || e}`);
    } finally {
      if (a.source) await a.source.close();
      await b.source.close();
    }
  }

  /** Scan every tensor of an open model, or of a file. */
  async function runScan(model, name, uri) {
    if (!(await pro.allowed('scan'))) return;
    let opened = null;
    if (!model) {
      if (!uri) uri = await pickFile('Scan all tensors of a file', 'Scan');
      if (!uri) return;
      try { opened = await openModel(uri); } catch (e) { vscode.window.showErrorMessage(`Could not open ${baseName(uri)}. ${e.message || e}`); return; }
      model = opened.model; name = baseName(uri);
    }
    await pro.useTrial('scan');
    try {
      const report = await withProgress(`Scanning every tensor of ${name}`, o => scanModel(model, name, o));
      report._uris = uri ? [uri] : [];
      showReport(report, uri);
      return report;
    } catch (e) {
      vscode.window.showErrorMessage(`The scan stopped. ${e.message || e}`);
    } finally {
      if (opened) await opened.source.close();
    }
  }

  context.subscriptions.push(
    vscode.commands.registerCommand(`${prefix}.compareFiles`, (arg, list) => {
      const two = Array.isArray(list) && list.length >= 2 && list.every(isUri) ? list : null;
      if (two) return runCompare(null, null, two[0], two[1]);
      return runCompare(null, null, isUri(arg) ? arg : tensorUri(activeUri()));
    }),
    vscode.commands.registerCommand(`${prefix}.scanTensors`, arg => runScan(null, null, isUri(arg) ? arg : tensorUri(activeUri()))),
    vscode.window.registerUriHandler({ handleUri: uri => pro.handleUri(uri) }),
  );
  return { pro, state, runCompare, runScan, onDidChange: pro.onDidChange };
}

module.exports = { registerPro, TENSOR_EXT };
