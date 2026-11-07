'use strict';
// Runs the extension host code with a mocked vscode module and real fixture files.
const test = require('node:test');
const assert = require('node:assert');
const Module = require('module');
const path = require('path');
const { FIX } = require('./helpers');

const root = path.join(__dirname, '..');
const registered = { editors: [], commands: {} };
let clipboard = '';
const watchers = [];
const executed = [];
const vscodeMock = {
  Uri: {
    joinPath: (u, ...p) => ({ ...u, path: path.join(u.path, ...p), fsPath: path.join(u.fsPath || u.path, ...p) }),
    file: p => ({ scheme: 'file', path: p, fsPath: p, toString() { return 'file://' + this.path; } }),
  },
  window: { registerCustomEditorProvider: (id, provider) => { registered.editors.push({ id, provider }); return { dispose() {} }; } },
  commands: { registerCommand: (id, fn) => { registered.commands[id] = fn; return { dispose() {} }; }, executeCommand: async (...a) => { executed.push(a); } },
  RelativePattern: class { constructor(base, pattern) { this.base = base; this.pattern = pattern; } },
  workspace: {
    getConfiguration: () => ({ get: (_k, d) => d }),
    fs: { stat: async u => { const st = require('fs').statSync(u.fsPath); return { size: st.size, mtime: st.mtimeMs }; } },
    createFileSystemWatcher: pattern => {
      const w = { pattern, change: [], create: [], disposed: false, onDidChange: fn => w.change.push(fn), onDidCreate: fn => w.create.push(fn), dispose() { w.disposed = true; } };
      watchers.push(w);
      return w;
    },
  },
  env: { clipboard: { writeText: async t => { clipboard = t; } } },
};
const origLoad = Module._load;
Module._load = function (req, ...rest) { return req === 'vscode' ? vscodeMock : origLoad.call(this, req, ...rest); };
const ext = require('../src/extension');

test('activate registers the editor and every command in package.json', () => {
  ext.activate({ subscriptions: [], extensionUri: vscodeMock.Uri.file(root), extensionPath: root });
  const pkg = require('../package.json');
  assert.deepStrictEqual(registered.editors.map(e => e.id), pkg.contributes.customEditors.map(e => e.viewType));
  assert.deepStrictEqual(Object.keys(registered.commands).sort(), pkg.contributes.commands.map(c => c.command).sort());
  assert.strictEqual(pkg.keywords.length, 30);
  assert.strictEqual(pkg.publisher, 'zovo');
});

function fakePanel() {
  const sent = [];
  const waiters = [];
  let onMsg, onDispose;
  const panel = {
    webview: {
      cspSource: 'vscode-resource:', options: {}, html: '',
      asWebviewUri: u => ({ toString: () => 'vscode-resource:' + u.path }),
      postMessage: m => { sent.push(JSON.parse(JSON.stringify(m))); waiters.slice().forEach(w => w()); return Promise.resolve(true); },
      onDidReceiveMessage: fn => { onMsg = fn; },
    },
    onDidDispose: fn => { onDispose = fn; },
    viewColumn: 1,
    dispose: () => { panel.disposedByExt = true; if (onDispose) onDispose(); },
  };
  const until = pred => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timeout: ' + JSON.stringify(sent.map(m => m.type)))), 10000);
    const check = () => { const m = sent.find(pred); if (m) { clearTimeout(timer); resolve(m); } };
    waiters.push(check); check();
  });
  return { panel, sent, until, send: m => onMsg(m), dispose: () => onDispose() };
}

test('a GGUF file opens: model, sample and copy flow through the webview protocol', async () => {
  const provider = registered.editors[0].provider;
  const f = fakePanel();
  const uri = vscodeMock.Uri.file(path.join(FIX, 'tiny-llama-q8.gguf'));
  await provider.resolveCustomEditor(provider.openCustomDocument(uri), f.panel);
  assert.match(f.panel.webview.html, /Content-Security-Policy/);
  assert.strictEqual(f.panel.webview.options.enableScripts, true);
  f.send({ type: 'ready' });
  const model = await f.until(m => m.type === 'model');
  assert.strictEqual(model.format, 'gguf');
  assert.strictEqual(model.tensors.length, 39);
  assert.ok(model.info.some(x => x.label === 'Architecture' && x.value === 'llama'));
  const i = model.tensors.findIndex(t => t.name === 'blk.1.ffn_gate.weight');
  f.send({ type: 'select', index: i });
  const s = await f.until(m => m.type === 'sample' && m.index === i);
  assert.strictEqual(s.result.view.kind, 'heatmap');
  assert.ok(s.result.stats.std > 0);
  f.send({ type: 'copy', index: i, what: 'shape' });
  await f.until(m => m.type === 'copied');
  assert.strictEqual(clipboard, '(704, 256)');
  const before = f.sent.length;
  f.send({ type: 'select', index: 9999 });
  f.send({ type: 'copy', index: 0, what: 'rm -rf' });
  f.send({ type: 'nope' });
  await new Promise(r => setTimeout(r, 50));
  assert.strictEqual(f.sent.length, before, 'invalid messages get no reply');
  f.dispose();
});

test('a broken file shows an error message in the webview', async () => {
  const provider = registered.editors[0].provider;
  const f = fakePanel();
  await provider.resolveCustomEditor(provider.openCustomDocument(vscodeMock.Uri.file(path.join(FIX, 'broken_header.safetensors'))), f.panel);
  f.send({ type: 'ready' });
  const e = await f.until(m => m.type === 'error');
  assert.match(e.message, /100 MB limit/);
  f.dispose();
});

test('an unreadable tensor returns a sampleError, not a crash', async () => {
  const provider = registered.editors[0].provider;
  const f = fakePanel();
  await provider.resolveCustomEditor(provider.openCustomDocument(vscodeMock.Uri.file(path.join(FIX, 'with_object.npz'))), f.panel);
  f.send({ type: 'ready' });
  const model = await f.until(m => m.type === 'model');
  const bad = model.tensors.findIndex(t => t.name === 'bad');
  f.send({ type: 'select', index: bad });
  const e = await f.until(m => m.type === 'sampleError');
  assert.match(e.message, /pickle/);
  f.dispose();
});

// ---------- support links (Sponsor, More free viewers) ----------
test('support: package.json has the sponsor field, the two commands and the setting', () => {
  const pkg = require('../package.json');
  const support = require('../src/support');
  assert.strictEqual(pkg.sponsor.url, 'https://github.com/sponsors/theluckystrike');
  assert.strictEqual(pkg.sponsor.url, support.sponsorUrl());
  const ids = pkg.contributes.commands.map(c => c.command);
  assert.ok(ids.includes('tensorViewer.supportDeveloper') && ids.includes('tensorViewer.moreViewers'));
  const cfg = Array.isArray(pkg.contributes.configuration) ? pkg.contributes.configuration : [pkg.contributes.configuration];
  const props = Object.assign({}, ...cfg.map(c => c.properties));
  assert.strictEqual(props['tensorViewer.showSupportLinks'].default, true);
  assert.ok(support.otherProducts('zovo.tensor-viewer').every(p => p.id !== 'zovo.tensor-viewer'));
});

test('support: the footer shows by default and tensorViewer.showSupportLinks=false hides it', async () => {
  const provider = registered.editors[0].provider;
  const orig = vscodeMock.workspace.getConfiguration;
  try {
    for (const [settings, shown] of [[{}, true], [{ showSupportLinks: false }, false]]) {
      vscodeMock.workspace.getConfiguration = () => ({ get: (k, d) => (k in settings ? settings[k] : d) });
      const f = fakePanel();
      await provider.resolveCustomEditor(provider.openCustomDocument(vscodeMock.Uri.file(path.join(FIX, 'tiny-llama-q8.gguf'))), f.panel);
      assert.strictEqual(/id="zovo-support"/.test(f.panel.webview.html), shown, JSON.stringify(settings));
      assert.strictEqual(f.panel.webview.html.includes('command:tensorViewer.supportDeveloper'), shown);
      assert.deepStrictEqual(f.panel.webview.options.enableCommandUris, ['tensorViewer.supportDeveloper', 'tensorViewer.moreViewers', 'workbench.action.openSettings']);
      f.dispose();
    }
  } finally { vscodeMock.workspace.getConfiguration = orig; }
});

// ---------- values table, copy values, save as .npy / CSV ----------
const os = require('os');
const fsx = require('fs');
const { BufferSource } = require('../src/core/source');
const { openTensorFile } = require('../src/core/formats');

test('values: a page of a GGUF tensor, then Copy values puts tab-separated text on the clipboard', async () => {
  const provider = registered.editors[0].provider;
  const f = fakePanel();
  await provider.resolveCustomEditor(provider.openCustomDocument(vscodeMock.Uri.file(path.join(FIX, 'tiny-llama-q8.gguf'))), f.panel);
  f.send({ type: 'ready' });
  const model = await f.until(m => m.type === 'model');
  const i = model.tensors.findIndex(t => t.name === 'blk.0.attn_q.weight');
  f.send({ type: 'values', index: i, slice: [], row: 3, col: 30, rows: 4, cols: 6 });
  const v = await f.until(m => m.type === 'values');
  assert.deepStrictEqual([v.row, v.col, v.rows, v.cols, v.totalRows, v.totalCols, v.kind], [3, 30, 4, 6, 256, 256, 'Q8_0']);
  const b = Buffer.from(v.re, 'base64');
  const re = new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
  assert.strictEqual(re.length, 24);
  f.send({ type: 'copyValues', index: i, slice: [], row: 3, col: 30, rows: 4, cols: 6 });
  await f.until(m => m.type === 'copied' && m.what === 'values');
  const lines = clipboard.split('\n');
  assert.strictEqual(lines.length, 4);
  assert.deepStrictEqual(lines[0].split('\t').map(x => Math.fround(Number(x))), Array.from(re.subarray(0, 6)));
  // A page outside the slice rules gets no reply.
  const before = f.sent.length;
  f.send({ type: 'values', index: i, slice: [0], row: 0, col: 0, rows: 1, cols: 1 });
  await new Promise(r => setTimeout(r, 50));
  assert.strictEqual(f.sent.length, before);
  f.dispose();
});

test('save: .npy and CSV go to the picked file, never over the source, and a cancel leaves no file', async () => {
  const provider = registered.editors[0].provider;
  const dir = fsx.mkdtempSync(path.join(os.tmpdir(), 'tensor-viewer-test-'));
  const shown = [];
  let pick = null, cancelNow = false;
  Object.assign(vscodeMock.window, {
    showSaveDialog: async o => { shown.push(o); return pick; },
    withProgress: async (_o, task) => task({ report() {} }, { get isCancellationRequested() { return cancelNow; } }),
    showErrorMessage: async t => { shown.push({ error: t }); },
    showInformationMessage: async t => { shown.push({ info: t }); },
  });
  vscodeMock.ProgressLocation = { Notification: 15 };
  try {
    const src = path.join(FIX, 'tiny-llama-q8.gguf');
    const f = fakePanel();
    await provider.resolveCustomEditor(provider.openCustomDocument(vscodeMock.Uri.file(src)), f.panel);
    f.send({ type: 'ready' });
    const model = await f.until(m => m.type === 'model');
    const i = model.tensors.findIndex(t => t.name === 'blk.1.ffn_gate.weight');
    const next = () => { const n = f.sent.length; return f.until(m => m.type === 'exported' && f.sent.indexOf(m) >= n); };

    pick = vscodeMock.Uri.file(path.join(dir, 'gate.npy'));
    let done = next();
    f.send({ type: 'export', index: i, format: 'npy' });
    let r = await done;
    assert.strictEqual(r.ok, true, r.text);
    assert.match(r.text, /Saved gate.npy .*Q4_0 has no NumPy type, so the values are saved as float32/);
    assert.match(shown[0].defaultUri.fsPath, /blk.1.ffn_gate.weight\.npy$/);
    assert.deepStrictEqual(shown[0].filters, { 'NumPy array': ['npy'] });
    const back = await openTensorFile(new BufferSource(fsx.readFileSync(pick.fsPath)), 'gate.npy');
    assert.deepStrictEqual([back.tensors[0].dtype, back.tensors[0].shape], ['float32', [704, 256]]);

    pick = vscodeMock.Uri.file(path.join(dir, 'gate.csv'));
    done = next();
    f.send({ type: 'export', index: i, format: 'csv' });
    r = await done;
    assert.strictEqual(r.ok, true, r.text);
    assert.strictEqual(fsx.readFileSync(pick.fsPath, 'utf8').trim().split('\n').length, 704);

    pick = vscodeMock.Uri.file(src); // the file being shown
    const size = fsx.statSync(src).size;
    done = next();
    f.send({ type: 'export', index: i, format: 'npy' });
    r = await done;
    assert.strictEqual(r.ok, false);
    assert.match(r.text, /never writes over the file it is showing/);
    assert.strictEqual(fsx.statSync(src).size, size);

    pick = null; // dialog closed
    done = next();
    f.send({ type: 'export', index: i, format: 'csv' });
    assert.strictEqual((await done).text, 'Not saved.');

    pick = vscodeMock.Uri.file(path.join(dir, 'cancel.npy'));
    cancelNow = true;
    done = next();
    f.send({ type: 'export', index: i, format: 'npy' });
    r = await done;
    assert.match(r.text, /cancelled. Nothing was written/);
    assert.deepStrictEqual(fsx.readdirSync(dir).sort(), ['gate.csv', 'gate.npy'], 'no partial file is left');
    f.dispose();
  } finally {
    fsx.rmSync(dir, { recursive: true, force: true });
  }
});

test('the open time is measured once, so a tab that comes back does not show a bigger number', async () => {
  const provider = registered.editors[0].provider;
  const f = fakePanel();
  await provider.resolveCustomEditor(provider.openCustomDocument(vscodeMock.Uri.file(path.join(FIX, 'arange_2x3_f4.npy'))), f.panel);
  f.send({ type: 'ready' });
  const a = await f.until(m => m.type === 'model');
  await new Promise(r => setTimeout(r, 60));
  const n = f.sent.length;
  f.send({ type: 'ready' }); // a hidden webview is rebuilt and asks again
  const b = await f.until(m => m.type === 'model' && f.sent.indexOf(m) >= n);
  assert.strictEqual(b.openMs, a.openMs);
  f.dispose();
});

test('auto-reload: a rewritten .pt file is read again and a half-written one is retried', async () => {
  const fs = require('fs');
  const os = require('os');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tv-reload-'));
  const file = path.join(dir, 'run.pt');
  const torchDir = path.join(FIX, 'torch');
  try {
    fs.copyFileSync(path.join(torchDir, 'state_dict.pt'), file);
    const provider = registered.editors[0].provider;
    const f = fakePanel();
    await provider.resolveCustomEditor(provider.openCustomDocument(vscodeMock.Uri.file(file)), f.panel);
    f.send({ type: 'ready' });
    const first = await f.until(m => m.type === 'model');
    assert.strictEqual(first.format, 'pytorch');
    assert.strictEqual(first.tensors.length, 9);
    assert.strictEqual(first.reloaded, false);
    const w = watchers[watchers.length - 1];
    assert.strictEqual(w.pattern.pattern, 'run.pt');
    // A writer that is half done: the zip directory is missing. The viewer must not show an error, it retries.
    fs.writeFileSync(file, fs.readFileSync(path.join(torchDir, 'truncated.pt')));
    w.change.forEach(fn => fn());
    await new Promise(r => setTimeout(r, 900));
    fs.copyFileSync(path.join(torchDir, 'checkpoint.pth'), file);
    w.change.forEach(fn => fn());
    const second = await f.until(m => m.type === 'model' && m.reloaded);
    assert.strictEqual(second.tensors.length, 27);
    assert.ok(second.info.some(x => x.label === 'epoch' && x.value === '7'));
    assert.ok(!f.sent.some(m => m.type === 'error'), 'no error while the file was half written');
    const i = second.tensors.findIndex(t => t.name === 'model_state_dict.fc.weight');
    f.send({ type: 'select', index: i });
    const s = await f.until(m => m.type === 'sample' && m.index === i);
    assert.strictEqual(s.result.stats.count, 2880);
    f.dispose();
    assert.strictEqual(w.disposed, true);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a plain-text Python .pth path file reopens in the text editor; PyTorch and binary .pth files do not', async () => {
  const fs = require('fs'); const os = require('os');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tens-pth-'));
  const provider = registered.editors[0].provider;
  const run = async (file, wait) => {
    executed.length = 0;
    const f = fakePanel();
    await provider.resolveCustomEditor(provider.openCustomDocument(vscodeMock.Uri.file(file)), f.panel);
    f.send({ type: 'ready' });
    const m = await f.until(x => x.type === wait);
    await new Promise(r => setTimeout(r, 200));
    const out = { m, opened: executed.filter(a => a[0] === 'vscode.openWith'), disposed: !!f.panel.disposedByExt };
    if (!out.disposed) f.dispose();
    return out;
  };
  try {
    fs.writeFileSync(path.join(dir, 'distutils-precedence.pth'), "import os; var = 'SETUPTOOLS_USE_DISTUTILS'\n");
    fs.writeFileSync(path.join(dir, 'empty.pth'), '');
    fs.writeFileSync(path.join(dir, 'junk.pth'), Buffer.from([0x13, 0x00, 0xff, 0x00, 0x42]));
    let r = await run(path.join(dir, 'distutils-precedence.pth'), 'error');
    assert.strictEqual(r.opened.length, 1); assert.strictEqual(r.opened[0][2], 'default'); assert.ok(r.disposed);
    r = await run(path.join(dir, 'empty.pth'), 'error');
    assert.strictEqual(r.opened.length, 1);
    r = await run(path.join(dir, 'junk.pth'), 'error');
    assert.strictEqual(r.opened.length, 0); assert.ok(!r.disposed);
    r = await run(path.join(FIX, 'torch', 'legacy.pth'), 'model');
    assert.strictEqual(r.opened.length, 0); assert.ok(!r.disposed);
    r = await run(path.join(FIX, 'torch', 'truncated.pt'), 'error');
    assert.strictEqual(r.opened.length, 0, '.pt files never fall back');
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});
