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
const vscodeMock = {
  Uri: {
    joinPath: (u, ...p) => ({ ...u, path: path.join(u.path, ...p), fsPath: path.join(u.fsPath || u.path, ...p) }),
    file: p => ({ scheme: 'file', path: p, fsPath: p }),
  },
  window: { registerCustomEditorProvider: (id, provider) => { registered.editors.push({ id, provider }); return { dispose() {} }; } },
  commands: { registerCommand: (id, fn) => { registered.commands[id] = fn; return { dispose() {} }; }, executeCommand: async () => {} },
  workspace: { getConfiguration: () => ({ get: (_k, d) => d }), fs: {} },
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
