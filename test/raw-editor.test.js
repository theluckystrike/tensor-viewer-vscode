'use strict';
// The raw-array editor through the extension host, with a mocked vscode module: the four prompts,
// the stored layout, "Change type and shape...", cancel, and the image request.
const test = require('node:test');
const assert = require('node:assert');
const Module = require('module');
const path = require('path');
const fs = require('fs');
const os = require('os');

const answers = [];
const registered = { editors: [], commands: {} };
const vscodeMock = {
  Uri: {
    joinPath: (u, ...p) => ({ ...u, path: path.join(u.path, ...p), fsPath: path.join(u.fsPath || u.path, ...p) }),
    file: p => ({ scheme: 'file', path: p, fsPath: p, toString() { return 'file://' + this.path; } }),
  },
  window: {
    registerCustomEditorProvider: (id, provider) => { registered.editors.push({ id, provider }); return { dispose() {} }; },
    showQuickPick: async (items, opts) => { const a = answers.shift(); assert.ok(a !== undefined, 'unexpected quick pick ' + opts.title); return a === null ? undefined : items.find(i => i.label === a); },
    showInputBox: async opts => {
      const a = answers.shift();
      assert.ok(a !== undefined, 'unexpected input box ' + opts.title);
      if (a === null) return undefined;
      assert.strictEqual(opts.validateInput(a), null, `answer ${a} should validate for ${opts.title}`);
      return a;
    },
    showErrorMessage: () => {},
    tabGroups: { activeTabGroup: { activeTab: null } },
  },
  commands: { registerCommand: (id, fn) => { registered.commands[id] = fn; return { dispose() {} }; }, executeCommand: async () => {} },
  workspace: { getConfiguration: () => ({ get: (_k, d) => d }), fs: { stat: async u => ({ size: fs.statSync(u.fsPath).size }) } },
  env: { clipboard: { writeText: async () => {} } },
};
const origLoad = Module._load;
Module._load = function (req, ...rest) { return req === 'vscode' ? vscodeMock : origLoad.call(this, req, ...rest); };
const ext = require('../src/extension');

const store = {};
const context = {
  subscriptions: [], extensionUri: vscodeMock.Uri.file(path.join(__dirname, '..')), extensionPath: path.join(__dirname, '..'),
  workspaceState: { get: (k, d) => (k in store ? store[k] : d), update: async (k, v) => { store[k] = v; } },
  globalState: { get: (_k, d) => d, update: async () => {}, setKeysForSync() {} },
};
ext.activate(context);
const rawProvider = () => registered.editors.find(e => e.id === ext.RAW_VIEW_TYPE).provider;

function fakePanel() {
  const sent = [], waiters = [];
  let onMsg, onDispose;
  const panel = {
    webview: { cspSource: 'vscode-resource:', options: {}, html: '', asWebviewUri: u => ({ toString: () => 'vscode-resource:' + u.path }),
      postMessage: m => { sent.push(JSON.parse(JSON.stringify(m))); waiters.slice().forEach(w => w()); return Promise.resolve(true); },
      onDidReceiveMessage: fn => { onMsg = fn; } },
    onDidDispose: fn => { onDispose = fn; },
  };
  const until = (pred, from = 0) => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timeout: ' + JSON.stringify(sent.map(m => m.type)))), 10000);
    const check = () => { const m = sent.slice(from).find(pred); if (m) { clearTimeout(timer); resolve(m); } };
    waiters.push(check); check();
  });
  return { panel, sent, until, send: m => onMsg(m), dispose: () => onDispose() };
}

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tv-raw-'));
test.after(() => fs.rmSync(dir, { recursive: true, force: true }));

test('package.json: raw editor is opt-in, the command exists, the Explorer menu is limited to binary extensions', () => {
  const pkg = require('../package.json');
  const raw = pkg.contributes.customEditors.find(e => e.viewType === 'tensorViewer.raw');
  assert.strictEqual(raw.priority, 'option');
  assert.ok(raw.selector.some(s => s.filenamePattern === '*.bin'));
  assert.ok(registered.commands['tensorViewer.openRaw']);
  const menu = pkg.contributes.menus['explorer/context'].find(m => m.command === 'tensorViewer.openRaw');
  assert.match(menu.when, /resourceExtname/);
  assert.match(menu.when, /bin\|raw\|dat/);
});

test('first open asks type, header, shape and byte order, then shows the array and keeps the layout', async () => {
  const file = path.join(dir, 'frames.bin');
  const a = new Float32Array(2 * 4 * 5);
  for (let i = 0; i < a.length; i++) a[i] = i;
  fs.writeFileSync(file, Buffer.from(a.buffer));
  const uri = vscodeMock.Uri.file(file);
  answers.push('float32', '0', '2, 4, 5', 'Little-endian');
  const f = fakePanel();
  await rawProvider().resolveCustomEditor(rawProvider().openCustomDocument(uri), f.panel);
  f.send({ type: 'ready' });
  const model = await f.until(m => m.type === 'model');
  assert.strictEqual(answers.length, 0);
  assert.strictEqual(model.format, 'raw');
  assert.deepStrictEqual(model.tensors[0].shape, [2, 4, 5]);
  assert.strictEqual(model.tensors[0].image.layouts[0].id, 'gray');
  f.send({ type: 'select', index: 0 });
  const s = await f.until(m => m.type === 'sample');
  assert.strictEqual(s.result.stats.max, 39);
  f.send({ type: 'image', index: 0, layout: 'gray', frame: 1, range: 'minmax' });
  const img = await f.until(m => m.type === 'image');
  assert.deepStrictEqual([img.result.lo, img.result.hi], [20, 39]);
  f.dispose();
  // Opening it again uses the stored layout: no prompts.
  const g = fakePanel();
  await rawProvider().resolveCustomEditor(rawProvider().openCustomDocument(uri), g.panel);
  g.send({ type: 'ready' });
  const again = await g.until(m => m.type === 'model');
  assert.deepStrictEqual(again.tensors[0].shape, [2, 4, 5]);
  // "Change type and shape..." asks again and reloads the open editor.
  answers.push('uint8', '0', '', );
  const n = g.sent.length;
  g.send({ type: 'rawLayout' });
  const changed = await g.until(m => m.type === 'model', n);
  assert.deepStrictEqual(changed.tensors[0].shape, [160]);
  assert.strictEqual(changed.tensors[0].dtype, 'uint8');
  g.dispose();
});

test('cancel on first open shows a message with a button, never a blank page', async () => {
  const file = path.join(dir, 'cancel.dat');
  fs.writeFileSync(file, Buffer.alloc(12));
  answers.push(null);
  const f = fakePanel();
  await rawProvider().resolveCustomEditor(rawProvider().openCustomDocument(vscodeMock.Uri.file(file)), f.panel);
  f.send({ type: 'ready' });
  const e = await f.until(m => m.type === 'error');
  assert.strictEqual(e.action, 'rawLayout');
  assert.match(e.message, /Pick the value type and shape/);
  // The button asks again; this time it works.
  answers.push('int32', '0', '-1', 'Big-endian');
  f.send({ type: 'rawLayout' });
  const m = await f.until(x => x.type === 'model');
  assert.deepStrictEqual(m.tensors[0].shape, [3]);
  f.dispose();
});

test('the command stores the layout and opens the raw editor', async () => {
  const file = path.join(dir, 'cmd.raw');
  fs.writeFileSync(file, Buffer.alloc(64 * 64));
  const calls = [];
  vscodeMock.commands.executeCommand = async (...args) => { calls.push(args); };
  answers.push('uint8', '0', '64, 64');
  await registered.commands['tensorViewer.openRaw'](vscodeMock.Uri.file(file));
  assert.strictEqual(calls[0][0], 'vscode.openWith');
  assert.strictEqual(calls[0][2], 'tensorViewer.raw');
  const saved = store['tensorViewer.rawLayouts']['file://' + file];
  assert.deepStrictEqual(saved, { kind: 'u8', shape: [64, 64], offset: 0, littleEndian: true });
});
