'use strict';
// Zovo Pro tools: compare two tensor files, scan all tensors, report tables and panel, free tries, package.json.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { FIX, open } = require('./helpers');
const { FileSource } = require('../src/core/source');
const { openTensorFile } = require('../src/core/formats');
const { compareModels, matchTensors, bareName, accumulate, emptyAcc, finish } = require('../src/pro/compare');
const { scanModel } = require('../src/pro/scan');
const { streamInfo, readAt, indexText } = require('../src/pro/stream');
const { describe } = require('../src/pro/describe');
const { toCsv, toMarkdown, toJson, cellText, render } = require('../src/pro/table');
const { panelHtml } = require('../src/pro/panel');
const { validateWebviewMessage } = require('../src/core/protocol');

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tv-pro-'));
test.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

/** Minimal .npy writer (v1.0) for float64/float32 data in C or Fortran order. */
function writeNpy(file, shape, data, { descr = '<f8', fortran = false } = {}) {
  const shapeTxt = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;
  let header = `{'descr': '${descr}', 'fortran_order': ${fortran ? 'True' : 'False'}, 'shape': ${shapeTxt}, }`;
  const total = 10 + header.length + 1;
  header += ' '.repeat((64 - (total % 64)) % 64) + '\n';
  const head = Buffer.alloc(10);
  head.write('\x93NUMPY', 0, 'latin1'); head[6] = 1; head[7] = 0; head.writeUInt16LE(header.length, 8);
  const body = descr === '<f4' ? Buffer.from(new Float32Array(data).buffer) : Buffer.from(new Float64Array(data).buffer);
  fs.writeFileSync(file, Buffer.concat([head, Buffer.from(header, 'latin1'), body]));
  return file;
}

/** Minimal .safetensors writer (F32). tensors: { name: { shape, data } } */
function writeSafetensors(file, tensors) {
  const header = {}; const bufs = []; let off = 0;
  for (const [name, t] of Object.entries(tensors)) {
    const b = Buffer.from(new Float32Array(t.data).buffer);
    header[name] = { dtype: 'F32', shape: t.shape, data_offsets: [off, off + b.length] };
    bufs.push(b); off += b.length;
  }
  let json = JSON.stringify(header);
  json += ' '.repeat((8 - (json.length % 8)) % 8);
  const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(json.length));
  fs.writeFileSync(file, Buffer.concat([len, Buffer.from(json), ...bufs]));
  return file;
}

async function openPath(p) {
  const src = await FileSource.open(p);
  const model = await openTensorFile(src, path.basename(p));
  model.close = () => src.close();
  return model;
}

test('names match exactly first, then without a wrapper prefix when unique', () => {
  assert.strictEqual(bareName('module._orig_mod.layer.weight'), 'layer.weight');
  const t = n => ({ name: n });
  const pairs = matchTensors([t('a'), t('module.b'), t('c')], [t('a'), t('b'), t('d')]);
  const key = p => `${p.a ? p.a.name : '-'}|${p.b ? p.b.name : '-'}|${p.how}`;
  assert.deepStrictEqual(pairs.map(key).sort(), ['-|d|none', 'a|a|name', 'c|-|none', 'module.b|b|prefix']);
  // Two A tensors that strip to the same name stay unmatched.
  const amb = matchTensors([t('module.x'), t('model.x')], [t('x')]);
  assert.ok(amb.every(p => p.how === 'none'));
});

test('difference metrics match a direct computation', () => {
  const x = [1, 2, 3, 4, NaN, NaN, Infinity], y = [1, 2.5, 2, 4, NaN, 0, Infinity];
  const acc = emptyAcc();
  accumulate(acc, Float64Array.from(x), Float64Array.from(y), 0);
  const r = finish(acc);
  assert.strictEqual(r.maxAbs, 1);
  assert.strictEqual(acc.maxAt, 2);
  assert.strictEqual(r.differing, 3); // 2 values + 1 NaN mismatch
  assert.strictEqual(r.nanMismatch, 1);
  assert.ok(Math.abs(r.rmse - Math.sqrt((0.25 + 1) / 4)) < 1e-12);
  const dot = 1 + 5 + 6 + 16, na = 1 + 4 + 9 + 16, nb = 1 + 6.25 + 4 + 16;
  assert.ok(Math.abs(r.cosine - dot / Math.sqrt(na * nb)) < 1e-12);
  assert.ok(Math.abs(r.relL2 - Math.sqrt(1.25) / Math.sqrt(na)) < 1e-12);
});

test('a file compared with itself is identical in every tensor', async () => {
  const a = await open('gpt2_tiny.safetensors'), b = await open('gpt2_tiny.safetensors');
  try {
    const r = await compareModels(a, b, 'A', 'B');
    assert.strictEqual(r.summary.changed, 0);
    assert.strictEqual(r.summary.identical, a.tensors.length);
    assert.ok(r.rows.every(x => x.maxAbs === 0 && x.cosine > 0.999999));
  } finally { a.close(); b.close(); }
});

test('compare finds a changed value, its index, extra tensors and shape changes, across formats', async () => {
  const w = Array.from({ length: 12 }, (_, i) => i / 10);
  const w2 = w.slice(); w2[7] += 0.5;
  const fa = writeSafetensors(path.join(tmp, 'a.safetensors'), { 'layer.weight': { shape: [3, 4], data: w }, bias: { shape: [4], data: [1, 2, 3, 4] }, gone: { shape: [2], data: [0, 0] }, resized: { shape: [6], data: [1, 2, 3, 4, 5, 6] } });
  const fb = writeSafetensors(path.join(tmp, 'b.safetensors'), { 'module.layer.weight': { shape: [3, 4], data: w2 }, bias: { shape: [4], data: [1, 2, 3, 4] }, added: { shape: [1], data: [9] }, resized: { shape: [7], data: [1, 2, 3, 4, 5, 6, 7] } });
  const a = await openPath(fa), b = await openPath(fb);
  try {
    const r = await compareModels(a, b, 'a.safetensors', 'b.safetensors');
    const row = n => r.rows.find(x => x.name === n);
    assert.strictEqual(row('layer.weight').status, 'changed');
    assert.ok(Math.abs(row('layer.weight').maxAbs - 0.5) < 1e-6);
    assert.strictEqual(row('layer.weight').maxAt, '[1, 3]');
    assert.strictEqual(row('layer.weight').differing, 1);
    assert.match(row('layer.weight').note, /without the prefix: module\.layer\.weight/);
    assert.strictEqual(row('bias').status, 'identical');
    assert.strictEqual(row('gone').status, 'only in A');
    assert.strictEqual(row('added').status, 'only in B');
    assert.strictEqual(row('resized').status, 'shape differs');
    assert.strictEqual(r.rows[0].name, 'layer.weight', 'changed tensors come first');
    // Cross-format: the safetensors file against .npy files of the same values.
    const npy = await openPath(writeNpy(path.join(tmp, 'w.npy'), [3, 4], w2, { descr: '<f4' }));
    const r2 = await compareModels(b, npy, 'b', 'w.npy');
    assert.strictEqual(r2.summary.onlyA, 4); // 'array' does not match any name
    assert.strictEqual(r2.summary.onlyB, 1);
    npy.close();
  } finally { a.close(); b.close(); }
});

test('Fortran-order arrays compare in logical order', async () => {
  const c = [1, 2, 3, 4, 5, 6]; // 2 x 3 C order: [[1,2,3],[4,5,6]]
  const f = [1, 4, 2, 5, 3, 6]; // the same array in Fortran order
  const fc = writeNpy(path.join(tmp, 'c.npy'), [2, 3], c);
  const ff = writeNpy(path.join(tmp, 'f.npy'), [2, 3], f, { fortran: true });
  const f2 = f.slice(); f2[3] = 50; // logical [1, 1]
  const ff2 = writeNpy(path.join(tmp, 'f2.npy'), [2, 3], f2, { fortran: true });
  const a = await openPath(fc), b = await openPath(ff), b2 = await openPath(ff2);
  try {
    assert.strictEqual((await compareModels(a, b, 'c', 'f')).rows[0].status, 'identical');
    const r = await compareModels(a, b2, 'c', 'f2');
    assert.strictEqual(r.rows[0].status, 'changed');
    assert.strictEqual(r.rows[0].maxAt, '[1, 1]');
    const r3 = await compareModels(b, b2, 'f', 'f2'); // both Fortran: memory order, index reported in logical order
    assert.strictEqual(r3.rows[0].maxAt, '[1, 1]');
  } finally { a.close(); b.close(); b2.close(); }
  assert.strictEqual(indexText(3, [2, 3], true), '[1, 1]');
  assert.strictEqual(indexText(4, [2, 3], false), '[1, 1]');
});

test('quantized GGUF tensors stream block by block and compare to themselves', async () => {
  const a = await open('quant_types.gguf'), b = await open('quant_types.gguf');
  try {
    const q = a.tensors.find(t => t._loc && t._loc.quant);
    assert.ok(q, 'fixture has a quantized tensor');
    const info = streamInfo(q);
    assert.ok(info.ok);
    const part = await readAt(q, 5, 40); // crosses a block border
    assert.strictEqual(part.length, 40);
    const r = await compareModels(a, b, 'A', 'B');
    assert.strictEqual(r.summary.changed, 0);
    assert.ok(r.summary.identical >= 1);
  } finally { a.close(); b.close(); }
});

test('cancel stops the compare and marks the rest', async () => {
  const a = await open('gpt2_tiny.safetensors'), b = await open('gpt2_tiny.safetensors');
  try {
    let calls = 0;
    const r = await compareModels(a, b, 'A', 'B', { cancelled: () => ++calls > 2 });
    assert.ok(r.stopped);
    assert.ok(r.summary.notCompared > 0);
  } finally { a.close(); b.close(); }
});

test('scan reads every value and flags NaN, Inf, zeros, constants and huge values', async () => {
  const f = writeSafetensors(path.join(tmp, 's.safetensors'), {
    ok: { shape: [4], data: [1, -2, 3, 0.5] }, nan: { shape: [3], data: [1, NaN, 2] }, inf: { shape: [2], data: [Infinity, 1] },
    zero: { shape: [3], data: [0, 0, 0] }, flat: { shape: [3], data: [7, 7, 7] }, big: { shape: [2], data: [1, 2e5] },
  });
  const m = await openPath(f);
  try {
    const r = await scanModel(m, 's.safetensors');
    const row = n => r.rows.find(x => x.name === n);
    assert.strictEqual(row('ok').status, 'ok');
    assert.strictEqual(row('ok').min, -2); assert.strictEqual(row('ok').max, 3);
    assert.ok(Math.abs(row('ok').mean - 0.625) < 1e-12);
    const sd = Math.sqrt([1, -2, 3, 0.5].reduce((s, v) => s + (v - 0.625) ** 2, 0) / 4);
    assert.ok(Math.abs(row('ok').std - sd) < 1e-9);
    assert.strictEqual(row('nan').flags, 'NaN');
    assert.strictEqual(row('inf').flags, 'Inf');
    assert.strictEqual(row('zero').flags, 'all zero');
    assert.strictEqual(row('flat').flags, 'constant');
    assert.strictEqual(row('big').flags, 'huge values');
    assert.deepStrictEqual([r.summary.withNaN, r.summary.withInf, r.summary.allZero, r.summary.constant, r.summary.huge, r.summary.flagged], [1, 1, 1, 1, 1, 5]);
    assert.ok(['nan', 'inf'].includes(r.rows[0].name), 'NaN and Inf come first');
  } finally { m.close(); }
});

test('scan of a large tensor is exact, not sampled', async () => {
  const n = 3 * 1048576 + 17; // more than 3 read runs
  const data = new Float64Array(n);
  for (let i = 0; i < n; i++) data[i] = (i % 1000) - 500;
  data[n - 1] = NaN;
  const m = await openPath(writeNpy(path.join(tmp, 'big.npy'), [n], data));
  try {
    const r = await scanModel(m, 'big.npy');
    assert.strictEqual(r.rows[0].nan, 1);
    assert.strictEqual(r.rows[0].min, -500);
    assert.strictEqual(r.rows[0].max, 499);
    assert.strictEqual(r.summary.values, n);
  } finally { m.close(); fs.rmSync(path.join(tmp, 'big.npy')); }
});

test('reports save as CSV (formula guard), Markdown (pipes escaped) and JSON (NaN as text)', async () => {
  const f = writeSafetensors(path.join(tmp, 'r.safetensors'), { '=cmd|x': { shape: [2], data: [NaN, 1] } });
  const m = await openPath(f);
  try {
    const r = await scanModel(m, 'r.safetensors');
    const d = describe(r);
    const csv = toCsv(d);
    assert.match(csv, /^Tensor,Flags,/);
    assert.match(csv, /\r\n'=cmd\|x,NaN,/);
    const md = toMarkdown(d);
    assert.match(md, /\| =cmd\\\|x \| NaN \|/);
    const json = JSON.parse(toJson(r));
    assert.strictEqual(json.rows[0].name, '=cmd|x');
    assert.ok(!('_mark' in json.rows[0]));
    assert.strictEqual(render('md', d, r), md);
    assert.throws(() => render('exe', d, r));
  } finally { m.close(); }
  assert.strictEqual(cellText(Infinity, 'num'), 'Inf');
  assert.strictEqual(cellText(12345678, 'int'), '12,345,678');
  assert.strictEqual(cellText(0.000012345, 'num'), '1.2345e-5');
});

test('report panel: strict CSP, data as JSON with < escaped, no inline handlers', () => {
  const d = { title: 'Scan: x</script><img src=x onerror=alert(1)>', facts: [['A', '1']], notes: ['n'], columns: [{ key: 'name', label: 'Tensor' }, { key: 'v', label: 'V', type: 'num' }],
    rows: [{ name: '</script><script>alert(1)</script>', v: 1, _mark: 'bad' }] };
  const html = panelHtml(d, 'vscode-resource:', 'NONCE');
  assert.match(html, /default-src 'none'; style-src 'nonce-NONCE'; script-src 'nonce-NONCE'/);
  assert.ok(!html.includes('</script><script>alert'));
  assert.ok(!html.includes('<img src=x'));
  const outsideData = html.replace(/<script id="data" type="application\/json">[^]*?<\/script>/, '');
  assert.ok(!/ on[a-z]+=/i.test(outsideData), 'no inline event handlers outside the JSON data');
  assert.ok(!/<script id="data"[^>]*>[^]*<\/script>[^]*<\/script>[^]*<script id/.test(html));
  assert.strictEqual((html.match(/<script nonce="NONCE">/g) || []).length, 1);
});

test('webview may ask for the Pro tools only by name', () => {
  assert.deepStrictEqual(validateWebviewMessage({ type: 'pro', action: 'compare' }, 1).msg, { type: 'pro', action: 'compare' });
  assert.deepStrictEqual(validateWebviewMessage({ type: 'pro', action: 'scan' }, 1).msg, { type: 'pro', action: 'scan' });
  assert.strictEqual(validateWebviewMessage({ type: 'pro', action: 'unlock' }, 1).ok, false);
});

// ---------- free tries and the unlock modal (mocked vscode) ----------

function mockVscode() {
  const calls = { modal: [], status: [], context: [] };
  const vscode = {
    EventEmitter: class { constructor() { this.l = []; this.event = f => { this.l.push(f); return { dispose() {} }; }; } fire(v) { this.l.forEach(f => f(v)); } dispose() {} },
    window: { setStatusBarMessage: t => calls.status.push(t), showInformationMessage: async (msg, opts) => { calls.modal.push({ msg, opts }); return undefined; } },
    commands: { executeCommand: async (...a) => calls.context.push(a), registerCommand: () => ({ dispose() {} }) },
  };
  const store = new Map();
  const context = { globalState: { get: k => store.get(k), update: async (k, v) => store.set(k, v) }, subscriptions: [] };
  return { vscode, context, calls };
}

test('3 free tries per feature, then one explanation with the ways to unlock; licensed never asks', async () => {
  const { createPro, TRIAL_USES } = require('../src/pro');
  const { vscode, context, calls } = mockVscode();
  let pro = false;
  const asked = [];
  const license = { isPro: () => pro, onDidChange: () => ({ dispose() {} }), ready: Promise.resolve(), refresh: () => Promise.resolve(), handleUri: () => false,
    requirePro: async (title, detail) => { asked.push({ title, detail }); return false; } };
  const p = createPro({ vscode, context, license });
  assert.strictEqual(TRIAL_USES, 3);
  for (let i = 0; i < 3; i++) { assert.strictEqual(await p.allowed('compare'), true); await p.useTrial('compare'); }
  assert.strictEqual(p.trialLeft('compare'), 0);
  assert.strictEqual(p.trialLeft('scan'), 3, 'tries are per feature');
  assert.strictEqual(await p.allowed('compare'), false);
  assert.strictEqual(asked.length, 1);
  assert.match(asked[0].detail, /used the 3 free tries/);
  assert.match(asked[0].detail, /USD 19 once/);
  assert.match(asked[0].detail, /stays free/);
  assert.ok(!/USD 15/.test(asked[0].detail));
  assert.strictEqual(calls.status.length, 3);
  pro = true;
  assert.strictEqual(await p.allowed('compare'), true);
  await p.useTrial('scan');
  assert.strictEqual(p.trialLeft('scan'), 3, 'a licensed run does not use a try');
  assert.strictEqual(asked.length, 1);
});

test('package.json: Pro commands, setting, menus, version, no old price', () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'package.json'), 'utf8'));
  const cmds = pkg.contributes.commands.map(c => c.command);
  for (const c of ['compareFiles', 'scanTensors', 'getPro', 'unlockWithSponsors', 'enterLicenseKey', 'removeLicense']) assert.ok(cmds.includes('tensorViewer.' + c), c);
  assert.strictEqual(pkg.contributes.configuration.properties['tensorViewer.showProFeatures'].default, true);
  for (const m of pkg.contributes.menus.commandPalette) assert.match(m.when, /proLicensed/);
  assert.match(pkg.contributes.commands.find(c => c.command === 'tensorViewer.getPro').title, /USD 19 once/);
  assert.strictEqual(pkg.engines.vscode, '^1.90.0');
  const text = JSON.stringify(pkg) + fs.readFileSync(path.join(__dirname, '..', 'README.md'), 'utf8');
  assert.ok(!/USD 15/.test(text));
  assert.ok(!/frequency=one-time/.test(text));
});

test('shared license module is byte-identical to the factory copy (when the factory is present)', () => {
  const factory = path.join(__dirname, '..', '..', '..', '..', '..', '..', 'factory', 'sprint', 'license');
  const alt = path.join(os.homedir(), 'plugin-market-intel-work', 'factory', 'sprint', 'license');
  const dir = fs.existsSync(factory) ? factory : alt;
  if (!fs.existsSync(dir)) return;
  for (const f of ['index.js', 'license.js', 'sponsors.js']) {
    assert.ok(fs.readFileSync(path.join(__dirname, '..', 'src', 'license', f)).equals(fs.readFileSync(path.join(dir, f))), f);
  }
});

test('fixture folder is untouched by the Pro tests', () => {
  assert.ok(fs.readdirSync(FIX).every(n => !/^(a|b|s|r|c|f|f2|w)\.(npy|safetensors)$/.test(n)));
});
