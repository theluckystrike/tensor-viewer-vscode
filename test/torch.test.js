'use strict';
// PyTorch files made by torch 2.8 (scripts/make-torch-fixtures.py). expected.json is torch's own view
// of each tensor: shape, dtype, the sum of all values and the first four values in row-major order.
const test = require('node:test');
const assert = require('node:assert');
const path = require('path');
const fs = require('fs');
const { open, close } = require('./helpers');
const { sampleTensor } = require('../src/core/sample');
const { readValues } = require('../src/core/values');
const { formatFromName } = require('../src/core/formats');
const { allowed } = require('../src/core/torch');

const DIR = path.join(__dirname, 'fixtures', 'torch');
const EXP = JSON.parse(fs.readFileSync(path.join(DIR, 'expected.json'), 'utf8'));
const LABEL = { float8_e4m3fn: 'float8_e4m3' };
const openT = name => open(path.join('torch', name));

async function allValues(t) {
  if (!t.shape.length) return Array.from((await readValues(t, { slice: [], row: 0, col: 0, rows: 1, cols: 1 })).re);
  const r = await sampleTensor(t, { sampleValues: 1e7 });
  return r;
}

for (const file of Object.keys(EXP)) {
  test(`${file}: every tensor matches torch (name, shape, dtype, sum, first values)`, async () => {
    const m = await openT(file);
    try {
      assert.strictEqual(m.format, 'pytorch');
      const want = EXP[file];
      const got = new Map(m.tensors.map(t => [t.name, t]));
      assert.deepStrictEqual([...got.keys()].sort(), Object.keys(want).sort(), 'tensor names');
      for (const [name, w] of Object.entries(want)) {
        const t = got.get(name);
        assert.deepStrictEqual(t.shape, w.shape, `${name} shape`);
        if (w.sum === undefined) { // sparse and quantized: listed, the shape is enough
          continue;
        }
        assert.strictEqual(t.dtype, LABEL[w.dtype] || w.dtype, `${name} dtype`);
        const numel = w.shape.reduce((a, b) => a * b, 1);
        if (!numel) { assert.strictEqual(t.readable, false); continue; }
        assert.ok(t.readable, `${name} readable`);
        const r = await allValues(t);
        if (Array.isArray(r)) { assert.ok(close(r[0], w.sum, 1e-5), `${name} scalar ${r[0]} vs ${w.sum}`); continue; }
        assert.ok(r.stats.exact !== false, `${name} exact`);
        const sum = r.stats.mean * r.stats.count;
        assert.ok(close(sum, w.sum, 1e-4) || Math.abs(sum - w.sum) < 1e-3, `${name} sum ${sum} vs ${w.sum}`);
        const page = await readValues(t, { slice: t.shape.length >= 3 ? t.shape.slice(0, -2).map(() => 0) : [], row: 0, col: 0, rows: 1, cols: Math.min(4, t.shape[t.shape.length - 1]) });
        const first = page.im ? Array.from(page.re, (re, i) => Math.hypot(re, page.im[i])) : Array.from(page.re);
        const wantFirst = w.first.slice(0, first.length);
        first.forEach((v, i) => assert.ok(close(v, wantFirst[i], 1e-6), `${name} value ${i}: ${v} vs ${wantFirst[i]}`));
      }
    } finally { m.close(); }
  });
}

test('checkpoints surface epoch, step and loss, and plain values go to Metadata', async () => {
  const m = await openT('lightning.ckpt');
  try {
    const info = Object.fromEntries(m.info.map(x => [x.label, x.value]));
    assert.strictEqual(info.epoch, '3');
    assert.strictEqual(info.global_step, '1200');
    assert.strictEqual(info['weights_only=True'], 'should load');
    const md = Object.fromEntries(m.metadata.map(x => [x.key, x.value]));
    assert.strictEqual(md['hyper_parameters.lr'], '0.001');
    assert.strictEqual(md['hyper_parameters.layers'], '[1, 2, 3]');
  } finally { m.close(); }
  const c = await openT('checkpoint.pth');
  try {
    const info = Object.fromEntries(c.info.map(x => [x.label, x.value]));
    assert.strictEqual(info.loss, '0.4321');
    assert.ok(c.tensors.some(t => t.name === 'optimizer_state_dict.state.0.exp_avg'));
  } finally { c.close(); }
});

test('formats: legacy, TorchScript and whole-module saves are named', async () => {
  const fmt = async f => { const m = await openT(f); m.close(); return m; };
  assert.strictEqual((await fmt('legacy.pth')).info[0].value, 'PyTorch legacy (before 1.6)');
  assert.strictEqual((await fmt('scripted.pt')).info[0].value, 'TorchScript (torch.jit.save)');
  const whole = await fmt('whole_module.pt');
  assert.match(whole.warnings.join(' '), /whole model object was saved.*__main__\.Net/);
  assert.ok(whole.tensors.filter(t => /^nn\.Parameter/.test(t.note || '')).length === 6);
});

test('strided views read the right values (transpose, column slice, shared storage)', async () => {
  const m = await openT('views.pt');
  try {
    const t = m.tensors.find(x => x.name === 'transposed');
    const page = await readValues(t, { slice: [], row: 0, col: 0, rows: 2, cols: 4 });
    assert.deepStrictEqual(Array.from(page.re), [0, 6, 12, 18, 1, 7, 13, 19]);
    const shared = m.tensors.find(x => x.name === 'shared_flat');
    assert.strictEqual(shared.offset, m.tensors.find(x => x.name === 'base').offset + 5 * 4);
  } finally { m.close(); }
});

test('a hostile pickle is never run: os.system is listed as risky and nothing is executed', async () => {
  const marker = '/tmp/tens2-pwned';
  try { fs.unlinkSync(marker); } catch { /* not there */ }
  const m = await openT('evil.pt');
  try {
    assert.strictEqual(m.tensors.length, 0);
    assert.match(m.warnings[0], /^Risky: this file refers to (os|posix)\.system/);
    assert.ok(m.metadata.some(x => /^python name: (os|posix)\.system$/.test(x.key) && /^RISKY/.test(x.value)));
    assert.ok(!m.info.some(x => x.label === 'weights_only=True'));
    assert.strictEqual(fs.existsSync(marker), false);
  } finally { m.close(); }
});

test('quantized and sparse tensors are listed with a note', async () => {
  const m = await openT('quant_sparse.pt');
  try {
    const q = m.tensors.find(t => t.name === 'q');
    assert.ok(q.readable);
    assert.match(q.note, /stored integers.*\(q - 0\) x 0\.1/);
    const s = m.tensors.find(t => t.name === 'sparse');
    assert.strictEqual(s.readable, false);
    assert.match(s.note, /Sparse/);
  } finally { m.close(); }
});

test('a truncated file gives a clear message', async () => {
  await assert.rejects(openT('truncated.pt'), /zip directory at the end is missing.*incomplete/);
});

test('PyTorch names and the allow-list', () => {
  for (const n of ['m.pt', 'm.PTH', 'last.ckpt', 'pytorch_model.bin', 'pytorch_model-00001-of-00002.bin', 'dir/optimizer.bin']) assert.strictEqual(formatFromName(n), 'pytorch', n);
  for (const n of ['x.bin', 'x.pkl', 'x.ptx', 'model.onnx']) assert.strictEqual(formatFromName(n), null, n);
  for (const n of ['torch._utils._rebuild_tensor_v2', 'torch.FloatStorage', 'collections.OrderedDict', 'torch.float8_e4m3fn', 'torch.Size']) assert.ok(allowed(n), n);
  for (const n of ['os.system', 'posix.system', 'builtins.eval', 'builtins.getattr', '__main__.Net', 'torch.load', 'subprocess.Popen']) assert.ok(!allowed(n), n);
});

// VER7 red team (2026-11-08): files that share one value many times (x1 = (x0, x0), x2 = (x1, x1), ... 60 levels)
// or contain themselves must open quickly. Before the fix, three of them kept the extension host busy for minutes.
for (const file of ['11-cyclic-list.pt', '11b-cyclic-dict.pt', '11c-dag-2pow60-tuples.pt', '11d-legacy-dag-2pow60.pth', '11e-dag-as-dict-key.pt', '11g-legacy-dag-lists.pth']) {
  test(`hostile shared or cyclic pickle opens in under 2 s: ${file}`, async () => {
    const t0 = Date.now();
    const m = await openT(path.join('hostile', file));
    try {
      assert.ok(Date.now() - t0 < 2000, `took ${Date.now() - t0} ms`);
      assert.strictEqual(m.format, 'pytorch');
      assert.strictEqual(m.tensors.length, 0);
    } finally { m.close(); }
  });
}

test('payload names that run commands, compile code or open files are rated risky', () => {
  const { rate } = require('../src/core/pickle/safety');
  for (const q of ['_posixsubprocess.fork_exec', 'pydoc.pipepager', 'torch.utils.cpp_extension.load_inline', 'io.open', '_io.FileIO',
    'codecs.open', 'pkgutil.resolve_name', 'torch.jit.load', 'numpy.f2py.compile', 'linecache.getline', 'torch.package.PackageImporter']) {
    assert.strictEqual(rate(q).level, 'danger', q);
  }
  // Names in real torch.save and TorchScript files stay allowed or data.
  for (const q of ['torch._utils._rebuild_tensor_v2', 'collections.OrderedDict', 'torch.FloatStorage']) assert.notStrictEqual(rate(q).level, 'danger', q);
  assert.ok(allowed('torch.jit._pickle.build_intlist'));
});
