'use strict';
// Exact median and percentiles (p1, p5, p25, median, p75, p95, p99) against numpy and exact integer math.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { FileSource, BufferSource } = require('../src/core/source');
const { openTensorFile } = require('../src/core/formats');
const { sampleTensor, exactPercentiles, statsText, percentileNote } = require('../src/core/sample');
const Q = require('../src/core/quantiles');
const { validateWebviewMessage } = require('../src/core/protocol');
const { close } = require('./helpers');

const DIR = path.join(__dirname, 'fixtures', 'objects');
const expected = JSON.parse(fs.readFileSync(path.join(DIR, 'expected.json'), 'utf8'));
async function open(dir, name) {
  const src = await FileSource.open(path.join(dir, name));
  const model = await openTensorFile(src, name);
  model.close = () => src.close();
  return model;
}
const texts = p => p.values.map(v => v.text);
const nums = p => p.values.map(v => v.value);

test('float32 with NaN and Inf: exact, NaN/Inf left out, same as numpy.percentile', async () => {
  const m = await open(DIR, 'normal_700x300_f4.npy');
  const r = await sampleTensor(m.tensors[0]);
  assert.strictEqual(r.exact, true);
  assert.strictEqual(r.pct.exact, true);
  assert.strictEqual(r.pct.n, expected['normal_700x300_f4.npy'].finite);
  nums(r.pct).forEach((v, i) => assert.ok(close(v, expected['normal_700x300_f4.npy'].pcts[i], 1e-12), `${v} vs ${expected['normal_700x300_f4.npy'].pcts[i]}`));
  assert.deepStrictEqual(r.pct.values.map(v => v.label), ['p1', 'p5', 'p25', 'median', 'p75', 'p95', 'p99']);
  await m.close();
});

test('a tensor bigger than the sample: stats sampled, percentiles exact from a second read', async () => {
  const m = await open(DIR, 'normal_700x300_f4.npy');
  const t = m.tensors[0];
  const r = await sampleTensor(t, { sampleValues: 30000 });
  assert.strictEqual(r.exact, false);
  assert.strictEqual(r.pct.pending, true);
  assert.match(percentileNote(r.pct), /reading all 210,000 values/);
  const p = await exactPercentiles(t, { chunkValues: 4096 });
  assert.strictEqual(p.exact, true);
  nums(p).forEach((v, i) => assert.ok(close(v, expected['normal_700x300_f4.npy'].pcts[i], 1e-12)));
  assert.strictEqual(await exactPercentiles(t, { cancelled: () => true }), null);
  await m.close();
});

test('above the exact limit, percentiles come from the sample and say so', async () => {
  const m = await open(DIR, 'normal_700x300_f4.npy');
  const r = await sampleTensor(m.tensors[0], { sampleValues: 30000, exactPercentileMax: 100000 });
  assert.strictEqual(r.pct.exact, false);
  assert.strictEqual(r.pct.sampled, true);
  assert.ok(r.pct.n <= 30000 && r.pct.n > 10000);
  assert.match(percentileNote(r.pct), new RegExp(`sampled from ${r.pct.n.toLocaleString('en-US')} values`));
  const med = r.pct.values[3].value;
  assert.ok(Math.abs(med - expected['normal_700x300_f4.npy'].pcts[3]) < 0.05, String(med));
  await m.close();
});

test('compressed npz entries too big to read whole get sampled percentiles', async () => {
  const m = await open(path.join(__dirname, 'fixtures'), 'bundle_compressed.npz');
  const t = m.tensors.find(x => x.name === 'signal');
  const r = await sampleTensor(t, { sampleValues: 1000 });
  assert.strictEqual(r.pct.sampled, true);
  await m.close();
});

test('int64 and uint64 percentiles are exact past 2^53', async () => {
  for (const f of ['int64_big.npy', 'uint64_big.npy']) {
    const m = await open(DIR, f);
    const r = await sampleTensor(m.tensors[0]);
    assert.strictEqual(r.pct.bigint, true);
    assert.deepStrictEqual(texts(r.pct), expected[f].pcts, f);
    const big = await sampleTensor(m.tensors[0], { sampleValues: 10000 });
    assert.deepStrictEqual(texts(big.pct), expected[f].pcts, f + ' (read again)');
    await m.close();
  }
});

test('ties, a single value and an all-NaN tensor', async () => {
  const m = await open(DIR, 'ties_i2.npy');
  const r = await sampleTensor(m.tensors[0]);
  assert.deepStrictEqual(nums(r.pct), expected['ties_i2.npy'].pcts);
  await m.close();
  const one = await Q.percentiles(Q.memoryScan([{ f: Float64Array.of(42) }]), 'float');
  assert.deepStrictEqual(nums(one), [42, 42, 42, 42, 42, 42, 42]);
  const none = await Q.percentiles(Q.memoryScan([{ f: Float64Array.of(NaN, Infinity, -Infinity) }]), 'float');
  assert.strictEqual(none.n, 0);
  assert.match(percentileNote({ ...none, exact: true }), /no finite values/);
});

test('complex arrays use the magnitude and say so', async () => {
  const m = await open(DIR, 'complex_501_c16.npy');
  const r = await sampleTensor(m.tensors[0]);
  assert.strictEqual(r.pct.complex, true);
  nums(r.pct).forEach((v, i) => assert.ok(close(v, expected['complex_501_c16.npy'].pcts[i], 1e-12)));
  assert.match(percentileNote(r.pct), /magnitude \|z\|/);
  await m.close();
});

test('a small existing fixture: percentiles of 0..5 match numpy', async () => {
  const m = await open(path.join(__dirname, 'fixtures'), 'arange_2x3_f4.npy');
  const r = await sampleTensor(m.tensors[0]);
  [0.05, 0.25, 1.25, 2.5, 3.75, 4.75, 4.95].forEach((x, i) => assert.ok(close(nums(r.pct)[i], x, 1e-12), `${nums(r.pct)[i]} vs ${x}`));
  await m.close();
});

test('radix select matches a full sort on awkward data', async () => {
  const cases = {
    narrow: n => 1 + (n % 7919) * 1e-12,
    negatives: n => -((n * 2654435761) % 1000003),
    zeros: n => (n % 3 ? 0 : -0),
    tiny: n => (n % 2 ? 5e-324 : -5e-324) * (n % 5),
    huge: n => (n % 2 ? 1.7e308 : -1.7e308) / ((n % 11) + 1),
  };
  for (const [name, f] of Object.entries(cases)) {
    const a = new Float64Array(250000);
    for (let i = 0; i < a.length; i++) a[i] = f(i);
    const s = Float64Array.from(a).sort();
    const n = s.length;
    const want = Q.PCTS.map(p => { const num = (n - 1) * p, i = Math.floor(num / 100), r = num % 100; return r ? Q.lerp(s[i], s[i + 1], r / 100) : s[i]; });
    const got = nums(await Q.percentiles(Q.memoryScan([{ f: a.subarray(0, 1000) }, { f: a.subarray(1000) }]), 'float'));
    got.forEach((v, i) => assert.ok(v === want[i] || (v === 0 && want[i] === 0), `${name} p${Q.PCTS[i]}: ${v} vs ${want[i]}`));
  }
});

test('Copy stats text has every percentile and how it was found; the message is checked', async () => {
  const m = await open(DIR, 'int64_big.npy');
  const t = m.tensors[0];
  const r = await sampleTensor(t);
  const text = statsText(t, r, null);
  for (const [i, label] of ['p1', 'p5', 'p25', 'median', 'p75', 'p95', 'p99'].entries()) assert.ok(text.includes(`${label} ${expected['int64_big.npy'].pcts[i]}\n`), label);
  assert.match(text, /exact over all 1,002 finite values/);
  assert.match(text, /^array\ndtype int64, shape \(1002,\)/);
  assert.deepStrictEqual(validateWebviewMessage({ type: 'copyStats', index: 0 }, 1), { ok: true, msg: { type: 'copyStats', index: 0 } });
  assert.strictEqual(validateWebviewMessage({ type: 'copyStats', index: 1 }, 1).ok, false);
  assert.strictEqual(validateWebviewMessage({ type: 'copyStats', index: '0' }, 1).ok, false);
  await m.close();
});

test('GGUF quantized and in-memory tensors stream in whole blocks', async () => {
  const m = await open(path.join(__dirname, 'fixtures'), 'quant_types.gguf');
  for (const t of m.tensors.filter(x => x.readable).slice(0, 4)) {
    const a = await sampleTensor(t);
    const b = await exactPercentiles(t, { chunkValues: 64 });
    assert.deepStrictEqual(nums(b), nums(a.pct), t.name);
  }
  await m.close();
  const src = new BufferSource(new Uint8Array(new Float32Array([3, 1, 2]).buffer));
  const t = { name: 'x', shape: [3], numel: 3, readable: true, _loc: { source: src, dataOffset: 0, kind: 'f32', elemSize: 4, littleEndian: true } };
  const got = nums(await exactPercentiles(t));
  [1.02, 1.1, 1.5, 2, 2.5, 2.9, 2.98].forEach((x, i) => assert.ok(close(got[i], x, 1e-12), `${got[i]} vs ${x}`));
});
