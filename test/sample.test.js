'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { expected, open, close } = require('./helpers');
const { sampleTensor, shapeAsPython } = require('../src/core/sample');
const { FileSource } = require('../src/core/source');
const { openTensorFile } = require('../src/core/formats');

function checkStats(got, want, tol = 1e-5) {
  for (const k of ['nan', 'posinf', 'neginf']) assert.strictEqual(got[k], want[k], k);
  for (const k of ['min', 'max', 'mean', 'std']) assert.ok(close(got[k], want[k], tol), `${k} ${got[k]} != ${want[k]}`);
}

for (const name of ['arange_2x3_f4.npy', 'specials_f2.npy', 'signal_5000_f4.npy', 'int64_2x2.npy', 'image_96x128x3_u1.npy', 'matrix_1024x512_f4.npy']) {
  test(`exact stats match numpy for ${name}`, async () => {
    const m = await open(name);
    const r = await sampleTensor(m.tensors[0]);
    assert.ok(r.exact);
    checkStats(r.stats, expected[name].stats);
    await m.close();
  });
}

test('a big tensor is sampled by whole rows spread over the tensor', async () => {
  const m = await open('matrix_1024x512_f4.npy');
  const r = await sampleTensor(m.tensors[0], { sampleValues: 64 * 512 });
  assert.strictEqual(r.exact, false);
  assert.ok(r.stats.count <= 64 * 512 && r.stats.count >= 32 * 512, String(r.stats.count));
  assert.match(r.how, /rows spread over all 1,024 rows/);
  assert.ok(Math.abs(r.stats.mean - expected['matrix_1024x512_f4.npy'].stats.mean) < 0.05);
  assert.strictEqual(r.view.kind, 'heatmap');
  assert.strictEqual(r.view.srcRows, 1024);
  await m.close();
});

test('views: heatmap for 2-D, line for 1-D, image for HxWx3 uint8', async () => {
  const cases = { 'arange_2x3_f4.npy': 'heatmap', 'signal_5000_f4.npy': 'line', 'image_96x128x3_u1.npy': 'image', 'scalar_f4.npy': null };
  for (const [f, kind] of Object.entries(cases)) {
    const m = await open(f);
    const r = await sampleTensor(m.tensors[0]);
    assert.strictEqual(r.view ? r.view.kind : null, kind, f);
    await m.close();
  }
});

test('heatmap cells hold the right values, also for Fortran order', async () => {
  for (const f of ['arange_2x3_f4.npy', 'fortran_3x4_f8.npy']) {
    const m = await open(f);
    const r = await sampleTensor(m.tensors[0]);
    const b = Buffer.from(r.view.data, 'base64');
    const g = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
    const want = expected[f].values;
    assert.deepStrictEqual([r.view.rows, r.view.cols], expected[f].shape);
    assert.deepStrictEqual(Array.from(g), want);
    await m.close();
  }
});

test('image keeps pixel order and colors', async () => {
  const m = await open('image_96x128x3_u1.npy');
  const r = await sampleTensor(m.tensors[0]);
  const px = Buffer.from(r.view.data, 'base64');
  const e = expected['image_96x128x3_u1.npy'];
  assert.deepStrictEqual([px[0], px[1], px[2], px[3]], [...e.first_pixel, 255]);
  const last = (96 * 128 - 1) * 4;
  assert.deepStrictEqual([px[last], px[last + 1], px[last + 2]], e.last_pixel);
  await m.close();
});

test('compressed npz: stats over a prefix when the budget is small, exact when it fits', async () => {
  const m = await open('bundle_compressed.npz');
  const sig = m.tensors.find(t => t.name === 'signal');
  const full = await sampleTensor(sig);
  assert.ok(full.exact);
  checkStats(full.stats, expected['bundle_compressed.npz'].signal_stats);
  const mat = m.tensors.find(t => t.name === 'matrix');
  const part = await sampleTensor(mat, { sampleValues: 10 * 512 });
  assert.strictEqual(part.exact, false);
  assert.strictEqual(part.stats.count, 10 * 512);
  assert.match(part.how, /first/);
  await m.close();
});

test('a 10 GB safetensors checkpoint opens and samples without reading it whole', async t => {
  const dir = path.join(__dirname, '.tmp-big');
  fs.mkdirSync(dir, { recursive: true });
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'big.safetensors');
  const rows = 40960, cols = 65536; // 2.68e9 float32 values = 10.7 GB
  const header = {};
  header['model.embed.weight'] = { dtype: 'F32', shape: [rows, cols], data_offsets: [0, rows * cols * 4] };
  for (let i = 0; i < 400; i++) header[`layers.${i}.norm.weight`] = { dtype: 'F32', shape: [1], data_offsets: [rows * cols * 4, rows * cols * 4 + 4] };
  let hb = Buffer.from(JSON.stringify(header));
  const pre = Buffer.alloc(8); pre.writeBigUInt64LE(BigInt(hb.length));
  fs.writeFileSync(file, Buffer.concat([pre, hb]));
  fs.truncateSync(file, 8 + hb.length + rows * cols * 4 + 4); // sparse: no real disk use
  const t0 = Date.now();
  const src = await FileSource.open(file);
  const m = await openTensorFile(src, 'big.safetensors');
  const openMs = Date.now() - t0;
  assert.strictEqual(m.tensors.length, 401);
  assert.ok(m.fileSize > 10e9);
  assert.ok(openMs < 1000, `opened in ${openMs} ms`);
  const r = await sampleTensor(m.tensors[0], { sampleValues: 1000000 });
  assert.strictEqual(r.exact, false);
  assert.ok(r.stats.count <= 1000000);
  assert.strictEqual(r.stats.zeros, r.stats.count);
  assert.ok(r.ms < 3000, `sampled in ${r.ms} ms`);
  await src.close();
});

test('shape as Python', () => {
  assert.strictEqual(shapeAsPython([]), '()');
  assert.strictEqual(shapeAsPython([5]), '(5,)');
  assert.strictEqual(shapeAsPython([2, 3]), '(2, 3)');
});
