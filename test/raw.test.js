'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { BufferSource } = require('../src/core/source');
const { openTensorFile } = require('../src/core/formats');
const { parseShape, resolveRawLayout, shapeHint, cleanLayout, RAW_TYPES } = require('../src/core/raw');
const { sampleTensor } = require('../src/core/sample');
const { readValues, planExport, writeExport } = require('../src/core/values');
const { readNpyHeader } = require('../src/core/npy');
const { readImage } = require('../src/core/image');

const f32 = arr => new Uint8Array(new Float32Array(arr).buffer);

test('parseShape reads the ways people type shapes', () => {
  assert.deepStrictEqual(parseShape('').shape, [-1]);
  assert.deepStrictEqual(parseShape('480, 640, 3').shape, [480, 640, 3]);
  assert.deepStrictEqual(parseShape('(480,640,3)').shape, [480, 640, 3]);
  assert.deepStrictEqual(parseShape('[2, 3]').shape, [2, 3]);
  assert.deepStrictEqual(parseShape('480x640').shape, [480, 640]);
  assert.deepStrictEqual(parseShape('480 640').shape, [480, 640]);
  assert.deepStrictEqual(parseShape('-1, 3').shape, [-1, 3]);
  assert.match(parseShape('-1, -1').error, /at most one/);
  assert.match(parseShape('a, 3').error, /not a whole number/);
  assert.match(parseShape('0, 3').error, /not a valid size/);
  assert.match(parseShape('2.5').error, /not a whole number/);
});

test('resolveRawLayout fills in -1, checks the size and reports bytes left over', () => {
  assert.deepStrictEqual(resolveRawLayout({ kind: 'f32', shape: [-1] }, 24).shape, [6]);
  assert.deepStrictEqual(resolveRawLayout({ kind: 'f32', shape: [-1, 3] }, 24).shape, [2, 3]);
  const r = resolveRawLayout({ kind: 'f32', shape: [-1] }, 26);
  assert.strictEqual(r.extra, 2);
  assert.deepStrictEqual(resolveRawLayout({ kind: 'f32', shape: [-1] }, 24 + 8).shape, [8]);
  assert.deepStrictEqual(resolveRawLayout({ kind: 'f32', shape: [-1], offset: 8 }, 32).shape, [6]);
  assert.match(resolveRawLayout({ kind: 'f32', shape: [4, 3] }, 24).error, /needs 48 bytes, but the file has 24/);
  assert.match(resolveRawLayout({ kind: 'f64', shape: [-1, 4] }, 24).error, /too few/);
  assert.match(resolveRawLayout({ kind: 'f32', shape: [2], offset: 30 }, 24).error, /larger than the file/);
  assert.match(resolveRawLayout({ kind: 'nope', shape: [2] }, 24).error, /value type/);
  assert.match(shapeHint('f32', 64 * 64 * 4), /4,096 values of 4 bytes, for example 64, 64/);
  assert.match(shapeHint('f32', 26), /2 bytes left over/);
});

test('cleanLayout drops anything that is not a valid layout', () => {
  assert.deepStrictEqual(cleanLayout({ kind: 'u8', shape: [2, 3], offset: 4, littleEndian: false, extra: 'x' }), { kind: 'u8', shape: [2, 3], offset: 4, littleEndian: false });
  for (const bad of [null, 'x', { kind: 'evil', shape: [2] }, { kind: 'u8', shape: [] }, { kind: 'u8', shape: [-1, -1] }, { kind: 'u8', shape: [0] }, { kind: 'u8', shape: [2], offset: -1 }, { kind: 'u8', shape: ['2'] }]) {
    assert.strictEqual(cleanLayout(bad), null, JSON.stringify(bad));
  }
  assert.ok(RAW_TYPES.length >= 14);
});

test('a raw float32 file opens like a .npy: stats, values and the NumPy line', async () => {
  const data = f32([1, 2, 3, 4, 5, 6]);
  const m = await openTensorFile(new BufferSource(data), 'x.bin', { raw: { kind: 'f32', shape: [-1, 3] } });
  assert.strictEqual(m.format, 'raw');
  const t = m.tensors[0];
  assert.deepStrictEqual(t.shape, [2, 3]);
  const s = await sampleTensor(t);
  assert.strictEqual(s.stats.mean, 3.5);
  assert.strictEqual(s.view.kind, 'heatmap');
  const v = await readValues(t, { row: 1, col: 0, rows: 1, cols: 3 });
  assert.deepStrictEqual(Array.from(v.re), [4, 5, 6]);
  assert.ok(m.metadata.some(x => x.key === 'NumPy equivalent' && x.value === 'np.fromfile(path, dtype="<f4").reshape(2, 3)'));
  assert.strictEqual(m.warnings.length, 0);
});

test('a header, big-endian int16 and bytes left over', async () => {
  const buf = Buffer.alloc(4 + 6 + 1);
  buf.write('HDR!', 0, 'latin1');
  buf.writeInt16BE(-2, 4); buf.writeInt16BE(300, 6); buf.writeInt16BE(7, 8);
  const m = await openTensorFile(new BufferSource(buf), 'x.dat', { raw: { kind: 'i16', shape: [-1], offset: 4, littleEndian: false } });
  const t = m.tensors[0];
  assert.deepStrictEqual(t.shape, [3]);
  const v = await readValues(t, { row: 0, col: 0, rows: 3, cols: 1 });
  assert.deepStrictEqual(Array.from(v.re), [-2, 300, 7]);
  assert.match(m.warnings[0], /1 byte at the end/);
  assert.ok(m.metadata.some(x => x.value.includes('dtype=">i2", offset=4, count=3')));
});

test('a layout that does not fit gives a readable error', async () => {
  await assert.rejects(openTensorFile(new BufferSource(f32([1, 2])), 'x.bin', { raw: { kind: 'f32', shape: [3, 3] } }), /needs 36 bytes/);
});

test('Save as .npy turns a raw file into a real .npy with the same values', async () => {
  const vals = [0.5, -1, 2, 8, 16, 32];
  const m = await openTensorFile(new BufferSource(f32(vals)), 'x.bin', { raw: { kind: 'f32', shape: [3, 2] } });
  const plan = planExport(m.tensors[0], 'npy', null);
  const chunks = [];
  assert.strictEqual(await writeExport(plan, async b => { chunks.push(Buffer.from(b)); }), true);
  const all = Buffer.concat(chunks);
  const h = await readNpyHeader(new BufferSource(all), 0);
  assert.deepStrictEqual(h.shape, [3, 2]);
  const back = new Float32Array(all.buffer.slice(all.byteOffset + h.dataOffset, all.byteOffset + all.length));
  assert.deepStrictEqual(Array.from(back), vals);
});

test('a raw uint8 image shows as a picture', async () => {
  const H = 4, W = 6, a = new Uint8Array(H * W * 3);
  for (let i = 0; i < H * W; i++) { a[i * 3] = 200; a[i * 3 + 1] = 10; a[i * 3 + 2] = 30; }
  const m = await openTensorFile(new BufferSource(a), 'x.raw', { raw: { kind: 'u8', shape: [H, W, 3] } });
  const r = await readImage(m.tensors[0], {});
  assert.strictEqual(r.layout, 'hwc');
  const d = Buffer.from(r.data, 'base64');
  assert.deepStrictEqual([d[0], d[1], d[2]], [200, 10, 30]);
});

test('an empty file gives a message, never a blank page', async () => {
  await assert.rejects(openTensorFile(new BufferSource(new Uint8Array(0)), 'empty.npy'), /empty \(0 bytes\)/);
});
