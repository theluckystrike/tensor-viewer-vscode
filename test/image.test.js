'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { open } = require('./helpers');
const { BufferSource, InflateSource } = require('../src/core/source');
const { openTensorFile } = require('../src/core/formats');
const { imageLayouts, suggestImage, readImage, imageInfo, unravel } = require('../src/core/image');
const { sampleTensor } = require('../src/core/sample');
const { validateWebviewMessage, publicModel } = require('../src/core/protocol');
const zlib = require('zlib');

/** A .npy file in memory. */
function npy(descr, shape, typed) {
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  const total = 10 + header.length + 1;
  header += ' '.repeat((64 - (total % 64)) % 64) + '\n';
  const h = Buffer.alloc(10);
  h.write('\x93NUMPY', 0, 'latin1'); h[6] = 1; h[7] = 0; h.writeUInt16LE(header.length, 8);
  return Buffer.concat([h, Buffer.from(header, 'latin1'), Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength)]);
}
async function model(descr, shape, typed) {
  return openTensorFile(new BufferSource(npy(descr, shape, typed)), 'x.npy');
}
const px = (r, x, y) => { const d = Buffer.from(r.data, 'base64'); const i = (y * r.width + x) * 4; return [d[i], d[i + 1], d[i + 2], d[i + 3]]; };

test('layouts: RGB channels last or first, grayscale stacks, and arrays that are not pictures', async () => {
  const ids = async (descr, shape, T) => imageLayouts((await model(descr, shape, new T(shape.reduce((a, b) => a * b, 1)))).tensors[0]).map(l => `${l.id}:${l.H}x${l.W}x${l.C}/${l.frames}`);
  assert.deepStrictEqual(await ids('|u1', [6, 8, 3], Uint8Array), ['hwc:6x8x3/1', 'gray:8x3x1/6']);
  assert.deepStrictEqual(await ids('<f4', [3, 6, 8], Float32Array), ['chw:6x8x3/1', 'gray:6x8x1/3']);
  assert.deepStrictEqual(await ids('<f4', [2, 3, 6, 8], Float32Array), ['chw:6x8x3/2', 'gray:6x8x1/6']);
  assert.deepStrictEqual(await ids('<f4', [10, 28, 28], Float32Array), ['gray:28x28x1/10']);
  assert.deepStrictEqual(await ids('|u1', [16, 16, 1], Uint8Array), ['gray:16x16x1/1']);
  assert.deepStrictEqual(await ids('|u1', [4, 16, 16], Uint8Array), ['gray:16x16x1/4', 'chw:16x16x4/1']);
  assert.deepStrictEqual(await ids('<f4', [50], Float32Array), []);
  assert.deepStrictEqual(await ids('<f4', [1, 50], Float32Array), []);
});

test('suggest: RGB and uint8 arrays show at once, float matrices wait for a click', async () => {
  const sug = async (descr, shape, T) => { const t = (await model(descr, shape, new T(shape.reduce((a, b) => a * b, 1)))).tensors[0]; return suggestImage(t); };
  assert.strictEqual(await sug('|u1', [6, 8, 3], Uint8Array), true);
  assert.strictEqual(await sug('<f4', [3, 6, 8], Float32Array), true);
  assert.strictEqual(await sug('|u1', [28, 28], Uint8Array), true);
  assert.strictEqual(await sug('<f4', [28, 28], Float32Array), false);
});

test('GGUF quant tensors and pickled arrays have no picture layout', async () => {
  const g = await open('tiny-llama-q8.gguf');
  const q = g.tensors.find(t => t._loc && t._loc.quant);
  assert.ok(q);
  assert.strictEqual(imageInfo(q), null);
  await g.close();
});

test('HxWx3 uint8: same pixels as the 0.2 image view', async () => {
  const m = await open('image_96x128x3_u1.npy');
  const t = m.tensors[0];
  const old = (await sampleTensor(t)).view;
  const r = await readImage(t, {});
  assert.strictEqual(r.layout, 'hwc');
  assert.strictEqual(r.width, old.width);
  assert.strictEqual(r.height, old.height);
  assert.strictEqual(r.data, old.data);
  assert.deepStrictEqual([r.lo, r.hi], [0, 255]);
  await m.close();
});

test('CxHxW float in 0..1 draws as RGB with the 0..1 range', async () => {
  const H = 4, W = 5, a = new Float32Array(3 * H * W);
  for (let c = 0; c < 3; c++) for (let i = 0; i < H * W; i++) a[c * H * W + i] = c / 2;
  const t = (await model('<f4', [3, H, W], a)).tensors[0];
  const r = await readImage(t, { layout: 'chw' });
  assert.deepStrictEqual([r.lo, r.hi], [0, 1]);
  assert.deepStrictEqual(px(r, 2, 3), [0, 128, 255, 255]);
});

test('frames: a stack shows one frame at a time, with its index', async () => {
  const N = 4, H = 8, W = 8, a = new Uint8Array(N * H * W);
  for (let k = 0; k < N; k++) a.fill(k * 10, k * H * W, (k + 1) * H * W);
  const t = (await model('|u1', [N, H, W], a)).tensors[0];
  const r = await readImage(t, { layout: 'gray', frame: 2 });
  assert.strictEqual(r.frames, 4);
  assert.deepStrictEqual(r.frameIndex, [2]);
  assert.deepStrictEqual(px(r, 0, 0).slice(0, 3), [20, 20, 20]);
  await assert.rejects(readImage(t, { layout: 'gray', frame: 4 }), /out of range \(0 to 3\)/);
  assert.deepStrictEqual(unravel(5, [2, 3]), [1, 2]);
});

test('a big picture reads only the rows it shows', async () => {
  const H = 2000, W = 1000, a = new Float32Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) a[y * W + x] = y;
  const src = new BufferSource(npy('<f4', [H, W], a));
  let reads = 0, bytes = 0;
  const orig = src.read.bind(src);
  const m = await openTensorFile(src, 'big.npy');
  src.read = (o, n) => { reads++; bytes += n; return orig(o, n); };
  const r = await readImage(m.tensors[0], { range: 'minmax' });
  assert.strictEqual(r.height, 512);
  assert.strictEqual(r.width, 256);
  assert.ok(reads <= 512, `reads ${reads}`);
  assert.ok(bytes <= 512 * W * 4, `bytes ${bytes}`);
  assert.ok(px(r, 0, 0)[0] === 0 && px(r, 0, 511)[0] > 250);
});

test('range modes, NaN in magenta, and the symmetric range', async () => {
  const a = new Float32Array([-1, 0, 1, 3, NaN, 2]);
  const t = (await model('<f4', [2, 3], a)).tensors[0];
  const mm = await readImage(t, { layout: 'gray', range: 'minmax' });
  assert.deepStrictEqual([mm.lo, mm.hi], [-1, 3]);
  assert.strictEqual(mm.nonFinite, 1);
  assert.deepStrictEqual(px(mm, 1, 1), [255, 0, 255, 255]);
  assert.deepStrictEqual(px(mm, 0, 0).slice(0, 3), [0, 0, 0]);
  const sym = await readImage(t, { layout: 'gray', range: 'sym' });
  assert.deepStrictEqual([sym.lo, sym.hi], [-3, 3]);
  const auto = await readImage(t, { layout: 'gray', range: 'auto' });
  assert.deepStrictEqual([auto.lo, auto.hi], [-1, 3]);
});

test('compressed npz entries: frames past the limit give a message, not a crash', async () => {
  const N = 4, H = 16, W = 16, a = new Uint8Array(N * H * W).fill(7);
  const raw = npy('|u1', [N, H, W], a);
  const def = zlib.deflateRawSync(raw);
  const src = new InflateSource(new BufferSource(def), 0, def.length, raw.length);
  const { readNpyHeader } = require('../src/core/npy');
  const h = await readNpyHeader(src, 0);
  const t = { name: 'x', dtype: 'uint8', shape: [N, H, W], numel: N * H * W, readable: true, _loc: { source: src, dataOffset: h.dataOffset, kind: 'u8', elemSize: 1, littleEndian: true, fortran: false } };
  const ok = await readImage(t, { layout: 'gray', frame: 0 }, { compressedMaxBytes: 600 });
  assert.strictEqual(ok.width, 16);
  await assert.rejects(readImage(t, { layout: 'gray', frame: 3 }, { compressedMaxBytes: 600 }), /compressed/);
});

test('protocol: image requests are checked, and the model carries plain image info', async () => {
  const m = await model('|u1', [6, 8, 3], new Uint8Array(144));
  assert.ok(validateWebviewMessage({ type: 'image', index: 0, layout: 'hwc', frame: 0, range: 'auto' }, 1).ok);
  for (const bad of [{ layout: 'xyz' }, { frame: -1 }, { frame: 1.5 }, { range: 'evil' }, { index: 3 }]) {
    assert.strictEqual(validateWebviewMessage({ type: 'image', index: 0, layout: 'hwc', frame: 0, range: 'auto', ...bad }, 1).ok, false, JSON.stringify(bad));
  }
  assert.deepStrictEqual(validateWebviewMessage({ type: 'rawLayout', junk: 1 }, 1), { ok: true, msg: { type: 'rawLayout' } });
  const pm = publicModel(m, 'x.npy');
  assert.strictEqual(pm.tensors[0].image.suggest, true);
  assert.strictEqual(pm.tensors[0].image.layouts[0].id, 'hwc');
  assert.doesNotThrow(() => JSON.stringify(pm));
  assert.ok(!JSON.stringify(pm).includes('_loc'));
});
