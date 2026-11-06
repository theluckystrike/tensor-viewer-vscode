'use strict';
// Values table, slices and export (.npy and CSV).
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { expected, open, close } = require('./helpers');
const V = require('../src/core/values');
const N = require('../src/core/numtext');
const { BufferSource } = require('../src/core/source');
const { openTensorFile } = require('../src/core/formats');
const { decodePlain, decodeQuant } = require('../src/core/dtypes');
const { sampleTensor } = require('../src/core/sample');
const { validateWebviewMessage } = require('../src/core/protocol');

// A 4 x 3 x 5 float32 array whose value is 100*i + 10*j + k, built in memory as a real .npy file.
function stackNpy(fortran = false) {
  const shape = [4, 3, 5];
  const data = new Float32Array(60);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 5; k++) {
    const at = fortran ? i + 4 * j + 12 * k : i * 15 + j * 5 + k;
    data[at] = 100 * i + 10 * j + k;
  }
  return Buffer.concat([V.npyHeader('<f4', fortran, shape), Buffer.from(data.buffer)]);
}
const openBytes = async (bytes, name) => openTensorFile(new BufferSource(new Uint8Array(bytes)), name);
async function exportBytes(plan, opts) {
  const parts = [];
  const done = await V.writeExport(plan, async b => { parts.push(Buffer.from(b)); }, opts);
  return { done, bytes: Buffer.concat(parts) };
}

test('grid shape and slice checks', () => {
  assert.deepStrictEqual(V.gridShape([]), { rows: 1, cols: 1, lead: [] });
  assert.deepStrictEqual(V.gridShape([7]), { rows: 7, cols: 1, lead: [] });
  assert.deepStrictEqual(V.gridShape([2, 3, 4, 5]), { rows: 4, cols: 5, lead: [2, 3] });
  assert.deepStrictEqual(V.checkSlice([2, 3, 4, 5], null), [0, 0]);
  assert.deepStrictEqual(V.checkSlice([2, 3, 4, 5], [1, 2]), [1, 2]);
  for (const bad of [[2, 0], [0], [0, -1], [0, 1.5], '0,0', [0, '1']]) assert.strictEqual(V.checkSlice([2, 3, 4, 5], bad), null, JSON.stringify(bad));
  assert.strictEqual(V.sliceLabel([2, 3, 4, 5], [1, 2]), '[1, 2, :, :]');
});

test('a page of a 2-D float32 array matches the bytes in the file', async () => {
  const m = await open('matrix_1024x512_f4.npy');
  const t = m.tensors[0];
  const v = await V.readValues(t, { slice: null, row: 1000, col: 3, rows: 10, cols: 7 });
  assert.deepStrictEqual([v.rows, v.cols, v.totalRows, v.totalCols], [10, 7, 1024, 512]);
  const L = t._loc;
  const row = decodePlain('f32', await L.source.read(L.dataOffset + 1000 * 512 * 4, 512 * 4), 512);
  for (let c = 0; c < 7; c++) assert.strictEqual(v.re[c], row[3 + c]);
  const first = await V.readValues(t, { slice: null, row: 1000, col: 0, rows: 1, cols: 1 });
  assert.ok(close(first.re[0], expected['matrix_1024x512_f4.npy'].row_1000_first));
  // Pages are clamped to the array and to the page limits.
  const edge = await V.readValues(t, { slice: null, row: 1020, col: 500, rows: 999, cols: 999 });
  assert.deepStrictEqual([edge.rows, edge.cols], [4, 12]);
  const big = await V.readValues(t, { slice: null, row: 0, col: 0, rows: 999, cols: 999 });
  assert.deepStrictEqual([big.rows, big.cols], [V.MAX_PAGE_ROWS, V.MAX_PAGE_COLS]);
  await m.close();
});

test('Fortran order, complex, bool, int64, big-endian and scalar values read right', async () => {
  const f = await open('fortran_3x4_f8.npy');
  const fv = await V.readValues(f.tensors[0], { slice: null, row: 0, col: 0, rows: 3, cols: 4 });
  assert.deepStrictEqual(Array.from(fv.re), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
  await f.close();
  const c = await open('complex_3_c8.npy');
  const cv = await V.readValues(c.tensors[0], { slice: null, row: 0, col: 0, rows: 3, cols: 1 });
  assert.deepStrictEqual(Array.from(cv.re), [3, 0, -6]);
  assert.deepStrictEqual(Array.from(cv.im), [4, 1, -8]);
  assert.strictEqual(V.pageText(cv), '3+4j\n0+1j\n-6-8j');
  await c.close();
  const b = await open('bool_4.npy');
  const bv = await V.readValues(b.tensors[0], { slice: null, row: 0, col: 0, rows: 4, cols: 1 });
  assert.strictEqual(bv.kind, 'bool');
  assert.deepStrictEqual(Array.from(bv.re).map(x => N.displayText(x, 'bool', 'auto')), ['True', 'False', 'True', 'True']);
  await b.close();
  const i = await open('int64_2x2.npy');
  assert.strictEqual(V.pageText(await V.readValues(i.tensors[0], { slice: null, row: 0, col: 0, rows: 2, cols: 2 })), '-1099511627776\t7\n1099511627776\t-7');
  await i.close();
  const be = await open('bigendian_i4.npy');
  assert.deepStrictEqual(Array.from((await V.readValues(be.tensors[0], { slice: null, row: 0, col: 0, rows: 5, cols: 1 })).re), [1, -2, 300000, -4, 5]);
  await be.close();
  const s = await open('scalar_f4.npy');
  const sv = await V.readValues(s.tensors[0], { slice: null, row: 0, col: 0, rows: 5, cols: 5 });
  assert.deepStrictEqual([sv.rows, sv.cols, sv.re[0]], [1, 1, 3.5]);
  await s.close();
});

test('GGUF quant pages that start inside a block match the dequantized row', async () => {
  const m = await open('tiny-llama-q8.gguf');
  for (const name of ['token_embd.weight', 'blk.0.attn_q.weight']) {
    const t = m.tensors.find(x => x.name === name);
    const L = t._loc, cols = t.shape[1];
    const rowBytes = (cols / L.block) * L.blockBytes;
    const row = decodeQuant(L.quant, await L.source.read(L.dataOffset + 5 * rowBytes, rowBytes), cols / L.block);
    const v = await V.readValues(t, { slice: null, row: 5, col: 29, rows: 2, cols: 9 }); // crosses the 32-value block edge
    for (let c = 0; c < 9; c++) assert.strictEqual(v.re[c], row[29 + c], `${name} col ${29 + c}`);
    assert.strictEqual(v.kind, L.quant);
  }
  await m.close();
});

test('slices: one 2-D slice of a 3-D array, for the table, the stats and the heatmap', async () => {
  const m = await openBytes(stackNpy(), 'stack.npy');
  const t = m.tensors[0];
  const v = await V.readValues(t, { slice: [2], row: 1, col: 0, rows: 2, cols: 5 });
  assert.deepStrictEqual(Array.from(v.re), [210, 211, 212, 213, 214, 220, 221, 222, 223, 224]);
  assert.deepStrictEqual(v.slice, [2]);
  const sub = V.sliceTensor(t, [3]);
  assert.deepStrictEqual(sub.shape, [3, 5]);
  const r = await sampleTensor(sub);
  assert.deepStrictEqual([r.stats.min, r.stats.max, r.exact, r.view.kind, r.view.reshaped], [300, 324, true, 'heatmap', false]);
  assert.throws(() => V.sliceTensor(t, [4]), /Bad slice/);
  assert.strictEqual(V.sliceTensor((await open('arange_2x3_f4.npy')).tensors[0], []), null);
  // Fortran order: the table still reads the right cells, the stats refuse a slice.
  const f = await openBytes(stackNpy(true), 'stack_f.npy');
  const fv = await V.readValues(f.tensors[0], { slice: [2], row: 1, col: 0, rows: 2, cols: 5 });
  assert.deepStrictEqual(Array.from(fv.re), [210, 211, 212, 213, 214, 220, 221, 222, 223, 224]);
  assert.throws(() => V.sliceTensor(f.tensors[0], [1]), /Fortran-order/);
});

test('compressed .npz entries refuse pages past the first 64 MB with a clear message', async () => {
  const m = await open('bundle_compressed.npz');
  const t = m.tensors.find(x => x.readable);
  const v = await V.readValues(t, { slice: null, row: 0, col: 0, rows: 2, cols: 3 });
  assert.deepStrictEqual(Array.from(v.re), [0, 1, 2, 3, 4, 5]);
  const far = { ...t, _loc: { ...t._loc, dataOffset: 65 * 1048576 } };
  await assert.rejects(V.readValues(far, { slice: null, row: 0, col: 0, rows: 1, cols: 1 }), /compressed .npz entry/);
  await m.close();
});

test('number text: exact float32 text, precision choices, specials', () => {
  assert.strictEqual(N.exactText(Math.fround(0.1), 'f32'), '0.1');
  assert.strictEqual(N.exactText(0.1, 'f64'), '0.1');
  assert.strictEqual(N.exactText(Math.fround(1 / 3), 'Q8_0'), '0.33333334');
  assert.strictEqual(N.exactText(NaN, 'f32'), 'nan');
  assert.strictEqual(N.exactText(-Infinity, 'f16'), '-inf');
  assert.strictEqual(N.exactText(7, 'i32'), '7');
  assert.strictEqual(N.displayText(0.123456789, 'f64', '2'), '0.12');
  assert.strictEqual(N.displayText(0.123456789, 'f64', '8'), '0.12345679');
  assert.strictEqual(N.displayText(0.123456789, 'f64', 'sci'), '1.235e-1');
  assert.strictEqual(N.displayText(0.123456789, 'f64', 'auto'), '0.123457');
  assert.strictEqual(N.displayText(12, 'i64', '4'), '12');
  assert.strictEqual(N.displayText(Infinity, 'f32', '4'), 'inf');
  assert.strictEqual(N.displayComplex(1.5, -2, 'c64', '2'), '1.50-2.00j');
  assert.deepStrictEqual(N.PRECISIONS, ['auto', 'exact', '2', '4', '8', 'sci']);
  // The webview bundles this file, so it must not use Node APIs.
  const src = fs.readFileSync(path.join(__dirname, '../src/core/numtext.js'), 'utf8');
  assert.ok(!/require\(|Buffer|process\./.test(src));
});

function parseNpy(bytes) {
  return openTensorFile(new BufferSource(new Uint8Array(bytes)), 'x.npy');
}

test('save as .npy: plain types keep their bytes, Fortran order and byte order', async () => {
  for (const name of ['matrix_1024x512_f4.npy', 'fortran_3x4_f8.npy', 'bigendian_i4.npy', 'complex_3_c8.npy', 'bool_4.npy', 'scalar_f4.npy', 'int64_2x2.npy']) {
    const m = await open(name);
    const t = m.tensors[0];
    const { done, bytes } = await exportBytes(V.planExport(t, 'npy', null));
    assert.ok(done);
    assert.strictEqual((10 + bytes.readUInt16LE(8)) % 64, 0, 'header is padded to 64 bytes');
    const back = await parseNpy(bytes);
    const b = back.tensors[0];
    assert.deepStrictEqual(b.shape, t.shape, name);
    assert.strictEqual(b.dtype, t.dtype, name);
    assert.strictEqual(b._loc.fortran, t._loc.fortran, name);
    assert.strictEqual(b._loc.littleEndian, t._loc.littleEndian, name);
    const src = Buffer.from(await t._loc.source.read(t._loc.dataOffset, t.nbytes));
    assert.ok(src.equals(bytes.subarray(b._loc.dataOffset)), `${name}: data bytes are copied as they are`);
    await m.close();
  }
});

test('save as .npy: bf16, fp8 and GGUF quants become float32, safetensors keep their dtype', async () => {
  const g = await open('tiny-llama-q8.gguf');
  const t = g.tensors.find(x => x.name === 'blk.0.attn_q.weight');
  const plan = V.planExport(t, 'npy', null);
  assert.match(plan.note, /Q8_0 has no NumPy type, so the values are saved as float32/);
  const back = (await parseNpy((await exportBytes(plan)).bytes)).tensors[0];
  assert.deepStrictEqual([back.dtype, back.shape], ['float32', t.shape]);
  const want = await V.readValues(t, { slice: null, row: 7, col: 0, rows: 1, cols: 64 });
  const got = await V.readValues(back, { slice: null, row: 7, col: 0, rows: 1, cols: 64 });
  assert.deepStrictEqual(Array.from(got.re), Array.from(want.re));
  await g.close();
  const s = await open('bf16_f8.safetensors');
  for (const x of s.tensors.filter(y => y.readable)) {
    const p = V.planExport(x, 'npy', null);
    const b = (await parseNpy((await exportBytes(p)).bytes)).tensors[0];
    const raw = /^(bfloat16|float8)/.test(x.dtype);
    assert.strictEqual(b.dtype, raw ? 'float32' : x.dtype, x.name);
    const a1 = await V.readValues(x, { slice: null, row: 0, col: 0, rows: 50, cols: 50 });
    const a2 = await V.readValues(b, { slice: null, row: 0, col: 0, rows: 50, cols: 50 });
    assert.deepStrictEqual(Array.from(a2.re), Array.from(a1.re), x.name);
  }
  await s.close();
});

test('save one slice as .npy and as CSV', async () => {
  const m = await openBytes(stackNpy(), 'stack.npy');
  const t = m.tensors[0];
  const plan = V.planExport(t, 'npy', [1]);
  assert.strictEqual(plan.label, '[1, :, :]');
  const back = (await parseNpy((await exportBytes(plan)).bytes)).tensors[0];
  assert.deepStrictEqual(back.shape, [3, 5]);
  assert.deepStrictEqual(Array.from((await V.readValues(back, { slice: null, row: 2, col: 0, rows: 1, cols: 5 })).re), [120, 121, 122, 123, 124]);
  const csv = (await exportBytes(V.planExport(t, 'csv', [3]))).bytes.toString('utf8');
  assert.strictEqual(csv, '300,301,302,303,304\n310,311,312,313,314\n320,321,322,323,324\n');
  // The whole 3-D array as CSV: leading dimensions become rows.
  const all = V.planExport(t, 'csv', null);
  assert.match(all.note, /flattened into rows/);
  const lines = (await exportBytes(all)).bytes.toString('utf8').trim().split('\n');
  assert.strictEqual(lines.length, 12);
  assert.strictEqual(lines[11], '320,321,322,323,324');
  assert.strictEqual(V.exportName('model.layers.0.weight', '[1, :, :]', 'npy'), 'model.layers.0.weight_1.npy');
  assert.strictEqual(V.exportName('../../etc/passwd', '', 'csv'), 'etc_passwd.csv');
  assert.strictEqual(V.exportName('', '', 'npy'), 'tensor.npy');
});

test('save as CSV: exact text, Fortran order, complex, quant rows, size limit and cancel', async () => {
  const f = await open('fortran_3x4_f8.npy');
  assert.strictEqual((await exportBytes(V.planExport(f.tensors[0], 'csv', null))).bytes.toString(), '0,1,2,3\n4,5,6,7\n8,9,10,11\n');
  await f.close();
  const c = await open('complex_3_c8.npy');
  assert.strictEqual((await exportBytes(V.planExport(c.tensors[0], 'csv', null))).bytes.toString(), '3+4j\n0+1j\n-6-8j\n');
  await c.close();
  const g = await open('tiny-llama-q8.gguf');
  const t = g.tensors.find(x => x.name === 'token_embd.weight');
  const lines = (await exportBytes(V.planExport(t, 'csv', null))).bytes.toString().trim().split('\n');
  assert.strictEqual(lines.length, t.shape[0]);
  const row9 = await V.readValues(t, { slice: null, row: 9, col: 0, rows: 1, cols: 64 });
  // Short text that reads back as the same float32, like NumPy prints it.
  assert.deepStrictEqual(lines[9].split(',').slice(0, 64).map(x => Math.fround(Number(x))), Array.from(row9.re));
  assert.strictEqual(lines[9].split(',').length, t.shape[1]);
  // Cancel stops before the next chunk and reports it.
  const big = await open('matrix_1024x512_f4.npy');
  const r = await exportBytes(V.planExport(big.tensors[0], 'csv', null), { cancelled: () => true });
  assert.strictEqual(r.done, false);
  assert.strictEqual(r.bytes.length, 0);
  const huge = { ...big.tensors[0], numel: V.CSV_MAX_VALUES + 1 };
  assert.throws(() => V.planExport(huge, 'csv', null), /CSV export stops at 20,000,000 values. Save it as .npy instead/);
  await big.close();
  await g.close();
});

test('protocol: values, copyValues and export messages are checked against the tensor shape', () => {
  const T = [{ shape: [4, 3, 5] }, { shape: [7] }];
  const ok = m => validateWebviewMessage(m, 2, T);
  assert.deepStrictEqual(ok({ type: 'values', index: 0, slice: [3], row: 0, col: 0, rows: 32, cols: 5, junk: 1 }).msg, { type: 'values', index: 0, slice: [3], row: 0, col: 0, rows: 32, cols: 5 });
  assert.deepStrictEqual(ok({ type: 'values', index: 1, row: 2, col: 0, rows: 5, cols: 1 }).msg.slice, []);
  assert.deepStrictEqual(ok({ type: 'copyValues', index: 0, slice: [0], row: 0, col: 0, rows: 1, cols: 1 }).msg.type, 'copyValues');
  assert.deepStrictEqual(ok({ type: 'select', index: 0, slice: [2] }).msg, { type: 'select', index: 0, slice: [2] });
  assert.deepStrictEqual(ok({ type: 'export', index: 0, format: 'csv', slice: [1] }).msg, { type: 'export', index: 0, format: 'csv', slice: [1] });
  assert.deepStrictEqual(ok({ type: 'export', index: 1, format: 'npy' }).msg, { type: 'export', index: 1, format: 'npy', slice: null });
  const bad = [
    { type: 'values', index: 0, slice: [4], row: 0, col: 0, rows: 1, cols: 1 },
    { type: 'values', index: 0, slice: [0, 0], row: 0, col: 0, rows: 1, cols: 1 },
    { type: 'values', index: 0, slice: [0], row: -1, col: 0, rows: 1, cols: 1 },
    { type: 'values', index: 0, slice: [0], row: 0, col: 0, rows: 201, cols: 1 },
    { type: 'values', index: 0, slice: [0], row: 0, col: 0, rows: 1, cols: 65 },
    { type: 'values', index: 0, slice: [0], row: 0, col: 0, rows: 0, cols: 1 },
    { type: 'values', index: 2, row: 0, col: 0, rows: 1, cols: 1 },
    { type: 'select', index: 1, slice: [] },
    { type: 'select', index: 0, slice: 'all' },
    { type: 'export', index: 0, format: 'pkl' },
    { type: 'export', index: 0, format: 'npy', slice: [9] },
    { type: 'export', index: 1, format: 'npy', slice: [] },
  ];
  for (const m of bad) assert.strictEqual(ok(m).ok, false, JSON.stringify(m));
});
