'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { expected, open, close } = require('./helpers');
const { parsePyLiteral } = require('../src/core/npy');
const { decodePlain } = require('../src/core/dtypes');
const { formatFromName } = require('../src/core/formats');

async function values(t) {
  const L = t._loc;
  const bytes = await L.source.read(L.dataOffset, t.nbytes);
  return Array.from(decodePlain(L.kind, bytes, t.numel, L.littleEndian));
}

test('python literal parser reads npy headers and refuses code', () => {
  assert.deepStrictEqual({ ...parsePyLiteral("{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }") }, { descr: '<f4', fortran_order: false, shape: [3, 4] });
  assert.deepStrictEqual(parsePyLiteral('(5,)'), [5]);
  assert.deepStrictEqual(parsePyLiteral('()'), []);
  assert.deepStrictEqual(parsePyLiteral("[('x', '<f4'), ('y', '<i2')]"), [['x', '<f4'], ['y', '<i2']]);
  assert.throws(() => parsePyLiteral("__import__('os').system('echo hi')"), /Bad .npy header/);
  assert.throws(() => parsePyLiteral("{'a': 1} + 1"), /Bad .npy header/);
  assert.throws(() => parsePyLiteral('['.repeat(100)), /nested too deep/);
});

test('format detection by extension', () => {
  assert.strictEqual(formatFromName('a/b/model.GGUF'), 'gguf');
  assert.strictEqual(formatFromName('x.safetensors'), 'safetensors');
  assert.strictEqual(formatFromName('x.pt'), 'pytorch');
  assert.strictEqual(formatFromName('x.pkl'), null);
});

for (const name of ['arange_2x3_f4.npy', 'bigendian_i4.npy', 'fortran_3x4_f8.npy', 'int64_2x2.npy', 'bool_4.npy', 'complex_3_c8.npy', 'v2_header.npy', 'scalar_f4.npy', 'specials_f2.npy']) {
  test(`npy ${name}: shape, dtype and values match numpy`, async () => {
    const m = await open(name);
    const e = expected[name];
    const t = m.tensors[0];
    assert.strictEqual(m.format, 'npy');
    assert.deepStrictEqual(t.shape, e.shape);
    assert.ok(t.readable);
    let v = await values(t);
    if (e.fortran) { // memory is column-major; reorder to C order to compare
      const [r, c] = e.shape; const cv = new Array(r * c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) cv[i * c + j] = v[j * r + i];
      v = cv;
    }
    e.values.forEach((x, i) => assert.ok(x === null ? Number.isNaN(v[i]) : typeof x === 'string' ? v[i] === (x === 'inf' ? Infinity : -Infinity) : close(v[i], x, 1e-6), `${name}[${i}] ${v[i]} != ${x}`));
    await m.close();
  });
}

test('npy object arrays (pickle) are listed but never read', async () => {
  const m = await open('object_pickle.npy');
  const t = m.tensors[0];
  assert.strictEqual(t.readable, false);
  assert.strictEqual(t.dtype, 'object');
  assert.match(t.note, /pickle/);
  await m.close();
});

test('npy text and structured arrays are listed without stats', async () => {
  for (const f of ['strings_U5.npy', 'structured.npy']) {
    const m = await open(f);
    assert.strictEqual(m.tensors[0].readable, false);
    assert.deepStrictEqual(m.tensors[0].shape, [3]);
    await m.close();
  }
});

test('broken files give clear errors', async () => {
  await assert.rejects(open('truncated.npy'), /Truncated/);
  await assert.rejects(open('broken_header.safetensors'), /100 MB limit/);
});

test('npz stored: every array is listed with its header', async () => {
  const m = await open('bundle_stored.npz');
  assert.deepStrictEqual(m.tensors.map(t => t.name), expected['bundle_stored.npz'].names);
  const img = m.tensors.find(t => t.name === 'image');
  assert.deepStrictEqual(img.shape, [96, 128, 3]);
  assert.strictEqual(img._loc.source.random, true);
  assert.strictEqual(m.tensors.find(t => t.name === 'words').readable, false);
  await m.close();
});

test('npz compressed: headers come from inflating only a prefix', async () => {
  const m = await open('bundle_compressed.npz');
  assert.deepStrictEqual(m.tensors.map(t => t.name), expected['bundle_compressed.npz'].names);
  assert.deepStrictEqual(m.tensors.find(t => t.name === 'signal').shape, [5000]);
  const mat = m.tensors.find(t => t.name === 'matrix');
  assert.deepStrictEqual(mat.shape, [256, 512]);
  assert.strictEqual(mat._loc.source.random, false);
  assert.ok(mat._loc.source.have < mat._loc.source.size, 'should not inflate the whole entry to list it');
  await m.close();
});

test('npz with a pickled entry refuses only that entry', async () => {
  const m = await open('with_object.npz');
  assert.strictEqual(m.tensors.find(t => t.name === 'ok').readable, true);
  assert.strictEqual(m.tensors.find(t => t.name === 'bad').readable, false);
  await m.close();
});

test('safetensors: names, params, offsets and metadata', async () => {
  const m = await open('gpt2_tiny.safetensors');
  const e = expected['gpt2_tiny.safetensors'];
  assert.deepStrictEqual(m.tensors.map(t => t.name).sort(), e.names.slice().sort());
  assert.strictEqual(m.totalParams, e.params);
  assert.deepStrictEqual(Object.fromEntries(m.metadata.map(x => [x.key, x.value])), e.metadata);
  const wte = m.tensors.find(t => t.name === 'wte.weight');
  assert.strictEqual(wte.dtype, 'float16');
  assert.deepStrictEqual(wte.shape, [1000, 256]);
  assert.strictEqual(wte.nbytes, 1000 * 256 * 2);
  await m.close();
});

test('safetensors BF16 and FP8 decode like the reference', async () => {
  const m = await open('bf16_f8.safetensors');
  const e = expected['bf16_f8.safetensors'];
  const get = async n => values(m.tensors.find(t => t.name === n));
  (await get('bf16')).forEach((v, i) => assert.ok(close(v, e.bf16[i]), `bf16[${i}]`));
  (await get('f8a')).forEach((v, i) => assert.ok(close(v, e.f8a[i]), `e4m3[${i}] ${v}`));
  const b = await get('f8b');
  assert.deepStrictEqual([b[0], b[1], b[2], b[3], b[5]], [1, -1, 57344, Infinity, 2 ** -16]);
  assert.ok(Number.isNaN(b[4]));
  await m.close();
});

test('gguf: metadata, architecture, tokenizer and per-tensor types', async () => {
  const m = await open('quant_types.gguf');
  const info = Object.fromEntries(m.info.map(x => [x.label, x.value]));
  assert.strictEqual(info.Architecture, 'llama');
  assert.strictEqual(info['Context length'], '4096');
  assert.match(info.Tokenizer, /300 tokens/);
  assert.match(info['File type'], /Q4_K_M/);
  assert.match(info['Chat template'], /^yes/);
  assert.deepStrictEqual(m.tensors.map(t => t.name), expected['quant_types.gguf'].names);
  assert.deepStrictEqual(m.tensors.find(t => t.name === 'q4_k').shape, [4, 256]);
  assert.strictEqual(m.tensors.find(t => t.name === 'q6_k').dtype, 'Q6_K');
  const tokens = m.metadata.find(x => x.key === 'tokenizer.ggml.tokens');
  assert.match(tokens.value, /^\[300 items\]/);
  await m.close();
});

test('gguf: every decoded type matches gguf-py dequantize', async () => {
  const m = await open('quant_types.gguf');
  const ref = await open('quant_types_ref.npz');
  const { sampleTensor } = require('../src/core/sample');
  const { decodePlain, decodeQuant } = require('../src/core/dtypes');
  for (const t of m.tensors) {
    const r = ref.tensors.find(x => x.name === t.name);
    const want = Array.from(decodePlain('f32', await r._loc.source.read(r._loc.dataOffset, r.nbytes), r.numel));
    const L = t._loc;
    const bytes = await L.source.read(L.dataOffset, t.nbytes);
    const got = L.quant ? decodeQuant(L.quant, bytes, t.numel / L.block) : decodePlain(L.kind, bytes, t.numel, true);
    assert.strictEqual(got.length, want.length, t.name);
    for (let i = 0; i < want.length; i++) assert.ok(close(got[i], want[i], 1e-5), `${t.name}[${i}] ${got[i]} != ${want[i]}`);
    const s = await sampleTensor(t);
    assert.ok(s.exact && s.stats.count === t.numel, t.name);
  }
  await m.close(); await ref.close();
});

test('gguf demo model: parameter count and bits per weight', async () => {
  const m = await open('tiny-llama-q8.gguf');
  assert.strictEqual(m.tensors.length, 39);
  assert.strictEqual(m.totalParams, m.tensors.reduce((a, t) => a + t.numel, 0));
  const bpw = Number(m.info.find(x => x.label === 'Bits per weight').value);
  assert.ok(bpw > 4 && bpw < 9, String(bpw));
  assert.deepStrictEqual(m.tensors.find(t => t.name === 'blk.0.ffn_down.weight').shape, [256, 704]);
  await m.close();
});
