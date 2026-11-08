'use strict';
// Object arrays (np.save of dicts, lists, strings; allow_pickle=True): decoded as data, never run.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { FileSource, BufferSource } = require('../src/core/source');
const { openTensorFile } = require('../src/core/formats');
const { sampleTensor } = require('../src/core/sample');
const { decodePlain } = require('../src/core/dtypes');
const { publicModel } = require('../src/core/protocol');
const pyobj = require('../src/core/pyobj');

const DIR = path.join(__dirname, 'fixtures', 'objects');
const expected = JSON.parse(fs.readFileSync(path.join(DIR, 'expected.json'), 'utf8'));

async function open(name) {
  const src = await FileSource.open(path.join(DIR, name));
  const model = await openTensorFile(src, name);
  model.close = () => src.close();
  return model;
}
async function values(t) {
  const L = t._loc;
  return Array.from(decodePlain(L.kind, await L.source.read(L.dataOffset, t.nbytes), t.numel, L.littleEndian));
}
const byName = (m, n) => m.tensors.find(t => t.name === n);
const row = (m, p) => m.objects.find(r => r.path === p);

test('np.save of a dict: every nested array is a tensor named by its Python path', async () => {
  const m = await open('dict_weights.npy');
  const e = expected['dict_weights.npy'];
  assert.deepStrictEqual(m.tensors.map(t => t.name).sort(), Object.keys(e.tensors).sort());
  for (const [name, want] of Object.entries(e.tensors)) {
    const t = byName(m, name);
    assert.strictEqual(t.dtype, want.dtype, name);
    assert.deepStrictEqual(t.shape, want.shape, name);
    if (want.values) assert.deepStrictEqual(await values(t), want.values, name);
  }
  assert.strictEqual(byName(m, "['words']").readable, false);
  assert.strictEqual(byName(m, "['fortran']")._loc.fortran, true);
  await m.close();
});

test('np.save of a dict: plain values are rows of a read-only tree, scalars exact', async () => {
  const m = await open('dict_weights.npy');
  for (const [p, v] of Object.entries(expected['dict_weights.npy'].rows)) assert.strictEqual(row(m, p) && row(m, p).value, v, p);
  assert.strictEqual(row(m, '(top level)').type, 'dict');
  assert.strictEqual(row(m, "['layers'][0]").depth, 2);
  assert.match(row(m, "['blob']").value, /^b'\\x00ab' \(3 bytes\)$/);
  assert.match(row(m, "['words']").value, /"alpha", "beta"/);
  const arr = row(m, "['layers'][0]['bias']");
  assert.strictEqual(m.tensors[arr.tensor].name, "['layers'][0]['bias']");
  // Nothing outside numpy's rebuild path and plain containers was referenced, so no warning.
  assert.ok(!m.warnings.some(w => /^Risky/.test(w)), m.warnings.join('\n'));
  assert.ok(m.metadata.some(x => x.key === 'python name: numpy.core.multiarray._reconstruct' && /^allowed/.test(x.value)));
  const pub = publicModel(m, 'dict_weights.npy');
  assert.strictEqual(pub.objects.length, m.objects.length);
  assert.ok(!JSON.stringify(pub.objects).includes('_loc'));
  await m.close();
});

test('nested arrays get stats and percentiles like any other tensor', async () => {
  const m = await open('dict_weights.npy');
  const r = await sampleTensor(byName(m, "['weights']"));
  assert.strictEqual(r.stats.min, 0);
  assert.strictEqual(r.stats.max, 11);
  assert.strictEqual(r.pct.exact, true);
  assert.strictEqual(r.pct.values.find(v => v.label === 'median').value, 5.5);
  const b = await sampleTensor(byName(m, "['layers'][0]['bias']"));
  assert.deepStrictEqual(b.pct.values.map(v => v.text).filter((_, i) => i === 0 || i === 6).length, 2);
  assert.strictEqual(b.pct.values.find(v => v.label === 'median').text, '5');
  assert.strictEqual(b.pct.bigint, true);
  await m.close();
});

test('a list of arrays, a 2-D object grid and protocol 2 / 5 pickles', async () => {
  let m = await open('list_arrays.npy');
  assert.deepStrictEqual(m.tensors.map(t => t.name), expected['list_arrays.npy'].names);
  assert.deepStrictEqual(m.tensors.map(t => t.shape), expected['list_arrays.npy'].shapes);
  await m.close();
  m = await open('grid_2x2.npy');
  assert.deepStrictEqual(m.tensors.map(t => t.name), expected['grid_2x2.npy'].names);
  assert.deepStrictEqual(await values(m.tensors[3]), [3, 3]);
  await m.close();
  for (const f of ['proto2.npy', 'proto5.npy']) {
    m = await open(f);
    assert.deepStrictEqual(m.tensors.map(t => t.name), expected[f].names, f);
    assert.deepStrictEqual(await values(m.tensors[0]), expected[f].values[0], f);
    assert.deepStrictEqual(await values(m.tensors[1]), expected[f].values[1], f);
    assert.strictEqual(m.tensors[1]._loc.littleEndian, false, 'big-endian >i4 kept');
    await m.close();
  }
});

test('an object array with no arrays inside stays listed and its values go to the tree', async () => {
  const m = await open('strings_only.npy');
  assert.strictEqual(m.tensors.length, 1);
  assert.strictEqual(m.tensors[0].readable, false);
  assert.match(m.tensors[0].note, /Objects tab/);
  for (const [p, v] of Object.entries(expected['strings_only.npy'].rows)) assert.strictEqual(row(m, p).value, v, p);
  assert.match(row(m, '[5]').value, /\["x", 1\]/);
  await m.close();
});

test('npz: a pickled dict entry is decoded next to normal arrays, stored or compressed', async () => {
  let m = await open('with_dict.npz');
  assert.deepStrictEqual(m.tensors.map(t => t.name), expected['with_dict.npz'].names);
  const t = byName(m, "meta['mean']");
  assert.strictEqual(m.objects.find(r => r.tensor !== undefined).tensor, m.tensors.indexOf(t));
  assert.ok(Math.abs((await values(t))[2] - 0.3) < 1e-6);
  assert.strictEqual(row(m, "meta['classes']").value, '2 items: ["cat", "dog"]');
  await m.close();
  m = await open('with_dict_compressed.npz');
  assert.deepStrictEqual(m.tensors.map(t => t.name), ["meta['mean']"]);
  await m.close();
});

test('hostile: os.system and eval REDUCE inside an object array run nothing and are flagged', async () => {
  const marker = expected['evil_reduce.npy'].marker;
  try { fs.unlinkSync(marker); } catch { /* not there */ }
  for (const f of ['evil_reduce.npy', 'evil_in_npz.npz']) {
    const m = await open(f);
    assert.ok(m.warnings.some(w => /^Risky: .*os\.system, builtins\.eval/.test(w)), f + ': ' + m.warnings.join(' | '));
    assert.ok(m.metadata.some(x => /python name: os\.system$/.test(x.key) && /^RISKY/.test(x.value)), f);
    assert.ok(m.objects.some(r => r.type === 'os.system' && /not run/.test(r.value)), f);
    const arr = m.tensors.find(t => /\[0\]$/.test(t.name));
    assert.deepStrictEqual(await values(arr), [1, 2, 3], f);
    await m.close();
  }
  assert.strictEqual(fs.existsSync(marker), false, 'the payload must never run');
});

test('hostile: huge declared shapes and cut-off array data are listed, not read', async () => {
  const m = await open('huge_shape.npy');
  assert.strictEqual(m.tensors[0].readable, false);
  assert.match(m.tensors[0].note, /too large/);
  assert.strictEqual(m.tensors[1].readable, false);
  assert.match(m.tensors[1].note, /cut off/);
  await m.close();
});

test('hostile: 100,000 nested lists, cyclic lists and 2^60 shared paths finish fast', async () => {
  const t0 = Date.now();
  let m = await open('deep_nesting.npy');
  assert.ok(m.warnings.some(w => /deeper than 64 levels/.test(w)));
  assert.ok(m.objects.length <= 70);
  await m.close();
  m = await open('cyclic.npy');
  assert.strictEqual(row(m, '[0][0]').type, 'same object');
  assert.deepStrictEqual(m.tensors.map(t => t.name), ['[1]']);
  await m.close();
  m = await open('shared_bomb.npy');
  assert.strictEqual(m.tensors.length, 1);
  assert.ok(m.objects.length < 200, String(m.objects.length));
  await m.close();
  assert.ok(Date.now() - t0 < 5000, `took ${Date.now() - t0} ms`);
});

test('hostile: a truncated object pickle gives a message, not a crash', async () => {
  const m = await open('truncated_object.npy');
  assert.strictEqual(m.tensors.length, 1);
  assert.strictEqual(m.tensors[0].readable, false);
  assert.match(m.tensors[0].note, /could not be decoded/);
  await m.close();
});

test('hostile: a pickle with more opcodes than the cap stops with a message', async () => {
  const header = "{'descr': '|O', 'fortran_order': False, 'shape': (1,), }";
  const hb = Buffer.from(header + ' '.repeat(64 - ((10 + header.length + 1) % 64)) + '\n', 'latin1');
  const n = 50000;
  const payload = Buffer.concat([Buffer.from([0x80, 0x02, 0x5d, 0x28]), Buffer.alloc(n, 0x4e), Buffer.from('e.', 'latin1')]); // [None] * n
  const file = Buffer.concat([Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, hb.length & 0xff, hb.length >> 8]), hb, payload]);
  const old = pyobj.LIMITS.ops;
  pyobj.LIMITS.ops = 10000;
  try {
    const m = await openTensorFile(new BufferSource(file), 'many.npy');
    assert.match(m.tensors[0].note, /more than 10,000 opcodes/);
  } finally { pyobj.LIMITS.ops = old; }
  const m2 = await openTensorFile(new BufferSource(file), 'many.npy');
  assert.match(row(m2, '(top level)').value, /^50,000 items/);
});

test('keys and scalars read like Python', () => {
  assert.strictEqual(pyobj.keyText('w'), "['w']");
  assert.strictEqual(pyobj.keyText("it's"), "['it\\'s']");
  assert.strictEqual(pyobj.keyText(3), '[3]');
  assert.strictEqual(pyobj.keyText(null), '[None]');
});
