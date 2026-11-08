'use strict';
// Object arrays in .npy and .npz files: np.save of a dict, a list of arrays, strings or any Python value
// (allow_pickle=True). numpy writes these as a pickle of the whole array after the .npy header.
//
// The pickle is decoded by the same stack machine as PyTorch files (./pickle/unpickler.js): no Python
// name is ever looked up or called, each one becomes inert data. numpy's own rebuild path
// (_reconstruct + BUILD, dtype + BUILD, scalar, _frombuffer) turns back into arrays, and every
// numeric array found anywhere in the object becomes a tensor named by its Python path, such as
// ['layers'][2]['bias']. Plain values (str, int, float, bool, None, bytes, lists, tuples, dicts)
// become rows of a read-only tree. Any other name stays a placeholder and gets a safety rating.
const { unpickle } = require('./pickle/unpickler');
const V = require('./pickle/values');
const { rate, norm } = require('./pickle/safety');
const { fromNumpyDescr, decodePlain } = require('./dtypes');
const { BufferSource } = require('./source');

const LIMITS = {
  pickleBytes: 512 * 1048576, // same cap as data.pkl in PyTorch files
  ops: 8000000, // opcodes per pickle: a hostile file cannot build an endless list
  depth: 64, // nesting levels walked
  nodes: 200000, // values visited per pickle (shared values count once)
  rows: 5000, // tree rows kept per file
  tensors: 200000,
};

// Names that rebuild numpy arrays, dtypes and scalars, and plain containers. Nothing here runs: the
// list only decides which names count as expected in a numpy object array.
const ALLOWED = [
  /^numpy\.core\.multiarray\.(_reconstruct|scalar)$/, /^numpy\.(ndarray|dtype)$/, /^numpy\.core\.numeric\._frombuffer$/,
  /^builtins\.(set|frozenset|dict|list|tuple|int|float|str|bool|bytes|bytearray|complex|slice)$/,
  /^_codecs\.encode$/, /^copyreg\._reconstructor$/, /^collections\.OrderedDict$/,
];
const allowed = q => ALLOWED.some(re => re.test(q));
const ARRAY_CLASSES = /^numpy\.(ndarray|matrix|memmap|ma\.core\.MaskedArray|recarray|core\.records\.recarray)$/;

const prod = a => a.reduce((x, y) => x * y, 1);
const intOf = v => (typeof v === 'number' ? v : typeof v === 'bigint' ? Number(v) : typeof v === 'boolean' ? (v ? 1 : 0) : NaN);

/** Python-style text for a dict key or index: ['name'], [3], [(1, 2)]. */
function keyText(k) {
  if (typeof k === 'string') return `['${k.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n').slice(0, 200)}']`;
  if (typeof k === 'number' || typeof k === 'bigint') return `[${k}]`;
  if (typeof k === 'boolean') return `[${k ? 'True' : 'False'}]`;
  if (k === null) return '[None]';
  return `[${short(k, 60)}]`;
}

/** Short Python-like text for a value, with a cap on the work for shared or deep values. */
function short(v, budget = 200, cost = { n: 0 }, depth = 0) {
  if (++cost.n > 400 || depth > 6) return '...';
  if (v === null) return 'None';
  if (v === undefined) return '?';
  if (typeof v === 'string') return v.length > budget ? JSON.stringify(v.slice(0, budget)) + '...' : JSON.stringify(v);
  if (typeof v === 'number') return Number.isInteger(v) || !Number.isFinite(v) ? (Number.isFinite(v) ? String(v) : v !== v ? 'nan' : v > 0 ? 'inf' : '-inf') : String(v);
  if (typeof v === 'bigint') return v.toString();
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (v instanceof V.PyFloat) return v.value.toFixed(1);
  if (v instanceof V.PyComplex) return `(${v.re}${v.im < 0 ? '-' : '+'}${Math.abs(v.im)}j)`;
  if (v instanceof V.PyGlobal) return v.qualname;
  if (v instanceof V.PyText) return v.text;
  if (v instanceof V.PyBytes) return bytesText(v.data, v.kind);
  if (v instanceof V.NpScalar) return scalarText(v);
  if (v instanceof V.NdArray) return `array ${v.dtype instanceof V.NpDtype ? v.dtype.code : '?'} ${shapeText(v.shape)}`;
  if (Array.isArray(v)) {
    const tup = V.isTuple(v);
    let s = tup ? '(' : '[';
    for (let i = 0; i < v.length; i++) {
      const p = short(v[i], 40, cost, depth + 1);
      if (s.length + p.length > budget) { s += `... ${v.length} items`; break; }
      s += (i ? ', ' : '') + p;
    }
    return s + (tup && v.length === 1 ? ',)' : tup ? ')' : ']');
  }
  if (v instanceof V.PyDict) return `${v.typeName} with ${v.size} keys`;
  if (v instanceof V.PySet) return `${v.typeName} of ${v.items.length}`;
  if (v instanceof V.PyObject) return `${v.className}(...) (not run)`;
  if (v instanceof V.PyPersistent) return 'persistent id';
  return Object.prototype.toString.call(v);
}

function bytesText(data, kind = 'bytes') {
  let s = '';
  const n = Math.min(data.length, 48);
  for (let i = 0; i < n; i++) {
    const c = data[i];
    s += c === 0x27 || c === 0x5c ? '\\' + String.fromCharCode(c) : c >= 0x20 && c < 0x7f ? String.fromCharCode(c) : '\\x' + c.toString(16).padStart(2, '0');
  }
  return `${kind === 'bytearray' ? 'bytearray(' : ''}b'${s}${data.length > n ? '...' : ''}'${kind === 'bytearray' ? ')' : ''} (${data.length} bytes)`;
}

const shapeText = s => (Array.isArray(s) ? (s.length === 1 ? `(${s[0]},)` : `(${s.join(', ')})`) : '(?)');

/** numpy dtype from the pickle to a .npy descr string such as '<f8', or null for a structured dtype. */
function descrOf(dt) {
  if (!(dt instanceof V.NpDtype)) return null;
  if (dt.fields || dt.subarray) return null;
  const e = dt.endian === '>' ? '>' : dt.endian === '|' ? '|' : '<';
  return e + dt.code;
}

/** A numpy scalar (np.float64(2.5), np.int64(7)) as text. int64 / uint64 stay exact. */
function scalarText(s) {
  const descr = descrOf(s.dtype);
  const type = descr ? fromNumpyDescr(descr) : null;
  const raw = s.raw instanceof Uint8Array ? s.raw : s.raw instanceof V.PyBytes ? s.raw.data : null;
  const name = type ? `numpy.${type.label}` : 'numpy scalar';
  if (!type || !type.kind || !raw || raw.length < type.size) return `${name} (not decoded)`;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  let text;
  if (type.kind === 'i64') text = dv.getBigInt64(0, type.littleEndian).toString();
  else if (type.kind === 'u64') text = dv.getBigUint64(0, type.littleEndian).toString();
  else if (type.kind === 'bool') text = raw[0] ? 'True' : 'False';
  else if (type.kind === 'c64' || type.kind === 'c128') {
    const w = type.size / 2, f = w === 4 ? (o => dv.getFloat32(o, type.littleEndian)) : (o => dv.getFloat64(o, type.littleEndian));
    const re = f(0), im = f(w);
    text = `(${re}${im < 0 ? '-' : '+'}${Math.abs(im)}j)`;
  } else text = String(decodePlain(type.kind, raw, 1, type.littleEndian)[0]);
  return `${text} (${name})`;
}

/** First items of a text or bytes array, for the tree. */
function textItems(arr, type, descr) {
  const raw = arr.raw;
  if (!raw || !type.size) return '';
  const n = Math.min(prod(arr.shape), 20, Math.floor(raw.length / type.size));
  const out = [];
  const isU = /U/.test(descr);
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  for (let i = 0; i < n; i++) {
    let s = '';
    if (isU) {
      for (let c = 0; c < type.size / 4; c++) { const cp = dv.getUint32(i * type.size + c * 4, type.littleEndian); if (!cp) break; s += cp <= 0x10ffff ? String.fromCodePoint(cp) : '?'; }
      out.push(JSON.stringify(s.slice(0, 80)));
    } else {
      for (let c = 0; c < type.size; c++) { const b = raw[i * type.size + c]; if (!b) break; s += b >= 0x20 && b < 0x7f ? String.fromCharCode(b) : '\\x' + b.toString(16).padStart(2, '0'); }
      out.push(`b'${s.slice(0, 80)}'`);
    }
  }
  return `[${out.join(', ')}${prod(arr.shape) > n ? ', ...' : ''}]`;
}

/**
 * Turn the decoded pickle of an object array into tensors and tree rows.
 * @param root the decoded value (normally a numpy.ndarray with dtype object)
 * @param base name prefix: '' for .npy, the entry name for .npz
 */
function collectObjects(root, base) {
  const tensors = [], rows = [], warnings = [];
  const seen = new Map();
  let nodes = 0, rowsSkipped = 0, deepWarned = false, nodeWarned = false;
  const addRow = (path, depth, type, value, tensor) => {
    if (rows.length >= LIMITS.rows) { rowsSkipped++; return; }
    const r = { path: path || '(top level)', depth, type, value: String(value).slice(0, 2000) };
    if (tensor !== undefined) r.tensor = tensor;
    rows.push(r);
  };
  const isLeaf = v => v === null || typeof v !== 'object' || v instanceof V.PyFloat || v instanceof V.PyText || v instanceof V.PyBytes
    || v instanceof V.PyGlobal || v instanceof V.PyComplex || v instanceof V.NpScalar || v instanceof V.PyPersistent || v instanceof V.PyExtension || v instanceof V.PyOutOfBand || v instanceof V.PySlice;
  const leafType = v => (v === null ? 'None' : typeof v === 'string' ? 'str' : typeof v === 'number' ? (Number.isInteger(v) ? 'int' : 'float')
    : typeof v === 'bigint' ? 'int' : typeof v === 'boolean' ? 'bool' : v instanceof V.PyFloat ? 'float' : v instanceof V.PyBytes ? v.kind
      : v instanceof V.PyComplex ? 'complex' : v instanceof V.PyText ? v.typeName : v instanceof V.PyGlobal ? 'name (not run)' : v instanceof V.NpScalar ? 'numpy scalar' : 'value');

  function arrayTensor(arr, name) {
    const shape = Array.isArray(arr.shape) ? arr.shape : null;
    const descr = descrOf(arr.dtype);
    const base = { name, shape: shape || [], numel: 0, nbytes: null, offset: null, readable: false };
    if (!shape || !shape.every(n => Number.isInteger(n) && n >= 0)) return { ...base, dtype: '?', note: 'A numpy array without a valid shape in the pickle.' };
    const numel = prod(shape);
    if (!(arr.dtype instanceof V.NpDtype)) return { ...base, dtype: '?', numel, note: 'A numpy array without a dtype in the pickle.' };
    const type = fromNumpyDescr(descr === null ? { structured: true } : descr);
    const t = { ...base, dtype: type.label, rawDtype: descr || 'structured', numel, nbytes: Number.isSafeInteger(numel * type.size) ? numel * type.size : null, note: type.reason };
    if (!Number.isSafeInteger(numel) || t.nbytes === null) { t.note = `The shape ${shapeText(shape)} is too large to be real.`; return t; }
    const raw = arr.raw;
    if (!raw) { t.note = 'The array has no data in the pickle.'; return t; }
    if (raw.length < t.nbytes) { t.note = `The array data is cut off: the shape and dtype need ${t.nbytes.toLocaleString('en-US')} bytes and the pickle holds ${raw.length.toLocaleString('en-US')}.`; return t; }
    if (!type.readable || !type.kind) return t;
    const fortran = !!arr.fortran && shape.length >= 2;
    t._loc = { source: new BufferSource(raw.subarray(0, t.nbytes)), dataOffset: 0, kind: type.kind, elemSize: type.size, littleEndian: type.littleEndian, fortran };
    t.readable = numel > 0;
    const notes = [];
    if (type.reason) notes.push(type.reason);
    if (fortran) notes.push('Fortran (column-major) order.');
    if (arr.clsName && arr.clsName !== 'numpy.ndarray') notes.push(`Saved as ${arr.clsName}; shown as a plain array.`);
    t.note = notes.join(' ') || undefined;
    return t;
  }

  function addTensor(t, path, depth) {
    if (tensors.length >= LIMITS.tensors) { if (tensors.length === LIMITS.tensors) warnings.push(`Only the first ${LIMITS.tensors} arrays are listed.`); return; }
    tensors.push(t);
    addRow(path, depth, 'ndarray', `${t.dtype} ${shapeText(t.shape)}${t.readable ? '' : t.numel === 0 ? ' (empty)' : ' (not plotted)'}`, tensors.length - 1);
  }

  function walk(v, path, depth) {
    if (++nodes > LIMITS.nodes) { if (!nodeWarned) { nodeWarned = true; warnings.push(`The object holds more than ${LIMITS.nodes.toLocaleString('en-US')} values. Only the first ones are listed.`); } return; }
    if (depth > LIMITS.depth) { if (!deepWarned) { deepWarned = true; warnings.push(`Values nested deeper than ${LIMITS.depth} levels are not listed (first at ${path.length > 80 ? path.slice(0, 80) + '...' : path}).`); } return; }
    if (isLeaf(v)) { addRow(path, depth, leafType(v), short(v, 400)); return; }
    if (seen.has(v)) { addRow(path, depth, 'same object', `same as ${seen.get(v) || '(top level)'}`); return; }
    seen.set(v, path);
    if (v instanceof V.NdArray) {
      if (Array.isArray(v.objects)) {
        const shape = Array.isArray(v.shape) ? v.shape : [v.objects.length];
        const items = v.objects;
        if (shape.length === 0) { walk(items[0], path, depth); return; }
        addRow(path, depth, 'object array', `${shapeText(shape)}, ${items.length.toLocaleString('en-US')} items`);
        const valid = shape.every(n => Number.isInteger(n) && n >= 0) && prod(shape) === items.length;
        for (let i = 0; i < items.length; i++) {
          if (nodes >= LIMITS.nodes) { walk(null, path, depth + 1); break; }
          walk(items[i], path + (valid && shape.length > 1 ? `[${unravel(i, shape).join(', ')}]` : `[${i}]`), depth + 1);
        }
        return;
      }
      if (!ARRAY_CLASSES.test(v.clsName || '')) { addRow(path, depth, v.clsName || 'array', 'an array-like object (not run)'); return; }
      const t = arrayTensor(v, path || base || 'array');
      addTensor(t, path, depth);
      if (!t.readable && /^(str|bytes)\d+$/.test(t.dtype) && rows.length && rows[rows.length - 1].tensor === tensors.length - 1) {
        const items = textItems(v, fromNumpyDescr(t.rawDtype), t.rawDtype);
        if (items) rows[rows.length - 1].value += ' ' + items;
      }
      return;
    }
    if (v instanceof V.PyDict) {
      addRow(path, depth, v.typeName, `${v.size.toLocaleString('en-US')} key${v.size === 1 ? '' : 's'}`);
      for (let i = 0; i < v.keys.length; i++) {
        if (nodes >= LIMITS.nodes) { walk(null, path, depth + 1); break; }
        walk(v.values[i], path + keyText(v.keys[i]), depth + 1);
      }
      return;
    }
    if (Array.isArray(v)) {
      const kind = V.isTuple(v) ? 'tuple' : (v.typeName || 'list');
      if (v.every(isLeaf)) { addRow(path, depth, kind, `${v.length.toLocaleString('en-US')} items: ${short(v, 400)}`); return; }
      addRow(path, depth, kind, `${v.length.toLocaleString('en-US')} items`);
      for (let i = 0; i < v.length; i++) {
        if (nodes >= LIMITS.nodes) { walk(null, path, depth + 1); break; }
        walk(v[i], `${path}[${i}]`, depth + 1);
      }
      return;
    }
    if (v instanceof V.PySet) { addRow(path, depth, v.typeName, `${v.items.length.toLocaleString('en-US')} items: ${short(v.items, 400)}`); return; }
    if (v instanceof V.NpDtype) { addRow(path, depth, 'numpy.dtype', descrOf(v) || 'structured'); return; }
    if (v instanceof V.PyObject) {
      const cls = v.className;
      const args = Array.isArray(v.args) ? v.args : [];
      const cost = { n: 0 };
      addRow(path, depth, cls, `object, not run. Python would call ${cls}(${args.map(a => short(a, 200, cost, 1)).join(', ')})`);
      const state = v.state instanceof V.PyDict ? v.state : Array.isArray(v.state) ? v.state.find(x => x instanceof V.PyDict) : null;
      if (state) state.keys.forEach((k, i) => walk(state.values[i], `${path}.${typeof k === 'string' ? k : short(k, 40)}`, depth + 1));
      if (v.dictItems) v.dictItems.keys.forEach((k, i) => walk(v.dictItems.values[i], path + keyText(k), depth + 1));
      if (v.listItems) v.listItems.forEach((x, i) => walk(x, `${path}[${i}]`, depth + 1));
      return;
    }
    addRow(path, depth, 'value', short(v, 400));
  }

  walk(root, base, 0);
  if (rowsSkipped) warnings.push(`Only the first ${LIMITS.rows.toLocaleString('en-US')} values are listed in the Objects tab (${rowsSkipped.toLocaleString('en-US')} more).`);
  return { tensors, rows, warnings };
}

function unravel(i, shape) {
  const out = new Array(shape.length);
  for (let k = shape.length - 1; k >= 0; k--) { out[k] = i % shape[k]; i = Math.floor(i / shape[k]); }
  return out;
}

/** Rate every Python name the pickle imports. */
function namesOf(events) {
  const names = new Map();
  for (const e of events) {
    if (e.op !== 'GLOBAL' && e.op !== 'STACK_GLOBAL' && e.op !== 'INST') continue;
    const q = norm(e.target);
    if (!names.has(q)) names.set(q, { name: q, count: 0, ok: allowed(q), rating: rate(q) });
    names.get(q).count++;
  }
  const all = [...names.values()];
  return { all, danger: all.filter(n => !n.ok && n.rating.level === 'danger'), other: all.filter(n => !n.ok && n.rating.level !== 'danger') };
}

/**
 * Read the pickle of one object array (the bytes after the .npy header) and turn it into tensors,
 * tree rows, metadata rows (one per Python name, with its rating) and warnings.
 * @param source byte source of the .npy (or .npz entry)
 * @param h the parsed .npy header
 * @param base name prefix ('' for .npy, the entry name for .npz)
 */
async function openObjectArray(source, h, base) {
  const len = source.size - h.dataOffset;
  const label = base || 'array';
  const fail = (note, extra = {}) => ({ tensors: [], rows: [], metadata: [], warnings: [], note, ...extra });
  if (len <= 0) return fail('Object array (Python pickle) with no data after the header. The file may be truncated.');
  if (len > LIMITS.pickleBytes) return fail(`Object array (Python pickle) of ${(len / 1048576).toFixed(0)} MB, more than the ${LIMITS.pickleBytes / 1048576} MB that Tensor Viewer decodes.`);
  const bytes = await source.read(h.dataOffset, len);
  const decoded = unpickle(bytes, { maxEvents: 20000, maxOps: LIMITS.ops });
  const s = namesOf(decoded.events);
  const metadata = s.all.map(n => ({ key: `${base ? base + ': ' : ''}python name: ${n.name}`, value: n.ok ? 'allowed: numpy array rebuild or a plain container' : `${n.rating.level === 'danger' ? 'RISKY' : 'not run'}: ${n.rating.why}` }));
  const warnings = [];
  if (s.danger.length) {
    warnings.push(`Risky: ${base ? `the entry ${base} ` : 'this file '}refers to ${s.danger.map(d => d.name).slice(0, 4).join(', ')}${s.danger.length > 4 ? ` and ${s.danger.length - 4} more` : ''}. np.load(allow_pickle=True) would call ${s.danger.length === 1 ? 'it' : 'them'}. Tensor Viewer did not. Do not load this file in Python unless you trust where it came from.`);
  } else if (s.other.length) {
    warnings.push(`${base ? `The entry ${base}` : 'This file'} holds Python objects of ${s.other.map(n => n.name).slice(0, 3).join(', ')}${s.other.length > 3 ? ` and ${s.other.length - 3} more` : ''}. They are listed, not run. np.load needs those packages to load it.`);
  }
  if (decoded.error) {
    warnings.push(`The pickle in ${label} could not be read to the end: ${decoded.error.message}`);
    return { tensors: [], rows: [], metadata, warnings, note: `Object array (Python pickle). It could not be decoded: ${decoded.error.message}`, events: decoded.eventCount };
  }
  const out = collectObjects(decoded.value, base);
  return { tensors: out.tensors, rows: out.rows, metadata, warnings: [...warnings, ...out.warnings], note: null, events: decoded.eventCount, trailing: decoded.trailing };
}

module.exports = { openObjectArray, collectObjects, keyText, short, scalarText, ALLOWED, allowed, LIMITS };
