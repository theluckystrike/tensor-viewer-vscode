'use strict';
// Opens PyTorch files (.pt, .pth, .ckpt, pytorch_model.bin) without running pickle.
//
// torch.save writes a zip: <name>/data.pkl holds the object tree as a pickle, and each tensor's
// bytes sit in <name>/data/<key>. Files saved before PyTorch 1.6 (or with
// _use_new_zipfile_serialization=False) are a row of pickles followed by the raw storages.
//
// The pickle is decoded by the PKL Viewer stack machine (./pickle/unpickler.js), which never looks
// up or calls a Python name: every class and function becomes inert data. This file then turns a
// short allow-list of tensor rebuilders back into tensors. Everything else stays a placeholder and
// is listed in the Metadata tab with a safety rating.
const { readCentralDirectory, entrySource } = require('./zip');
const { unpickle } = require('./pickle/unpickler');
const V = require('./pickle/values');
const { rate, norm } = require('./pickle/safety');
const { PLAIN } = require('./dtypes');

const MAX_PICKLE_BYTES = 512 * 1048576;
const MAX_GATHER_BYTES = 256 * 1048576;
const MAX_TENSORS = 200000;
const MAX_METADATA = 2000;
const MAX_DEPTH = 64;
const SHORT_NODES = 400;
const SHORT_DEPTH = 6;

// torch.<Name>Storage -> element kind. Quantized storages hold the stored integers.
const STORAGE_KIND = {
  FloatStorage: 'f32', DoubleStorage: 'f64', HalfStorage: 'f16', BFloat16Storage: 'bf16', LongStorage: 'i64', IntStorage: 'i32',
  ShortStorage: 'i16', CharStorage: 'i8', ByteStorage: 'u8', BoolStorage: 'bool', ComplexFloatStorage: 'c64', ComplexDoubleStorage: 'c128',
  QUInt8Storage: 'u8', QInt8Storage: 'i8', QInt32Storage: 'i32', QUInt4x2Storage: 'u8', QUInt2x4Storage: 'u8',
  UInt16Storage: 'u16', UInt32Storage: 'u32', UInt64Storage: 'u64',
};
// torch.<dtype> (from _rebuild_tensor_v3 and torch.storage.UntypedStorage) -> element kind.
const DTYPE_KIND = {
  float32: 'f32', float: 'f32', float64: 'f64', double: 'f64', float16: 'f16', half: 'f16', bfloat16: 'bf16', int64: 'i64', long: 'i64',
  int32: 'i32', int: 'i32', int16: 'i16', short: 'i16', int8: 'i8', uint8: 'u8', bool: 'bool', complex64: 'c64', cfloat: 'c64',
  complex128: 'c128', cdouble: 'c128', uint16: 'u16', uint32: 'u32', uint64: 'u64', float8_e4m3fn: 'f8e4m3', float8_e5m2: 'f8e5m2',
};
const TORCH_NAME = { f32: 'float32', f64: 'float64', f16: 'float16', bf16: 'bfloat16', i64: 'int64', i32: 'int32', i16: 'int16', i8: 'int8', u8: 'uint8',
  bool: 'bool', c64: 'complex64', c128: 'complex128', u16: 'uint16', u32: 'uint32', u64: 'uint64', f8e4m3: 'float8_e4m3fn', f8e5m2: 'float8_e5m2' };

// The names this reader understands. torch.load(weights_only=True) accepts roughly the same set.
const ALLOWED = [
  /^torch\._utils\._rebuild_(tensor|tensor_v2|tensor_v3|parameter|parameter_with_state|qtensor|sparse_tensor|sparse_csr_tensor|meta_tensor_no_storage|wrapper_subclass|nested_tensor|device_tensor_from_numpy)$/,
  /^torch\.\w+Storage$/, /^torch\.storage\.(UntypedStorage|TypedStorage|_load_from_bytes)$/, /^torch\.(Size|device|dtype|layout|memory_format|qscheme)$/,
  /^torch\.(per_tensor_affine|per_channel_affine|per_tensor_symmetric|per_channel_symmetric|per_channel_affine_float_qparams|sparse_coo|sparse_csr|strided|contiguous_format|channels_last|preserve_format)$/,
  /^torch\.serialization\._get_layout$/, /^torch\.nn\.parameter\.Parameter$/, /^torch\.jit\._pickle\.(build_\w+|restore_type_tag)$/,
  /^collections\.OrderedDict$/, /^builtins\.(set|frozenset|dict|list|tuple|int|float|str|bool|slice|complex|bytes|bytearray)$/,
  /^_codecs\.encode$/, /^copyreg\._reconstructor$/,
];
for (const k of Object.keys(DTYPE_KIND)) ALLOWED.push(new RegExp(`^torch\\.${k}$`));
const allowed = q => ALLOWED.some(re => re.test(q));

const qualname = g => (g instanceof V.PyGlobal ? norm(g.qualname) : null);
const intOf = v => (typeof v === 'number' ? v : typeof v === 'bigint' ? Number(v) : typeof v === 'boolean' ? (v ? 1 : 0) : v instanceof V.PyFloat ? v.value : NaN);
const prod = a => a.reduce((x, y) => x * y, 1);

function isTorchFileName(name) {
  return /\.(pt|pth|ckpt|pkl\.pt|tar\.pt)$/i.test(name || '') || /(^|[\\/])(pytorch_model|training_args|optimizer|scheduler|rng_state)[^\\/]*\.bin$/i.test(name || '') || /\.pt\.bin$/i.test(name || '');
}

/** A tuple of ints, or torch.Size(tuple). */
function intList(v) {
  if (v instanceof V.PyObject && qualname(v.cls) === 'torch.Size' && v.args && Array.isArray(v.args[0])) v = v.args[0];
  if (!Array.isArray(v)) return null;
  const out = v.map(intOf);
  return out.every(n => Number.isInteger(n) && n >= 0) ? out : null;
}

/** Element kind from a storage type (torch.FloatStorage) or a dtype (torch.float8_e4m3fn). */
function kindOf(storageType, dtype) {
  const d = qualname(dtype);
  if (d && d.startsWith('torch.') && DTYPE_KIND[d.slice(6)]) return DTYPE_KIND[d.slice(6)];
  const s = qualname(storageType);
  if (s && s.startsWith('torch.') && STORAGE_KIND[s.slice(6)]) return STORAGE_KIND[s.slice(6)];
  return null;
}

/** A Source that gathers a strided view into row-major order the first time it is read. */
class GatherSource {
  constructor(storage, offset, shape, stride, elemSize) {
    Object.assign(this, { storage, offset, shape, stride, elemSize });
    this.size = prod(shape) * elemSize; this.random = true; this.buf = null;
  }
  async load() {
    if (this.buf) return;
    const { shape, stride, elemSize, offset } = this;
    let lo = offset, hi = offset;
    shape.forEach((n, k) => { if (n > 0) { const d = (n - 1) * stride[k]; if (d < 0) lo += d; else hi += d; } });
    const span = await this.storage.read(lo * elemSize, (hi - lo + 1) * elemSize);
    const out = new Uint8Array(this.size);
    const n = shape.length, idx = new Array(n).fill(0);
    const total = prod(shape);
    let src = offset - lo;
    for (let i = 0; i < total; i++) {
      out.set(span.subarray(src * elemSize, src * elemSize + elemSize), i * elemSize);
      for (let k = n - 1; k >= 0; k--) {
        idx[k]++; src += stride[k];
        if (idx[k] < shape[k]) break;
        src -= stride[k] * shape[k]; idx[k] = 0;
      }
    }
    this.buf = out;
  }
  async read(offset, length) {
    await this.load();
    if (offset < 0 || offset + length > this.size) throw new RangeError('Read past the end of the tensor.');
    return this.buf.subarray(offset, offset + length);
  }
  async close() { this.buf = null; }
}

const rowMajor = shape => { const s = new Array(shape.length); let m = 1; for (let k = shape.length - 1; k >= 0; k--) { s[k] = m; m *= shape[k]; } return s; };
const colMajor = shape => { const s = new Array(shape.length); let m = 1; for (let k = 0; k < shape.length; k++) { s[k] = m; m *= shape[k]; } return s; };
const sameStrides = (shape, a, b) => shape.every((n, k) => n <= 1 || a[k] === b[k]);

/**
 * Walks the decoded object tree and collects tensors and plain values.
 * storages(key) returns {source, start, numel?} for a storage key or null.
 */
function collect(root, ctx) {
  const tensors = [], metadata = [], warnings = [];
  const seen = new Map(); // object -> first path, so shared and cyclic references are listed once
  let skippedMeta = 0;

  const addMeta = (key, value) => { if (metadata.length < MAX_METADATA) metadata.push({ key: key || '(top level)', value }); else skippedMeta++; };

  function storageOf(pid) {
    // ('storage', storage_type, key, location, numel[, view_metadata])
    if (!(pid instanceof V.PyPersistent)) return null;
    const t = pid.pid;
    if (!Array.isArray(t) || t[0] !== 'storage') return null;
    return { type: t[1], key: String(t[2]), location: typeof t[3] === 'string' ? t[3] : '', numel: intOf(t[4]), view: t[5] };
  }

  function tensorInfo(o, name) {
    const fn = qualname(o.cls);
    const a = o.args || [];
    if (fn === 'torch._utils._rebuild_parameter' || fn === 'torch._utils._rebuild_parameter_with_state') {
      const inner = a[0] instanceof V.PyObject ? tensorInfo(a[0], name) : null;
      if (inner) inner.param = true;
      return inner;
    }
    if (fn === 'torch._utils._rebuild_tensor' || fn === 'torch._utils._rebuild_tensor_v2' || fn === 'torch._utils._rebuild_tensor_v3') {
      // v2: (storage, storage_offset, size, stride, requires_grad, backward_hooks[, metadata]); v3 adds dtype after backward_hooks.
      const st = storageOf(a[0]);
      return { name, st, offset: intOf(a[1]), shape: intList(a[2]), stride: intList(a[3]) || null, dtype: fn.endsWith('v3') ? a[6] : null };
    }
    if (fn === 'torch._utils._rebuild_qtensor') {
      // (storage, storage_offset, size, stride, quantizer_params, requires_grad, backward_hooks)
      const st = storageOf(a[0]);
      const qp = Array.isArray(a[4]) ? a[4] : [];
      const scheme = qualname(qp[0]) || '';
      let note = 'Quantized tensor: the values shown are the stored integers.';
      if (/per_tensor/.test(scheme) && typeof qp[1] !== 'object') note += ` Real value = (q - ${intOf(qp[2])}) x ${Number(qp[1] instanceof V.PyFloat ? qp[1].value : qp[1])}.`;
      else if (scheme) note += ` Scheme ${scheme.replace(/^torch\./, '')}.`;
      return { name, st, offset: intOf(a[1]), shape: intList(a[2]), stride: intList(a[3]), dtype: null, note, quant: true };
    }
    if (fn === 'torch._utils._rebuild_sparse_tensor' || fn === 'torch._utils._rebuild_sparse_csr_tensor') {
      const data = Array.isArray(a[1]) ? a[1] : [];
      const shape = intList(data[2]) || intList(data[3]) || [];
      return { name, shape, unreadable: 'Sparse tensor: listed with its shape. The values are not decoded. Call .to_dense() before saving to see them.' };
    }
    if (fn && /^torch\._utils\._rebuild_(meta_tensor_no_storage|wrapper_subclass|nested_tensor|device_tensor_from_numpy)$/.test(fn)) {
      return { name, shape: intList(a[1]) || [], unreadable: `${fn.replace('torch._utils.', '')}: listed, the values are not decoded.` };
    }
    return null;
  }

  function addTensor(info) {
    if (tensors.length >= MAX_TENSORS) { if (tensors.length === MAX_TENSORS) warnings.push(`Only the first ${MAX_TENSORS} tensors are listed.`); return; }
    const shape = info.shape || [];
    const numel = prod(shape);
    const base = { name: info.name || '(tensor)', shape, numel, offset: null, readable: false };
    if (info.unreadable) { tensors.push({ ...base, dtype: '?', nbytes: null, note: info.unreadable }); return; }
    const kind = info.st ? kindOf(info.st.type, info.dtype) : null;
    const label = kind ? PLAIN[kind].label : (qualname(info.dtype) || qualname(info.st && info.st.type) || '?').replace(/^torch\./, '');
    const t = { ...base, dtype: label, rawDtype: kind ? `torch.${TORCH_NAME[kind]}` : label, nbytes: kind ? numel * PLAIN[kind].size : null };
    const notes = [];
    if (info.note) notes.push(info.note);
    if (info.st && info.st.location && !/^cpu/.test(info.st.location)) notes.push(`Saved from ${info.st.location}.`);
    if (!info.st) { t.note = 'The tensor has no storage reference. The file may be damaged.'; tensors.push(t); return; }
    if (!kind) { t.note = `Storage type ${label} is listed but not decoded.`; tensors.push(t); return; }
    const es = PLAIN[kind].size;
    const store = ctx.storage(info.st.key);
    if (!store) { t.note = `Storage ${info.st.key} is missing from the file.`; tensors.push(t); return; }
    const off = Number.isInteger(info.offset) && info.offset >= 0 ? info.offset : 0;
    const stride = info.stride && info.stride.length === shape.length ? info.stride : rowMajor(shape);
    let maxEl = off;
    shape.forEach((n, k) => { if (n > 0) maxEl += (n - 1) * Math.abs(stride[k]); });
    if (numel > 0 && (maxEl + 1) * es > store.source.size) {
      t.note = [...notes, `The data needs ${(maxEl + 1) * es} bytes of storage ${info.st.key} but only ${store.source.size} are in the file. The file may be truncated.`].join(' ');
      tensors.push(t); return;
    }
    const loc = { source: store.source, dataOffset: off * es, kind, elemSize: es, littleEndian: ctx.littleEndian, fortran: false };
    if (numel === 0 || sameStrides(shape, stride, rowMajor(shape))) {
      t.offset = store.start + off * es;
    } else if (numel * es <= MAX_GATHER_BYTES && stride.every(s => s >= 0)) {
      loc.source = new GatherSource(store.source, off, shape, stride, es); loc.dataOffset = 0;
      notes.push(`A strided view (stride ${stride.join(', ')}). The values are read in row-major order.`);
    } else if (sameStrides(shape, stride, colMajor(shape))) {
      loc.fortran = true; t.offset = store.start + off * es;
      notes.push('Column-major (transposed) layout.');
    } else {
      t.note = [...notes, `A strided view (stride ${stride.join(', ')}) larger than ${MAX_GATHER_BYTES / 1048576} MB. Listed with its shape; call .contiguous() before saving to see the values.`].join(' ');
      tensors.push(t); return;
    }
    if (!ctx.littleEndian && es > 1) notes.push('Big-endian file.');
    t._loc = loc; t.readable = numel > 0;
    if (info.param) notes.unshift('nn.Parameter.');
    if (notes.length) t.note = notes.join(' ');
    tensors.push(t);
  }

  // Text for one value. A file can share one tuple many times (x1 = (x0, x0), x2 = (x1, x1), ...), so the work is
  // capped: at most SHORT_NODES values and SHORT_DEPTH levels per call, which also stops cycles.
  function short(v, budget = 200, cost = { n: 0 }, depth = 0) {
    if (++cost.n > SHORT_NODES || depth > SHORT_DEPTH) return '...';
    if (v === null) return 'None';
    if (typeof v === 'string') return v.length > budget ? JSON.stringify(v.slice(0, budget)) + '...' : JSON.stringify(v);
    if (typeof v === 'number' || typeof v === 'bigint') return String(v);
    if (typeof v === 'boolean') return v ? 'True' : 'False';
    if (v instanceof V.PyFloat) return v.value.toFixed(1);
    if (v instanceof V.PyGlobal) return v.qualname;
    if (v instanceof V.PyText) return v.text;
    if (v instanceof V.PyBytes) return `${v.kind} (${v.data.length} bytes)`;
    if (Array.isArray(v)) {
      const open = V.isTuple(v) ? '(' : '[', close = V.isTuple(v) ? ')' : ']';
      let s = open;
      for (let i = 0; i < v.length; i++) { const p = short(v[i], 40, cost, depth + 1); if (s.length + p.length > budget) { s += `... ${v.length} items`; break; } s += (i ? ', ' : '') + p; }
      return s + close;
    }
    if (v instanceof V.PyDict) return `${v.typeName} with ${v.size} keys`;
    if (v instanceof V.PyObject) return `${v.className} object (not run)`;
    if (v instanceof V.PySet) return `${v.typeName} of ${v.items.length}`;
    return Object.prototype.toString.call(v);
  }

  const isLeaf = v => v === null || typeof v !== 'object' || v instanceof V.PyFloat || v instanceof V.PyText || v instanceof V.PyBytes || v instanceof V.PyGlobal || v instanceof V.PyComplex;
  const join = (p, k) => (p ? `${p}.${k}` : String(k));
  const isEmpty = v => (v instanceof V.PyDict && !v.size) || (Array.isArray(v) && !v.length) || (v instanceof V.PySet && !v.items.length);

  function walk(v, path, depth) {
    if (depth > MAX_DEPTH) { warnings.push(`Nesting deeper than ${MAX_DEPTH} levels at ${path} is not listed.`); return; }
    if (isLeaf(v)) { addMeta(path, short(v)); return; }
    if (seen.has(v)) {
      const first = seen.get(v);
      if (v instanceof V.PyObject && tensorInfo(v, path)) addMeta(path, `same tensor as ${first || '(top level)'}`);
      return;
    }
    seen.set(v, path);
    if (v instanceof V.PyObject) {
      const info = tensorInfo(v, path);
      if (info) { addTensor(info); return; }
      const cls = v.className;
      // An nn.Module saved whole: its weights live in _parameters, _buffers and _modules.
      const state = v.state instanceof V.PyDict ? v.state : Array.isArray(v.state) && v.state.find(x => x instanceof V.PyDict);
      if (state) {
        addMeta(path, `${cls} object (not run)`);
        state.keys.forEach((k, i) => {
          const val = state.values[i];
          if (k === '_parameters' || k === '_buffers' || k === '_modules') { if (!isEmpty(val)) walk(val, path, depth + 1); return; }
          if (typeof k === 'string' && k.startsWith('_') && (isLeaf(val) || isEmpty(val))) return; // module internals such as _backward_hooks
          walk(val, join(path, k), depth + 1);
        });
      } else if (cls === 'collections.OrderedDict' || cls === 'builtins.dict') {
        // empty dict rebuilt by REDUCE with no args
      } else addMeta(path, short(v));
      if (v.dictItems) walk(v.dictItems, path, depth + 1);
      if (v.listItems) v.listItems.forEach((x, i) => walk(x, join(path, i), depth + 1));
      return;
    }
    if (v instanceof V.PyDict) {
      if (v.has('_metadata') && v.get('_metadata') instanceof V.PyDict) {
        // state_dict() keeps per-module version numbers here; one line is enough.
        const md = v.get('_metadata');
        addMeta(join(path, '_metadata'), `state_dict version info for ${md.size} modules`);
        seen.set(md, path);
      }
      if (!v.size) { addMeta(path, `${v.typeName === 'dict' ? '{}' : v.typeName + '()'}`); return; }
      v.keys.forEach((k, i) => { if (k !== '_metadata' || !(v.values[i] instanceof V.PyDict)) walk(v.values[i], join(path, typeof k === 'string' ? k : short(k, 60)), depth + 1); });
      return;
    }
    if (Array.isArray(v)) {
      if (!v.length || v.every(isLeaf)) { addMeta(path, short(v)); return; }
      v.forEach((x, i) => walk(x, join(path, i), depth + 1));
      return;
    }
    if (v instanceof V.PySet) { addMeta(path, short(v)); return; }
    if (v instanceof V.NdArray) { addMeta(path, `numpy array ${v.shape ? '(' + v.shape.join(', ') + ')' : ''}`.trim()); return; }
    addMeta(path, short(v));
  }

  walk(root, '', 0);
  if (skippedMeta) warnings.push(`Only the first ${MAX_METADATA} plain values are listed in Metadata (${skippedMeta} more).`);
  return { tensors, metadata, warnings };
}

/** Rate every Python name the pickle refers to. */
function safetyOf(events) {
  const names = new Map();
  for (const e of events) {
    if (e.op !== 'GLOBAL' && e.op !== 'STACK_GLOBAL' && e.op !== 'INST') continue;
    const q = norm(e.target);
    if (!names.has(q)) names.set(q, { name: q, count: 0, ok: allowed(q), rating: rate(q) });
    names.get(q).count++;
  }
  const all = [...names.values()];
  const danger = all.filter(n => !n.ok && n.rating.level === 'danger');
  const other = all.filter(n => !n.ok && n.rating.level !== 'danger');
  return { all, danger, other };
}

function checkpointInfo(v) {
  if (!(v instanceof V.PyDict)) return [];
  const out = [];
  for (const k of CHECKPOINT_KEYS) {
    const x = v.get(k);
    if (x === undefined) continue;
    if (typeof x === 'number' || typeof x === 'bigint' || typeof x === 'string' || x instanceof V.PyFloat) out.push({ label: k, value: x instanceof V.PyFloat ? x.value.toFixed(1) : String(x).slice(0, 200) });
  }
  return out;
}

function topType(v) {
  if (v instanceof V.PyDict) return v.typeName === 'dict' ? 'dict' : v.typeName.replace(/^collections\./, '');
  if (v instanceof V.PyObject) return v.className;
  if (Array.isArray(v)) return V.isTuple(v) ? 'tuple' : 'list';
  return typeof v;
}

const CHECKPOINT_KEYS = ['epoch', 'global_step', 'step', 'iteration', 'iter', 'loss', 'best_loss', 'val_loss', 'best_acc', 'best_score', 'lr', 'pytorch-lightning_version', 'arch'];

function finish(kind, decoded, ctx, extraInfo, extraWarnings) {
  const { tensors, metadata, warnings } = decoded.value === undefined ? { tensors: [], metadata: [], warnings: [] } : collect(decoded.value, ctx);
  const s = safetyOf(decoded.events);
  const info = [
    { label: 'Format', value: kind },
    { label: 'Top level', value: tensors.length === 1 && tensors[0].name === '(tensor)' ? 'one tensor' : topType(decoded.value) },
    ...checkpointInfo(decoded.value),
    ...extraInfo,
    { label: 'Pickle', value: `read as data, not run (protocol ${decoded.protocol})` },
  ];
  const params = tensors.filter(t => t.note && t.note.startsWith('nn.Parameter')).length;
  if (params) info.push({ label: 'Parameters (nn.Parameter)', value: String(params) });
  const outWarn = [...extraWarnings];
  if (s.danger.length) {
    outWarn.unshift(`Risky: this file refers to ${s.danger.map(d => d.name).slice(0, 4).join(', ')}${s.danger.length > 4 ? ` and ${s.danger.length - 4} more` : ''}. torch.load(weights_only=False) or pickle.load would call ${s.danger.length === 1 ? 'it' : 'them'}. Tensor Viewer did not. Do not load this file in Python unless you trust where it came from.`);
  }
  const custom = s.other.filter(n => !/^torch\./.test(n.name) || n.rating.level !== 'data');
  const needs = custom.map(n => n.name);
  if (!s.danger.length) {
    info.push({ label: 'weights_only=True', value: needs.length ? `likely fails, needs ${needs.slice(0, 3).join(', ')}${needs.length > 3 ? ` and ${needs.length - 3} more` : ''}` : 'should load' });
  }
  if (needs.length && !s.danger.length && /__main__\./.test(needs.join(' '))) outWarn.push(`The whole model object was saved (torch.save(model)). Python needs the class ${needs.filter(n => n.startsWith('__main__.')).slice(0, 2).join(', ')} from the script that saved it to load this file. Tensor Viewer lists the weights without it.`);
  const md = [...metadata];
  for (const n of s.all) md.push({ key: `python name: ${n.name}`, value: n.ok ? 'allowed: a tensor rebuilder, dtype or plain container' : `${n.rating.level === 'danger' ? 'RISKY' : 'not run'}: ${n.rating.why}` });
  if (decoded.error) outWarn.push(`The pickle in this file could not be read to the end: ${decoded.error.message}`);
  return { format: 'pytorch', tensors, metadata: md, info, warnings: [...outWarn, ...warnings] };
}

async function openTorchZip(source) {
  const entries = (await readCentralDirectory(source, 'PyTorch')).filter(e => !e.name.endsWith('/'));
  const pkls = entries.filter(e => /(^|\/)data\.pkl$/.test(e.name)).sort((a, b) => a.name.length - b.name.length);
  if (!pkls.length) throw new Error(entries.some(e => /\.npy$/.test(e.name)) ? 'This zip holds .npy arrays, not a PyTorch file. Rename it to .npz to open it here.' : 'This zip has no data.pkl, so it is not a torch.save file.');
  const pk = pkls[0];
  const prefix = pk.name.slice(0, pk.name.length - 'data.pkl'.length);
  if (pk.size > MAX_PICKLE_BYTES) throw new Error(`data.pkl is ${(pk.size / 1048576).toFixed(0)} MB, more than the ${MAX_PICKLE_BYTES / 1048576} MB limit.`);
  const pkSrc = await entrySource(source, pk);
  const decoded = unpickle(await pkSrc.read(0, pk.size), { maxEvents: 50000 });
  const byName = new Map(entries.map(e => [e.name, e]));
  const read = async n => { const e = byName.get(prefix + n); if (!e || e.size > 4096) return null; const s = await entrySource(source, e); return new TextDecoder().decode(await s.read(0, e.size)).trim(); };
  const byteorder = await read('byteorder');
  const version = await read('version') || await read('.format_version');
  let compressed = 0;
  // entrySource is async; resolve every referenced storage before walking.
  const keys = new Set();
  for (const e of entries) if (e.name.startsWith(prefix + 'data/')) keys.add(e.name.slice(prefix.length + 5));
  const stores = new Map();
  for (const k of keys) {
    const e = byName.get(`${prefix}data/${k}`);
    try {
      const src = await entrySource(source, e);
      if (!src.random) compressed++;
      stores.set(k, { source: src, start: src.start !== undefined ? src.start : e.localOffset });
    } catch { /* listed as missing */ }
  }
  const ctx = { littleEndian: byteorder !== 'big', storage: key => stores.get(key) || null };
  const isScript = entries.some(e => e.name.startsWith(prefix + 'code/')) || entries.some(e => e.name === prefix + 'constants.pkl');
  const extraInfo = [
    { label: 'Storages', value: `${keys.size.toLocaleString('en-US')}${compressed ? ` (${compressed} compressed)` : ''}` },
    { label: 'Byte order', value: byteorder || 'little (not stated)' },
  ];
  if (version) extraInfo.push({ label: 'Archive version', value: version });
  const warnings = [];
  if (compressed) warnings.push(`${compressed} storages are compressed in the zip. Large tensors in them are sampled from the start.`);
  return finish(isScript ? 'TorchScript (torch.jit.save)' : 'PyTorch zip (torch.save)', decoded, ctx, extraInfo, warnings);
}

async function openTorchLegacy(source) {
  // magic number, protocol version, sys_info, then the object, then the list of storage keys.
  let want = Math.min(source.size, 4 * 1048576);
  for (;;) {
    const bytes = await source.read(0, want);
    const parts = [];
    let pos = 0, err = null;
    for (let i = 0; i < 5; i++) {
      const r = unpickle(bytes, { start: pos, maxEvents: 50000 });
      if (r.error) { err = r; break; }
      parts.push(r); pos = r.endPos;
    }
    if (err && want < source.size && want < MAX_PICKLE_BYTES) { want = Math.min(source.size, want * 4, MAX_PICKLE_BYTES); continue; }
    if (parts.length < 1 || !(typeof parts[0].value === 'bigint' || typeof parts[0].value === 'number') || BigInt(parts[0].value) !== 0x1950a86a20f9469cfc6cn) {
      throw new Error('Not a PyTorch file: it is neither a zip (torch.save since 1.6) nor the older pickle format with the PyTorch magic number.');
    }
    if (err) {
      if (parts.length >= 3) return finish('PyTorch legacy (before 1.6)', err, { littleEndian: true, storage: () => null }, [], ['The storage list could not be read, so no values are shown.']);
      throw new Error(`The file header could not be read: ${err.error.message}`);
    }
    const sys = parts[2].value;
    const le = !(sys instanceof V.PyDict) || sys.get('little_endian') !== false;
    const keys = Array.isArray(parts[4].value) ? parts[4].value.map(String) : [];
    // Each storage: int64 element count, then the bytes. Element sizes come from the storage types in the object pickle.
    const types = new Map();
    const scanned = new Set(); // shared or cyclic values are scanned once (a file can share one tuple 2^60 ways)
    (function scan(v, d) {
      if (d > MAX_DEPTH || v === null || typeof v !== 'object' || scanned.has(v)) return;
      scanned.add(v);
      if (v instanceof V.PyPersistent && Array.isArray(v.pid) && v.pid[0] === 'storage') { types.set(String(v.pid[2]), kindOf(v.pid[1], null)); return; }
      if (v instanceof V.PyObject) { (v.args || []).forEach(x => scan(x, d + 1)); scan(v.state, d + 1); if (v.dictItems) scan(v.dictItems, d + 1); if (v.listItems) v.listItems.forEach(x => scan(x, d + 1)); return; }
      if (v instanceof V.PyDict) { v.values.forEach(x => scan(x, d + 1)); return; }
      if (Array.isArray(v)) v.forEach(x => scan(x, d + 1));
    })(parts[3].value, 0);
    const { SliceSource } = require('./source');
    const stores = new Map();
    const warnings = [];
    let p = pos;
    for (const k of keys) {
      if (p + 8 > source.size) { warnings.push('The file ends before all storages. It may be truncated.'); break; }
      const hdr = await source.read(p, 8);
      const n = Number(new DataView(hdr.buffer, hdr.byteOffset, 8).getBigInt64(0, le));
      const kind = types.get(k);
      const es = kind ? PLAIN[kind].size : 1;
      const len = n * es;
      if (n < 0 || p + 8 + len > source.size) { warnings.push(`Storage ${k} runs past the end of the file. It may be truncated.`); break; }
      stores.set(k, { source: new SliceSource(source, p + 8, len), start: p + 8 });
      p += 8 + len;
    }
    return finish('PyTorch legacy (before 1.6)', parts[3], { littleEndian: le, storage: k => stores.get(k) || null },
      [{ label: 'Storages', value: String(keys.length) }, { label: 'Byte order', value: le ? 'little' : 'big' }], warnings);
  }
}

async function openTorch(source) {
  const head = await source.read(0, Math.min(4, source.size));
  if (head[0] === 0x50 && head[1] === 0x4b) return openTorchZip(source);
  if (head[0] === 0x80) return openTorchLegacy(source);
  throw new Error('Not a PyTorch file: torch.save files start with a zip header (PK) or a pickle (0x80). This file starts with other bytes. TorchScript and ONNX are other formats; safetensors files open here too.');
}

module.exports = { openTorch, isTorchFileName, collect, ALLOWED, allowed, GatherSource };
