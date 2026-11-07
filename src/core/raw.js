'use strict';
// Raw binary arrays: files written by ndarray.tofile(), array.array.tofile() or C code, with no header.
// The user says the type and shape; the file is then read like the data part of a .npy file.
const { PLAIN } = require('./dtypes');

/** Types offered for raw files, most common first. `np` is the NumPy name used in labels and notes. */
const RAW_TYPES = [
  { kind: 'f32', np: 'float32' }, { kind: 'f64', np: 'float64' }, { kind: 'f16', np: 'float16' }, { kind: 'bf16', np: 'bfloat16' },
  { kind: 'u8', np: 'uint8' }, { kind: 'i8', np: 'int8' }, { kind: 'i16', np: 'int16' }, { kind: 'u16', np: 'uint16' },
  { kind: 'i32', np: 'int32' }, { kind: 'u32', np: 'uint32' }, { kind: 'i64', np: 'int64' }, { kind: 'u64', np: 'uint64' },
  { kind: 'c64', np: 'complex64' }, { kind: 'c128', np: 'complex128' }, { kind: 'bool', np: 'bool' },
];
const KINDS = new Set(RAW_TYPES.map(t => t.kind));
const MAX_DIMS = 32;

/**
 * Parse a shape typed by the user: "480, 640, 3", "(480, 640, 3)", "-1, 3", "480x640" or empty (1-D).
 * At most one -1, which is worked out from the file size.
 * @returns {{ok:true, shape:number[]} | {ok:false, error:string}}
 */
function parseShape(text) {
  const s = String(text == null ? '' : text).trim().replace(/^[([]\s*/, '').replace(/\s*[)\]]$/, '').trim();
  if (s === '') return { ok: true, shape: [-1] };
  const parts = s.split(/\s*[,x×*]\s*|\s+/).filter(p => p !== '');
  if (parts.length > MAX_DIMS) return { ok: false, error: `At most ${MAX_DIMS} dimensions.` };
  const shape = [];
  for (const p of parts) {
    if (!/^-?\d+$/.test(p)) return { ok: false, error: `"${p}" is not a whole number. Type sizes like 480, 640, 3.` };
    const n = Number(p);
    if (n === -1) shape.push(-1);
    else if (n < 1 || !Number.isSafeInteger(n)) return { ok: false, error: `${p} is not a valid size. Use 1 or more, or -1 once.` };
    else shape.push(n);
  }
  if (shape.filter(n => n === -1).length > 1) return { ok: false, error: 'Use -1 for at most one dimension.' };
  return { ok: true, shape };
}

/**
 * Check a raw layout against the file size and work out a -1.
 * @param {{kind:string, shape:number[], offset?:number, littleEndian?:boolean}} layout
 * @returns {{ok:true, shape:number[], numel:number, nbytes:number, offset:number, extra:number, littleEndian:boolean, kind:string} | {ok:false, error:string}}
 */
function resolveRawLayout(layout, fileSize) {
  if (!layout || !KINDS.has(layout.kind)) return { ok: false, error: 'Pick a value type.' };
  const size = PLAIN[layout.kind].size;
  const offset = layout.offset === undefined ? 0 : layout.offset;
  if (!Number.isSafeInteger(offset) || offset < 0) return { ok: false, error: 'The header size must be 0 or a positive whole number of bytes.' };
  if (offset > fileSize) return { ok: false, error: `The header size (${offset} bytes) is larger than the file (${fileSize} bytes).` };
  if (!Array.isArray(layout.shape) || !layout.shape.length) return { ok: false, error: 'Type a shape.' };
  const avail = fileSize - offset;
  const known = layout.shape.filter(n => n !== -1).reduce((a, b) => a * b, 1);
  const shape = layout.shape.slice();
  const free = shape.indexOf(-1);
  const np = RAW_TYPES.find(t => t.kind === layout.kind).np;
  if (free >= 0) {
    const fit = Math.floor(avail / size / known);
    if (fit < 1) return { ok: false, error: `The file has ${avail} bytes after the header, too few for one ${shape.map(n => (n === -1 ? 1 : n)).join(' x ')} block of ${np}.` };
    shape[free] = fit;
  }
  const numel = shape.reduce((a, b) => a * b, 1);
  const nbytes = numel * size;
  if (!Number.isSafeInteger(nbytes)) return { ok: false, error: 'That shape is too large.' };
  if (nbytes > avail) {
    return { ok: false, error: `${shape.join(' x ')} ${np} needs ${nbytes.toLocaleString('en-US')} bytes, but the file has ${avail.toLocaleString('en-US')} after the header. Check the type and shape.` };
  }
  return { ok: true, shape, numel, nbytes, offset, extra: avail - nbytes, littleEndian: layout.littleEndian !== false, kind: layout.kind };
}

/** Shapes that fit the file exactly, for the hint under the shape box. */
function shapeHint(kind, fileSize, offset = 0) {
  const size = PLAIN[kind] ? PLAIN[kind].size : 1;
  const avail = Math.max(0, fileSize - offset);
  const n = Math.floor(avail / size);
  const rest = avail - n * size;
  let hint = `${n.toLocaleString('en-US')} values of ${size} byte${size > 1 ? 's' : ''}`;
  if (rest) hint += ` (${rest} byte${rest > 1 ? 's' : ''} left over)`;
  const r = Math.round(Math.sqrt(n));
  if (n > 3 && r * r === n) hint += `, for example ${r}, ${r}`;
  else if (n % 3 === 0 && n > 3) hint += `, for example -1, 3`;
  return hint;
}

/** Text for the title bar and notes, like "float32, little-endian, shape (480, 640, 3), header 0 bytes". */
function layoutText(r) {
  const np = RAW_TYPES.find(t => t.kind === r.kind).np;
  const order = PLAIN[r.kind].size > 1 && !r.littleEndian ? ', big-endian' : '';
  return `${np}${order}, shape (${r.shape.join(', ')}${r.shape.length === 1 ? ',' : ''})${r.offset ? `, after a ${r.offset}-byte header` : ''}`;
}

/** Build the model for a raw file. Throws with a readable message when the layout doesn't fit. */
function openRaw(source, layout) {
  const r = resolveRawLayout(layout, source.size);
  if (!r.ok) throw new Error(r.error);
  const np = RAW_TYPES.find(t => t.kind === r.kind).np;
  const t = {
    name: 'array', dtype: PLAIN[r.kind].label, rawDtype: np === PLAIN[r.kind].label ? undefined : np, shape: r.shape, numel: r.numel, nbytes: r.nbytes,
    offset: r.offset, readable: r.numel > 0,
    _loc: { source, dataOffset: r.offset, kind: r.kind, elemSize: PLAIN[r.kind].size, littleEndian: r.littleEndian, fortran: false },
  };
  const warnings = [];
  if (r.extra) warnings.push(`${r.extra.toLocaleString('en-US')} byte${r.extra > 1 ? 's' : ''} at the end of the file are not part of this shape.`);
  return {
    format: 'raw',
    tensors: [t],
    metadata: [
      { key: 'type', value: np },
      { key: 'byte order', value: PLAIN[r.kind].size > 1 ? (r.littleEndian ? 'little-endian' : 'big-endian') : 'not used (1 byte per value)' },
      { key: 'shape', value: `(${r.shape.join(', ')}${r.shape.length === 1 ? ',' : ''})` },
      { key: 'header bytes skipped', value: String(r.offset) },
      { key: 'bytes after the data', value: String(r.extra) },
      { key: 'NumPy equivalent', value: `np.fromfile(path, dtype=${npDtype(r)}${r.offset ? `, offset=${r.offset}` : ''}${r.extra ? `, count=${r.numel}` : ''}).reshape(${r.shape.join(', ')}${r.shape.length === 1 ? ',' : ''})` },
    ],
    info: [{ label: 'Read as', value: layoutText(r) }],
    warnings,
  };
}

function npDtype(r) {
  if (r.kind === 'bf16') return 'ml_dtypes.bfloat16';
  const code = { f32: 'f4', f64: 'f8', f16: 'f2', u8: 'u1', i8: 'i1', i16: 'i2', u16: 'u2', i32: 'i4', u32: 'u4', i64: 'i8', u64: 'u8', c64: 'c8', c128: 'c16', bool: 'b1' }[r.kind];
  return `"${PLAIN[r.kind].size > 1 ? (r.littleEndian ? '<' : '>') : '|'}${code}"`;
}

/** Check a layout that came from storage or from the webview before using it. */
function cleanLayout(x) {
  if (!x || typeof x !== 'object') return null;
  if (!KINDS.has(x.kind)) return null;
  if (!Array.isArray(x.shape) || !x.shape.length || x.shape.length > MAX_DIMS || !x.shape.every(n => n === -1 || (Number.isSafeInteger(n) && n >= 1))) return null;
  if (x.shape.filter(n => n === -1).length > 1) return null;
  const offset = x.offset === undefined ? 0 : x.offset;
  if (!Number.isSafeInteger(offset) || offset < 0) return null;
  return { kind: x.kind, shape: x.shape.slice(), offset, littleEndian: x.littleEndian !== false };
}

module.exports = { RAW_TYPES, parseShape, resolveRawLayout, shapeHint, layoutText, openRaw, cleanLayout };
