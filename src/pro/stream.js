'use strict';
// Reads every value of one tensor, a run at a time, for the Pro tools (compare two files, scan all tensors).
// The free viewer samples big tensors; these tools read all of them, so they work in runs of about 1M values and
// never hold a whole tensor in memory (except to reorder a Fortran-order array of up to 16M values).
const { decodePlain, decodeQuant, PLAIN } = require('../core/dtypes');

const RUN_VALUES = 1 << 20;
const MAX_COMPRESSED_BYTES = 512 * 1048576; // a deflated .npz entry is inflated from its start and kept in memory
const MAX_REORDER_VALUES = 16 * 1048576;

const isComplex = k => k === 'c64' || k === 'c128';

/**
 * Can this tensor be read value by value? Complex tensors give 2 values per element (real, imaginary).
 * @returns {{ok:true, perElement:number, total:number} | {ok:false, why:string}}
 */
function streamInfo(t) {
  if (!t || !t.readable || !t._loc) return { ok: false, why: (t && t.note) || 'This tensor cannot be read.' };
  const loc = t._loc;
  if (loc.quant) {
    if (!loc.block || !loc.blockBytes || t.numel % loc.block) return { ok: false, why: `${t.dtype}: the value count is not a whole number of blocks.` };
    return { ok: true, perElement: 1, total: t.numel };
  }
  if (!loc.kind || !PLAIN[loc.kind]) return { ok: false, why: `No decoder for ${t.dtype}.` };
  if (loc.source && loc.source.random === false && (t.nbytes || 0) > MAX_COMPRESSED_BYTES) {
    return { ok: false, why: `Compressed .npz entry of ${Math.round(t.nbytes / 1048576)} MB. Entries up to ${MAX_COMPRESSED_BYTES / 1048576} MB are read; save it with np.savez (not savez_compressed) to check it.` };
  }
  const per = isComplex(loc.kind) ? 2 : 1;
  return { ok: true, perElement: per, total: t.numel * per };
}

/** Values [start, start + n) in memory order. Complex tensors are read as interleaved real and imaginary parts. */
async function readAt(t, start, n) {
  const loc = t._loc;
  if (n <= 0) return new Float64Array(0);
  if (loc.quant) {
    const b0 = Math.floor(start / loc.block), b1 = Math.floor((start + n - 1) / loc.block) + 1;
    const bytes = await loc.source.read(loc.dataOffset + b0 * loc.blockBytes, (b1 - b0) * loc.blockBytes);
    const all = decodeQuant(loc.quant, bytes, b1 - b0);
    const from = start - b0 * loc.block;
    return all.subarray(from, from + n);
  }
  const complex = isComplex(loc.kind);
  const kind = complex ? (loc.kind === 'c64' ? 'f32' : 'f64') : loc.kind;
  const size = complex ? loc.elemSize / 2 : loc.elemSize;
  const bytes = await loc.source.read(loc.dataOffset + start * size, n * size);
  return decodePlain(kind, bytes, n, loc.littleEndian);
}

/** Every value of a Fortran-order tensor, reordered to C order (for comparing it with a C-order tensor). */
async function readAllCOrder(t, perElement) {
  const total = t.numel * perElement;
  if (total > MAX_REORDER_VALUES) throw new Error(`One file stores this tensor in Fortran order and the other in C order. Such pairs are compared up to ${MAX_REORDER_VALUES.toLocaleString('en-US')} values.`);
  const mem = new Float64Array(total);
  for (let i = 0; i < total; i += RUN_VALUES) mem.set(await readAt(t, i, Math.min(RUN_VALUES, total - i)), i);
  if (!t._loc.fortran || t.shape.length < 2) return mem;
  const shape = t.shape, nd = shape.length;
  const fStride = new Array(nd);
  let s = 1;
  for (let k = 0; k < nd; k++) { fStride[k] = s; s *= shape[k]; }
  const out = new Float64Array(total);
  const idx = new Array(nd).fill(0);
  for (let c = 0; c < t.numel; c++) {
    let f = 0;
    for (let k = 0; k < nd; k++) f += idx[k] * fStride[k];
    for (let p = 0; p < perElement; p++) out[c * perElement + p] = mem[f * perElement + p];
    for (let k = nd - 1; k >= 0; k--) { if (++idx[k] < shape[k]) break; idx[k] = 0; }
  }
  return out;
}

/** Flat memory index -> index text such as [3, 17] (C or Fortran order). */
function indexText(flat, shape, fortran) {
  if (!shape.length) return '[]';
  const out = new Array(shape.length);
  let r = flat;
  if (fortran) for (let k = 0; k < shape.length; k++) { out[k] = r % shape[k]; r = Math.floor(r / shape[k]); }
  else for (let k = shape.length - 1; k >= 0; k--) { out[k] = r % shape[k]; r = Math.floor(r / shape[k]); }
  return '[' + out.join(', ') + ']';
}

module.exports = { streamInfo, readAt, readAllCOrder, indexText, RUN_VALUES, MAX_COMPRESSED_BYTES, MAX_REORDER_VALUES };
