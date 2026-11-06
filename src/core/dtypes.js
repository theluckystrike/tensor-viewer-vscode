'use strict';
// Element types and decoders. Everything decodes to float64 so stats and plots share one path.
// No decoder ever interprets bytes as code: object and pickle data are refused before this point.

const f16buf = new DataView(new ArrayBuffer(4));

function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 5.960464477539063e-8; // 2^-24
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * Math.pow(2, e - 15);
}

function bf16ToFloat(h) {
  f16buf.setUint32(0, (h & 0xffff) << 16 >>> 0);
  return f16buf.getFloat32(0);
}

function e4m3ToFloat(b) {
  const s = (b & 0x80) ? -1 : 1;
  const e = (b >> 3) & 0xf;
  const m = b & 7;
  if (e === 15 && m === 7) return NaN; // E4M3FN has no infinities
  if (e === 0) return s * (m / 8) * Math.pow(2, -6);
  return s * (1 + m / 8) * Math.pow(2, e - 7);
}

function e5m2ToFloat(b) {
  const s = (b & 0x80) ? -1 : 1;
  const e = (b >> 2) & 0x1f;
  const m = b & 3;
  if (e === 31) return m ? NaN : s * Infinity;
  if (e === 0) return s * (m / 4) * Math.pow(2, -14);
  return s * (1 + m / 4) * Math.pow(2, e - 15);
}

// Plain element types. size = bytes per element.
const PLAIN = {
  bool: { size: 1, label: 'bool' },
  i8: { size: 1, label: 'int8' },
  u8: { size: 1, label: 'uint8' },
  i16: { size: 2, label: 'int16' },
  u16: { size: 2, label: 'uint16' },
  i32: { size: 4, label: 'int32' },
  u32: { size: 4, label: 'uint32' },
  i64: { size: 8, label: 'int64' },
  u64: { size: 8, label: 'uint64' },
  f16: { size: 2, label: 'float16' },
  bf16: { size: 2, label: 'bfloat16' },
  f32: { size: 4, label: 'float32' },
  f64: { size: 8, label: 'float64' },
  c64: { size: 8, label: 'complex64' },
  c128: { size: 16, label: 'complex128' },
  f8e4m3: { size: 1, label: 'float8_e4m3' },
  f8e5m2: { size: 1, label: 'float8_e5m2' },
};

/**
 * Decode `count` elements of a plain type from bytes. Complex values decode to their magnitude.
 * @returns {Float64Array}
 */
function decodePlain(kind, bytes, count, littleEndian = true) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float64Array(count);
  const le = littleEndian;
  switch (kind) {
    case 'bool': for (let i = 0; i < count; i++) out[i] = bytes[i] ? 1 : 0; break;
    case 'i8': for (let i = 0; i < count; i++) out[i] = dv.getInt8(i); break;
    case 'u8': for (let i = 0; i < count; i++) out[i] = bytes[i]; break;
    case 'i16': for (let i = 0; i < count; i++) out[i] = dv.getInt16(i * 2, le); break;
    case 'u16': for (let i = 0; i < count; i++) out[i] = dv.getUint16(i * 2, le); break;
    case 'i32': for (let i = 0; i < count; i++) out[i] = dv.getInt32(i * 4, le); break;
    case 'u32': for (let i = 0; i < count; i++) out[i] = dv.getUint32(i * 4, le); break;
    case 'i64': for (let i = 0; i < count; i++) out[i] = Number(dv.getBigInt64(i * 8, le)); break;
    case 'u64': for (let i = 0; i < count; i++) out[i] = Number(dv.getBigUint64(i * 8, le)); break;
    case 'f16': for (let i = 0; i < count; i++) out[i] = halfToFloat(dv.getUint16(i * 2, le)); break;
    case 'bf16': for (let i = 0; i < count; i++) out[i] = bf16ToFloat(dv.getUint16(i * 2, le)); break;
    case 'f32': for (let i = 0; i < count; i++) out[i] = dv.getFloat32(i * 4, le); break;
    case 'f64': for (let i = 0; i < count; i++) out[i] = dv.getFloat64(i * 8, le); break;
    case 'c64': for (let i = 0; i < count; i++) out[i] = Math.hypot(dv.getFloat32(i * 8, le), dv.getFloat32(i * 8 + 4, le)); break;
    case 'c128': for (let i = 0; i < count; i++) out[i] = Math.hypot(dv.getFloat64(i * 16, le), dv.getFloat64(i * 16 + 8, le)); break;
    case 'f8e4m3': for (let i = 0; i < count; i++) out[i] = e4m3ToFloat(bytes[i]); break;
    case 'f8e5m2': for (let i = 0; i < count; i++) out[i] = e5m2ToFloat(bytes[i]); break;
    default: throw new Error(`No decoder for ${kind}`);
  }
  return out;
}

// ---------- GGML block-quantized types (llama.cpp layout, little-endian) ----------

const QK_K = 256;

function getScaleMinK4(j, q, o) {
  if (j < 4) return [q[o + j] & 63, q[o + j + 4] & 63];
  return [(q[o + j + 4] & 0xf) | ((q[o + j - 4] >> 6) << 4), (q[o + j + 4] >> 4) | ((q[o + j] >> 6) << 4)];
}

const h16 = (b, o) => halfToFloat(b[o] | (b[o + 1] << 8));

// Each entry: block = elements per block, bytes = bytes per block, dq = dequantize one block into out[oi..].
const QUANT = {
  Q4_0: { block: 32, bytes: 18, dq(b, o, out, oi) {
    const d = h16(b, o);
    for (let j = 0; j < 16; j++) { const q = b[o + 2 + j]; out[oi + j] = ((q & 0xf) - 8) * d; out[oi + j + 16] = ((q >> 4) - 8) * d; }
  } },
  Q4_1: { block: 32, bytes: 20, dq(b, o, out, oi) {
    const d = h16(b, o), m = h16(b, o + 2);
    for (let j = 0; j < 16; j++) { const q = b[o + 4 + j]; out[oi + j] = (q & 0xf) * d + m; out[oi + j + 16] = (q >> 4) * d + m; }
  } },
  Q5_0: { block: 32, bytes: 22, dq(b, o, out, oi) {
    const d = h16(b, o);
    const qh = (b[o + 2] | (b[o + 3] << 8) | (b[o + 4] << 16) | (b[o + 5] << 24)) >>> 0;
    for (let j = 0; j < 16; j++) {
      const xh0 = ((qh >>> j) << 4) & 0x10, xh1 = (qh >>> (j + 12)) & 0x10, q = b[o + 6 + j];
      out[oi + j] = (((q & 0xf) | xh0) - 16) * d; out[oi + j + 16] = (((q >> 4) | xh1) - 16) * d;
    }
  } },
  Q5_1: { block: 32, bytes: 24, dq(b, o, out, oi) {
    const d = h16(b, o), m = h16(b, o + 2);
    const qh = (b[o + 4] | (b[o + 5] << 8) | (b[o + 6] << 16) | (b[o + 7] << 24)) >>> 0;
    for (let j = 0; j < 16; j++) {
      const xh0 = ((qh >>> j) << 4) & 0x10, xh1 = (qh >>> (j + 12)) & 0x10, q = b[o + 8 + j];
      out[oi + j] = ((q & 0xf) | xh0) * d + m; out[oi + j + 16] = ((q >> 4) | xh1) * d + m;
    }
  } },
  Q8_0: { block: 32, bytes: 34, dq(b, o, out, oi) {
    const d = h16(b, o);
    for (let j = 0; j < 32; j++) out[oi + j] = ((b[o + 2 + j] << 24) >> 24) * d;
  } },
  Q4_K: { block: QK_K, bytes: 144, dq(b, o, out, oi) {
    const d = h16(b, o), dmin = h16(b, o + 2), sc = o + 4;
    let q = o + 16, is = 0, y = oi;
    for (let j = 0; j < QK_K; j += 64) {
      const [s1, m1] = getScaleMinK4(is, b, sc), [s2, m2] = getScaleMinK4(is + 1, b, sc);
      const d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
      for (let l = 0; l < 32; l++) out[y++] = d1 * (b[q + l] & 0xf) - mm1;
      for (let l = 0; l < 32; l++) out[y++] = d2 * (b[q + l] >> 4) - mm2;
      q += 32; is += 2;
    }
  } },
  Q5_K: { block: QK_K, bytes: 176, dq(b, o, out, oi) {
    const d = h16(b, o), dmin = h16(b, o + 2), sc = o + 4, qh = o + 16;
    let ql = o + 48, is = 0, y = oi, u1 = 1, u2 = 2;
    for (let j = 0; j < QK_K; j += 64) {
      const [s1, m1] = getScaleMinK4(is, b, sc), [s2, m2] = getScaleMinK4(is + 1, b, sc);
      const d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
      for (let l = 0; l < 32; l++) out[y++] = d1 * ((b[ql + l] & 0xf) + (b[qh + l] & u1 ? 16 : 0)) - mm1;
      for (let l = 0; l < 32; l++) out[y++] = d2 * ((b[ql + l] >> 4) + (b[qh + l] & u2 ? 16 : 0)) - mm2;
      ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
  } },
  Q6_K: { block: QK_K, bytes: 210, dq(b, o, out, oi) {
    const d = h16(b, o + 208);
    let ql = o, qh = o + 128, sc = o + 192, y = oi;
    for (let n = 0; n < QK_K; n += 128) {
      for (let l = 0; l < 32; l++) {
        const is = l >> 4;
        const q1 = ((b[ql + l] & 0xf) | (((b[qh + l] >> 0) & 3) << 4)) - 32;
        const q2 = ((b[ql + l + 32] & 0xf) | (((b[qh + l] >> 2) & 3) << 4)) - 32;
        const q3 = ((b[ql + l] >> 4) | (((b[qh + l] >> 4) & 3) << 4)) - 32;
        const q4 = ((b[ql + l + 32] >> 4) | (((b[qh + l] >> 6) & 3) << 4)) - 32;
        const s = i => (b[sc + is + i] << 24) >> 24;
        out[y + l] = d * s(0) * q1; out[y + l + 32] = d * s(2) * q2; out[y + l + 64] = d * s(4) * q3; out[y + l + 96] = d * s(6) * q4;
      }
      y += 128; ql += 64; qh += 32; sc += 8;
    }
  } },
};

/** Dequantize `nblocks` consecutive blocks starting at bytes[0]. */
function decodeQuant(kind, bytes, nblocks) {
  const q = QUANT[kind];
  const out = new Float64Array(nblocks * q.block);
  for (let i = 0; i < nblocks; i++) q.dq(bytes, i * q.bytes, out, i * q.block);
  return out;
}

// GGML type ids (ggml.h). size info for every known type so row sizes are right even
// for types we can list but not decode.
const GGML = {
  0: { name: 'F32', block: 1, bytes: 4, kind: 'f32' },
  1: { name: 'F16', block: 1, bytes: 2, kind: 'f16' },
  2: { name: 'Q4_0', block: 32, bytes: 18, quant: 'Q4_0' },
  3: { name: 'Q4_1', block: 32, bytes: 20, quant: 'Q4_1' },
  6: { name: 'Q5_0', block: 32, bytes: 22, quant: 'Q5_0' },
  7: { name: 'Q5_1', block: 32, bytes: 24, quant: 'Q5_1' },
  8: { name: 'Q8_0', block: 32, bytes: 34, quant: 'Q8_0' },
  9: { name: 'Q8_1', block: 32, bytes: 36 },
  10: { name: 'Q2_K', block: 256, bytes: 84 },
  11: { name: 'Q3_K', block: 256, bytes: 110 },
  12: { name: 'Q4_K', block: 256, bytes: 144, quant: 'Q4_K' },
  13: { name: 'Q5_K', block: 256, bytes: 176, quant: 'Q5_K' },
  14: { name: 'Q6_K', block: 256, bytes: 210, quant: 'Q6_K' },
  15: { name: 'Q8_K', block: 256, bytes: 292 },
  16: { name: 'IQ2_XXS', block: 256, bytes: 66 },
  17: { name: 'IQ2_XS', block: 256, bytes: 74 },
  18: { name: 'IQ3_XXS', block: 256, bytes: 98 },
  19: { name: 'IQ1_S', block: 256, bytes: 50 },
  20: { name: 'IQ4_NL', block: 32, bytes: 18 },
  21: { name: 'IQ3_S', block: 256, bytes: 110 },
  22: { name: 'IQ2_S', block: 256, bytes: 82 },
  23: { name: 'IQ4_XS', block: 256, bytes: 136 },
  24: { name: 'I8', block: 1, bytes: 1, kind: 'i8' },
  25: { name: 'I16', block: 1, bytes: 2, kind: 'i16' },
  26: { name: 'I32', block: 1, bytes: 4, kind: 'i32' },
  27: { name: 'I64', block: 1, bytes: 8, kind: 'i64' },
  28: { name: 'F64', block: 1, bytes: 8, kind: 'f64' },
  29: { name: 'IQ1_M', block: 256, bytes: 56 },
  30: { name: 'BF16', block: 1, bytes: 2, kind: 'bf16' },
  34: { name: 'TQ1_0', block: 256, bytes: 54 },
  35: { name: 'TQ2_0', block: 256, bytes: 66 },
  39: { name: 'MXFP4', block: 32, bytes: 17 },
};

// safetensors dtype names.
const SAFETENSORS = {
  BOOL: 'bool', U8: 'u8', I8: 'i8', I16: 'i16', U16: 'u16', I32: 'i32', U32: 'u32', I64: 'i64', U64: 'u64',
  F16: 'f16', BF16: 'bf16', F32: 'f32', F64: 'f64', F8_E4M3: 'f8e4m3', F8_E5M2: 'f8e5m2', C64: 'c64',
};

/**
 * Turn a numpy descr string (for example '<f4', '>i8', '|b1', '<U5') into a type record.
 * @returns {{kind?:string, size:number, littleEndian:boolean, label:string, readable:boolean, reason?:string}}
 */
function fromNumpyDescr(descr) {
  if (typeof descr !== 'string') return { size: 0, littleEndian: true, label: 'structured', readable: false, reason: 'Structured (record) arrays are listed but not plotted.' };
  const m = /^([<>|=]?)([a-zA-Z])(\d*)$/.exec(descr);
  if (!m) return { size: 0, littleEndian: true, label: descr, readable: false, reason: `Unknown dtype ${descr}.` };
  const order = m[1], code = m[2], n = m[3] ? parseInt(m[3], 10) : 0;
  const littleEndian = order !== '>';
  if (code === 'O') return { size: 8, littleEndian, label: 'object', readable: false, reason: 'Object arrays are stored as Python pickle. Tensor Viewer never loads pickle data, because it can run code.' };
  if (code === 'U') return { size: n * 4, littleEndian, label: `str${n}`, readable: false, reason: 'Text arrays have no numeric stats.' };
  if (code === 'S' || code === 'a') return { size: n, littleEndian, label: `bytes${n}`, readable: false, reason: 'Byte-string arrays have no numeric stats.' };
  if (code === 'V') return { size: n, littleEndian, label: `void${n}`, readable: false, reason: 'Raw void arrays have no numeric stats.' };
  if (code === 'M' || code === 'm') return { size: n || 8, littleEndian, label: code === 'M' ? 'datetime64' : 'timedelta64', kind: 'i64', readable: true, reason: 'Shown as raw int64 ticks.' };
  const map = { b: { 1: 'bool' }, i: { 1: 'i8', 2: 'i16', 4: 'i32', 8: 'i64' }, u: { 1: 'u8', 2: 'u16', 4: 'u32', 8: 'u64' }, f: { 2: 'f16', 4: 'f32', 8: 'f64' }, c: { 8: 'c64', 16: 'c128' } };
  const kind = map[code] && map[code][n];
  if (!kind) return { size: n, littleEndian, label: descr, readable: false, reason: `dtype ${descr} is not supported.` };
  return { kind, size: n, littleEndian, label: PLAIN[kind].label, readable: true };
}

module.exports = { PLAIN, QUANT, GGML, SAFETENSORS, decodePlain, decodeQuant, fromNumpyDescr, halfToFloat, bf16ToFloat, e4m3ToFloat, e5m2ToFloat };
