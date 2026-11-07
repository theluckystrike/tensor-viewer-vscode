'use strict';
// Opens .npy, .npz, .safetensors and .gguf files into one model: a tensor list plus metadata.
// Only headers are read here. Tensor bytes are read later, one tensor at a time, by sample.js.
const { readNpyHeader, descrLabel } = require('./npy');
const { readCentralDirectory, entrySource } = require('./zip');
const { SAFETENSORS, PLAIN, GGML } = require('./dtypes');
const { openRaw } = require('./raw');

const MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024;
const MAX_TENSORS = 200000;

function formatFromName(name) {
  const m = /\.(npy|npz|safetensors|gguf)$/i.exec(name || '');
  return m ? m[1].toLowerCase() : null;
}

const prod = shape => shape.reduce((a, b) => a * b, 1);

function npyTensor(name, h, source, displayOffset) {
  const numel = prod(h.shape);
  const t = {
    name, dtype: h.type.label, rawDtype: descrLabel(h.descr), shape: h.shape.slice(), numel,
    nbytes: h.type.size ? numel * h.type.size : null, offset: displayOffset, readable: h.type.readable && numel > 0,
    note: h.type.reason || (h.fortranOrder ? 'Fortran (column-major) order.' : undefined),
    _loc: { source, dataOffset: h.dataOffset, kind: h.type.kind, elemSize: h.type.size, littleEndian: h.type.littleEndian, fortran: h.fortranOrder },
  };
  if (h.type.readable && h.dataOffset + t.nbytes > source.size) {
    t.readable = false;
    t.note = `The file is truncated: the data needs ${t.nbytes} bytes but only ${Math.max(0, source.size - h.dataOffset)} are there.`;
  }
  return t;
}

async function openNpy(source) {
  const h = await readNpyHeader(source, 0);
  return {
    format: 'npy',
    tensors: [npyTensor('array', h, source, h.dataOffset)],
    metadata: [
      { key: 'npy version', value: h.version },
      { key: 'descr', value: descrLabel(h.descr) },
      { key: 'fortran_order', value: String(h.fortranOrder) },
      { key: 'data offset', value: String(h.dataOffset) },
    ],
    info: [],
    warnings: [],
  };
}

async function openNpz(source) {
  const entries = (await readCentralDirectory(source)).filter(e => !e.name.endsWith('/'));
  const tensors = [];
  const warnings = [];
  let compressed = 0;
  for (const e of entries) {
    const name = e.name.replace(/\.npy$/, '');
    if (!e.name.endsWith('.npy')) { warnings.push(`Skipped ${e.name}: not a .npy entry.`); continue; }
    try {
      const sub = await entrySource(source, e);
      if (!sub.random) compressed++;
      const h = await readNpyHeader(sub, 0);
      const t = npyTensor(name, h, sub, e.localOffset);
      t.note = [t.note, sub.random ? '' : 'Compressed entry: large arrays are sampled from the start.'].filter(Boolean).join(' ') || undefined;
      tensors.push(t);
    } catch (err) {
      tensors.push({ name, dtype: '?', shape: [], numel: 0, nbytes: e.size, offset: e.localOffset, readable: false, note: `Could not read this entry: ${err.message}` });
    }
  }
  return {
    format: 'npz', tensors,
    metadata: [
      { key: 'arrays', value: String(tensors.length) },
      { key: 'compression', value: compressed ? `deflate (${compressed} of ${entries.length} entries)` : 'stored (np.savez)' },
    ],
    info: [], warnings,
  };
}

async function openSafetensors(source) {
  if (source.size < 8) throw new Error('Not a safetensors file: it is shorter than 8 bytes.');
  const pre = await source.read(0, 8);
  const n = Number(new DataView(pre.buffer, pre.byteOffset, 8).getBigUint64(0, true));
  if (n > MAX_SAFETENSORS_HEADER) throw new Error(`The safetensors header says it is ${n} bytes, more than the 100 MB limit. The file may be damaged.`);
  if (8 + n > source.size) throw new Error(`Not a valid safetensors file: the header (${n} bytes) runs past the end of the file (${source.size} bytes).`);
  let header;
  try { header = JSON.parse(new TextDecoder().decode(await source.read(8, n))); } catch (e) { throw new Error(`Not a valid safetensors file: the JSON header does not parse (${e.message}).`); }
  if (!header || typeof header !== 'object' || Array.isArray(header)) throw new Error('Not a valid safetensors file: the header is not a JSON object.');
  const base = 8 + n;
  const tensors = [];
  const warnings = [];
  const metadata = [];
  for (const [name, v] of Object.entries(header)) {
    if (name === '__metadata__') {
      if (v && typeof v === 'object') for (const [k, val] of Object.entries(v)) metadata.push({ key: k, value: typeof val === 'string' ? val : JSON.stringify(val) });
      continue;
    }
    if (tensors.length >= MAX_TENSORS) { warnings.push(`Only the first ${MAX_TENSORS} tensors are listed.`); break; }
    const ok = v && typeof v.dtype === 'string' && Array.isArray(v.shape) && v.shape.every(d => Number.isInteger(d) && d >= 0) && Array.isArray(v.data_offsets) && v.data_offsets.length === 2;
    if (!ok) { tensors.push({ name, dtype: '?', shape: [], numel: 0, nbytes: null, offset: null, readable: false, note: 'Bad header entry.' }); continue; }
    const kind = SAFETENSORS[v.dtype];
    const [b, e] = v.data_offsets.map(Number);
    const numel = prod(v.shape);
    const t = { name, dtype: kind ? PLAIN[kind].label : v.dtype, rawDtype: v.dtype, shape: v.shape.slice(), numel, nbytes: e - b, offset: base + b, readable: !!kind && numel > 0,
      _loc: { source, dataOffset: base + b, kind, elemSize: kind ? PLAIN[kind].size : 0, littleEndian: true, fortran: false } };
    if (!kind) t.note = `dtype ${v.dtype} is listed but not decoded.`;
    else if (e - b !== numel * PLAIN[kind].size) { t.readable = false; t.note = `The byte range (${e - b} bytes) does not match the shape and dtype (${numel * PLAIN[kind].size} bytes).`; }
    if (base + e > source.size || b > e) { t.readable = false; t.note = 'The data range runs past the end of the file. The file may be truncated.'; }
    tensors.push(t);
  }
  return { format: 'safetensors', tensors, metadata, info: [{ label: 'Header', value: `${n.toLocaleString('en-US')} bytes` }], warnings };
}

// ---------- GGUF (metadata through @huggingface/gguf) ----------

function displayValue(v, depth = 0) {
  if (typeof v === 'bigint') return v.toString();
  if (Array.isArray(v)) {
    const head = v.slice(0, 16).map(x => displayValue(x, depth + 1));
    return `[${v.length.toLocaleString('en-US')} items] ${JSON.stringify(head).slice(0, 600)}${v.length > 16 ? ' ...' : ''}`;
  }
  if (typeof v === 'string') return v.length > 20000 ? v.slice(0, 20000) + ` ... (${v.length.toLocaleString('en-US')} characters)` : v;
  return String(v);
}

function fileTypeName(n) {
  // llama.cpp LLAMA_FTYPE names (src/llama.h); the file type is the "mostly" type of the big tensors.
  const names = { 0: 'ALL_F32', 1: 'MOSTLY_F16', 2: 'MOSTLY_Q4_0', 3: 'MOSTLY_Q4_1', 7: 'MOSTLY_Q8_0', 8: 'MOSTLY_Q5_0', 9: 'MOSTLY_Q5_1', 10: 'MOSTLY_Q2_K', 11: 'MOSTLY_Q3_K_S', 12: 'MOSTLY_Q3_K_M', 13: 'MOSTLY_Q3_K_L', 14: 'MOSTLY_Q4_K_S', 15: 'MOSTLY_Q4_K_M', 16: 'MOSTLY_Q5_K_S', 17: 'MOSTLY_Q5_K_M', 18: 'MOSTLY_Q6_K', 19: 'MOSTLY_IQ2_XXS', 20: 'MOSTLY_IQ2_XS', 21: 'MOSTLY_Q2_K_S', 22: 'MOSTLY_IQ3_XS', 23: 'MOSTLY_IQ3_XXS', 24: 'MOSTLY_IQ1_S', 25: 'MOSTLY_IQ4_NL', 26: 'MOSTLY_IQ3_S', 27: 'MOSTLY_IQ3_M', 28: 'MOSTLY_IQ2_S', 29: 'MOSTLY_IQ2_M', 30: 'MOSTLY_IQ4_XS', 31: 'MOSTLY_IQ1_M', 32: 'MOSTLY_BF16', 36: 'MOSTLY_TQ1_0', 37: 'MOSTLY_TQ2_0', 38: 'MOSTLY_MXFP4_MOE' };
  return names[n] ? `${names[n]} (${n})` : String(n);
}

async function openGguf(source) {
  const { gguf } = require('@huggingface/gguf');
  // The library reads through a fetch-style callback. We hand it our own reader, so the
  // ".invalid" URL is only a label: no request ever leaves the machine.
  const localFetch = async (_url, init) => {
    const m = /bytes=(\d+)-(\d+)/.exec((init && init.headers && init.headers.Range) || '');
    const start = m ? Number(m[1]) : 0;
    const end = Math.min(m ? Number(m[2]) : source.size - 1, source.size - 1);
    const bytes = start <= end ? await source.read(start, end - start + 1) : new Uint8Array(0);
    return new Response(bytes, { status: 206, headers: { 'content-range': `bytes ${start}-${end}/${source.size}` } });
  };
  const r = await gguf('https://tensor-viewer.invalid/local.gguf', { fetch: localFetch });
  const md = r.metadata;
  const le = r.littleEndian !== false;
  const dataStart = Number(r.tensorDataOffset);
  const tensors = [];
  const typeCount = new Map();
  let dataBytes = 0;
  for (const ti of r.tensorInfos) {
    const ne = ti.shape.map(Number); // ne[0] varies fastest
    const shape = ne.slice().reverse(); // row-major order, as in PyTorch
    const g = GGML[ti.dtype];
    const numel = prod(ne);
    const name = g ? g.name : `type ${ti.dtype}`;
    typeCount.set(name, (typeCount.get(name) || 0) + 1);
    const nbytes = g && ne[0] % g.block === 0 ? (numel / g.block) * g.bytes : null;
    if (nbytes) dataBytes += nbytes;
    const off = dataStart + Number(ti.offset);
    const t = { name: ti.name, dtype: name, shape, numel, nbytes, offset: off, readable: !!(g && (g.kind || g.quant)) && !!nbytes && numel > 0,
      _loc: { source, dataOffset: off, kind: g && g.kind, quant: g && g.quant, block: g ? g.block : 1, blockBytes: g ? g.bytes : 0, elemSize: g && g.kind ? g.bytes : 0, littleEndian: le, fortran: false } };
    if (g && !t.readable && nbytes) t.note = `${name} is listed with its size, but Tensor Viewer does not decode it yet.`;
    if (!g) t.note = `Unknown GGML type ${ti.dtype}.`;
    if (nbytes && off + nbytes > source.size) { t.readable = false; t.note = 'The data range runs past the end of the file. The file may be truncated or a split shard.'; }
    tensors.push(t);
  }
  const arch = md['general.architecture'];
  const metadata = [];
  for (const [k, v] of Object.entries(md)) {
    if (k === 'version' || k === 'tensor_count' || k === 'kv_count') continue;
    metadata.push({ key: k, value: k === 'general.file_type' ? fileTypeName(Number(v)) : displayValue(v) });
  }
  const pick = k => (md[k] !== undefined ? displayValue(md[k]) : undefined);
  const info = [
    { label: 'Architecture', value: arch !== undefined ? String(arch) : undefined },
    { label: 'Model name', value: pick('general.name') },
    { label: 'Context length', value: arch ? pick(`${arch}.context_length`) : undefined },
    { label: 'Embedding size', value: arch ? pick(`${arch}.embedding_length`) : undefined },
    { label: 'Layers', value: arch ? pick(`${arch}.block_count`) : undefined },
    { label: 'Attention heads', value: arch ? pick(`${arch}.attention.head_count`) : undefined },
    { label: 'KV heads', value: arch ? pick(`${arch}.attention.head_count_kv`) : undefined },
    { label: 'File type', value: md['general.file_type'] !== undefined ? fileTypeName(Number(md['general.file_type'])) : undefined },
    { label: 'Tokenizer', value: md['tokenizer.ggml.model'] !== undefined ? `${md['tokenizer.ggml.model']}, ${Array.isArray(md['tokenizer.ggml.tokens']) ? md['tokenizer.ggml.tokens'].length.toLocaleString('en-US') : '?'} tokens` : undefined },
    { label: 'Chat template', value: typeof md['tokenizer.chat_template'] === 'string' ? `yes (${md['tokenizer.chat_template'].length.toLocaleString('en-US')} characters)` : undefined },
    { label: 'Tensor types', value: [...typeCount.entries()].sort((a, b) => b[1] - a[1]).map(([k, c]) => `${k} x${c}`).join(', ') },
    { label: 'GGUF version', value: String(md.version) },
  ].filter(x => x.value !== undefined && x.value !== '');
  const params = tensors.reduce((a, t) => a + t.numel, 0);
  if (params && dataBytes) info.push({ label: 'Bits per weight', value: (dataBytes * 8 / params).toFixed(2) });
  const warnings = [];
  if (md['split.count'] !== undefined && Number(md['split.count']) > 1) warnings.push(`This is shard ${Number(md['split.no']) + 1} of ${md['split.count']}. Only the tensors in this file are listed.`);
  return { format: 'gguf', tensors, metadata, info, warnings };
}

/**
 * Open a tensor file from a Source. With opts.raw ({kind, shape, offset, littleEndian}) the file is read as a raw array.
 * @returns {Promise<{format:string, tensors:object[], metadata:object[], info:object[], warnings:string[], totalParams:number, fileSize:number}>}
 */
async function openTensorFile(source, fileName, opts = {}) {
  if (opts.raw) {
    const model = openRaw(source, opts.raw);
    model.fileSize = source.size;
    model.totalParams = model.tensors.reduce((a, t) => a + (t.numel || 0), 0);
    return model;
  }
  if (source.size === 0) throw new Error(`${fileName} is empty (0 bytes), so there is nothing to show.`);
  const format = formatFromName(fileName);
  if (!format) throw new Error(`Tensor Viewer opens .npy, .npz, .safetensors and .gguf files, not ${fileName}.`);
  const fn = { npy: openNpy, npz: openNpz, safetensors: openSafetensors, gguf: openGguf }[format];
  const model = await fn(source);
  model.fileSize = source.size;
  model.totalParams = model.tensors.reduce((a, t) => a + (t.numel || 0), 0);
  return model;
}

module.exports = { openTensorFile, formatFromName, displayValue, fileTypeName };
