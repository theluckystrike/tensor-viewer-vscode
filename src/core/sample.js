'use strict';
// Reads a bounded sample of one tensor and turns it into stats, a histogram and a picture.
// The sample is whole rows spread evenly over the tensor, so a 10 GB tensor costs about as
// much as a 10 MB one.
const { decodePlain, decodeQuant } = require('./dtypes');

const DEFAULTS = { sampleValues: 2000000, imageMaxBytes: 64 * 1024 * 1024, grid: 256, linePoints: 1000, bins: 64, imageMax: 512 };

function layoutOf(t) {
  const loc = t._loc;
  let shape = t.shape.length ? t.shape.slice() : [1];
  let transposed = false;
  if (loc.fortran && shape.length >= 2) { shape = shape.slice().reverse(); transposed = true; }
  const cols = shape[shape.length - 1];
  const rows = t.numel / cols;
  const block = loc.quant ? loc.block : 1;
  const unitBytes = loc.quant ? loc.blockBytes : loc.elemSize; // bytes per block (1 element for plain types)
  if (cols % block) throw new Error(`Row length ${cols} is not a multiple of the ${block}-value block.`);
  const rowBytes = (cols / block) * unitBytes;
  return { rows, cols, block, unitBytes, rowBytes, transposed, memShape: shape };
}

function decodeRange(loc, bytes, values) {
  if (loc.quant) return decodeQuant(loc.quant, bytes, values / loc.block);
  return decodePlain(loc.kind, bytes, values, loc.littleEndian);
}

/**
 * Plan which pieces to read. Each piece is a run of whole rows (or, for very long rows, of whole blocks).
 * @returns {{pieces:{start:number, values:number}[], exact:boolean, how:string}}  start/values count values (flat, memory order)
 */
function planPieces(t, L, opts, random) {
  const budget = Math.max(L.block, Math.floor(opts.sampleValues / L.block) * L.block);
  if (t.numel <= budget) return { pieces: [{ start: 0, values: t.numel }], exact: true, how: `all ${t.numel.toLocaleString('en-US')} values` };
  if (!random) {
    const rows = Math.max(1, Math.floor(budget / L.cols));
    const values = L.cols > budget ? budget : rows * L.cols;
    return { pieces: [{ start: 0, values }], exact: false, how: `the first ${values.toLocaleString('en-US')} values (compressed entry)` };
  }
  if (L.cols <= budget) {
    const rowsWanted = Math.floor(budget / L.cols);
    const k = Math.min(rowsWanted, 512); // pieces
    const per = Math.max(1, Math.floor(rowsWanted / k));
    const pieces = [];
    let last = -1;
    for (let i = 0; i < k; i++) {
      let r = Math.floor((i * (L.rows - per)) / Math.max(1, k - 1));
      if (r <= last) r = last + 1;
      if (r + per > L.rows) break;
      pieces.push({ start: r * L.cols, values: per * L.cols });
      last = r + per - 1;
    }
    const n = pieces.reduce((a, p) => a + p.values, 0);
    return { pieces, exact: false, how: `${n.toLocaleString('en-US')} values in ${(n / L.cols).toLocaleString('en-US')} rows spread over all ${L.rows.toLocaleString('en-US')} rows` };
  }
  // Rows longer than the budget: 256 runs of blocks spread over the flat tensor.
  const k = 256;
  const per = Math.max(L.block, Math.floor(budget / k / L.block) * L.block);
  const pieces = [];
  for (let i = 0; i < k; i++) {
    let s = Math.floor(((t.numel - per) * i) / (k - 1) / L.block) * L.block;
    pieces.push({ start: s, values: per });
  }
  return { pieces, exact: false, how: `${(per * k).toLocaleString('en-US')} values in ${k} runs spread over the tensor` };
}

async function readPieces(t, L, plan) {
  const loc = t._loc;
  const out = [];
  for (const p of plan.pieces) {
    const off = loc.dataOffset + (p.start / L.block) * L.unitBytes;
    const len = (p.values / L.block) * L.unitBytes;
    const bytes = await loc.source.read(off, len);
    out.push({ start: p.start, data: decodeRange(loc, bytes, p.values) });
  }
  return out;
}

function computeStats(pieces, bins) {
  // Two passes over the sample (sum, then squared deviations) keep mean and std accurate
  // even for large integers, where a running update drifts.
  let n = 0, nan = 0, pinf = 0, ninf = 0, zeros = 0, sum = 0, min = Infinity, max = -Infinity;
  for (const { data } of pieces) {
    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      if (x !== x) { nan++; continue; }
      if (x === Infinity) { pinf++; continue; }
      if (x === -Infinity) { ninf++; continue; }
      if (x === 0) zeros++;
      n++; sum += x;
      if (x < min) min = x;
      if (x > max) max = x;
    }
  }
  const mean = n ? sum / n : 0;
  let ss = 0;
  if (n) for (const { data } of pieces) for (let i = 0; i < data.length; i++) { const x = data[i]; if (Number.isFinite(x)) ss += (x - mean) * (x - mean); }
  const count = n + nan + pinf + ninf;
  const stats = { count, finite: n, nan, posinf: pinf, neginf: ninf, zeros, min: n ? min : null, max: n ? max : null, mean: n ? mean : null, std: n ? Math.sqrt(ss / n) : null };
  let hist = null;
  if (n) {
    const counts = new Array(bins).fill(0);
    const span = max - min;
    for (const { data } of pieces) for (let i = 0; i < data.length; i++) {
      const x = data[i];
      if (!Number.isFinite(x)) continue;
      const b = span > 0 ? Math.min(bins - 1, Math.floor(((x - min) / span) * bins)) : 0;
      counts[b]++;
    }
    hist = { min, max, counts };
  }
  return { stats, hist };
}

function b64(typed) { return Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength).toString('base64'); }

/** 2-D grid of means, at most grid x grid, from the sampled rows. NaN where no sample fell. */
function heatmap(pieces, L, opts) {
  const R = L.rows, C = L.cols;
  const gh = Math.min(opts.grid, R), gw = Math.min(opts.grid, C);
  const sum = new Float64Array(gh * gw), cnt = new Uint32Array(gh * gw);
  for (const { start, data } of pieces) {
    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      if (!Number.isFinite(x)) continue;
      const flat = start + i;
      const r = Math.floor(flat / C), c = flat - r * C;
      const gi = Math.floor((r * gh) / R) * gw + Math.floor((c * gw) / C);
      sum[gi] += x; cnt[gi]++;
    }
  }
  let g = new Float32Array(gh * gw);
  for (let i = 0; i < g.length; i++) g[i] = cnt[i] ? sum[i] / cnt[i] : NaN;
  let h = gh, w = gw;
  if (L.transposed) { // Fortran order: memory rows are display columns.
    const tg = new Float32Array(g.length);
    for (let r = 0; r < gh; r++) for (let c = 0; c < gw; c++) tg[c * gh + r] = g[r * gw + c];
    g = tg; h = gw; w = gh;
  }
  return { kind: 'heatmap', rows: h, cols: w, srcRows: L.transposed ? C : R, srcCols: L.transposed ? R : C, data: b64(g), reshaped: L.memShape.length > 2 };
}

function linePlot(pieces, numel, opts) {
  const n = Math.min(opts.linePoints, numel);
  const mn = new Float32Array(n).fill(NaN), mx = new Float32Array(n).fill(NaN), mean = new Float32Array(n).fill(NaN);
  const sum = new Float64Array(n), cnt = new Uint32Array(n);
  for (const { start, data } of pieces) for (let i = 0; i < data.length; i++) {
    const x = data[i];
    if (!Number.isFinite(x)) continue;
    const b = Math.floor(((start + i) * n) / numel);
    if (!(mn[b] <= x)) mn[b] = x;
    if (!(mx[b] >= x)) mx[b] = x;
    sum[b] += x; cnt[b]++;
  }
  for (let i = 0; i < n; i++) if (cnt[i]) mean[i] = sum[i] / cnt[i];
  return { kind: 'line', points: n, length: numel, min: b64(mn), max: b64(mx), mean: b64(mean) };
}

function isImage(t) {
  const s = t.shape;
  return t._loc.kind === 'u8' && !t._loc.fortran && s.length === 3 && (s[2] === 3 || s[2] === 4 || s[2] === 1) && s[0] >= 2 && s[1] >= 2;
}

async function image(t, opts) {
  const [H, W, C] = t.shape;
  const bytes = await t._loc.source.read(t._loc.dataOffset, H * W * C);
  const scale = Math.min(1, opts.imageMax / Math.max(H, W));
  const h = Math.max(1, Math.round(H * scale)), w = Math.max(1, Math.round(W * scale));
  const rgba = new Uint8Array(h * w * 4);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(H - 1, Math.floor(y / scale));
    for (let x = 0; x < w; x++) {
      const sx = Math.min(W - 1, Math.floor(x / scale));
      const si = (sy * W + sx) * C, di = (y * w + x) * 4;
      rgba[di] = bytes[si]; rgba[di + 1] = bytes[si + (C > 1 ? 1 : 0)]; rgba[di + 2] = bytes[si + (C > 1 ? 2 : 0)]; rgba[di + 3] = C === 4 ? bytes[si + 3] : 255;
    }
  }
  return { kind: 'image', width: w, height: h, srcWidth: W, srcHeight: H, channels: C, data: Buffer.from(rgba).toString('base64') };
}

/**
 * Sample one tensor.
 * @returns {Promise<{stats:object, hist:object|null, view:object|null, how:string, exact:boolean, ms:number}>}
 */
async function sampleTensor(t, options = {}) {
  const t0 = Date.now();
  if (!t.readable || !t._loc) throw new Error(t.note || 'This tensor cannot be read.');
  const opts = { ...DEFAULTS, ...options };
  const L = layoutOf(t);
  const random = t._loc.source.random !== false;
  const plan = planPieces(t, L, opts, random);
  const pieces = await readPieces(t, L, plan);
  const { stats, hist } = computeStats(pieces, opts.bins);
  let view = null;
  if (isImage(t) && random && t.nbytes <= opts.imageMaxBytes) view = await image(t, opts);
  else if (t.shape.length >= 2 && L.cols > 1 && L.rows > 1 && L.cols <= opts.sampleValues) view = heatmap(pieces, L, opts);
  else if (t.numel > 1) view = linePlot(pieces, t.numel, opts);
  return { stats, hist, view, how: plan.how, exact: plan.exact, ms: Date.now() - t0 };
}

/** Python tuple for a shape: (), (5,), (2, 3). */
function shapeAsPython(shape) {
  if (shape.length === 0) return '()';
  if (shape.length === 1) return `(${shape[0]},)`;
  return `(${shape.join(', ')})`;
}

module.exports = { sampleTensor, planPieces, layoutOf, computeStats, shapeAsPython, DEFAULTS };
