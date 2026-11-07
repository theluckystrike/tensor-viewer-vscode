'use strict';
// Draws one frame of an array as a picture: H x W (grayscale), H x W x C or C x H x W (C = 1, 3 or 4),
// with any number of leading "frame" dimensions (N x H x W, N x C x H x W, ...).
// Only the source rows that the downscaled picture needs are read, so a 20,000 x 20,000 image or
// frame 9,000 of a big stack costs a few MB.
const { decodePlain, PLAIN } = require('./dtypes');

const IMAGE_KINDS = new Set(['bool', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'f16', 'bf16', 'f32', 'f64', 'f8e4m3', 'f8e5m2']);
const LAYOUT_LABELS = { hwc: 'H x W x C (channels last)', chw: 'C x H x W (channels first)', gray: 'H x W (grayscale)' };
const RANGES = ['auto', 'minmax', '01', '0255', 'sym'];
const DEFAULTS = { imageMax: 512, compressedMaxBytes: 64 * 1024 * 1024, maxRowBytes: 64 * 1024 * 1024 };
const prod = a => a.reduce((x, y) => x * y, 1);
const isChannels = c => c === 1 || c === 3 || c === 4;

/**
 * The picture layouts a tensor can be shown in, best guess first. Empty when it can't be a picture.
 * @returns {{id:string, label:string, H:number, W:number, C:number, frames:number, frameDims:number[]}[]}
 */
function imageLayouts(t) {
  if (!t || !t.readable || !t._loc || t._loc.quant || t._loc.fortran || !IMAGE_KINDS.has(t._loc.kind)) return [];
  const s = t.shape;
  const out = [];
  const add = (id, H, W, C, frameDims) => { if (H >= 2 && W >= 2) out.push({ id, label: LAYOUT_LABELS[id], H, W, C, frames: prod(frameDims), frameDims }); };
  const n = s.length;
  if (n >= 3 && isChannels(s[n - 1]) && s[n - 1] > 1) add('hwc', s[n - 3], s[n - 2], s[n - 1], s.slice(0, n - 3));
  // C x H x W with 3 channels is a common tensor layout; 4 leading planes are more often 4 frames, so gray goes first then.
  const chw = n >= 3 && isChannels(s[n - 3]) && s[n - 3] > 1;
  if (chw && s[n - 3] === 3) add('chw', s[n - 2], s[n - 1], s[n - 3], s.slice(0, n - 3));
  if (n >= 2) add('gray', s[n - 2], s[n - 1], 1, s.slice(0, n - 2));
  if (chw && s[n - 3] === 4) add('chw', s[n - 2], s[n - 1], s[n - 3], s.slice(0, n - 3));
  // A (H, W, 1) array is a grayscale picture; drop the trailing 1 so it isn't read as frames of H rows.
  if (n >= 3 && s[n - 1] === 1) {
    const g = out.findIndex(l => l.id === 'gray');
    if (g >= 0) out.splice(g, 1);
    add('gray', s[n - 3], s[n - 2], 1, s.slice(0, n - 3));
  }
  return out;
}

/** True when the picture should show without a click: RGB(A) layouts and uint8/bool arrays. */
function suggestImage(t, layouts = imageLayouts(t)) {
  if (!layouts.length) return false;
  if (layouts[0].id !== 'gray') return true;
  return t._loc.kind === 'u8' || t._loc.kind === 'bool';
}

function rangeFor(kind, mode, lo, hi) {
  if (mode === '01') return [0, 1];
  if (mode === '0255') return [0, 255];
  if (mode === 'sym') { const a = Math.max(Math.abs(lo), Math.abs(hi)) || 1; return [-a, a]; }
  if (mode === 'auto') {
    if (kind === 'u8') return [0, 255];
    if (kind === 'bool') return [0, 1];
    if (lo >= 0 && hi <= 1) return [0, 1];
    if (lo >= 0 && hi <= 255 && kind !== 'f16' && kind !== 'bf16' && kind !== 'f32' && kind !== 'f64') return [0, 255];
  }
  if (!(hi > lo)) return [lo, lo + 1];
  return [lo, hi];
}

/**
 * Read one frame and turn it into RGBA pixels, at most opts.imageMax on the long side.
 * @param {{layout?:string, frame?:number, range?:string}} req
 */
async function readImage(t, req = {}, options = {}) {
  const opts = { ...DEFAULTS, ...options };
  const layouts = imageLayouts(t);
  if (!layouts.length) throw new Error('This tensor has no picture layout. Pictures need at least 2 dimensions of 2 or more.');
  const L = layouts.find(l => l.id === req.layout) || layouts[0];
  const frame = Number.isInteger(req.frame) ? req.frame : 0;
  if (frame < 0 || frame >= L.frames) throw new Error(`Frame ${frame} is out of range (0 to ${L.frames - 1}).`);
  const range = RANGES.includes(req.range) ? req.range : 'auto';
  const loc = t._loc;
  const es = loc.elemSize || PLAIN[loc.kind].size;
  const { H, W, C } = L;
  const frameValues = H * W * C;
  const frameStart = frame * frameValues; // in values, memory order
  const random = loc.source.random !== false;
  if (!random && (frameStart + frameValues) * es > opts.compressedMaxBytes) {
    throw new Error(`This entry is compressed, so only pictures in its first ${Math.round(opts.compressedMaxBytes / 1048576)} MB can be shown. Save it uncompressed (np.savez) to see this frame.`);
  }
  const rowValues = L.id === 'chw' ? W : W * C;
  if (rowValues * es > opts.maxRowBytes) throw new Error('One row of this picture is too large to read.');
  const scale = Math.min(1, opts.imageMax / Math.max(H, W));
  const h = Math.max(1, Math.round(H * scale)), w = Math.max(1, Math.round(W * scale));
  const srcY = new Int32Array(h), srcX = new Int32Array(w);
  for (let y = 0; y < h; y++) srcY[y] = Math.min(H - 1, Math.floor(y / scale));
  for (let x = 0; x < w; x++) srcX[x] = Math.min(W - 1, Math.floor(x / scale));
  // vals[(y*w + x)*C + c]: the picked source values, as float64.
  const vals = new Float64Array(h * w * C);
  const read = async (startValue, count) => decodePlain(loc.kind, await loc.source.read(loc.dataOffset + startValue * es, count * es), count, loc.littleEndian);
  for (let y = 0; y < h; y++) {
    if (L.id === 'chw') {
      for (let c = 0; c < C; c++) {
        const row = await read(frameStart + c * H * W + srcY[y] * W, W);
        for (let x = 0; x < w; x++) vals[(y * w + x) * C + c] = row[srcX[x]];
      }
    } else {
      const row = await read(frameStart + srcY[y] * W * C, W * C);
      for (let x = 0; x < w; x++) for (let c = 0; c < C; c++) vals[(y * w + x) * C + c] = row[srcX[x] * C + c];
    }
  }
  let lo = Infinity, hi = -Infinity, nonFinite = 0;
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i];
    if (!Number.isFinite(v)) { nonFinite++; continue; }
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo === Infinity) { lo = 0; hi = 0; }
  const [a, b] = rangeFor(loc.kind, range, lo, hi);
  const k = 255 / (b - a);
  const px = v => (Number.isFinite(v) ? Math.max(0, Math.min(255, Math.round((v - a) * k))) : 0);
  const rgba = new Uint8Array(h * w * 4);
  for (let i = 0, n = h * w; i < n; i++) {
    const s = i * C, d = i * 4;
    if (C >= 3) { rgba[d] = px(vals[s]); rgba[d + 1] = px(vals[s + 1]); rgba[d + 2] = px(vals[s + 2]); }
    else { const g = px(vals[s]); rgba[d] = g; rgba[d + 1] = g; rgba[d + 2] = g; }
    // NaN and Inf show as magenta, so they stand out on any picture.
    if (C === 1 && !Number.isFinite(vals[s])) { rgba[d] = 255; rgba[d + 1] = 0; rgba[d + 2] = 255; }
    rgba[d + 3] = C === 4 ? px(vals[s + 3]) : 255;
  }
  return {
    kind: 'image', layout: L.id, frame, frames: L.frames, frameDims: L.frameDims, frameIndex: unravel(frame, L.frameDims),
    width: w, height: h, srcWidth: W, srcHeight: H, channels: C, dtype: t.dtype, range, lo: a, hi: b, dataMin: lo, dataMax: hi, nonFinite,
    data: Buffer.from(rgba).toString('base64'),
  };
}

/** Flat frame number to one index per frame dimension. */
function unravel(i, dims) {
  const out = new Array(dims.length).fill(0);
  for (let k = dims.length - 1; k >= 0; k--) { out[k] = i % dims[k]; i = Math.floor(i / dims[k]); }
  return out;
}

/** Plain data for the webview: which layouts a tensor offers and whether to show the picture at once. */
function imageInfo(t) {
  const layouts = imageLayouts(t);
  if (!layouts.length) return null;
  return { suggest: suggestImage(t, layouts), layouts: layouts.map(l => ({ id: l.id, label: l.label, H: l.H, W: l.W, C: l.C, frames: l.frames })) };
}

module.exports = { imageLayouts, suggestImage, readImage, imageInfo, unravel, RANGES, rangeFor };
