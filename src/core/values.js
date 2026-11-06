'use strict';
// Exact values, slices and export for one tensor.
// - readValues: one page of a 2-D view (a slice of an N-D tensor), read by offset, so a page
//   of a 10 GB tensor costs a few KB.
// - sliceTensor: a C-order slice as its own tensor, so stats and the heatmap can use it.
// - exportTensor: writes a tensor or a slice as .npy or CSV in chunks, never all in memory.
const { decodePlain, decodeQuant, PLAIN } = require('./dtypes');
const { exactText, complexText } = require('./numtext');

const MAX_PAGE_ROWS = 200;
const MAX_PAGE_COLS = 64;
const MAX_INFLATE_BYTES = 64 * 1024 * 1024; // compressed .npz entries are inflated from the start
const CSV_MAX_VALUES = 20000000;
const EXPORT_CHUNK_BYTES = 4 * 1024 * 1024;

const prod = a => a.reduce((x, y) => x * y, 1);

/** The 2-D view of a tensor: rows x cols, and how many leading indices pick a slice. */
function gridShape(shape) {
  const n = shape.length;
  if (n === 0) return { rows: 1, cols: 1, lead: [] };
  if (n === 1) return { rows: shape[0], cols: 1, lead: [] };
  return { rows: shape[n - 2], cols: shape[n - 1], lead: shape.slice(0, n - 2) };
}

/** Checks a slice (one index per leading dimension). Returns the normalized array or null. */
function checkSlice(shape, slice) {
  const { lead } = gridShape(shape);
  if (slice === null || slice === undefined) return lead.map(() => 0);
  if (!Array.isArray(slice) || slice.length !== lead.length) return null;
  for (let k = 0; k < lead.length; k++) if (!Number.isInteger(slice[k]) || slice[k] < 0 || slice[k] >= lead[k]) return null;
  return slice.slice();
}

/** Flat memory index of element (lead..., r, c), C or Fortran order. */
function memIndexer(shape, fortran) {
  const n = shape.length;
  const strides = new Array(n);
  if (fortran) { let s = 1; for (let k = 0; k < n; k++) { strides[k] = s; s *= shape[k]; } } else { let s = 1; for (let k = n - 1; k >= 0; k--) { strides[k] = s; s *= shape[k]; } }
  return (slice, r, c) => {
    if (n === 0) return 0;
    if (n === 1) return r * strides[0];
    let i = 0;
    for (let k = 0; k < n - 2; k++) i += slice[k] * strides[k];
    return i + r * strides[n - 2] + c * strides[n - 1];
  };
}

function isComplex(kind) { return kind === 'c64' || kind === 'c128'; }

/** Decode `count` values starting at flat memory index `start`. Complex gives {re, im}. */
async function readRun(loc, start, count) {
  const src = loc.source;
  if (loc.quant) {
    const b0 = Math.floor(start / loc.block), b1 = Math.floor((start + count - 1) / loc.block) + 1;
    const off = loc.dataOffset + b0 * loc.blockBytes, len = (b1 - b0) * loc.blockBytes;
    guardInflate(src, off + len);
    const all = decodeQuant(loc.quant, await src.read(off, len), b1 - b0);
    return { re: all.subarray(start - b0 * loc.block, start - b0 * loc.block + count), im: null };
  }
  const size = loc.elemSize;
  const off = loc.dataOffset + start * size, len = count * size;
  guardInflate(src, off + len);
  const bytes = await src.read(off, len);
  if (isComplex(loc.kind)) {
    const half = loc.kind === 'c64' ? 'f32' : 'f64';
    const parts = decodePlain(half, bytes, count * 2, loc.littleEndian);
    const re = new Float64Array(count), im = new Float64Array(count);
    for (let i = 0; i < count; i++) { re[i] = parts[2 * i]; im[i] = parts[2 * i + 1]; }
    return { re, im };
  }
  return { re: decodePlain(loc.kind, bytes, count, loc.littleEndian), im: null };
}

function guardInflate(src, end) {
  if (src.random === false && end > MAX_INFLATE_BYTES) throw new Error(`This array is in a compressed .npz entry, which can only be read from the start. Values past the first ${MAX_INFLATE_BYTES / 1048576} MB are not shown. Save the entry uncompressed (np.savez) to see all of it.`);
}

/**
 * One page of values.
 * @param t tensor with _loc
 * @param q {slice:number[]|null, row:number, col:number, rows:number, cols:number}
 */
async function readValues(t, q) {
  if (!t.readable || !t._loc) throw new Error(t.note || 'This tensor cannot be read.');
  const g = gridShape(t.shape);
  const slice = checkSlice(t.shape, q.slice);
  if (!slice) throw new Error('Bad slice.');
  const row = Math.max(0, Math.min(q.row | 0, Math.max(0, g.rows - 1)));
  const col = Math.max(0, Math.min(q.col | 0, Math.max(0, g.cols - 1)));
  const rows = Math.max(0, Math.min(q.rows | 0, MAX_PAGE_ROWS, g.rows - row));
  const cols = Math.max(0, Math.min(q.cols | 0, MAX_PAGE_COLS, g.cols - col));
  const loc = t._loc;
  const idx = memIndexer(t.shape, !!loc.fortran);
  const re = new Float64Array(rows * cols);
  const im = isComplex(loc.kind) ? new Float64Array(rows * cols) : null;
  // Group cells into runs of consecutive memory positions: a row in C order, a column in Fortran order.
  const cells = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) cells.push([idx(slice, row + r, col + c), r * cols + c]);
  cells.sort((a, b) => a[0] - b[0]);
  let i = 0;
  while (i < cells.length) {
    let j = i + 1;
    while (j < cells.length && cells[j][0] === cells[j - 1][0] + 1) j++;
    const run = await readRun(loc, cells[i][0], j - i);
    for (let k = i; k < j; k++) { re[cells[k][1]] = run.re[k - i]; if (im) im[cells[k][1]] = run.im[k - i]; }
    i = j;
  }
  return { slice, row, col, rows, cols, totalRows: g.rows, totalCols: g.cols, kind: loc.kind || loc.quant, re, im };
}

/** A C-order slice [s0, s1, ..., :, :] as a tensor of shape [rows, cols]. Null when not possible. */
function sliceTensor(t, slice) {
  if (!t._loc || t.shape.length < 3) return null;
  const s = checkSlice(t.shape, slice);
  if (!s) throw new Error('Bad slice.');
  if (t._loc.fortran) throw new Error('Slices of Fortran-order arrays are shown in the values table only. The stats and the heatmap cover the whole array.');
  const g = gridShape(t.shape);
  const per = g.rows * g.cols;
  let lin = 0;
  for (let k = 0; k < g.lead.length; k++) lin = lin * g.lead[k] + s[k];
  const start = lin * per;
  const loc = t._loc;
  const off = loc.quant ? (start / loc.block) * loc.blockBytes : start * loc.elemSize;
  const nbytes = loc.quant ? (per / loc.block) * loc.blockBytes : per * loc.elemSize;
  return { ...t, shape: [g.rows, g.cols], numel: per, nbytes, _loc: { ...loc, dataOffset: loc.dataOffset + off } };
}

function sliceLabel(shape, slice) {
  const g = gridShape(shape);
  if (!g.lead.length) return '';
  return '[' + slice.concat([':', ':']).join(', ') + ']';
}

// ---------- export ----------

const NPY_DESCR = { bool: 'b1', i8: 'i1', u8: 'u1', i16: 'i2', u16: 'u2', i32: 'i4', u32: 'u4', i64: 'i8', u64: 'u8', f16: 'f2', f32: 'f4', f64: 'f8', c64: 'c8', c128: 'c16' };

/** How a tensor is written to .npy: the bytes as they are, or decoded to float32. */
function npyPlan(t) {
  const loc = t._loc;
  const d = !loc.quant && NPY_DESCR[loc.kind];
  if (d) {
    const one = PLAIN[loc.kind].size === 1;
    return { raw: true, descr: (one ? '|' : loc.littleEndian ? '<' : '>') + d, fortran: !!loc.fortran, note: '' };
  }
  return { raw: false, descr: '<f4', fortran: false, note: `${t.dtype} has no NumPy type, so the values are saved as float32.` };
}

function npyHeader(descr, fortran, shape) {
  const sh = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;
  let dict = `{'descr': '${descr}', 'fortran_order': ${fortran ? 'True' : 'False'}, 'shape': ${sh}, }`;
  const pad = 64 - ((10 + dict.length + 1) % 64);
  dict += ' '.repeat(pad % 64) + '\n';
  const h = Buffer.alloc(10 + dict.length);
  h.write('\x93NUMPY', 0, 'latin1'); h[6] = 1; h[7] = 0; h.writeUInt16LE(dict.length, 8); h.write(dict, 10, 'latin1');
  return h;
}

/**
 * What an export will write, before any bytes are read. Throws with a clear message when it cannot.
 * @returns {{tensor:object, shape:number[], label:string, format:string, note:string}}
 */
function planExport(t, format, slice) {
  if (!t.readable || !t._loc) throw new Error(t.note || 'This tensor cannot be read.');
  let src = t, label = '';
  if (slice) {
    src = sliceTensor(t, slice);
    if (!src) throw new Error('Only tensors with 3 or more dimensions have slices.');
    label = sliceLabel(t.shape, checkSlice(t.shape, slice));
  }
  if (src._loc.source.random === false && src._loc.dataOffset + src.nbytes > 512 * 1048576) throw new Error('This array is in a compressed .npz entry larger than 512 MB. Save it uncompressed (np.savez) to export it.');
  if (format === 'csv' && src.numel > CSV_MAX_VALUES) throw new Error(`This tensor has ${src.numel.toLocaleString('en-US')} values. CSV export stops at ${CSV_MAX_VALUES.toLocaleString('en-US')} values. Save it as .npy instead${t.shape.length >= 3 && !slice ? ', or export one slice' : ''}.`);
  const note = format === 'npy' ? npyPlan(src).note : (src.shape.length >= 3 ? 'Leading dimensions are flattened into rows.' : '');
  return { tensor: src, shape: src.shape.slice(), label, format, note };
}

/**
 * Writes the planned export through `write(buffer)` in chunks.
 * @param plan from planExport
 * @param write async (Buffer) => void
 * @param opts {cancelled?:()=>boolean, progress?:(fraction:number)=>void}
 */
async function writeExport(plan, write, opts = {}) {
  const t = plan.tensor, loc = t._loc;
  const cancelled = opts.cancelled || (() => false), progress = opts.progress || (() => {});
  const total = Math.max(1, t.numel);
  const unit = loc.quant ? loc.block : 1;
  const unitBytes = loc.quant ? loc.blockBytes : loc.elemSize;
  // Chunk in whole blocks for quant types, and in whole rows for CSV so each line is complete.
  const cols = plan.format === 'csv' ? (t.shape.length >= 2 ? t.shape[t.shape.length - 1] : 1) : unit;
  let step = Math.max(unit, Math.floor(EXPORT_CHUNK_BYTES / unitBytes) * unit);
  if (plan.format === 'csv') step = Math.max(cols, Math.floor(step / cols) * cols); // readRun handles block edges
  if (plan.format === 'npy') await write(npyHeader(npyPlan(t).descr, npyPlan(t).fortran, t.shape));
  const fortranCsv = plan.format === 'csv' && loc.fortran && t.shape.length >= 2;
  if (fortranCsv) return writeFortranCsv(t, write, cancelled, progress);
  const raw = plan.format === 'npy' && npyPlan(t).raw;
  for (let start = 0; start < t.numel; start += step) {
    if (cancelled()) return false;
    const count = Math.min(step, t.numel - start);
    if (raw) {
      await write(Buffer.from(await loc.source.read(loc.dataOffset + start * loc.elemSize, count * loc.elemSize)));
    } else {
      const { re, im } = await readRun(loc, start, count);
      if (plan.format === 'npy') {
        const f = new Float32Array(count);
        f.set(re);
        await write(Buffer.from(f.buffer));
      } else {
        await write(Buffer.from(csvLines(re, im, cols, loc.kind || loc.quant), 'utf8'));
      }
    }
    progress((start + count) / total);
  }
  return true;
}

function writeFortranCsv(t, write, cancelled, progress) {
  // Rare case: read row by row through the values reader (Fortran order keeps columns together).
  const g = gridShape(t.shape);
  const lead = g.lead, nslices = prod(lead);
  return (async () => {
    let done = 0;
    for (let s = 0; s < nslices; s++) {
      const slice = []; let rem = s;
      for (let k = lead.length - 1; k >= 0; k--) { slice.unshift(rem % lead[k]); rem = Math.floor(rem / lead[k]); }
      for (let r = 0; r < g.rows; r += MAX_PAGE_ROWS) {
        if (cancelled()) return false;
        let text = '';
        const n = Math.min(MAX_PAGE_ROWS, g.rows - r);
        const rowsRe = [];
        for (let c = 0; c < g.cols; c += MAX_PAGE_COLS) rowsRe.push(await readValues(t, { slice, row: r, col: c, rows: n, cols: MAX_PAGE_COLS }));
        for (let i = 0; i < n; i++) {
          const parts = [];
          for (const v of rowsRe) for (let j = 0; j < v.cols; j++) parts.push(v.im ? complexText(v.re[i * v.cols + j], v.im[i * v.cols + j], v.kind) : exactText(v.re[i * v.cols + j], v.kind));
          text += parts.join(',') + '\n';
        }
        await write(Buffer.from(text, 'utf8'));
        done += n * g.cols;
        progress(done / Math.max(1, t.numel));
      }
    }
    return true;
  })();
}

function csvLines(re, im, cols, kind) {
  const out = [];
  for (let i = 0; i < re.length; i += cols) {
    const parts = new Array(Math.min(cols, re.length - i));
    for (let j = 0; j < parts.length; j++) parts[j] = im ? complexText(re[i + j], im[i + j], kind) : exactText(re[i + j], kind);
    out.push(parts.join(','));
  }
  return out.join('\n') + '\n';
}

/** Tab-separated text of one page (pastes into a spreadsheet). */
function pageText(v) {
  const lines = [];
  for (let r = 0; r < v.rows; r++) {
    const parts = [];
    for (let c = 0; c < v.cols; c++) { const i = r * v.cols + c; parts.push(v.im ? complexText(v.re[i], v.im[i], v.kind) : exactText(v.re[i], v.kind)); }
    lines.push(parts.join('\t'));
  }
  return lines.join('\n');
}

/** A safe default file name for an export. */
function exportName(tensorName, label, ext) {
  let base = String(tensorName || 'tensor').replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[._]+/, '').slice(0, 120) || 'tensor';
  if (label) base += '_' + label.replace(/[^0-9]+/g, '_').replace(/^_+|_+$/g, '');
  return `${base}.${ext}`;
}

module.exports = {
  MAX_PAGE_ROWS, MAX_PAGE_COLS, CSV_MAX_VALUES,
  gridShape, checkSlice, readValues, sliceTensor, sliceLabel, exactText, complexText,
  npyHeader, npyPlan, planExport, writeExport, pageText, exportName,
};
