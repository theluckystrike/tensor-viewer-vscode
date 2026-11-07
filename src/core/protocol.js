'use strict';
// Webview message protocol. The webview is treated as untrusted input: every message is checked
// here before the host acts on it, and host messages never carry internal file handles.

const PROTOCOL_VERSION = 1;
const COPY_WHAT = new Set(['shape', 'name', 'torch']);
const EXPORT_FORMATS = new Set(['npy', 'csv']);
const { checkSlice, MAX_PAGE_ROWS, MAX_PAGE_COLS } = require('./values');
const { imageInfo, RANGES } = require('./image');
const LAYOUTS = new Set(['hwc', 'chw', 'gray']);
const nat = (v, max) => Number.isInteger(v) && v >= 0 && v <= max;

/**
 * @param tensors optional tensor list; when given, slices are checked against the tensor's shape.
 * @returns {{ok:true, msg:object} | {ok:false, error:string}}
 */
function validateWebviewMessage(m, tensorCount, tensors) {
  if (!m || typeof m !== 'object' || Array.isArray(m)) return { ok: false, error: 'not an object' };
  const idx = i => Number.isInteger(i) && i >= 0 && i < tensorCount;
  switch (m.type) {
    case 'ready': return { ok: true, msg: { type: 'ready' } };
    case 'rawLayout': return { ok: true, msg: { type: 'rawLayout' } };
    case 'image': {
      if (!idx(m.index)) return { ok: false, error: 'bad index' };
      if (!LAYOUTS.has(m.layout)) return { ok: false, error: 'bad layout' };
      if (!nat(m.frame, 2 ** 52)) return { ok: false, error: 'bad frame' };
      if (!RANGES.includes(m.range)) return { ok: false, error: 'bad range' };
      return { ok: true, msg: { type: 'image', index: m.index, layout: m.layout, frame: m.frame, range: m.range } };
    }
    case 'select': {
      if (!idx(m.index)) return { ok: false, error: 'bad index' };
      if (m.slice === undefined || m.slice === null) return { ok: true, msg: { type: 'select', index: m.index } };
      const s = slice(m);
      if (!s) return { ok: false, error: 'bad slice' };
      return { ok: true, msg: { type: 'select', index: m.index, slice: s } };
    }
    case 'values':
    case 'copyValues': {
      if (!idx(m.index)) return { ok: false, error: 'bad index' };
      const s = slice(m, true);
      if (!s) return { ok: false, error: 'bad slice' };
      if (!nat(m.row, 2 ** 52) || !nat(m.col, 2 ** 52) || !nat(m.rows, MAX_PAGE_ROWS) || !nat(m.cols, MAX_PAGE_COLS) || m.rows < 1 || m.cols < 1) return { ok: false, error: 'bad page' };
      return { ok: true, msg: { type: m.type, index: m.index, slice: s, row: m.row, col: m.col, rows: m.rows, cols: m.cols } };
    }
    case 'export': {
      if (!idx(m.index)) return { ok: false, error: 'bad index' };
      if (!EXPORT_FORMATS.has(m.format)) return { ok: false, error: 'bad format' };
      if (m.slice === undefined || m.slice === null) return { ok: true, msg: { type: 'export', index: m.index, format: m.format, slice: null } };
      const s = slice(m);
      if (!s) return { ok: false, error: 'bad slice' };
      return { ok: true, msg: { type: 'export', index: m.index, format: m.format, slice: s } };
    }
    case 'copy':
      if (!idx(m.index)) return { ok: false, error: 'bad index' };
      if (!COPY_WHAT.has(m.what)) return { ok: false, error: 'bad copy kind' };
      return { ok: true, msg: { type: 'copy', index: m.index, what: m.what } };
    default: return { ok: false, error: 'unknown type' };
  }
  // A slice is one index per leading dimension (all but the last two). With `orDefault`,
  // a missing slice means the first one.
  function slice(msg, orDefault) {
    if (!Array.isArray(msg.slice) && !(orDefault && (msg.slice === undefined || msg.slice === null))) return null;
    if (Array.isArray(msg.slice) && (msg.slice.length > 32 || !msg.slice.every(v => Number.isInteger(v) && v >= 0))) return null;
    if (!tensors) return Array.isArray(msg.slice) ? msg.slice.slice() : [];
    const t = tensors[msg.index];
    if (!t || !Array.isArray(t.shape)) return null;
    if (!orDefault && t.shape.length < 3) return null;
    return checkSlice(t.shape, Array.isArray(msg.slice) ? msg.slice : null);
  }
}

/** The model as sent to the webview: plain data only. */
function publicModel(model, fileName) {
  return {
    type: 'model', version: PROTOCOL_VERSION, fileName, format: model.format, fileSize: model.fileSize,
    totalParams: model.totalParams, info: model.info, metadata: model.metadata, warnings: model.warnings,
    tensors: model.tensors.map(t => ({ name: t.name, dtype: t.dtype, rawDtype: t.rawDtype, shape: t.shape, numel: t.numel, nbytes: t.nbytes, offset: t.offset, readable: !!t.readable, note: t.note, image: imageInfo(t) })),
  };
}

function shapeText(t, what) {
  const { shapeAsPython } = require('./sample');
  if (what === 'name') return t.name;
  if (what === 'torch') return `torch.Size([${t.shape.join(', ')}])`;
  return shapeAsPython(t.shape);
}

module.exports = { PROTOCOL_VERSION, validateWebviewMessage, publicModel, shapeText };
