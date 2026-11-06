'use strict';
// Webview message protocol. The webview is treated as untrusted input: every message is checked
// here before the host acts on it, and host messages never carry internal file handles.

const PROTOCOL_VERSION = 1;
const COPY_WHAT = new Set(['shape', 'name', 'torch']);

/** @returns {{ok:true, msg:object} | {ok:false, error:string}} */
function validateWebviewMessage(m, tensorCount) {
  if (!m || typeof m !== 'object' || Array.isArray(m)) return { ok: false, error: 'not an object' };
  const idx = i => Number.isInteger(i) && i >= 0 && i < tensorCount;
  switch (m.type) {
    case 'ready': return { ok: true, msg: { type: 'ready' } };
    case 'select':
      if (!idx(m.index)) return { ok: false, error: 'bad index' };
      return { ok: true, msg: { type: 'select', index: m.index } };
    case 'copy':
      if (!idx(m.index)) return { ok: false, error: 'bad index' };
      if (!COPY_WHAT.has(m.what)) return { ok: false, error: 'bad copy kind' };
      return { ok: true, msg: { type: 'copy', index: m.index, what: m.what } };
    default: return { ok: false, error: 'unknown type' };
  }
}

/** The model as sent to the webview: plain data only. */
function publicModel(model, fileName) {
  return {
    type: 'model', version: PROTOCOL_VERSION, fileName, format: model.format, fileSize: model.fileSize,
    totalParams: model.totalParams, info: model.info, metadata: model.metadata, warnings: model.warnings,
    tensors: model.tensors.map(t => ({ name: t.name, dtype: t.dtype, rawDtype: t.rawDtype, shape: t.shape, numel: t.numel, nbytes: t.nbytes, offset: t.offset, readable: !!t.readable, note: t.note })),
  };
}

function shapeText(t, what) {
  const { shapeAsPython } = require('./sample');
  if (what === 'name') return t.name;
  if (what === 'torch') return `torch.Size([${t.shape.join(', ')}])`;
  return shapeAsPython(t.shape);
}

module.exports = { PROTOCOL_VERSION, validateWebviewMessage, publicModel, shapeText };
