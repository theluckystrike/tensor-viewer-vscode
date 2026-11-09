'use strict';
// Pro: health scan of a whole checkpoint. Reads every value of every tensor (not a sample) and flags NaN, Inf,
// all-zero and constant tensors and very large float values: the usual signs of a training run that diverged or
// a conversion that went wrong. Pure Node, no vscode.
const { streamInfo, readAt, RUN_VALUES } = require('./stream');

const HUGE = 1e4; // |x| at or above this in a float tensor is flagged
const INT_KIND = /^(bool|int|uint)/i;

/**
 * opts: { cancelled(), progress(fraction, text), now() }
 * @returns {Promise<object>} report (kind 'scan'); flagged tensors first.
 */
async function scanModel(model, fileName, opts = {}) {
  const cancelled = opts.cancelled || (() => false);
  const progress = opts.progress || (() => {});
  const now = opts.now || Date.now;
  const t0 = now();
  const work = model.tensors.reduce((s, t) => s + (t.readable ? Math.max(1, t.numel) : 0), 0) || 1;
  let done = 0, stopped = false;
  const rows = [];
  for (const t of model.tensors) {
    const row = { name: t.name, dtype: t.dtype, shape: '[' + t.shape.join(', ') + ']', numel: t.numel, status: '', flags: '', note: '',
      min: null, max: null, mean: null, std: null, absMax: null, l2: null, nan: null, inf: null, zeros: null, zeroPct: null };
    rows.push(row);
    const info = streamInfo(t);
    if (!info.ok) { row.status = 'not scanned'; row.note = info.why; continue; }
    if (stopped || cancelled()) { stopped = true; row.status = 'not scanned'; row.note = 'Stopped before this tensor.'; continue; }
    let n = 0, nan = 0, pinf = 0, ninf = 0, zeros = 0, sumSq = 0, min = Infinity, max = -Infinity;
    // Mean and std in one pass: shifted sums around the first finite value keep the error small.
    let shift = null, s1 = 0, s2 = 0;
    try {
      for (let i = 0; i < info.total; i += RUN_VALUES) {
        if (cancelled()) { stopped = true; break; }
        const k = Math.min(RUN_VALUES, info.total - i);
        const x = await readAt(t, i, k);
        for (let j = 0; j < x.length; j++) {
          const v = x[j];
          if (v !== v) { nan++; continue; }
          if (v === Infinity) { pinf++; continue; }
          if (v === -Infinity) { ninf++; continue; }
          if (v === 0) zeros++;
          if (shift === null) shift = v;
          const d = v - shift;
          s1 += d; s2 += d * d; sumSq += v * v; n++;
          if (v < min) min = v;
          if (v > max) max = v;
        }
        done += k / info.perElement;
        progress(Math.min(1, done / work), t.name);
      }
    } catch (e) {
      row.status = 'not scanned'; row.note = e.message || String(e); continue;
    }
    if (stopped) { row.status = 'not scanned'; row.note = 'Stopped while this tensor was read.'; continue; }
    const total = n + nan + pinf + ninf;
    const mean = n ? shift + s1 / n : null;
    const variance = n ? Math.max(0, (s2 - (s1 * s1) / n) / n) : null;
    Object.assign(row, {
      min: n ? min : null, max: n ? max : null, mean, std: variance === null ? null : Math.sqrt(variance),
      absMax: n ? Math.max(Math.abs(min), Math.abs(max)) : null, l2: Math.sqrt(sumSq),
      nan, inf: pinf + ninf, zeros, zeroPct: total ? (100 * zeros) / total : 0,
    });
    const flags = [];
    if (nan) flags.push('NaN');
    if (pinf + ninf) flags.push('Inf');
    if (total && zeros === total) flags.push('all zero');
    else if (n > 1 && min === max) flags.push('constant');
    if (!INT_KIND.test(String(t.dtype)) && n && row.absMax >= HUGE) flags.push('huge values');
    row.flags = flags.join(', ');
    row.status = flags.length ? 'check' : 'ok';
    if (info.perElement === 2) row.note = 'Complex: stats over the real and imaginary parts.';
  }
  const rank = r => (r.status === 'check' ? (/NaN|Inf/.test(r.flags) ? 0 : 1) : r.status === 'not scanned' ? 2 : 3);
  rows.sort((a, b) => rank(a) - rank(b));
  const has = f => rows.filter(r => r.flags.split(', ').includes(f)).length;
  const summary = {
    tensors: model.tensors.length,
    scanned: rows.filter(r => r.status === 'ok' || r.status === 'check').length,
    flagged: rows.filter(r => r.status === 'check').length,
    withNaN: has('NaN'), withInf: has('Inf'), allZero: has('all zero'), constant: has('constant'), huge: has('huge values'),
    notScanned: rows.filter(r => r.status === 'not scanned').length,
    values: rows.reduce((s, r) => s + (r.status === 'ok' || r.status === 'check' ? r.numel : 0), 0),
    nanTotal: rows.reduce((s, r) => s + (r.nan || 0), 0), infTotal: rows.reduce((s, r) => s + (r.inf || 0), 0),
  };
  return { kind: 'scan', file: fileName, format: model.format, created: new Date(now()).toISOString(), ms: now() - t0, stopped, summary, rows };
}

module.exports = { scanModel, HUGE };
