'use strict';
// Pro: compare two tensor files (two checkpoints, a model before and after conversion or quantization).
// Tensors are matched by name, then by name without a wrapper prefix (module., _orig_mod., model.).
// Every value of every matched pair is read: max and mean absolute difference, RMSE, relative L2 error, cosine
// similarity, the count of values that differ and NaN mismatches. Pure Node, no vscode, so the tests call it directly.
const { streamInfo, readAt, readAllCOrder, indexText, RUN_VALUES } = require('./stream');

const PREFIXES = ['module.', '_orig_mod.', 'model.', 'state_dict.'];

function bareName(name) {
  let n = String(name);
  for (let again = true; again;) {
    again = false;
    for (const p of PREFIXES) if (n.startsWith(p) && n.length > p.length) { n = n.slice(p.length); again = true; }
  }
  return n;
}

/** Pairs of tensors: [{a, b, how}] with a or b null when a tensor is in one file only. */
function matchTensors(ta, tb) {
  const pairs = [];
  const byName = new Map();
  tb.forEach((t, i) => { if (!byName.has(t.name)) byName.set(t.name, i); });
  const usedB = new Set();
  const restA = [];
  for (const t of ta) {
    const j = byName.get(t.name);
    if (j !== undefined && !usedB.has(j)) { usedB.add(j); pairs.push({ a: t, b: tb[j], how: 'name' }); } else restA.push(t);
  }
  const restB = tb.filter((_, j) => !usedB.has(j));
  // Second pass: the same name once a wrapper prefix is removed, when that is unique on both sides.
  const count = list => { const m = new Map(); for (const t of list) { const k = bareName(t.name); m.set(k, (m.get(k) || 0) + 1); } return m; };
  const ca = count(restA), cb = count(restB);
  const bByBare = new Map(restB.map(t => [bareName(t.name), t]));
  const pairedB = new Set();
  for (const t of restA) {
    const k = bareName(t.name);
    if (ca.get(k) === 1 && cb.get(k) === 1) { const u = bByBare.get(k); pairedB.add(u); pairs.push({ a: t, b: u, how: 'prefix' }); } else pairs.push({ a: t, b: null, how: 'none' });
  }
  for (const t of restB) if (!pairedB.has(t)) pairs.push({ a: null, b: t, how: 'none' });
  return pairs;
}

const shapeText = s => '[' + s.join(', ') + ']';
const sameShape = (x, y) => x.length === y.length && x.every((v, k) => v === y[k]);

function emptyAcc() { return { n: 0, nDiff: 0, nanMismatch: 0, infMismatch: 0, bothNan: 0, sumAbs: 0, sumSq: 0, maxAbs: 0, maxAt: -1, dot: 0, na: 0, nb: 0, maxA: 0, maxB: 0 }; }

/** Adds values x[i] vs y[i]; base is the flat index of x[0]. Inline loop: this is the hot path. */
function accumulate(acc, x, y, base) {
  let { n, nDiff, nanMismatch, infMismatch, bothNan, sumAbs, sumSq, maxAbs, maxAt, dot, na, nb } = acc;
  for (let i = 0; i < x.length; i++) {
    const a = x[i], b = y[i];
    if (a !== a || b !== b) { if ((a !== a) !== (b !== b)) nanMismatch++; else bothNan++; continue; }
    if (a === Infinity || a === -Infinity || b === Infinity || b === -Infinity) { if (a !== b) infMismatch++; continue; }
    const d = a - b;
    const ad = d < 0 ? -d : d;
    if (ad > 0) { nDiff++; if (ad > maxAbs) { maxAbs = ad; maxAt = base + i; } }
    sumAbs += ad; sumSq += d * d; dot += a * b; na += a * a; nb += b * b; n++;
  }
  Object.assign(acc, { n, nDiff, nanMismatch, infMismatch, bothNan, sumAbs, sumSq, maxAbs, maxAt, dot, na, nb });
}

function finish(acc) {
  const r = {
    compared: acc.n + acc.nanMismatch + acc.infMismatch + acc.bothNan,
    differing: acc.nDiff + acc.nanMismatch + acc.infMismatch,
    maxAbs: acc.maxAbs,
    meanAbs: acc.n ? acc.sumAbs / acc.n : 0,
    rmse: acc.n ? Math.sqrt(acc.sumSq / acc.n) : 0,
    relL2: acc.na > 0 ? Math.sqrt(acc.sumSq) / Math.sqrt(acc.na) : (acc.sumSq > 0 ? Infinity : 0),
    cosine: acc.na > 0 && acc.nb > 0 ? acc.dot / Math.sqrt(acc.na * acc.nb) : (acc.na === 0 && acc.nb === 0 ? 1 : 0),
    nanMismatch: acc.nanMismatch,
    infMismatch: acc.infMismatch,
  };
  return r;
}

/**
 * Compare two opened models (formats.openTensorFile).
 * opts: { cancelled(): boolean, progress(fraction, text), now(): ms }
 * @returns {Promise<object>} the report (kind 'compare'); rows are sorted with the largest change first.
 */
async function compareModels(ma, mb, nameA, nameB, opts = {}) {
  const cancelled = opts.cancelled || (() => false);
  const progress = opts.progress || (() => {});
  const now = opts.now || Date.now;
  const t0 = now();
  const pairs = matchTensors(ma.tensors, mb.tensors);
  const work = pairs.reduce((s, p) => s + (p.a && p.b ? Math.max(1, p.a.numel) : 0), 0) || 1;
  let done = 0, stopped = false;
  const rows = [];
  for (const p of pairs) {
    const row = {
      name: p.a ? p.a.name : p.b.name, nameB: p.a && p.b && p.a.name !== p.b.name ? p.b.name : '',
      dtypeA: p.a ? p.a.dtype : '', dtypeB: p.b ? p.b.dtype : '',
      shapeA: p.a ? shapeText(p.a.shape) : '', shapeB: p.b ? shapeText(p.b.shape) : '',
      numel: p.a ? p.a.numel : p.b.numel, status: '', note: '',
      maxAbs: null, meanAbs: null, rmse: null, relL2: null, cosine: null, differing: null, maxAt: '',
    };
    rows.push(row);
    if (!p.a) { row.status = 'only in B'; continue; }
    if (!p.b) { row.status = 'only in A'; continue; }
    if (stopped || cancelled()) { stopped = true; row.status = 'not compared'; row.note = 'Stopped before this tensor.'; continue; }
    if (p.a.numel !== p.b.numel) { row.status = 'shape differs'; row.note = `${shapeText(p.a.shape)} vs ${shapeText(p.b.shape)}: values not compared.`; done += p.a.numel; continue; }
    const ia = streamInfo(p.a), ib = streamInfo(p.b);
    if (!ia.ok || !ib.ok) { row.status = 'not compared'; row.note = ia.ok ? `B: ${ib.why}` : `A: ${ia.why}`; done += p.a.numel; continue; }
    if (ia.perElement !== ib.perElement) { row.status = 'not compared'; row.note = 'One tensor is complex and the other is not.'; done += p.a.numel; continue; }
    const reshaped = !sameShape(p.a.shape, p.b.shape);
    const fa = !!p.a._loc.fortran && p.a.shape.length >= 2, fb = !!p.b._loc.fortran && p.b.shape.length >= 2;
    const reorder = fa !== fb || (reshaped && (fa || fb));
    const acc = emptyAcc();
    try {
      if (reorder) {
        const x = await readAllCOrder(p.a, ia.perElement), y = await readAllCOrder(p.b, ib.perElement);
        accumulate(acc, x, y, 0);
        done += p.a.numel;
        progress(Math.min(1, done / work), row.name);
      } else {
        const total = ia.total;
        for (let i = 0; i < total; i += RUN_VALUES) {
          if (cancelled()) { stopped = true; break; }
          const k = Math.min(RUN_VALUES, total - i);
          const x = await readAt(p.a, i, k), y = await readAt(p.b, i, k);
          accumulate(acc, x, y, i);
          done += k / ia.perElement;
          progress(Math.min(1, done / work), row.name);
        }
      }
    } catch (e) {
      row.status = 'not compared'; row.note = e.message || String(e); continue;
    }
    if (stopped) { row.status = 'not compared'; row.note = 'Stopped while this tensor was read.'; continue; }
    Object.assign(row, finish(acc));
    if (acc.maxAt >= 0) row.maxAt = indexText(Math.floor(acc.maxAt / ia.perElement), p.a.shape, reorder ? false : fa);
    row.status = row.differing === 0 ? 'identical' : 'changed';
    const notes = [];
    if (reshaped) notes.push(`Shapes differ (${shapeText(p.a.shape)} vs ${shapeText(p.b.shape)}), compared as flat arrays.`);
    if (p.how === 'prefix') notes.push(`Matched by name without the prefix: ${p.b.name}.`);
    if (p.a.dtype !== p.b.dtype) notes.push(`Type ${p.a.dtype} vs ${p.b.dtype}.`);
    if (ia.perElement === 2) notes.push('Complex: real and imaginary parts compared as separate values.');
    if (acc.nanMismatch) notes.push(`${acc.nanMismatch.toLocaleString('en-US')} NaN in one file only.`);
    if (acc.bothNan) notes.push(`${acc.bothNan.toLocaleString('en-US')} NaN in both.`);
    row.note = notes.join(' ');
  }
  const order = { changed: 0, 'shape differs': 1, 'only in A': 2, 'only in B': 3, 'not compared': 4, identical: 5 };
  rows.sort((x, y) => (order[x.status] - order[y.status]) || ((y.relL2 || 0) - (x.relL2 || 0)) || ((y.maxAbs || 0) - (x.maxAbs || 0)));
  const count = s => rows.filter(r => r.status === s).length;
  const summary = {
    tensorsA: ma.tensors.length, tensorsB: mb.tensors.length,
    identical: count('identical'), changed: count('changed'), shapeDiffers: count('shape differs'),
    onlyA: count('only in A'), onlyB: count('only in B'), notCompared: count('not compared'),
    valuesCompared: rows.reduce((s, r) => s + (r.compared || 0), 0),
    worst: rows.filter(r => r.status === 'changed').slice(0, 1).map(r => ({ name: r.name, relL2: r.relL2, maxAbs: r.maxAbs }))[0] || null,
  };
  return { kind: 'compare', fileA: nameA, fileB: nameB, formatA: ma.format, formatB: mb.format, created: new Date(now()).toISOString(), ms: now() - t0, stopped, summary, rows };
}

module.exports = { compareModels, matchTensors, bareName, accumulate, emptyAcc, finish };
