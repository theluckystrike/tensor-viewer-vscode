'use strict';
// Turns a compare or scan report into the table description the report panel and the CSV/Markdown export use.
const { cellText } = require('./table');

const int = v => cellText(v, 'int');
const sec = ms => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);

function describeCompare(r) {
  const s = r.summary;
  const facts = [
    ['Identical', int(s.identical)], ['Changed', int(s.changed)], ['Shape differs', int(s.shapeDiffers)],
    ['Only in A', int(s.onlyA)], ['Only in B', int(s.onlyB)], ['Not compared', int(s.notCompared)],
    ['Values compared', int(s.valuesCompared)], ['Time', sec(r.ms)],
  ];
  const notes = [];
  if (r.stopped) notes.push('Stopped before the end. The tensors after that point are listed as "not compared".');
  if (s.worst) notes.push(`Largest relative change: ${s.worst.name} (relative L2 ${cellText(s.worst.relL2, 'num')}, max abs diff ${cellText(s.worst.maxAbs, 'num')}).`);
  if (!s.changed && !s.shapeDiffers && !s.onlyA && !s.onlyB && !s.notCompared) notes.push('Every tensor is identical in both files.');
  const rows = r.rows.map(x => ({ ...x, _mark: x.status === 'changed' || x.status === 'shape differs' ? 'bad' : x.status === 'identical' ? '' : 'warn' }));
  return {
    title: `Compare: ${r.fileA} vs ${r.fileB}`,
    subtitle: `A = ${r.fileA} (${r.formatA}), B = ${r.fileB} (${r.formatB}). Every value of every tensor with the same name was read. Relative L2 = ||A - B|| / ||A||.`,
    facts, notes, filterLabel: 'Only tensors that differ',
    columns: [
      { key: 'name', label: 'Tensor' }, { key: 'status', label: 'Status' }, { key: 'dtypeA', label: 'Type A' }, { key: 'dtypeB', label: 'Type B' },
      { key: 'shapeA', label: 'Shape A' }, { key: 'shapeB', label: 'Shape B' }, { key: 'maxAbs', label: 'Max abs diff', type: 'num' },
      { key: 'meanAbs', label: 'Mean abs diff', type: 'num' }, { key: 'rmse', label: 'RMSE', type: 'num' }, { key: 'relL2', label: 'Relative L2', type: 'num' },
      { key: 'cosine', label: 'Cosine', type: 'num' }, { key: 'differing', label: 'Values that differ', type: 'int' }, { key: 'maxAt', label: 'Max diff at' },
      { key: 'note', label: 'Note' },
    ],
    rows,
  };
}

function describeScan(r) {
  const s = r.summary;
  const facts = [
    ['Tensors', int(s.tensors)], ['Scanned', int(s.scanned)], ['To check', int(s.flagged)], ['With NaN', int(s.withNaN)], ['With Inf', int(s.withInf)],
    ['All zero', int(s.allZero)], ['Constant', int(s.constant)], ['Huge values', int(s.huge)], ['Values read', int(s.values)], ['Time', sec(r.ms)],
  ];
  const notes = [];
  if (r.stopped) notes.push('Stopped before the end. The tensors after that point are listed as "not scanned".');
  if (!s.flagged && !s.notScanned) notes.push('No NaN, no Inf, no all-zero or constant tensor and no float value of 10,000 or more.');
  if (s.nanTotal || s.infTotal) notes.push(`${int(s.nanTotal)} NaN and ${int(s.infTotal)} Inf values in all.`);
  const rows = r.rows.map(x => ({ ...x, _mark: /NaN|Inf/.test(x.flags) ? 'bad' : x.status === 'check' ? 'warn' : x.status === 'not scanned' ? 'warn' : '' }));
  return {
    title: `Scan: ${r.file}`,
    subtitle: `Every value of every tensor was read (no sampling). Flags: NaN, Inf, all zero, constant, huge values (a float of 10,000 or more).`,
    facts, notes, filterLabel: 'Only tensors to check',
    columns: [
      { key: 'name', label: 'Tensor' }, { key: 'flags', label: 'Flags' }, { key: 'dtype', label: 'Type' }, { key: 'shape', label: 'Shape' },
      { key: 'numel', label: 'Values', type: 'int' }, { key: 'min', label: 'Min', type: 'num' }, { key: 'max', label: 'Max', type: 'num' },
      { key: 'mean', label: 'Mean', type: 'num' }, { key: 'std', label: 'Std', type: 'num' }, { key: 'l2', label: 'L2 norm', type: 'num' },
      { key: 'nan', label: 'NaN', type: 'int' }, { key: 'inf', label: 'Inf', type: 'int' }, { key: 'zeroPct', label: 'Zeros', type: 'pct' },
      { key: 'note', label: 'Note' },
    ],
    rows,
  };
}

const describe = r => (r.kind === 'compare' ? describeCompare(r) : describeScan(r));

module.exports = { describe, describeCompare, describeScan };
