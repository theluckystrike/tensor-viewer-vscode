'use strict';
// Report tables for the Zovo Pro tools: one description ({title, subtitle, facts, notes, columns, rows}) is shown in
// the report panel and saved as CSV, Markdown or JSON. Pure functions, shared by the Zovo viewers.
//   column: { key, label, type: 'text' | 'num' | 'int' | 'pct' }
//   row: plain object; row._mark ('bad' | 'warn' | '') colors the row and feeds the "Only rows to check" filter.

const nf = new Intl.NumberFormat('en-US');

/** Display text for one cell. */
function cellText(v, type) {
  if (v === null || v === undefined || v === '') return '';
  if (typeof v !== 'number') return String(v);
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? 'Inf' : '-Inf';
  if (type === 'int') return nf.format(v);
  if (type === 'pct') return (v === 0 ? '0' : v < 0.01 ? '<0.01' : v.toFixed(2)) + ' %';
  if (type === 'num') {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (Number.isInteger(v) && a < 1e15) return nf.format(v);
    if (a >= 1e6 || a < 1e-4) return v.toExponential(4);
    return String(Number(v.toPrecision(6)));
  }
  return String(v);
}

// A text cell that starts with one of these could run as a formula when the CSV is opened in a spreadsheet.
const FORMULA = /^[=+\-@\t\r]/;

function csvField(v) {
  if (v === null || v === undefined) return '';
  let s;
  if (typeof v === 'number') s = Number.isNaN(v) ? 'NaN' : Number.isFinite(v) ? String(v) : (v > 0 ? 'Inf' : '-Inf');
  else { s = String(v); if (FORMULA.test(s) && !/^-?(\d|Inf|NaN)/.test(s)) s = "'" + s; }
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

/** CSV: one header line, then one line per row, numbers at full precision. */
function toCsv(desc) {
  const lines = [desc.columns.map(c => csvField(c.label)).join(',')];
  for (const r of desc.rows) lines.push(desc.columns.map(c => csvField(r[c.key])).join(','));
  return lines.join('\r\n') + '\r\n';
}

const mdCell = s => String(s).replace(/\\/g, '\\\\').replace(/\|/g, '\\|').replace(/[\r\n]+/g, ' ');

/** Markdown: title, facts, notes and the table. */
function toMarkdown(desc) {
  const out = [`# ${mdCell(desc.title)}`, ''];
  if (desc.subtitle) out.push(mdCell(desc.subtitle), '');
  for (const [k, v] of desc.facts || []) out.push(`- ${mdCell(k)}: ${mdCell(v)}`);
  if ((desc.facts || []).length) out.push('');
  for (const n of desc.notes || []) out.push(`> ${mdCell(n)}`, '');
  out.push('| ' + desc.columns.map(c => mdCell(c.label)).join(' | ') + ' |');
  out.push('| ' + desc.columns.map(c => (c.type && c.type !== 'text' ? '---:' : '---')).join(' | ') + ' |');
  for (const r of desc.rows) out.push('| ' + desc.columns.map(c => mdCell(cellText(r[c.key], c.type))).join(' | ') + ' |');
  return out.join('\n') + '\n';
}

/** JSON: the raw report, with NaN and Inf written as strings (JSON has no such numbers). */
function toJson(report) {
  return JSON.stringify(report, (k, v) => (k.startsWith('_') ? undefined : typeof v === 'number' && !Number.isFinite(v) ? (Number.isNaN(v) ? 'NaN' : v > 0 ? 'Inf' : '-Inf') : v), 2) + '\n';
}

/** Text for one format. */
function render(format, desc, report) {
  if (format === 'csv') return toCsv(desc);
  if (format === 'md') return toMarkdown(desc);
  if (format === 'json') return toJson(report);
  throw new Error('unknown format');
}

module.exports = { cellText, csvField, toCsv, toMarkdown, toJson, render, FORMATS: ['csv', 'md', 'json'] };
