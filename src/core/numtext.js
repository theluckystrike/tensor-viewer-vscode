'use strict';
// Number text shared by the host (copy, CSV) and the webview (values table). No Node APIs here.

/** Shortest text that reads back as the same value: float32 for narrow float types, float64 otherwise. */
function exactText(x, kind) {
  if (x !== x) return 'nan';
  if (x === Infinity) return 'inf';
  if (x === -Infinity) return '-inf';
  if (kind === 'f64' || kind === 'c128' || !kind || /^[iu]\d|^bool$/.test(kind)) return String(x);
  // Every narrower float type (f16, bf16, fp8, GGUF quants) fits in float32 exactly.
  const f = Math.fround(x);
  if (f !== x) return String(x);
  for (let p = 1; p <= 9; p++) { const s = Number(x.toPrecision(p)); if (Math.fround(s) === f) return String(s); }
  return String(x);
}

function complexText(re, im, kind) {
  const a = exactText(re, kind), b = exactText(Math.abs(im), kind);
  return `${a}${im < 0 || Object.is(im, -0) ? '-' : '+'}${b}j`;
}

const PRECISIONS = ['auto', 'exact', '2', '4', '8', 'sci'];

/** Display text for the values table. Integers are always shown whole. */
function displayText(x, kind, prec) {
  if (!Number.isFinite(x)) return exactText(x, kind);
  if (kind === 'bool') return x ? 'True' : 'False';
  if (/^[iu]\d/.test(kind || '') || (prec === 'auto' && Number.isInteger(x) && Math.abs(x) < 1e15)) return String(x);
  switch (prec) {
    case 'exact': return exactText(x, kind);
    case '2': case '4': case '8': return x.toFixed(Number(prec));
    case 'sci': return x.toExponential(3);
    default: {
      if (x === 0) return '0';
      const a = Math.abs(x);
      if (a >= 1e6 || a < 1e-4) return x.toExponential(4);
      return String(Number(x.toPrecision(6)));
    }
  }
}

function displayComplex(re, im, kind, prec) {
  const a = displayText(re, kind, prec), b = displayText(Math.abs(im), kind, prec);
  return `${a}${im < 0 || Object.is(im, -0) ? '-' : '+'}${b}j`;
}

module.exports = { exactText, complexText, displayText, displayComplex, PRECISIONS };
