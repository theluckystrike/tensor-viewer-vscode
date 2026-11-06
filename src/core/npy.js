'use strict';
// .npy header parser. The header is a Python dict literal; it is parsed with a small literal
// reader (strings, numbers, True/False/None, tuples, lists, dicts). Nothing is evaluated.
const { fromNumpyDescr } = require('./dtypes');

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY
const MAX_HEADER = 1024 * 1024;

function parsePyLiteral(text) {
  let i = 0;
  const ws = () => { while (i < text.length && /\s/.test(text[i])) i++; };
  const fail = msg => { throw new Error(`Bad .npy header: ${msg} at character ${i}.`); };
  let depth = 0;
  function value() {
    ws();
    const ch = text[i];
    if (ch === "'" || ch === '"') {
      const q = ch; i++;
      let s = '';
      while (i < text.length && text[i] !== q) {
        if (text[i] === '\\') { i++; const e = text[i]; s += e === 'n' ? '\n' : e === 't' ? '\t' : e; i++; } else s += text[i++];
      }
      if (text[i] !== q) fail('unterminated string');
      i++;
      return s;
    }
    if (ch === '(' || ch === '[') {
      if (++depth > 32) fail('nested too deep');
      const close = ch === '(' ? ')' : ']';
      i++;
      const arr = [];
      for (;;) {
        ws();
        if (text[i] === close) { i++; break; }
        arr.push(value());
        ws();
        if (text[i] === ',') { i++; continue; }
        if (text[i] === close) { i++; break; }
        fail('expected , or ' + close);
      }
      depth--;
      return arr;
    }
    if (ch === '{') {
      if (++depth > 32) fail('nested too deep');
      i++;
      const obj = Object.create(null);
      for (;;) {
        ws();
        if (text[i] === '}') { i++; break; }
        const k = value();
        ws();
        if (text[i] !== ':') fail('expected :');
        i++;
        obj[String(k)] = value();
        ws();
        if (text[i] === ',') { i++; continue; }
        if (text[i] === '}') { i++; break; }
        fail('expected , or }');
      }
      depth--;
      return obj;
    }
    const m = /^(True|False|None|-?\d+L?|-?\d*\.\d+(?:[eE][-+]?\d+)?)/.exec(text.slice(i, i + 40));
    if (!m) fail('unexpected token');
    i += m[0].length;
    if (m[0] === 'True') return true;
    if (m[0] === 'False') return false;
    if (m[0] === 'None') return null;
    return Number(m[0].replace(/L$/, ''));
  }
  const v = value();
  ws();
  if (i < text.length) fail('trailing text');
  return v;
}

/**
 * Parse the .npy header from a source at `base`.
 * @returns {Promise<{version:string, descr:any, fortranOrder:boolean, shape:number[], dataOffset:number, type:object}>}
 */
async function readNpyHeader(source, base = 0) {
  if (source.size - base < 10) throw new Error('Not a .npy file: it is shorter than the 10-byte preamble.');
  const pre = await source.read(base, 12 > source.size - base ? 10 : 12);
  for (let k = 0; k < 6; k++) if (pre[k] !== MAGIC[k]) throw new Error('Not a .npy file: the magic bytes \\x93NUMPY are missing.');
  const major = pre[6], minor = pre[7];
  let hlen, hstart;
  if (major === 1) { hlen = pre[8] | (pre[9] << 8); hstart = 10; } else if (major === 2 || major === 3) {
    if (pre.length < 12) throw new Error('Truncated .npy header.');
    hlen = (pre[8] | (pre[9] << 8) | (pre[10] << 16) | (pre[11] << 24)) >>> 0; hstart = 12;
  } else throw new Error(`Unsupported .npy version ${major}.${minor}.`);
  if (hlen > MAX_HEADER) throw new Error(`The .npy header says it is ${hlen} bytes, more than the 1 MB limit.`);
  if (base + hstart + hlen > source.size) throw new Error('Truncated .npy file: the header runs past the end.');
  const hbytes = await source.read(base + hstart, hlen);
  const text = new TextDecoder(major === 3 ? 'utf-8' : 'latin1').decode(hbytes).trim();
  const d = parsePyLiteral(text);
  if (!d || typeof d !== 'object' || Array.isArray(d)) throw new Error('Bad .npy header: not a dict.');
  if (!Array.isArray(d.shape) || !d.shape.every(n => Number.isInteger(n) && n >= 0)) throw new Error('Bad .npy header: shape must be a tuple of non-negative integers.');
  const type = fromNumpyDescr(d.descr);
  return { version: `${major}.${minor}`, descr: d.descr, fortranOrder: d.fortran_order === true, shape: d.shape, dataOffset: base + hstart + hlen, type };
}

function descrLabel(descr) {
  if (typeof descr === 'string') return descr;
  try { return JSON.stringify(descr).slice(0, 120); } catch { return 'structured'; }
}

module.exports = { readNpyHeader, parsePyLiteral, descrLabel };
