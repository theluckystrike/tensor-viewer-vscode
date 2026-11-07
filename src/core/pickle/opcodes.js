'use strict';
// Pickle opcode table and a reader that walks the stream one opcode at a time.
// The argument formats follow CPython's Lib/pickletools.py. Nothing here runs code: it only reads bytes.

// [code, name, argument kind, protocol]
const TABLE = [
  ['(', 'MARK', null, 0], ['.', 'STOP', null, 0], ['0', 'POP', null, 0], ['1', 'POP_MARK', null, 1],
  ['2', 'DUP', null, 0], ['F', 'FLOAT', 'floatnl', 0], ['I', 'INT', 'decimalnl_short', 0], ['J', 'BININT', 'int4', 1],
  ['K', 'BININT1', 'uint1', 1], ['L', 'LONG', 'decimalnl_long', 0], ['M', 'BININT2', 'uint2', 1], ['N', 'NONE', null, 0],
  ['P', 'PERSID', 'stringnl_noescape', 0], ['Q', 'BINPERSID', null, 1], ['R', 'REDUCE', null, 0], ['S', 'STRING', 'stringnl', 0],
  ['T', 'BINSTRING', 'string4', 1], ['U', 'SHORT_BINSTRING', 'string1', 1], ['V', 'UNICODE', 'unicodestringnl', 0],
  ['X', 'BINUNICODE', 'unicodestring4', 1], ['a', 'APPEND', null, 0], ['b', 'BUILD', null, 0],
  ['c', 'GLOBAL', 'stringnl_noescape_pair', 0], ['d', 'DICT', null, 0], ['}', 'EMPTY_DICT', null, 1], ['e', 'APPENDS', null, 1],
  ['g', 'GET', 'decimalnl_short', 0], ['h', 'BINGET', 'uint1', 1], ['i', 'INST', 'stringnl_noescape_pair', 0],
  ['j', 'LONG_BINGET', 'uint4', 1], ['l', 'LIST', null, 0], [']', 'EMPTY_LIST', null, 1], ['o', 'OBJ', null, 1],
  ['p', 'PUT', 'decimalnl_short', 0], ['q', 'BINPUT', 'uint1', 1], ['r', 'LONG_BINPUT', 'uint4', 1], ['s', 'SETITEM', null, 0],
  ['t', 'TUPLE', null, 0], [')', 'EMPTY_TUPLE', null, 1], ['u', 'SETITEMS', null, 1], ['G', 'BINFLOAT', 'float8', 1],
  ['\x80', 'PROTO', 'uint1', 2], ['\x81', 'NEWOBJ', null, 2], ['\x82', 'EXT1', 'uint1', 2], ['\x83', 'EXT2', 'uint2', 2],
  ['\x84', 'EXT4', 'int4', 2], ['\x85', 'TUPLE1', null, 2], ['\x86', 'TUPLE2', null, 2], ['\x87', 'TUPLE3', null, 2],
  ['\x88', 'NEWTRUE', null, 2], ['\x89', 'NEWFALSE', null, 2], ['\x8a', 'LONG1', 'long1', 2], ['\x8b', 'LONG4', 'long4', 2],
  ['B', 'BINBYTES', 'bytes4', 3], ['C', 'SHORT_BINBYTES', 'bytes1', 3], ['\x8c', 'SHORT_BINUNICODE', 'unicodestring1', 4],
  ['\x8d', 'BINUNICODE8', 'unicodestring8', 4], ['\x8e', 'BINBYTES8', 'bytes8', 4], ['\x8f', 'EMPTY_SET', null, 4],
  ['\x90', 'ADDITEMS', null, 4], ['\x91', 'FROZENSET', null, 4], ['\x92', 'NEWOBJ_EX', null, 4], ['\x93', 'STACK_GLOBAL', null, 4],
  ['\x94', 'MEMOIZE', null, 4], ['\x95', 'FRAME', 'uint8', 4], ['\x96', 'BYTEARRAY8', 'bytes8', 5], ['\x97', 'NEXT_BUFFER', null, 5],
  ['\x98', 'READONLY_BUFFER', null, 5],
];

const BY_CODE = new Array(256).fill(null);
const OP = {};
for (const [ch, name, arg, proto] of TABLE) {
  const code = ch.charCodeAt(0);
  BY_CODE[code] = { code, ch, name, arg, proto };
  OP[name] = code;
}

const utf8 = new TextDecoder('utf-8');

function latin1Text(bytes) {
  // TextDecoder('latin1') is windows-1252 in browsers and Node; map bytes 1:1 instead.
  let s = '';
  for (let i = 0; i < bytes.length; i += 8192) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
  return s;
}

/** Python's raw-unicode-escape (used by the protocol 0 UNICODE opcode). */
function rawUnicodeEscape(bytes) {
  const s = latin1Text(bytes);
  if (s.indexOf('\\u') < 0 && s.indexOf('\\U') < 0) return s;
  return s.replace(/\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})/g, (m, a, b) => String.fromCodePoint(parseInt(a || b, 16)));
}

/** A Python 2 string literal from the STRING opcode, such as 'abc\n' or "it's". */
function pyStringLiteral(text) {
  if (text.length < 2 || text[0] !== text[text.length - 1] || (text[0] !== '"' && text[0] !== "'")) {
    throw new Error('The STRING opcode has no quotes around its text.');
  }
  const body = text.slice(1, -1);
  const simple = { n: '\n', r: '\r', t: '\t', '\\': '\\', "'": "'", '"': '"', a: '\x07', b: '\b', f: '\f', v: '\v', 0: '\0' };
  return body.replace(/\\(x[0-9a-fA-F]{2}|[0-7]{1,3}|.)/g, (m, e) => {
    if (e[0] === 'x') return String.fromCharCode(parseInt(e.slice(1), 16));
    if (/^[0-7]+$/.test(e)) return String.fromCharCode(parseInt(e, 8) & 0xff);
    return Object.prototype.hasOwnProperty.call(simple, e) ? simple[e] : '\\' + e;
  });
}

/** Two's complement little-endian integer of any length, as Number when safe, else BigInt. */
function decodeLong(bytes) {
  if (bytes.length === 0) return 0;
  let n = 0n;
  for (let i = bytes.length - 1; i >= 0; i--) n = (n << 8n) | BigInt(bytes[i]);
  if (bytes[bytes.length - 1] & 0x80) n -= 1n << BigInt(bytes.length * 8);
  return smallest(n);
}

function smallest(n) {
  return n >= BigInt(Number.MIN_SAFE_INTEGER) && n <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(n) : n;
}

function parseDecimal(text, what) {
  const t = text.trim().replace(/L$/, '');
  if (!/^[+-]?\d+$/.test(t)) throw new Error(`The ${what} opcode has a bad number: ${JSON.stringify(text.slice(0, 40))}.`);
  return smallest(BigInt(t));
}

class PickleError extends Error {
  constructor(message, pos) { super(message); this.pos = pos; }
}

/**
 * Walk the opcodes of a pickle. Yields { pos, code, name, arg, proto } for each opcode, STOP included.
 * Throws PickleError on an unknown opcode or a cut-off argument.
 */
function* iterOps(bytes, start = 0) {
  const b = bytes;
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let p = start;
  const need = (n, pos, name) => {
    if (n < 0 || p + n > b.length) throw new PickleError(`The file ends inside the ${name} opcode at byte ${pos}: it needs ${n} more bytes and ${Math.max(0, b.length - p)} are left.`, pos);
  };
  const line = (pos, name) => {
    const i = b.indexOf(0x0a, p);
    if (i < 0) throw new PickleError(`The ${name} opcode at byte ${pos} has no line end.`, pos);
    const s = b.subarray(p, i);
    p = i + 1;
    return s;
  };
  const len64 = (pos, name) => {
    need(8, pos, name);
    const lo = dv.getUint32(p, true), hi = dv.getUint32(p + 4, true);
    p += 8;
    const n = hi * 4294967296 + lo;
    if (!Number.isSafeInteger(n)) throw new PickleError(`The ${name} opcode at byte ${pos} has a length that is too large.`, pos);
    return n;
  };
  while (p < b.length) {
    const pos = p;
    const code = b[p++];
    const info = BY_CODE[code];
    if (!info) throw new PickleError(`Unknown opcode 0x${code.toString(16).padStart(2, '0')} at byte ${pos}. This may not be a pickle file, or it is damaged.`, pos);
    const name = info.name;
    let arg;
    switch (info.arg) {
      case null: arg = undefined; break;
      case 'uint1': need(1, pos, name); arg = b[p]; p += 1; break;
      case 'uint2': need(2, pos, name); arg = dv.getUint16(p, true); p += 2; break;
      case 'int4': need(4, pos, name); arg = dv.getInt32(p, true); p += 4; break;
      case 'uint4': need(4, pos, name); arg = dv.getUint32(p, true); p += 4; break;
      case 'uint8': arg = len64(pos, name); break;
      case 'float8': need(8, pos, name); arg = dv.getFloat64(p, false); p += 8; break;
      case 'decimalnl_short': {
        const t = latin1Text(line(pos, name));
        if (name === 'INT' && (t === '00' || t === '01')) arg = t === '01';
        else arg = parseDecimal(t, name);
        break;
      }
      case 'decimalnl_long': arg = parseDecimal(latin1Text(line(pos, name)), name); break;
      case 'floatnl': {
        const t = latin1Text(line(pos, name)).trim();
        const v = /^nan$/i.test(t) ? NaN : /^[+-]?inf$/i.test(t) ? (t[0] === '-' ? -Infinity : Infinity) : Number(t);
        if (Number.isNaN(v) && !/^nan$/i.test(t)) throw new PickleError(`The FLOAT opcode at byte ${pos} has a bad number.`, pos);
        arg = v;
        break;
      }
      case 'stringnl': arg = pyStringLiteral(latin1Text(line(pos, name))); break;
      case 'stringnl_noescape': arg = latin1Text(line(pos, name)); break;
      case 'stringnl_noescape_pair': {
        const mod = utf8.decode(line(pos, name));
        const nm = utf8.decode(line(pos, name));
        arg = [mod, nm];
        break;
      }
      case 'unicodestringnl': arg = rawUnicodeEscape(line(pos, name)); break;
      case 'string1': case 'bytes1': case 'unicodestring1': {
        need(1, pos, name); const n = b[p]; p += 1; need(n, pos, name);
        const s = b.subarray(p, p + n); p += n;
        arg = info.arg === 'bytes1' ? s : info.arg === 'string1' ? latin1Text(s) : utf8.decode(s);
        break;
      }
      case 'string4': case 'bytes4': case 'unicodestring4': {
        need(4, pos, name);
        const n = info.arg === 'string4' ? dv.getInt32(p, true) : dv.getUint32(p, true); p += 4;
        if (n < 0) throw new PickleError(`The ${name} opcode at byte ${pos} has a negative length.`, pos);
        need(n, pos, name);
        const s = b.subarray(p, p + n); p += n;
        arg = info.arg === 'bytes4' ? s : info.arg === 'string4' ? latin1Text(s) : utf8.decode(s);
        break;
      }
      case 'bytes8': case 'unicodestring8': {
        const n = len64(pos, name); need(n, pos, name);
        const s = b.subarray(p, p + n); p += n;
        arg = info.arg === 'bytes8' ? s : utf8.decode(s);
        break;
      }
      case 'long1': { need(1, pos, name); const n = b[p]; p += 1; need(n, pos, name); arg = decodeLong(b.subarray(p, p + n)); p += n; break; }
      case 'long4': {
        need(4, pos, name); const n = dv.getInt32(p, true); p += 4;
        if (n < 0) throw new PickleError(`The LONG4 opcode at byte ${pos} has a negative length.`, pos);
        need(n, pos, name); arg = decodeLong(b.subarray(p, p + n)); p += n; break;
      }
      default: throw new PickleError(`Internal: no reader for ${info.arg}.`, pos);
    }
    yield { pos, code, name, arg, proto: info.proto, end: p };
    if (code === 0x2e) return; // STOP
  }
  throw new PickleError('The file ends before the STOP opcode. It may be cut off.', p);
}

module.exports = { TABLE, BY_CODE, OP, iterOps, PickleError, decodeLong, rawUnicodeEscape, pyStringLiteral, latin1Text, smallest };
