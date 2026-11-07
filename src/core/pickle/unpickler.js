'use strict';
// Copied from Zovo PKL Viewer (src/core/unpickler.js). Change: opts.start, to read the pickles that follow one another in legacy torch.save files.
// A pickle reader that never runs code.
//
// The stack machine follows pickleparser 0.2.1 by ewfian (MIT, https://github.com/ewfian/pickleparser),
// whose source we read first: it never evaluates code and turns unknown classes into placeholder objects.
// This copy keeps that design and changes what the viewer needs:
//  - every class and function name becomes inert data (PyGlobal), and calling one makes a PyObject
//    placeholder that only records its arguments. No name is ever looked up or called;
//  - a short list of data-only rebuilders turns well-known types back into values (bytes from
//    _codecs.encode, set, datetime, numpy arrays and dtypes, pandas blocks and indexes);
//  - tuples, bytes, sets and dicts with any keys keep their Python type;
//  - LONG1/LONG4 are signed, large lists do not overflow the call stack, persistent ids and
//    out-of-band buffers become placeholders instead of errors;
//  - every GLOBAL, STACK_GLOBAL, INST, REDUCE, NEWOBJ, OBJ and BUILD is recorded for the safety panel.
const { iterOps, PickleError } = require('./opcodes');
const V = require('./values');

const ALIASES = [
  [/^__builtin__$/, 'builtins'], [/^copy_reg$/, 'copyreg'], [/^numpy\._core(\.|$)/, 'numpy.core$1'],
];

function normalName(g) {
  let m = g.module;
  for (const [re, to] of ALIASES) m = m.replace(re, to);
  return `${m}.${g.name}`;
}

const isStr = v => typeof v === 'string';
const intOf = v => (typeof v === 'number' && Number.isInteger(v) ? v : typeof v === 'bigint' ? Number(v) : typeof v === 'boolean' ? (v ? 1 : 0) : v instanceof V.PyFloat ? v.value : NaN);
const bytesOf = v => (v instanceof V.PyBytes ? v.data : v instanceof Uint8Array ? v : isStr(v) ? latin1Bytes(v) : null);

function latin1Bytes(s) {
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
  return out;
}

const pad = (n, w = 2) => String(n).padStart(w, '0');

function datetimeText(b) {
  const y = b[0] * 256 + b[1];
  const us = (b[7] << 16) | (b[8] << 8) | b[9];
  return `${pad(y, 4)}-${pad(b[2])}-${pad(b[3])} ${pad(b[4])}:${pad(b[5])}:${pad(b[6])}${us ? '.' + pad(us, 6) : ''}`;
}

/** Data-only rebuilders for well-known callables. Each returns a value, or undefined to keep a placeholder. */
const REBUILD = {
  'builtins.set': a => new V.PySet(a.length ? listOf(a[0]) : [], false),
  'builtins.frozenset': a => new V.PySet(a.length ? listOf(a[0]) : [], true),
  'builtins.bytes': a => (a.length === 0 ? new V.PyBytes(new Uint8Array(0)) : bytesFromArgs(a, 'bytes')),
  'builtins.bytearray': a => (a.length === 0 ? new V.PyBytes(new Uint8Array(0), 'bytearray') : bytesFromArgs(a, 'bytearray')),
  'builtins.complex': a => new V.PyComplex(num(a[0]), a.length > 1 ? num(a[1]) : 0),
  'builtins.slice': a => new V.PySlice(a[0], a[1], a[2]),
  'builtins.range': a => new V.PyText('range', `range(${a.map(x => String(intOf(x))).join(', ')})`),
  'builtins.object': () => undefined,
  '_codecs.encode': a => {
    if (!isStr(a[0])) return undefined;
    const enc = isStr(a[1]) ? a[1].toLowerCase().replace(/[-_]/g, '') : 'utf8';
    if (enc === 'latin1' || enc === 'iso88591') return new V.PyBytes(latin1Bytes(a[0]));
    if (enc === 'utf8') return new V.PyBytes(new TextEncoder().encode(a[0]));
    return undefined;
  },
  'copyreg._reconstructor': a => {
    const cls = a[0];
    if (cls instanceof V.PyGlobal) {
      const special = NEWOBJ[normalName(cls)];
      if (special) { const v = special([]); if (v !== undefined) return v; }
    }
    return new V.PyObject(cls, 'REDUCE', V.tuple([]));
  },
  'copyreg.__newobj__': a => newObject(a[0], V.tuple(a.slice(1)), 'NEWOBJ'),
  'collections.OrderedDict': a => dictFrom(a[0], 'collections.OrderedDict'),
  'collections.defaultdict': a => { const d = new V.PyDict('collections.defaultdict'); d.factory = a[0]; return d; },
  'collections.Counter': a => dictFrom(a[0], 'collections.Counter'),
  'collections.deque': a => { const l = listOf(a[0]).slice(); l.typeName = 'collections.deque'; return l; },
  'datetime.datetime': a => {
    const b = bytesOf(a[0]);
    if (!b || b.length !== 10 || b[2] < 1 || b[2] > 12) return undefined;
    return new V.PyText('datetime.datetime', datetimeText(b) + (a[1] != null ? ' (with tzinfo)' : ''), { tz: a[1] });
  },
  'datetime.date': a => {
    const b = bytesOf(a[0]);
    if (!b || b.length !== 4) return undefined;
    return new V.PyText('datetime.date', `${pad(b[0] * 256 + b[1], 4)}-${pad(b[2])}-${pad(b[3])}`);
  },
  'datetime.time': a => {
    const b = bytesOf(a[0]);
    if (!b || b.length !== 6) return undefined;
    const us = (b[3] << 16) | (b[4] << 8) | b[5];
    return new V.PyText('datetime.time', `${pad(b[0])}:${pad(b[1])}:${pad(b[2])}${us ? '.' + pad(us, 6) : ''}`);
  },
  'datetime.timedelta': a => {
    const [d, s, us] = [intOf(a[0] ?? 0), intOf(a[1] ?? 0), intOf(a[2] ?? 0)];
    if (![d, s, us].every(Number.isFinite)) return undefined;
    const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
    return new V.PyText('datetime.timedelta', `${d ? d + (Math.abs(d) === 1 ? ' day, ' : ' days, ') : ''}${hh}:${pad(mm)}:${pad(ss)}${us ? '.' + pad(us, 6) : ''}`);
  },
  'decimal.Decimal': a => (isStr(a[0]) ? new V.PyText('decimal.Decimal', a[0]) : undefined),
  'numpy.dtype': a => (isStr(a[0]) ? new V.NpDtype(a[0]) : undefined),
  'numpy.core.multiarray._reconstruct': a => new V.NdArray(a[0] instanceof V.PyGlobal ? normalName(a[0]) : 'numpy.ndarray'),
  'numpy.core.numeric._frombuffer': a => {
    // _frombuffer(buffer, dtype, shape, order)
    const raw = bytesOf(a[0]);
    if (!raw || !(a[1] instanceof V.NpDtype) || !Array.isArray(a[2])) return undefined;
    const arr = new V.NdArray('numpy.ndarray');
    arr.raw = raw; arr.dtype = a[1]; arr.shape = a[2].map(intOf); arr.fortran = a[3] === 'F';
    return arr;
  },
  'numpy.core.multiarray.scalar': a => (a[0] instanceof V.NpDtype ? new V.NpScalar(a[0], bytesOf(a[1]) || a[1]) : undefined),
  'pandas._libs.internals._unpickle_block': a => new V.PdBlock(a[0], a[1], intOf(a[2])),
  'pandas.core.internals.managers.BlockManager': a => (Array.isArray(a[0]) && Array.isArray(a[1]) ? new V.PdBlockManager(a[0], a[1]) : undefined),
  'pandas.core.indexes.base._new_Index': a => {
    if (!(a[0] instanceof V.PyGlobal) || !(a[1] instanceof V.PyDict)) return undefined;
    return new V.PdIndex(normalName(a[0]), a[1]);
  },
};

/** Rebuilders for NEWOBJ / copyreg.__newobj__ (cls.__new__(cls, *args)). */
const NEWOBJ = {
  'collections.OrderedDict': () => new V.PyDict('collections.OrderedDict'),
  'collections.defaultdict': () => new V.PyDict('collections.defaultdict'),
  'collections.Counter': () => new V.PyDict('collections.Counter'),
  'builtins.set': () => new V.PySet([], false),
  'builtins.dict': () => new V.PyDict('dict'),
  'builtins.list': () => [],
};

function num(v) { return v instanceof V.PyFloat ? v.value : typeof v === 'bigint' ? Number(v) : Number(v); }
function listOf(v) {
  if (Array.isArray(v)) return v;
  if (v instanceof V.PySet) return v.items;
  return [];
}
function dictFrom(v, typeName) {
  const d = new V.PyDict(typeName);
  if (v instanceof V.PyDict) v.keys.forEach((k, i) => d.set(k, v.values[i]));
  else if (Array.isArray(v)) for (const pair of v) if (Array.isArray(pair) && pair.length === 2) d.set(pair[0], pair[1]);
  return d;
}
function bytesFromArgs(a, kind) {
  if (a[0] instanceof V.PyBytes) return new V.PyBytes(a[0].data, kind);
  if (isStr(a[0])) {
    const enc = isStr(a[1]) ? a[1].toLowerCase().replace(/[-_]/g, '') : 'utf8';
    return new V.PyBytes(enc === 'latin1' ? latin1Bytes(a[0]) : new TextEncoder().encode(a[0]), kind);
  }
  if (Array.isArray(a[0]) && a[0].every(x => Number.isInteger(x) && x >= 0 && x < 256)) return new V.PyBytes(Uint8Array.from(a[0]), kind);
  return undefined;
}

function newObject(cls, args, how) {
  if (cls instanceof V.PyGlobal) {
    const f = NEWOBJ[normalName(cls)];
    if (f) { const v = f(args); if (v !== undefined) return v; }
  }
  return new V.PyObject(cls, how, args);
}

function call(func, args, how) {
  if (func instanceof V.PyGlobal) {
    const f = REBUILD[normalName(func)];
    if (f) {
      try { const v = f(args); if (v !== undefined) return v; } catch { /* fall back to a placeholder */ }
    }
  }
  return new V.PyObject(func, how, args);
}

function describe(v) {
  if (v instanceof V.PyGlobal) return v.qualname;
  if (v instanceof V.PyObject) return `${v.className} object`;
  if (v instanceof V.NdArray) return v.clsName;
  if (v instanceof V.NpDtype) return 'numpy.dtype';
  if (v instanceof V.PyDict) return v.typeName;
  if (v instanceof V.PySet) return v.typeName;
  if (Array.isArray(v)) return V.isTuple(v) ? 'tuple' : 'list';
  if (v === null) return 'None';
  return typeof v;
}

function applyBuild(obj, state) {
  if (obj instanceof V.NdArray && obj.shape === null) {
    // (version, shape, dtype, is_fortran, rawdata), version optional
    const s = Array.isArray(state) ? (state.length === 5 ? state.slice(1) : state.slice(0, 4)) : null;
    if (s && Array.isArray(s[0]) && s[1] instanceof V.NpDtype) {
      obj.shape = s[0].map(intOf); obj.dtype = s[1]; obj.fortran = !!s[2];
      if (Array.isArray(s[3])) obj.objects = s[3]; else obj.raw = bytesOf(s[3]);
      return;
    }
  }
  if (obj instanceof V.NpDtype && Array.isArray(state)) {
    // (version, endian, subarray, names, fields, elsize, alignment, flags[, metadata])
    if (isStr(state[1])) obj.endian = state[1];
    obj.subarray = state[2] ?? null;
    obj.fields = state[3] ?? null;
    if (state.length > 8) obj.metadata = state[8];
    return;
  }
  if (obj instanceof V.PyObject) {
    obj.state = obj.state === undefined ? state : V.tuple([obj.state, state]);
    return;
  }
  if (obj instanceof V.PyDict && state instanceof V.PyDict) {
    state.keys.forEach((k, i) => obj.set(k, state.values[i]));
    return;
  }
  if (obj && typeof obj === 'object') obj.__state = state;
}

/**
 * Decode a pickle. Returns { value, protocol, events, opCount, error }.
 * On a damaged file, value is undefined and error explains where it stopped; events read so far are kept.
 * opts.maxEvents caps the recorded events (the count keeps going).
 */
function unpickle(bytes, opts = {}) {
  const maxEvents = opts.maxEvents || 20000;
  const events = [];
  let eventCount = 0;
  const record = (op, pos, target, extra) => { eventCount++; if (events.length < maxEvents) events.push({ op, pos, target, ...extra }); };
  let stack = [];
  const metastack = [];
  const memo = new Map();
  let protocol = 0, opCount = 0, pos = 0;
  const pop = () => {
    if (!stack.length) throw new PickleError(`The stack is empty at byte ${pos}. The file is damaged.`, pos);
    return stack.pop();
  };
  const top = () => {
    if (!stack.length) throw new PickleError(`The stack is empty at byte ${pos}. The file is damaged.`, pos);
    return stack[stack.length - 1];
  };
  const popMark = () => {
    if (!metastack.length) throw new PickleError(`A MARK is missing before byte ${pos}. The file is damaged.`, pos);
    const items = stack;
    stack = metastack.pop();
    return items;
  };
  const appendTo = (target, items) => {
    if (Array.isArray(target)) { for (let i = 0; i < items.length; i++) target.push(items[i]); return; }
    if (target instanceof V.PyObject) { if (!target.listItems) target.listItems = []; for (let i = 0; i < items.length; i++) target.listItems.push(items[i]); return; }
    throw new PickleError(`APPEND at byte ${pos} targets a ${describe(target)}, not a list.`, pos);
  };
  const setItems = (target, items) => {
    if (target instanceof V.PyDict) { for (let i = 0; i + 1 < items.length; i += 2) target.set(items[i], items[i + 1]); return; }
    if (target instanceof V.PyObject) { if (!target.dictItems) target.dictItems = new V.PyDict(); for (let i = 0; i + 1 < items.length; i += 2) target.dictItems.set(items[i], items[i + 1]); return; }
    throw new PickleError(`SETITEM at byte ${pos} targets a ${describe(target)}, not a dict.`, pos);
  };
  try {
    for (const op of iterOps(bytes, opts.start || 0)) {
      opCount++;
      pos = op.pos;
      const a = op.arg;
      switch (op.name) {
        case 'PROTO': protocol = a; if (a > 5) throw new PickleError(`Pickle protocol ${a} is newer than 5, the highest this viewer reads.`, pos); break;
        case 'FRAME': break;
        case 'STOP': return { value: pop(), protocol, events, eventCount, opCount, error: null, endPos: op.end, trailing: bytes.length - op.end };
        case 'MARK': metastack.push(stack); stack = []; break;
        case 'POP': if (stack.length) stack.pop(); else popMark(); break;
        case 'POP_MARK': popMark(); break;
        case 'DUP': stack.push(top()); break;
        case 'PUT': case 'BINPUT': case 'LONG_BINPUT': memo.set(Number(a), top()); break;
        case 'MEMOIZE': memo.set(memo.size, top()); break;
        case 'GET': case 'BINGET': case 'LONG_BINGET':
          if (!memo.has(Number(a))) throw new PickleError(`Memo entry ${a} is read at byte ${pos} before it was stored.`, pos);
          stack.push(memo.get(Number(a))); break;
        case 'NONE': stack.push(null); break;
        case 'NEWTRUE': stack.push(true); break;
        case 'NEWFALSE': stack.push(false); break;
        case 'INT': case 'LONG': case 'BININT': case 'BININT1': case 'BININT2': case 'LONG1': case 'LONG4': stack.push(a); break;
        case 'FLOAT': case 'BINFLOAT': stack.push(Number.isInteger(a) ? new V.PyFloat(a) : a); break;
        case 'STRING': case 'BINSTRING': case 'SHORT_BINSTRING': stack.push(a); break;
        case 'UNICODE': case 'BINUNICODE': case 'SHORT_BINUNICODE': case 'BINUNICODE8': stack.push(a); break;
        case 'BINBYTES': case 'SHORT_BINBYTES': case 'BINBYTES8': stack.push(new V.PyBytes(a, 'bytes')); break;
        case 'BYTEARRAY8': stack.push(new V.PyBytes(a, 'bytearray')); break;
        case 'NEXT_BUFFER': stack.push(new V.PyOutOfBand(memo.size)); break;
        case 'READONLY_BUFFER': break;
        case 'EMPTY_TUPLE': stack.push(V.tuple([])); break;
        case 'TUPLE': { const t = V.tuple(popMark()); stack.push(t); break; }
        case 'TUPLE1': { const x = pop(); stack.push(V.tuple([x])); break; }
        case 'TUPLE2': { const y = pop(), x = pop(); stack.push(V.tuple([x, y])); break; }
        case 'TUPLE3': { const z = pop(), y = pop(), x = pop(); stack.push(V.tuple([x, y, z])); break; }
        case 'EMPTY_LIST': stack.push([]); break;
        case 'LIST': { const l = popMark(); stack.push(l); break; }
        case 'APPEND': { const x = pop(); appendTo(top(), [x]); break; }
        case 'APPENDS': { const items = popMark(); appendTo(top(), items); break; }
        case 'EMPTY_DICT': stack.push(new V.PyDict()); break;
        case 'DICT': { const items = popMark(); const d = new V.PyDict(); for (let i = 0; i + 1 < items.length; i += 2) d.set(items[i], items[i + 1]); stack.push(d); break; }
        case 'SETITEM': { const v = pop(), k = pop(); setItems(top(), [k, v]); break; }
        case 'SETITEMS': { const items = popMark(); setItems(top(), items); break; }
        case 'EMPTY_SET': stack.push(new V.PySet([], false)); break;
        case 'FROZENSET': { const s = new V.PySet(popMark(), true); stack.push(s); break; }
        case 'ADDITEMS': {
          const items = popMark(); const s = top();
          if (!(s instanceof V.PySet)) throw new PickleError(`ADDITEMS at byte ${pos} targets a ${describe(s)}, not a set.`, pos);
          for (let i = 0; i < items.length; i++) s.items.push(items[i]);
          break;
        }
        case 'GLOBAL': { const g = new V.PyGlobal(a[0], a[1]); record('GLOBAL', pos, g.qualname); stack.push(g); break; }
        case 'STACK_GLOBAL': {
          const name = pop(), mod = pop();
          if (!isStr(name) || !isStr(mod)) throw new PickleError(`STACK_GLOBAL at byte ${pos} needs two strings.`, pos);
          const g = new V.PyGlobal(mod, name); record('STACK_GLOBAL', pos, g.qualname); stack.push(g); break;
        }
        case 'INST': {
          const args = V.tuple(popMark()); const g = new V.PyGlobal(a[0], a[1]);
          record('INST', pos, g.qualname, { args });
          stack.push(call(g, args, 'INST')); break;
        }
        case 'OBJ': {
          const items = popMark(); const cls = items.shift(); const args = V.tuple(items);
          record('OBJ', pos, describe(cls), { args });
          stack.push(call(cls, args, 'OBJ')); break;
        }
        case 'NEWOBJ': {
          const args = pop(), cls = pop();
          record('NEWOBJ', pos, describe(cls), { args });
          stack.push(newObject(cls, Array.isArray(args) ? args : V.tuple([args]), 'NEWOBJ')); break;
        }
        case 'NEWOBJ_EX': {
          const kwargs = pop(), args = pop(), cls = pop();
          record('NEWOBJ_EX', pos, describe(cls), { args });
          const o = newObject(cls, Array.isArray(args) ? args : V.tuple([args]), 'NEWOBJ_EX');
          if (o instanceof V.PyObject) o.kwargs = kwargs;
          stack.push(o); break;
        }
        case 'REDUCE': {
          const args = pop(), func = pop();
          record('REDUCE', pos, describe(func), { args });
          stack.push(call(func, Array.isArray(args) ? args : V.tuple([args]), 'REDUCE')); break;
        }
        case 'BUILD': {
          const state = pop(); const obj = top();
          record('BUILD', pos, describe(obj));
          applyBuild(obj, state); break;
        }
        case 'PERSID': stack.push(new V.PyPersistent(a)); break;
        case 'BINPERSID': stack.push(new V.PyPersistent(pop())); break;
        case 'EXT1': case 'EXT2': case 'EXT4': record(op.name, pos, `extension code ${a}`); stack.push(new V.PyExtension(a)); break;
        default: throw new PickleError(`Opcode ${op.name} at byte ${pos} is not handled.`, pos);
      }
    }
  } catch (e) {
    const where = e instanceof PickleError ? e.pos : pos;
    const msg = e instanceof RangeError ? `The file is too deep or too large to decode here (${e.message}).` : e.message;
    return { value: undefined, protocol, events, eventCount, opCount, error: { message: msg, pos: where } };
  }
  return { value: undefined, protocol, events, eventCount, opCount, error: { message: 'The file ends before the STOP opcode.', pos } };
}

module.exports = { unpickle, normalName, describe, REBUILD };
