'use strict';
// Plain data holders for decoded Python values. None of them has behaviour beyond storing data.

const TUPLE = Symbol('tuple');

/** A Python float with a whole value (2.0). Other floats stay plain JS numbers. */
class PyFloat { constructor(value) { this.value = value; } }

class PyBytes {
  /** kind: 'bytes' | 'bytearray' */
  constructor(data, kind = 'bytes') { this.data = data; this.kind = kind; }
}

class PyComplex { constructor(re, im) { this.re = re; this.im = im; } }

/** dict with any keys. Primitive keys (str, int, float, bool, None) are de-duplicated like Python does. */
class PyDict {
  constructor(typeName) { this.keys = []; this.values = []; this.index = new Map(); this.typeName = typeName || 'dict'; }
  static keyOf(k) {
    if (k === null) return 'N';
    switch (typeof k) {
      case 'string': return 's' + k;
      case 'number': return 'n' + (k === 0 ? 0 : k);
      case 'bigint': return 'n' + k.toString();
      case 'boolean': return 'n' + (k ? 1 : 0);
      default: return k instanceof PyFloat ? 'n' + k.value : null;
    }
  }
  set(k, v) {
    const h = PyDict.keyOf(k);
    if (h !== null && this.index.has(h)) { this.values[this.index.get(h)] = v; return; }
    if (h !== null) this.index.set(h, this.keys.length);
    this.keys.push(k); this.values.push(v);
  }
  get(k) { const h = PyDict.keyOf(k); return h !== null && this.index.has(h) ? this.values[this.index.get(h)] : undefined; }
  has(k) { const h = PyDict.keyOf(k); return h !== null && this.index.has(h); }
  get size() { return this.keys.length; }
}

class PySet {
  constructor(items, frozen, typeName) { this.items = items || []; this.frozen = !!frozen; this.typeName = typeName || (frozen ? 'frozenset' : 'set'); }
}

/** A module-level name from GLOBAL or STACK_GLOBAL. Never resolved to anything that runs. */
class PyGlobal {
  constructor(module, name) { this.module = String(module); this.name = String(name); }
  get qualname() { return `${this.module}.${this.name}`; }
}

/** An object we cannot rebuild. Holds what the pickle passed to it. */
class PyObject {
  constructor(cls, how, args) {
    this.cls = cls; // PyGlobal or any decoded value that was "called"
    this.how = how; // REDUCE, NEWOBJ, NEWOBJ_EX, INST, OBJ
    this.args = args; // array (tuple)
    this.kwargs = undefined;
    this.state = undefined; // from BUILD
    this.listItems = undefined; // from APPEND(S) on the object
    this.dictItems = undefined; // from SETITEM(S) on the object
    this.setItems = undefined;
  }
  get className() { return this.cls instanceof PyGlobal ? this.cls.qualname : 'call result'; }
}

class PyPersistent { constructor(pid) { this.pid = pid; } }
class PyExtension { constructor(code) { this.code = code; } }
class PyOutOfBand { constructor(index) { this.index = index; } }

/** A value of a known type that we show as text, such as datetime.datetime or decimal.Decimal. */
class PyText { constructor(typeName, text, parts) { this.typeName = typeName; this.text = text; this.parts = parts; } }

class PySlice { constructor(start, stop, step) { this.start = start; this.stop = stop; this.step = step; } }

/** numpy.dtype rebuilt from its pickle state. */
class NpDtype {
  constructor(code) { this.code = String(code); this.endian = '='; this.fields = null; this.subarray = null; this.metadata = null; }
}

/** numpy.ndarray rebuilt from its pickle data. raw is a Uint8Array for plain dtypes or an array for object dtype. */
class NdArray {
  constructor(clsName) { this.clsName = clsName || 'numpy.ndarray'; this.shape = null; this.dtype = null; this.fortran = false; this.raw = null; this.objects = null; }
}

class NpScalar { constructor(dtype, raw) { this.dtype = dtype; this.raw = raw; } }

/** pandas internals: one block of a BlockManager. */
class PdBlock { constructor(values, locs, ndim) { this.values = values; this.locs = locs; this.ndim = ndim; } }
class PdBlockManager { constructor(blocks, axes) { this.blocks = blocks; this.axes = axes; } }
/** pandas Index (from _new_Index). */
class PdIndex { constructor(clsName, fields) { this.clsName = clsName; this.fields = fields; } }

const isTuple = v => Array.isArray(v) && v[TUPLE] === true;
function tuple(items) { items[TUPLE] = true; return items; }

module.exports = {
  TUPLE, tuple, isTuple, PyFloat, PyBytes, PyComplex, PyDict, PySet, PyGlobal, PyObject, PyPersistent, PyExtension,
  PyOutOfBand, PyText, PySlice, NpDtype, NdArray, NpScalar, PdBlock, PdBlockManager, PdIndex,
};
