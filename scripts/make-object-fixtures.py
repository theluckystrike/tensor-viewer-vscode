"""Fixtures for object arrays (np.save of dicts, lists, strings) and for percentiles.

Run: python3 scripts/make-object-fixtures.py test/fixtures/objects
Needs numpy. Writes .npy/.npz files and expected.json. The hostile files are built from pickle bytes;
nothing in them is ever loaded here.
"""
import io
import json
import os
import pickle
import struct
import sys
import zipfile
from fractions import Fraction

import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "test/fixtures/objects"
os.makedirs(out, exist_ok=True)
p = lambda n: os.path.join(out, n)
expected = {}


def obj_array(items, shape=None):
    a = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        a[i] = x
    return a.reshape(shape) if shape else a


# ---------- object arrays that hold real data ----------
w = np.arange(12, dtype="<f4").reshape(3, 4)
layer_w = np.array([[1.5, -2.0], [0.25, 4.0]], dtype="<f8")
layer_b = np.array([2**62 + 1, -(2**63), 5], dtype="<i8")
d = {
    "weights": w,
    "bias": np.array([0.5, -1.5], dtype="<f8"),
    "name": "model-a",
    "epoch": 7,
    "lr": 0.001,
    "flag": True,
    "nothing": None,
    "blob": b"\x00ab",
    "layers": [{"w": layer_w, "bias": layer_b}, {"w": np.ones((2, 2), dtype="<f2"), "bias": np.zeros(3, dtype="<i4")}],
    "tags": ("a", "b"),
    "big": np.int64(2**62 + 1),
    "scalar": np.float32(2.5),
    "words": np.array(["alpha", "beta"]),
    "fortran": np.asfortranarray(np.arange(6, dtype="<i2").reshape(2, 3)),
    3: "int key",
}
np.save(p("dict_weights.npy"), d, allow_pickle=True)
expected["dict_weights.npy"] = {
    "tensors": {
        "['weights']": {"dtype": "float32", "shape": [3, 4], "values": w.ravel().tolist()},
        "['bias']": {"dtype": "float64", "shape": [2], "values": [0.5, -1.5]},
        "['layers'][0]['w']": {"dtype": "float64", "shape": [2, 2], "values": layer_w.ravel().tolist()},
        "['layers'][0]['bias']": {"dtype": "int64", "shape": [3]},
        "['layers'][1]['w']": {"dtype": "float16", "shape": [2, 2], "values": [1, 1, 1, 1]},
        "['layers'][1]['bias']": {"dtype": "int32", "shape": [3], "values": [0, 0, 0]},
        "['words']": {"dtype": "str5", "shape": [2]},
        "['fortran']": {"dtype": "int16", "shape": [2, 3], "values": [0, 3, 1, 4, 2, 5]},  # memory (column-major) order
    },
    "rows": {"['name']": '"model-a"', "['epoch']": "7", "['lr']": "0.001", "['flag']": "True", "['nothing']": "None",
             "['big']": "4611686018427387905 (numpy.int64)", "['scalar']": "2.5 (numpy.float32)", "[3]": '"int key"'},
    "layer_bias_exact": [str(x) for x in layer_b.tolist()],
}

lst = obj_array([np.zeros((2, 3), dtype="<f4"), np.arange(5, dtype="<i4"), np.arange(4, dtype="u1")])
np.save(p("list_arrays.npy"), lst, allow_pickle=True)
expected["list_arrays.npy"] = {"names": ["[0]", "[1]", "[2]"], "shapes": [[2, 3], [5], [4]]}

grid = obj_array([np.full(2, i, dtype="<f8") for i in range(4)], (2, 2))
np.save(p("grid_2x2.npy"), grid, allow_pickle=True)
expected["grid_2x2.npy"] = {"names": ["[0, 0]", "[0, 1]", "[1, 0]", "[1, 1]"]}

strs = obj_array(["a", "bb", None, 3, 2.5, ["x", 1]])
np.save(p("strings_only.npy"), strs, allow_pickle=True)
expected["strings_only.npy"] = {"rows": {"[0]": '"a"', "[1]": '"bb"', "[2]": "None", "[3]": "3", "[4]": "2.5"}}

meta = {"mean": np.array([0.1, 0.2, 0.3], dtype="<f4"), "classes": ["cat", "dog"]}
np.savez(p("with_dict.npz"), data=np.linspace(0, 1, 10, dtype="<f4"), meta=np.array(meta, dtype=object))
expected["with_dict.npz"] = {"names": ["data", "meta['mean']"]}
np.savez_compressed(p("with_dict_compressed.npz"), meta=np.array(meta, dtype=object))



# ---------- hostile files, from hand-written pickle bytes ----------
def npy_with_pickle(payload, shape=(1,)):
    header = "{'descr': '|O', 'fortran_order': False, 'shape': %s, }" % (str(tuple(shape)),)
    hb = header.encode("latin1")
    pad = 64 - ((10 + len(hb) + 1) % 64)
    hb = hb + b" " * pad + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hb)) + hb + payload


def g(mod, name):
    return b"c" + mod.encode() + b"\n" + name.encode() + b"\n"


def u(s):
    b = s.encode()
    return b"X" + struct.pack("<I", len(b)) + b


def ndarray_ops(shape, code, raw):
    """numpy's own rebuild path: _reconstruct(ndarray, (0,), b'b') then BUILD (1, shape, dtype, False, raw)."""
    o = g("numpy.core.multiarray", "_reconstruct") + g("numpy", "ndarray") + b"K\x00\x85" + b"C\x01b" + b"\x87R"
    o += b"(K\x01" + b"(" + b"".join(b"\x8a\x08" + struct.pack("<q", n) for n in shape) + b"t"
    o += g("numpy", "dtype") + u(code) + b"\x89\x88\x87R" + b"(K\x03" + u("<") + b"NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb"
    o += b"\x89" + b"B" + struct.pack("<I", len(raw)) + raw + b"tb"
    return o


def object_array_ops(items_ops, n):
    """An object ndarray of n items whose BUILD state holds a list."""
    o = g("numpy.core.multiarray", "_reconstruct") + g("numpy", "ndarray") + b"K\x00\x85" + b"C\x01b" + b"\x87R"
    o += b"(K\x01" + b"(" + b"\x8a\x08" + struct.pack("<q", n) + b"t"
    o += g("numpy", "dtype") + u("O8") + b"\x89\x88\x87R" + b"(K\x03" + u("|") + b"NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?tb"
    o += b"\x89" + b"]" + b"(" + b"".join(items_ops) + b"e" + b"tb"
    return o


# np.save always pickles with protocol 3; protocol 2 and 5 files come from other writers.
for proto in (2, 5):
    arr = obj_array([np.arange(6, dtype="<f8").reshape(2, 3), {"k": np.arange(3, dtype=">i4")}])
    open(p(f"proto{proto}.npy"), "wb").write(npy_with_pickle(pickle.dumps(arr, protocol=proto), (2,)))
expected["proto2.npy"] = expected["proto5.npy"] = {"names": ["[0]", "[1]['k']"], "values": [[0, 1, 2, 3, 4, 5], [0, 1, 2]]}

f8 = struct.pack("<3d", 1.0, 2.0, 3.0)
# os.system / builtins.eval REDUCE inside an object array, next to a real array.
evil_cmd = "touch /tmp/tensor-viewer-pwned-acq1"
evil = g("os", "system") + b"(" + u(evil_cmd) + b"tR"
evil2 = g("builtins", "eval") + b"(" + u("__import__('os').system('touch /tmp/tensor-viewer-pwned-acq1')") + b"tR"
payload = b"\x80\x03" + object_array_ops([ndarray_ops([3], "f8", f8), evil, evil2], 3) + b"."
open(p("evil_reduce.npy"), "wb").write(npy_with_pickle(payload, (3,)))
expected["evil_reduce.npy"] = {"marker": "/tmp/tensor-viewer-pwned-acq1"}
with open(p("evil_in_npz.npz"), "wb") as fh:
    with zipfile.ZipFile(fh, "w") as z:
        ok = io.BytesIO()
        np.save(ok, np.arange(2.0))
        z.writestr("ok.npy", ok.getvalue())
        z.writestr("bad.npy", npy_with_pickle(payload, (3,)))

# A nested array that declares a huge shape but holds 24 bytes.
payload = b"\x80\x03" + object_array_ops([ndarray_ops([10**12, 10**12], "f8", f8), ndarray_ops([4], "f8", f8)], 2) + b"."
open(p("huge_shape.npy"), "wb").write(npy_with_pickle(payload, (2,)))

# 100,000 nested lists: [[[[...]]]].
N = 100000
payload = b"\x80\x02" + b"]" * N + b"a" * (N - 1) + b"."
open(p("deep_nesting.npy"), "wb").write(npy_with_pickle(payload))

# A list that holds itself, next to an array.
payload = b"\x80\x03" + object_array_ops([b"]q\x01h\x01a", ndarray_ops([3], "f8", f8)], 2) + b"."
open(p("cyclic.npy"), "wb").write(npy_with_pickle(payload, (2,)))

# Shared values: x0 = [array], x1 = [x0, x0], ... x60. Walking every path would never end.
o = b"\x80\x03" + b"]" + b"(" + ndarray_ops([3], "f8", f8) + b"e" + b"q\x00" + b"0"
for i in range(1, 61):
    o += b"]" + b"(h" + bytes([i - 1]) + b"h" + bytes([i - 1]) + b"e" + b"q" + bytes([i]) + b"0"
o += b"h" + bytes([60]) + b"."
open(p("shared_bomb.npy"), "wb").write(npy_with_pickle(o))

# Cut off in the middle of the pickle.
full = open(p("dict_weights.npy"), "rb").read()
open(p("truncated_object.npy"), "wb").write(full[: int(len(full) * 0.6)])

# ---------- percentiles ----------
PCTS = [1, 5, 25, 50, 75, 95, 99]


def exact_pcts(values):
    s = sorted(values)
    n = len(s)
    out = []
    for q in PCTS:
        num = (n - 1) * q
        i, r = divmod(num, 100)
        v = Fraction(s[i]) if not r else Fraction(s[i]) + (Fraction(s[i + 1]) - Fraction(s[i])) * Fraction(r, 100)
        out.append(v)
    return out


def frac_text(v):
    if v.denominator == 1:
        return str(v.numerator)
    whole = abs(v.numerator) // v.denominator
    rest = Fraction(abs(v.numerator), v.denominator) - whole
    digits = str(int(rest * 100)).rjust(2, "0").rstrip("0")
    return ("-" if v < 0 else "") + f"{whole}.{digits}"


rng = np.random.default_rng(7)
big_i = np.concatenate([rng.integers(-(2**62), 2**62, 999, dtype=np.int64), np.array([2**63 - 1, -(2**63), 2**53 + 1], dtype=np.int64)])
np.save(p("int64_big.npy"), big_i)
expected["int64_big.npy"] = {"pcts": [frac_text(v) for v in exact_pcts([int(x) for x in big_i])]}
big_u = np.array([2**64 - 1, 0, 1, 2**64 - 2, 2**63, 12345], dtype=np.uint64)
np.save(p("uint64_big.npy"), big_u)
expected["uint64_big.npy"] = {"pcts": [frac_text(v) for v in exact_pcts([int(x) for x in big_u])]}

wide = rng.standard_normal((700, 300)).astype("<f4")
wide[0, :5] = [np.nan, np.inf, -np.inf, np.nan, 0]
np.save(p("normal_700x300_f4.npy"), wide)
fin = wide[np.isfinite(wide)].astype(np.float64)
expected["normal_700x300_f4.npy"] = {"pcts": np.percentile(fin, PCTS).tolist(), "finite": int(fin.size)}
ties = np.repeat(np.arange(5, dtype="<i2"), [1, 1000, 3, 2, 1])
np.save(p("ties_i2.npy"), ties)
expected["ties_i2.npy"] = {"pcts": np.percentile(ties.astype(np.float64), PCTS).tolist()}
cx = (rng.standard_normal(501) + 1j * rng.standard_normal(501)).astype("<c16")
np.save(p("complex_501_c16.npy"), cx)
expected["complex_501_c16.npy"] = {"pcts": np.percentile(np.abs(cx), PCTS).tolist()}

with open(p("expected.json"), "w") as fh:
    json.dump(expected, fh, indent=1, sort_keys=True)
print("wrote", len(os.listdir(out)), "files to", out)
