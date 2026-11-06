"""Writes real sample files for the tests and screenshots with the official writers
(numpy, safetensors, gguf-py). Every file is small; the whole folder stays under 5 MB.

Usage: python3 scripts/make-fixtures.py test/fixtures
"""
import json
import os
import struct
import sys

import numpy as np
from safetensors.numpy import save_file
import gguf
from gguf import quants

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(42)
expected = {}


def p(name):
    return os.path.join(out, name)


def stats(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    fin = a[np.isfinite(a)]
    return {
        "count": int(a.size),
        "nan": int(np.isnan(a).sum()),
        "posinf": int(np.isposinf(a).sum()),
        "neginf": int(np.isneginf(a).sum()),
        "min": float(fin.min()) if fin.size else None,
        "max": float(fin.max()) if fin.size else None,
        "mean": float(fin.mean()) if fin.size else None,
        "std": float(fin.std()) if fin.size else None,
    }


# ---------- .npy ----------
a = np.arange(6, dtype="<f4").reshape(2, 3)
np.save(p("arange_2x3_f4.npy"), a)
expected["arange_2x3_f4.npy"] = {"shape": [2, 3], "descr": "<f4", "values": a.ravel().tolist(), "stats": stats(a)}

b = np.array([1, -2, 300000, -4, 5], dtype=">i4")
np.save(p("bigendian_i4.npy"), b)
expected["bigendian_i4.npy"] = {"shape": [5], "descr": ">i4", "values": b.astype(np.int64).tolist(), "stats": stats(b)}

f = np.asfortranarray(np.arange(12, dtype="<f8").reshape(3, 4))
np.save(p("fortran_3x4_f8.npy"), f)
expected["fortran_3x4_f8.npy"] = {"shape": [3, 4], "descr": "<f8", "fortran": True, "values": f.ravel(order="C").tolist(), "stats": stats(f)}

h = np.array([0.5, -1.25, np.nan, np.inf, -np.inf, 65504, 6e-8, 0], dtype="<f2")
np.save(p("specials_f2.npy"), h)
expected["specials_f2.npy"] = {"shape": [8], "descr": "<f2", "values": [None if np.isnan(x) else ("inf" if x == np.inf else "-inf" if x == -np.inf else float(x)) for x in h.astype(np.float64)], "stats": stats(h)}

i64 = np.array([[-(2 ** 40), 7], [2 ** 40, -7]], dtype="<i8")
np.save(p("int64_2x2.npy"), i64)
expected["int64_2x2.npy"] = {"shape": [2, 2], "descr": "<i8", "values": i64.ravel().tolist(), "stats": stats(i64)}

bo = np.array([True, False, True, True])
np.save(p("bool_4.npy"), bo)
expected["bool_4.npy"] = {"shape": [4], "descr": "|b1", "values": [1, 0, 1, 1], "stats": stats(bo)}

c = np.array([3 + 4j, 0 + 1j, -6 - 8j], dtype="<c8")
np.save(p("complex_3_c8.npy"), c)
expected["complex_3_c8.npy"] = {"shape": [3], "descr": "<c8", "values": np.abs(c).astype(np.float64).tolist(), "stats": stats(np.abs(c))}

s = np.array(["alpha", "beta", "gamma"])
np.save(p("strings_U5.npy"), s)
expected["strings_U5.npy"] = {"shape": [3], "descr": "<U5"}

# An object array is stored with pickle. Tensor Viewer must refuse to read its data.
o = np.array([{"a": 1}, [1, 2], "x"], dtype=object)
np.save(p("object_pickle.npy"), o, allow_pickle=True)
expected["object_pickle.npy"] = {"shape": [3], "descr": "|O"}

st = np.zeros(3, dtype=[("x", "<f4"), ("y", "<i2")])
np.save(p("structured.npy"), st)
expected["structured.npy"] = {"shape": [3]}

sc = np.array(3.5, dtype="<f4")
np.save(p("scalar_f4.npy"), sc)
expected["scalar_f4.npy"] = {"shape": [], "descr": "<f4", "values": [3.5], "stats": stats(sc)}

# Version 2.0 header (4-byte header length).
v2 = np.linspace(-1, 1, 7, dtype="<f4")
with open(p("v2_header.npy"), "wb") as fh:
    np.lib.format.write_array(fh, v2, version=(2, 0))
expected["v2_header.npy"] = {"shape": [7], "descr": "<f4", "values": v2.astype(np.float64).tolist(), "stats": stats(v2)}

# 1-D signal for the line plot.
t = np.linspace(0, 8 * np.pi, 5000, dtype=np.float32)
sig = (np.sin(t) * np.exp(-t / 20) + 0.08 * rng.standard_normal(5000)).astype("<f4")
np.save(p("signal_5000_f4.npy"), sig)
expected["signal_5000_f4.npy"] = {"shape": [5000], "descr": "<f4", "stats": stats(sig)}

# HxWx3 uint8 image.
H, W = 96, 128
yy, xx = np.mgrid[0:H, 0:W]
img = np.zeros((H, W, 3), dtype=np.uint8)
img[..., 0] = (xx * 255 // (W - 1)).astype(np.uint8)
img[..., 1] = (yy * 255 // (H - 1)).astype(np.uint8)
img[..., 2] = (((xx // 16 + yy // 16) % 2) * 200).astype(np.uint8)
np.save(p("image_96x128x3_u1.npy"), img)
expected["image_96x128x3_u1.npy"] = {"shape": [96, 128, 3], "descr": "|u1", "first_pixel": img[0, 0].tolist(), "last_pixel": img[-1, -1].tolist(), "stats": stats(img)}

# A 2-D matrix bigger than the default sample budget in tests (sampled by rows).
big = rng.standard_normal((1024, 512)).astype("<f4")
np.save(p("matrix_1024x512_f4.npy"), big)
expected["matrix_1024x512_f4.npy"] = {"shape": [1024, 512], "descr": "<f4", "stats": stats(big), "row_1000_first": float(big[1000, 0])}

# ---------- .npz ----------
np.savez(p("bundle_stored.npz"), weights=a, labels=np.array([0, 1, 1, 0], dtype="<i8"), image=img, words=s)
expected["bundle_stored.npz"] = {"names": ["weights", "labels", "image", "words"]}
np.savez_compressed(p("bundle_compressed.npz"), weights=a, signal=sig, matrix=big[:256])
expected["bundle_compressed.npz"] = {"names": ["weights", "signal", "matrix"], "signal_stats": stats(sig), "matrix_stats": stats(big[:256])}
np.savez(p("with_object.npz"), ok=a, bad=o)

# ---------- .safetensors ----------
pos = np.arange(512)[:, None]
dim = np.arange(256)[None, :]
angle = pos / np.power(10000, (2 * (dim // 2)) / 256)
wpe = np.where(dim % 2 == 0, np.sin(angle), np.cos(angle)).astype(np.float32)
tensors = {
    "wpe.weight": wpe,
    "wte.weight": (rng.standard_normal((1000, 256)) * 0.02).astype(np.float16),
    "h.0.ln_1.weight": (1 + 0.05 * rng.standard_normal(256)).astype(np.float32),
    "h.0.ln_1.bias": (0.01 * rng.standard_normal(256)).astype(np.float32),
    "h.0.attn.c_attn.weight": (rng.standard_normal((256, 768)) * 0.02).astype(np.float16),
    "h.0.attn.bias": np.tril(np.ones((1, 1, 64, 64), dtype=np.uint8)),
    "h.0.mlp.c_fc.weight": (rng.standard_normal((256, 1024)) * 0.02).astype(np.float16),
    "step": np.array([12000], dtype=np.int64),
    "mask": np.array([True, False, True]),
}
save_file(tensors, p("gpt2_tiny.safetensors"), metadata={"format": "pt", "source": "make-fixtures.py"})
expected["gpt2_tiny.safetensors"] = {
    "names": list(tensors.keys()),
    "params": int(sum(v.size for v in tensors.values())),
    "wpe_stats": stats(wpe),
    "metadata": {"format": "pt", "source": "make-fixtures.py"},
}


# Hand-written safetensors with BF16 and F8 tensors (numpy has no such dtypes).
def f32_to_bf16_bytes(x):
    u = np.asarray(x, dtype="<f4").view("<u4")
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()


bf_vals = np.array([1.0, -2.5, 3.140625, 0.0, 1e-3, -65280.0], dtype=np.float32)
bf = f32_to_bf16_bytes(bf_vals)
e4m3 = bytes([0x38, 0xB8, 0x40, 0x7E, 0x00, 0x01])  # 1.0, -1.0, 2.0, 448.0, 0.0, 2^-9
e5m2 = bytes([0x3C, 0xBC, 0x7B, 0x7C, 0x7E, 0x01])  # 1.0, -1.0, 57344.0, inf, nan, 2^-16
header = {
    "__metadata__": {"note": "hand written"},
    "bf16": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, len(bf)]},
    "f8a": {"dtype": "F8_E4M3", "shape": [6], "data_offsets": [len(bf), len(bf) + 6]},
    "f8b": {"dtype": "F8_E5M2", "shape": [6], "data_offsets": [len(bf) + 6, len(bf) + 12]},
}
hb = json.dumps(header).encode()
hb += b" " * ((8 - len(hb) % 8) % 8)
with open(p("bf16_f8.safetensors"), "wb") as fh:
    fh.write(struct.pack("<Q", len(hb)) + hb + bf + e4m3 + e5m2)
bf_round = np.frombuffer(bf, dtype="<u2").astype("<u4") << 16
expected["bf16_f8.safetensors"] = {
    "bf16": bf_round.view("<f4").astype(np.float64).tolist(),
    "f8a": [1.0, -1.0, 2.0, 448.0, 0.0, 2 ** -9],
    "f8b": [1.0, -1.0, 57344.0, "inf", "nan", 2 ** -16],
}

# Broken files for error handling.
with open(p("broken_header.safetensors"), "wb") as fh:
    fh.write(struct.pack("<Q", 10 ** 12) + b"{}")
with open(p("truncated.npy"), "wb") as fh:
    fh.write(open(p("arange_2x3_f4.npy"), "rb").read()[:40])

# ---------- .gguf ----------
# Test file: one tensor per supported type, with reference values from gguf-py.
w = gguf.GGUFWriter(p("quant_types.gguf"), "llama")
w.add_name("quant-types-test")
w.add_context_length(4096)
w.add_block_count(1)
w.add_embedding_length(256)
w.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_K_M)
w.add_tokenizer_model("llama")
w.add_token_list([f"tok{i}" for i in range(300)])
w.add_array("test.int_list", [1, 2, 3])
w.add_string("tokenizer.chat_template", "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}")
ref = {}
base = (rng.standard_normal((4, 256)) * 0.5).astype(np.float32)
w.add_tensor("f32", base)
ref["f32"] = base
w.add_tensor("f16", base.astype(np.float16))
ref["f16"] = base.astype(np.float16).astype(np.float32)
for name in ["Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "BF16"]:
    qt = getattr(gguf.GGMLQuantizationType, name)
    q = quants.quantize(base, qt)
    w.add_tensor(name.lower(), q, raw_dtype=qt)
    ref[name.lower()] = quants.dequantize(q, qt).astype(np.float32)
# K-quants: gguf-py can only dequantize them, so write random blocks with sane f16 scales.
for name in ["Q4_K", "Q5_K", "Q6_K"]:
    qt = getattr(gguf.GGMLQuantizationType, name)
    blck, tsize = gguf.GGML_QUANT_SIZES[qt]
    raw = rng.integers(0, 256, size=(4, tsize), dtype=np.uint8)
    scale = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
    if name == "Q6_K":
        raw[:, tsize - 2:tsize] = scale
    else:
        raw[:, 0:2] = scale
        raw[:, 2:4] = np.frombuffer(np.float16(0.002).tobytes(), dtype=np.uint8)
    w.add_tensor(name.lower(), raw, raw_dtype=qt)
    ref[name.lower()] = quants.dequantize(raw, qt).astype(np.float32).reshape(4, blck)
w.add_tensor("i32", np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32))
ref["i32"] = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.float32)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
np.savez(p("quant_types_ref.npz"), **ref)
expected["quant_types.gguf"] = {"names": list(ref.keys()), "context_length": 4096, "tokens": 300, "architecture": "llama"}

# Screenshot file: a small llama-style model with realistic tensor names and mixed quant types.
w = gguf.GGUFWriter(p("tiny-llama-q8.gguf"), "llama")
w.add_name("TinyStories Llama 3M (demo)")
w.add_context_length(2048)
w.add_embedding_length(256)
w.add_block_count(4)
w.add_feed_forward_length(704)
w.add_head_count(8)
w.add_head_count_kv(4)
w.add_rope_freq_base(10000.0)
w.add_layer_norm_rms_eps(1e-5)
w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
w.add_tokenizer_model("llama")
vocab = 2048
w.add_token_list(["<unk>", "<s>", "</s>"] + [f"▁w{i}" for i in range(vocab - 3)])
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_chat_template("{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}")


def weight(rows, cols, outliers=3):
    m = rng.standard_normal((rows, cols)).astype(np.float32) * 0.02
    m *= (1 + 0.6 * np.sin(np.arange(rows) / 9.0))[:, None].astype(np.float32)
    for c in rng.choice(cols, outliers, replace=False):
        m[:, c] *= 6
    return m


def add(name, arr, qt):
    if qt in ("F32", "F16"):
        w.add_tensor(name, arr.astype(np.float32 if qt == "F32" else np.float16))
    else:
        t = getattr(gguf.GGMLQuantizationType, qt)
        w.add_tensor(name, quants.quantize(arr, t), raw_dtype=t)


add("token_embd.weight", weight(vocab, 256), "Q8_0")
for L in range(4):
    add(f"blk.{L}.attn_norm.weight", (1 + 0.1 * rng.standard_normal(256)).astype(np.float32), "F32")
    add(f"blk.{L}.attn_q.weight", weight(256, 256), "Q8_0")
    add(f"blk.{L}.attn_k.weight", weight(128, 256), "Q8_0")
    add(f"blk.{L}.attn_v.weight", weight(128, 256), "Q4_1")
    add(f"blk.{L}.attn_output.weight", weight(256, 256), "Q8_0")
    add(f"blk.{L}.ffn_norm.weight", (1 + 0.1 * rng.standard_normal(256)).astype(np.float32), "F32")
    add(f"blk.{L}.ffn_gate.weight", weight(704, 256), "Q4_0")
    add(f"blk.{L}.ffn_up.weight", weight(704, 256), "Q4_0")
    add(f"blk.{L}.ffn_down.weight", weight(256, 704), "Q8_0")
add("output_norm.weight", (1 + 0.1 * rng.standard_normal(256)).astype(np.float32), "F32")
add("output.weight", weight(vocab, 256), "Q8_0")
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()

with open(p("expected.json"), "w") as fh:
    json.dump(expected, fh, indent=1, allow_nan=False)
total = sum(os.path.getsize(p(n)) for n in os.listdir(out))
print(f"wrote {len(os.listdir(out))} files, {total / 1e6:.2f} MB")
