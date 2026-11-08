'use strict';
// Exact percentiles (p1, p5, p25, median, p75, p95, p99) without sorting the whole tensor.
//
// Every finite value maps to a 64-bit key whose unsigned order is the value order (IEEE float bits with
// the sign handled, two's complement int64 with the sign bit flipped, uint64 as is). A radix select
// then finds the values at the wanted ranks: one pass counts the top 16 key bits, later passes count
// the next 16 bits only for values that share the prefix of a wanted rank, and once a prefix holds few
// enough values they are collected and sorted. Memory stays at a few MB for any tensor size, and
// int64 / uint64 values come out exact (as text) instead of rounded through float64.
// NaN and +-Inf are left out. Results use linear interpolation, numpy.percentile's default.

const PCTS = [1, 5, 25, 50, 75, 95, 99];
const PCT_LABELS = ['p1', 'p5', 'p25', 'median', 'p75', 'p95', 'p99'];
const COLLECT_MAX = 2 * 1024 * 1024; // keys gathered in one pass, at most
const TWO32 = 4294967296;
const TWO16 = 65536;

const f64 = new Float64Array(1);
const u32 = new Uint32Array(f64.buffer); // little-endian hosts: u32[0] = low word, u32[1] = high word

/**
 * A chunk of values: { f: Float64Array } for decoded values (NaN/Inf skipped),
 * or { raw: Uint8Array, count, signed, littleEndian } for int64 / uint64 bytes read as they are.
 * Fills hi/lo (Uint32Array) with keys and returns how many finite values it wrote.
 */
function keysOf(chunk, hi, lo) {
  let n = 0;
  if (chunk.f) {
    const d = chunk.f;
    for (let i = 0; i < d.length; i++) {
      const x = d[i];
      if (x - x !== 0) continue; // NaN or +-Inf
      f64[0] = x;
      let h = u32[1], l = u32[0];
      if (h & 0x80000000) { h = ~h >>> 0; l = ~l >>> 0; } else h = (h | 0x80000000) >>> 0;
      hi[n] = h; lo[n] = l; n++;
    }
    return n;
  }
  const dv = new DataView(chunk.raw.buffer, chunk.raw.byteOffset, chunk.raw.byteLength);
  const le = chunk.littleEndian !== false;
  for (let i = 0; i < chunk.count; i++) {
    const a = dv.getUint32(i * 8, le), b = dv.getUint32(i * 8 + 4, le);
    let h = le ? b : a;
    const l = le ? a : b;
    if (chunk.signed) h = (h ^ 0x80000000) >>> 0;
    hi[n] = h; lo[n] = l; n++;
  }
  return n;
}

function chunkLength(chunk) { return chunk.f ? chunk.f.length : chunk.count; }

/** Key (hi, lo) back to a value: a Number for floats, a BigInt for int64 / uint64. */
function valueOf(hi, lo, mode) {
  if (mode === 'float') {
    let h = hi, l = lo;
    if (h & 0x80000000) h = (h & 0x7fffffff) >>> 0; else { h = ~h >>> 0; l = ~l >>> 0; }
    u32[1] = h; u32[0] = l;
    return f64[0];
  }
  let h = hi;
  if (mode === 'i64') h = (h ^ 0x80000000) >>> 0;
  const v = (BigInt(h) << 32n) | BigInt(lo);
  return mode === 'i64' ? BigInt.asIntN(64, v) : v;
}

// Prefix of a key after `level` 16-bit digits, as a Number (at most 48 bits, so always exact).
function prefixOf(level, h, l) {
  if (level === 1) return h >>> 16;
  if (level === 2) return h;
  return h * TWO16 + (l >>> 16);
}
function digitOf(level, h, l) {
  if (level === 0) return h >>> 16;
  if (level === 1) return h & 0xffff;
  if (level === 2) return l >>> 16;
  return l & 0xffff;
}
// The key bits below a level-L prefix, as a Number (exact: at most 48 bits).
function restOf(level, h, l) {
  if (level === 1) return (h & 0xffff) * TWO32 + l;
  if (level === 2) return l;
  return l & 0xffff;
}
function keyFrom(level, prefix, rest) {
  if (level === 1) { const h = prefix * TWO16 + Math.floor(rest / TWO32); return [h >>> 0, (rest % TWO32) >>> 0]; }
  if (level === 2) return [prefix >>> 0, rest >>> 0];
  const h = Math.floor(prefix / TWO16), lTop = prefix % TWO16;
  return [h >>> 0, ((lTop * TWO16) + rest) >>> 0];
}

const tick = () => new Promise(r => setImmediate(r));

/**
 * Exact values at the given ranks (0-based, in the sorted list of finite values).
 * scan(onChunk) must call onChunk for every chunk of the data, in any order, and may be called several times.
 * @returns {Promise<{n:number, byRank:Map<number,number|bigint>}|null>} null when cancelled
 */
async function selectRanks(scan, mode, ranksFor, opts = {}) {
  const cancelled = opts.cancelled || (() => false);
  let hi = new Uint32Array(0), lo = new Uint32Array(0);
  const keyed = chunk => {
    const len = chunkLength(chunk);
    if (hi.length < len) { hi = new Uint32Array(len); lo = new Uint32Array(len); }
    return keysOf(chunk, hi, lo);
  };
  // Pass 1: top 16 bits of every key.
  const top = new Float64Array(TWO16);
  let n = 0;
  await scan(chunk => { const k = keyed(chunk); n += k; for (let i = 0; i < k; i++) top[hi[i] >>> 16]++; });
  if (cancelled()) return null;
  if (!n) return { n: 0, byRank: new Map() };
  const ranks = [...new Set(ranksFor(n))].sort((a, b) => a - b);
  // Each wanted rank: its prefix so far, its rank inside that prefix and how many values share the prefix.
  const want = ranks.map(r => ({ rank: r, prefix: 0, rr: r, count: n, done: false, value: undefined }));
  const descend = (w, hist) => {
    let acc = 0;
    for (let d = 0; d < TWO16; d++) {
      const c = hist[d];
      if (w.rr < acc + c) { w.rr -= acc; w.count = c; return d; }
      acc += c;
    }
    throw new Error('Internal: rank out of range.');
  };
  for (const w of want) w.prefix = descend(w, top);
  let level = 1;
  while (want.some(w => !w.done)) {
    if (cancelled()) return null;
    const open = want.filter(w => !w.done);
    if (level > 3) throw new Error('Internal: radix select did not finish.');
    // Group the open ranks by prefix. Small groups are collected and sorted, big ones are split again.
    const groups = new Map();
    for (const w of open) {
      if (!groups.has(w.prefix)) groups.set(w.prefix, { prefix: w.prefix, count: w.count, wants: [] });
      groups.get(w.prefix).wants.push(w);
    }
    let budget = COLLECT_MAX;
    for (const g of [...groups.values()].sort((a, b) => a.count - b.count)) {
      if (g.count <= budget) { g.collect = new Float64Array(g.count); g.fill = 0; budget -= g.count; } else g.hist = new Float64Array(TWO16);
    }
    const filter = new Uint8Array(TWO16);
    for (const g of groups.values()) filter[level === 1 ? g.prefix : level === 2 ? g.prefix >>> 16 : Math.floor(g.prefix / TWO32)] = 1;
    const lv = level;
    await scan(chunk => {
      const k = keyed(chunk);
      for (let i = 0; i < k; i++) {
        const h = hi[i];
        if (!filter[h >>> 16]) continue;
        const l = lo[i];
        const g = groups.get(prefixOf(lv, h, l));
        if (!g) continue;
        if (g.collect) { if (g.fill < g.collect.length) g.collect[g.fill++] = restOf(lv, h, l); } else g.hist[digitOf(lv, h, l)]++;
      }
    });
    if (cancelled()) return null;
    for (const g of groups.values()) {
      if (g.collect) {
        const arr = g.collect.subarray(0, g.fill).sort();
        for (const w of g.wants) {
          const [h, l] = keyFrom(lv, g.prefix, arr[Math.min(w.rr, arr.length - 1)]);
          w.value = valueOf(h, l, mode); w.done = true;
        }
      } else {
        for (const w of g.wants) {
          const d = descend(w, g.hist);
          if (lv === 3) { // all 64 bits are fixed now: every value in this bucket has the same key
            const [h, l] = keyFrom(3, w.prefix, d);
            w.value = valueOf(h, l, mode); w.done = true;
          } else w.prefix = w.prefix * TWO16 + d;
        }
      }
    }
    level++;
  }
  return { n, byRank: new Map(want.map(w => [w.rank, w.value])) };
}

/** Ranks needed for PCTS with linear interpolation: floor and floor+1 of (n-1)*p/100. */
function ranksFor(n) {
  const out = [];
  for (const p of PCTS) {
    const num = (n - 1) * p, i = Math.floor(num / 100);
    out.push(i);
    if (num % 100 && i + 1 < n) out.push(i + 1);
  }
  return out;
}

/** numpy's _lerp: a + (b - a) * t, computed from the b side when t >= 0.5 so that t = 1 gives b exactly. */
function lerp(a, b, t) {
  const d = b - a;
  const v = t >= 0.5 ? b - d * (1 - t) : a + d * t;
  return a === b ? a : v;
}

/** Exact decimal text of a + (b - a) * r / 100 for BigInts. */
function bigLerpText(a, b, r) {
  const num = a * 100n + (b - a) * BigInt(r); // value * 100
  const neg = num < 0n;
  const abs = neg ? -num : num;
  const whole = abs / 100n, frac = abs % 100n;
  let s = whole.toString();
  if (frac) s += '.' + frac.toString().padStart(2, '0').replace(/0+$/, '');
  return (neg ? '-' : '') + s;
}

/**
 * Percentiles over all finite values that scan() delivers.
 * mode: 'float' (decoded float64 chunks), 'i64' or 'u64' (raw 8-byte chunks).
 * @returns {Promise<{n:number, values:{p:number, label:string, value:number, text:string}[]}|null>}
 */
async function percentiles(scan, mode, opts = {}) {
  const sel = await selectRanks(scan, mode, ranksFor, opts);
  if (!sel) return null;
  const { n, byRank } = sel;
  if (!n) return { n: 0, values: [] };
  const values = PCTS.map((p, k) => {
    const num = (n - 1) * p, i = Math.floor(num / 100), r = num % 100;
    const a = byRank.get(i), b = r && i + 1 < n ? byRank.get(i + 1) : a;
    if (mode === 'float') {
      const v = r ? lerp(a, b, r / 100) : a;
      return { p, label: PCT_LABELS[k], value: v, text: String(v) };
    }
    const text = r ? bigLerpText(a, b, r) : a.toString();
    return { p, label: PCT_LABELS[k], value: Number(text), text };
  });
  return { n, values };
}

/** A scan over chunks that are already in memory. */
function memoryScan(chunks) {
  return async onChunk => { for (const c of chunks) onChunk(c); };
}

module.exports = { PCTS, PCT_LABELS, percentiles, selectRanks, ranksFor, memoryScan, lerp, bigLerpText, keysOf, valueOf, tick };
