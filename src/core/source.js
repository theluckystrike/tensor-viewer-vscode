'use strict';
// Byte sources. Parsers only ever ask for (offset, length) ranges, so a 10 GB file is never read whole.
const fs = require('fs');
const { Inflate } = require('fflate');

class FileSource {
  static async open(path) {
    const fh = await fs.promises.open(path, 'r');
    const st = await fh.stat();
    return new FileSource(fh, st.size);
  }
  constructor(fh, size) { this.fh = fh; this.size = size; this.random = true; }
  async read(offset, length) {
    if (offset < 0 || length < 0 || offset + length > this.size) throw new RangeError(`Read past the end of the file (offset ${offset}, length ${length}, size ${this.size}).`);
    const buf = Buffer.allocUnsafe(length);
    let done = 0;
    while (done < length) {
      const { bytesRead } = await this.fh.read(buf, done, length - done, offset + done);
      if (bytesRead === 0) throw new RangeError('The file got shorter while it was being read.');
      done += bytesRead;
    }
    return new Uint8Array(buf.buffer, buf.byteOffset, length);
  }
  async close() { await this.fh.close().catch(() => {}); }
}

class BufferSource {
  constructor(bytes) { this.bytes = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes); this.size = this.bytes.length; this.random = true; }
  async read(offset, length) {
    if (offset < 0 || length < 0 || offset + length > this.size) throw new RangeError(`Read past the end of the data (offset ${offset}, length ${length}, size ${this.size}).`);
    return this.bytes.subarray(offset, offset + length);
  }
  async close() {}
}

/** A window [start, start+size) of another source (a stored zip entry). */
class SliceSource {
  constructor(parent, start, size) { this.parent = parent; this.start = start; this.size = size; this.random = true; }
  read(offset, length) {
    if (offset < 0 || offset + length > this.size) return Promise.reject(new RangeError('Read past the end of the entry.'));
    return this.parent.read(this.start + offset, length);
  }
  async close() {}
}

/**
 * A deflated zip entry. Only a prefix can be read: it inflates from the start and stops as soon
 * as it has enough bytes, so listing a compressed .npz never inflates whole arrays.
 */
class InflateSource {
  constructor(parent, start, compSize, size) {
    this.parent = parent; this.start = start; this.compSize = compSize; this.size = size; this.random = false;
    this.chunks = []; this.have = 0; this.pos = 0; this.ended = false;
    this.inf = new Inflate((chunk, final) => { this.chunks.push(chunk); this.have += chunk.length; if (final) this.ended = true; });
  }
  async ensure(n) {
    const CH = 256 * 1024;
    while (this.have < n && !this.ended && this.pos < this.compSize) {
      const len = Math.min(CH, this.compSize - this.pos);
      const data = await this.parent.read(this.start + this.pos, len);
      this.pos += len;
      this.inf.push(data.slice(), this.pos >= this.compSize);
    }
    if (this.chunks.length > 1) { this.chunks = [concat(this.chunks, this.have)]; }
  }
  async read(offset, length) {
    await this.ensure(offset + length);
    if (offset + length > this.have) throw new RangeError('The compressed entry ended early.');
    return this.chunks[0].subarray(offset, offset + length);
  }
  async close() { this.chunks = []; }
}

function concat(chunks, total) {
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

module.exports = { FileSource, BufferSource, SliceSource, InflateSource };
