'use strict';
// Minimal zip directory reader for .npz files (np.savez stores, np.savez_compressed deflates).
// It reads only the central directory at the end of the file, plus each local header on demand.
const { SliceSource, InflateSource } = require('./source');

const MAX_ENTRIES = 100000;

async function readCentralDirectory(source) {
  const tailLen = Math.min(source.size, 65557);
  const tail = await source.read(source.size - tailLen, tailLen);
  const dv = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
  let eocd = -1;
  for (let i = tail.length - 22; i >= 0; i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error('Not a valid .npz file: the zip directory is missing.');
  let count = dv.getUint16(eocd + 10, true);
  let cdSize = dv.getUint32(eocd + 12, true);
  let cdOffset = dv.getUint32(eocd + 16, true);
  // Zip64: the locator sits 20 bytes before the EOCD record.
  if ((count === 0xffff || cdSize === 0xffffffff || cdOffset === 0xffffffff) && eocd >= 20 && dv.getUint32(eocd - 20, true) === 0x07064b50) {
    const z64Off = Number(dv.getBigUint64(eocd - 20 + 8, true));
    const z = await source.read(z64Off, 56);
    const zv = new DataView(z.buffer, z.byteOffset, 56);
    if (zv.getUint32(0, true) !== 0x06064b50) throw new Error('Bad zip64 directory.');
    count = Number(zv.getBigUint64(32, true));
    cdSize = Number(zv.getBigUint64(40, true));
    cdOffset = Number(zv.getBigUint64(48, true));
  }
  if (count > MAX_ENTRIES) throw new Error(`The .npz file lists ${count} arrays; the limit is ${MAX_ENTRIES}.`);
  if (cdOffset + cdSize > source.size) throw new Error('Not a valid .npz file: the zip directory points past the end.');
  const cd = await source.read(cdOffset, cdSize);
  const cv = new DataView(cd.buffer, cd.byteOffset, cd.byteLength);
  const entries = [];
  let p = 0;
  for (let n = 0; n < count; n++) {
    if (p + 46 > cd.length || cv.getUint32(p, true) !== 0x02014b50) throw new Error('Bad zip directory entry.');
    const method = cv.getUint16(p + 10, true);
    let compSize = cv.getUint32(p + 20, true);
    let size = cv.getUint32(p + 24, true);
    const nameLen = cv.getUint16(p + 28, true), extraLen = cv.getUint16(p + 30, true), commentLen = cv.getUint16(p + 32, true);
    let localOffset = cv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(cd.subarray(p + 46, p + 46 + nameLen));
    // Zip64 extra field: values appear only for the fields that are 0xffffffff, in this order.
    let e = p + 46 + nameLen;
    const eEnd = e + extraLen;
    while (e + 4 <= eEnd) {
      const id = cv.getUint16(e, true), len = cv.getUint16(e + 2, true);
      if (id === 0x0001) {
        let q = e + 4;
        if (size === 0xffffffff) { size = Number(cv.getBigUint64(q, true)); q += 8; }
        if (compSize === 0xffffffff) { compSize = Number(cv.getBigUint64(q, true)); q += 8; }
        if (localOffset === 0xffffffff) { localOffset = Number(cv.getBigUint64(q, true)); q += 8; }
      }
      e += 4 + len;
    }
    entries.push({ name, method, compSize, size, localOffset });
    p += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

/** A Source over the uncompressed bytes of one entry. */
async function entrySource(source, entry) {
  const lh = await source.read(entry.localOffset, 30);
  const lv = new DataView(lh.buffer, lh.byteOffset, 30);
  if (lv.getUint32(0, true) !== 0x04034b50) throw new Error(`Bad local header for ${entry.name}.`);
  const start = entry.localOffset + 30 + lv.getUint16(26, true) + lv.getUint16(28, true);
  if (start + entry.compSize > source.size) throw new Error(`Entry ${entry.name} runs past the end of the file.`);
  if (entry.method === 0) return new SliceSource(source, start, entry.size);
  if (entry.method === 8) return new InflateSource(source, start, entry.compSize, entry.size);
  throw new Error(`Entry ${entry.name} uses zip method ${entry.method}, which is not supported.`);
}

module.exports = { readCentralDirectory, entrySource };
