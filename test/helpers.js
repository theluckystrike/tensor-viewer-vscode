'use strict';
const path = require('path');
const fs = require('fs');
const { FileSource } = require('../src/core/source');
const { openTensorFile } = require('../src/core/formats');
const FIX = path.join(__dirname, 'fixtures');
const expected = JSON.parse(fs.readFileSync(path.join(FIX, 'expected.json'), 'utf8'));
async function open(name) {
  const src = await FileSource.open(path.join(FIX, name));
  const model = await openTensorFile(src, name);
  model.close = () => src.close();
  return model;
}
function close(a, b, tol = 1e-6) {
  if (a === null || b === null) return a === b;
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}
module.exports = { FIX, expected, open, close };
