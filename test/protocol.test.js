'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { validateWebviewMessage, publicModel, shapeText } = require('../src/core/protocol');
const { open } = require('./helpers');

test('valid webview messages pass and are normalized', () => {
  assert.deepStrictEqual(validateWebviewMessage({ type: 'ready', extra: 1 }, 3), { ok: true, msg: { type: 'ready' } });
  assert.deepStrictEqual(validateWebviewMessage({ type: 'select', index: 2 }, 3).msg, { type: 'select', index: 2 });
  assert.deepStrictEqual(validateWebviewMessage({ type: 'copy', index: 0, what: 'torch' }, 3).msg, { type: 'copy', index: 0, what: 'torch' });
});

test('bad webview messages are rejected', () => {
  const bad = [null, 'ready', [], {}, { type: 'eval', code: '1' }, { type: 'select' }, { type: 'select', index: 3 }, { type: 'select', index: -1 }, { type: 'select', index: 1.5 }, { type: 'select', index: '1' },
    { type: 'copy', index: 0, what: 'text' }, { type: 'copy', index: 9, what: 'shape' }, { type: 'copy', index: 0 }];
  for (const m of bad) assert.strictEqual(validateWebviewMessage(m, 3).ok, false, JSON.stringify(m));
});

test('the public model has no file handles and survives JSON', async () => {
  const m = await open('gpt2_tiny.safetensors');
  const p = publicModel(m, 'gpt2_tiny.safetensors');
  assert.strictEqual(p.type, 'model');
  assert.strictEqual(p.version, 1);
  for (const t of p.tensors) assert.ok(!('_loc' in t));
  const back = JSON.parse(JSON.stringify(p));
  assert.strictEqual(back.tensors.length, m.tensors.length);
  assert.deepStrictEqual(back.tensors[0].shape, p.tensors[0].shape);
  await m.close();
});

test('copy text is built on the host from the tensor, not from the webview', async () => {
  const m = await open('gpt2_tiny.safetensors');
  const t = m.tensors.find(x => x.name === 'wpe.weight');
  assert.strictEqual(shapeText(t, 'shape'), '(512, 256)');
  assert.strictEqual(shapeText(t, 'torch'), 'torch.Size([512, 256])');
  assert.strictEqual(shapeText(t, 'name'), 'wpe.weight');
  await m.close();
});
