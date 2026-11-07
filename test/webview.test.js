'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { buildHtml } = require('../src/html');

const root = path.join(__dirname, '..');

test('webview HTML has a strict CSP with a nonce and no remote sources', () => {
  const html = buildHtml({ cspSource: 'vscode-resource:', scriptUri: 'vscode-resource:/dist/webview.js', styleUri: 'vscode-resource:/media/webview.css', title: 'm.gguf', nonce: 'abc' });
  const csp = /Content-Security-Policy" content="([^"]+)"/.exec(html)[1];
  assert.match(csp, /default-src 'none'/);
  assert.match(csp, /script-src 'nonce-abc'/);
  assert.doesNotMatch(csp, /unsafe-inline|unsafe-eval|https?:|\*/);
  assert.match(html, /<script nonce="abc" src=/);
  assert.strictEqual((html.match(/<script/g) || []).length, 1);
});

test('file names are escaped in the HTML', () => {
  const html = buildHtml({ cspSource: 'x', scriptUri: 's', styleUri: 'c', title: '"><script>alert(1)</script>.npy' });
  assert.doesNotMatch(html, /<script>alert/);
  assert.match(html, /&quot;&gt;&lt;script&gt;/);
});

test('the webview never turns file content into HTML or code', () => {
  const src = fs.readFileSync(path.join(root, 'src/webview/main.js'), 'utf8');
  for (const bad of ['innerHTML', 'outerHTML', 'insertAdjacentHTML', 'document.write', 'eval(', 'new Function', 'setTimeout(\'', 'fetch(', 'XMLHttpRequest', 'WebSocket']) {
    assert.ok(!src.includes(bad), bad);
  }
});

test('extension code makes no network calls and never runs pickle', () => {
  const files = fs.readdirSync(path.join(root, 'src/core'), { recursive: true }).map(f => path.join(root, 'src/core', f)).filter(f => f.endsWith('.js')).concat([path.join(root, 'src/extension.js')]);
  assert.ok(files.some(f => f.endsWith(path.join('pickle', 'unpickler.js'))));
  for (const f of files) {
    const s = fs.readFileSync(f, 'utf8');
    for (const bad of ["require('http')", "require('https')", "require('net')", "require('child_process')", 'XMLHttpRequest', 'eval(', 'new Function']) assert.ok(!s.includes(bad), `${path.basename(f)}: ${bad}`);
    // The only fetch-like call is the local reader handed to @huggingface/gguf.
    if (!f.endsWith('formats.js')) assert.ok(!/\bfetch\s*\(/.test(s), `${path.basename(f)}: fetch`);
  }
});
