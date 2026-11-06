// Builds static pages that run the real webview bundle with real host messages for fixture files.
// Usage: node scripts/screenshot/harness.js <outDir>
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..', '..');
const { FileSource } = require(root + '/src/core/source');
const { openTensorFile } = require(root + '/src/core/formats');
const { sampleTensor } = require(root + '/src/core/sample');
const { publicModel } = require(root + '/src/core/protocol');
const out = process.argv[2];
const shots = [
  { name: 'gguf', file: 'tiny-llama-q8.gguf', select: 'blk.2.attn_q.weight', filter: '' },
  { name: 'safetensors', file: 'gpt2_tiny.safetensors', select: 'wpe.weight' },
  { name: 'npy-image', file: 'image_96x128x3_u1.npy', select: 'array' },
  { name: 'npz-signal', file: 'bundle_compressed.npz', select: 'signal' },
  { name: 'gguf-metadata', file: 'tiny-llama-q8.gguf', select: 'token_embd.weight', tab: 'metadata' },
];
const theme = `:root{--vscode-font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;--vscode-font-size:13px;--vscode-editor-font-family:Menlo,Monaco,monospace;
--vscode-foreground:#cccccc;--vscode-descriptionForeground:#9d9d9d;--vscode-editor-background:#1f1f1f;--vscode-panel-border:#2b2b2b;--vscode-input-background:#313131;--vscode-input-foreground:#cccccc;--vscode-input-border:#3c3c3c;
--vscode-badge-background:#616161;--vscode-badge-foreground:#f8f8f8;--vscode-button-secondaryBackground:#313131;--vscode-button-secondaryForeground:#ccc;--vscode-focusBorder:#0078d4;
--vscode-list-hoverBackground:#2a2d2e;--vscode-list-activeSelectionBackground:#04395e;--vscode-list-activeSelectionForeground:#fff;}`;
(async () => {
  for (const s of shots) {
    const src = await FileSource.open(path.join(root, 'test/fixtures', s.file));
    const t0 = Date.now();
    const model = await openTensorFile(src, s.file);
    const openMs = Date.now() - t0;
    const idx = model.tensors.findIndex(t => t.name === s.select);
    const sample = await sampleTensor(model.tensors[idx]);
    const first = model.tensors.findIndex(t => t.readable);
    const firstSample = first === idx ? sample : await sampleTensor(model.tensors[first]);
    await src.close();
    const D = { model: { ...publicModel(model, s.file), openMs }, idx, first, sample, firstSample, tab: s.tab || null };
    const script = `window.__D = ${JSON.stringify(D)};
window.acquireVsCodeApi = () => ({ postMessage(m) {
  const D = window.__D, send = x => setTimeout(() => window.dispatchEvent(new MessageEvent('message', { data: x })), 0);
  if (m.type === 'ready') { send(D.model); setTimeout(() => { if (D.idx !== D.first) document.querySelector('tr[data-i="' + D.idx + '"]').click(); if (D.tab) document.querySelectorAll('.tab')[1].click(); }, 30); }
  if (m.type === 'select') send({ type: 'sample', index: m.index, result: m.index === D.idx ? D.sample : D.firstSample });
}});`;
    const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><style>${theme}</style><link rel="stylesheet" href="file://${root}/media/webview.css"></head>
<body class="vscode-dark"><div id="app"></div><script>${script}</script><script src="file://${root}/dist/webview.js"></script></body></html>`;
    fs.writeFileSync(path.join(out, s.name + '.html'), html);
    console.log(s.name, 'open', openMs, 'ms, sample', sample.ms, 'ms');
  }
})();
