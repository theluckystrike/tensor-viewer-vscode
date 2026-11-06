// Captures each harness page through CDP on an agent-chrome instance.
// Usage: node scripts/screenshot/shoot.js <port> <harnessDir> <outDir>
const fs = require('fs');
const [port, dir, out] = process.argv.slice(2);
(async () => {
  for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.html'))) {
    const t = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' })).json();
    const ws = new WebSocket(t.webSocketDebuggerUrl);
    await new Promise(r => ws.addEventListener('open', r, { once: true }));
    let id = 0; const waits = new Map();
    ws.addEventListener('message', e => { const m = JSON.parse(e.data); if (m.id && waits.has(m.id)) { waits.get(m.id)(m); waits.delete(m.id); } });
    const send = (method, params = {}) => new Promise(r => { const i = ++id; waits.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
    await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 760, deviceScaleFactor: 2, mobile: false });
    await send('Page.enable');
    await send('Page.navigate', { url: 'file://' + dir + '/' + f });
    await new Promise(r => setTimeout(r, 1500));
    const r = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(`${out}/${f.replace('.html', '.png')}`, Buffer.from(r.result.data, 'base64'));
    await send('Target.closeTarget', { targetId: t.id }).catch(() => {});
    ws.close();
  }
  console.log('ok', fs.readdirSync(out));
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
