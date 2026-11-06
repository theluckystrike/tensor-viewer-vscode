'use strict';
// Builds the webview HTML. Kept free of the vscode module so tests can check the CSP.
const crypto = require('crypto');
const { footerHtml, FOOTER_CSS } = require('./support');

function escapeAttr(s) {
  return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/**
 * @param {{cspSource:string, scriptUri:string, styleUri:string, title:string, nonce?:string, showSupportLinks?:boolean}} p
 * showSupportLinks false hides the small Sponsor and More free viewers footer (setting tensorViewer.showSupportLinks).
 */
function buildHtml(p) {
  const n = p.nonce || crypto.randomBytes(24).toString('base64url');
  const footer = footerHtml('tensorViewer', p.showSupportLinks !== false, 'zovo-support-fixed');
  const csp = ["default-src 'none'", footer ? `style-src ${p.cspSource} 'nonce-${n}'` : `style-src ${p.cspSource}`, `script-src 'nonce-${n}'`, `img-src ${p.cspSource} data:`, `font-src ${p.cspSource}`].join('; ');
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="${escapeAttr(p.styleUri)}">
${footer ? `<style nonce="${n}">${FOOTER_CSS}</style>` : ''}
<title>${escapeAttr(p.title)}</title>
</head>
<body>
<div id="app" data-title="${escapeAttr(p.title)}"><p class="loading">Reading the header...</p></div>
${footer}
<script nonce="${n}" src="${escapeAttr(p.scriptUri)}"></script>
</body>
</html>`;
}

module.exports = { buildHtml, escapeAttr };
