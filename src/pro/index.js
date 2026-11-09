'use strict';
// NumPy Viewer's Pro layer on top of the shared Zovo license module (src/license, copied unchanged from
// factory/sprint/license). The module answers isPro('numpy-pro') from local caches (GitHub Sponsors unlock or an
// offline ZOVO1 license key) and owns the unlock, Get Pro and remove commands. This file adds the feature texts and
// 3 free tries per feature.
// Rules: everything that was free stays free; nothing about Pro shows on open; a Pro feature explains itself only
// after the user clicks it with no free tries left; no network except what the user starts.
const { createLicense } = require('../license');

const SKU = 'numpy-pro';
const TRIAL_USES = 3;
const TRIAL_KEY = 'tensorViewer.proTrial';

const FEATURES = {
  compare: {
    title: 'Compare two checkpoints',
    detail: 'Reads every value of two tensor files (NumPy, PyTorch, Safetensors or GGUF, also two different formats) and lists, per tensor: max and mean absolute difference, RMSE, relative L2 error, cosine similarity and how many values differ. Tensors that are in one file only, or that changed shape, are listed too. Save the report as CSV, Markdown or JSON.',
  },
  scan: {
    title: 'Scan all tensors',
    detail: 'Reads every value of every tensor in the file (no sampling) and flags NaN, Inf, all-zero and constant tensors and very large values, with exact min, max, mean, std and L2 norm per tensor. Save the report as CSV, Markdown or JSON.',
  },
};

// The shared module adds the "How to buy" steps (GitHub Sponsors, PayPal, team) under this text.
const PRICE_LINE = 'Zovo Pro is USD 19 once and unlocks the Pro features in every Zovo viewer.';
const FREE_LINE = 'Everything else in NumPy Viewer stays free: every format, stats, histograms, heatmaps and images, the values table, slices, raw arrays and Save as .npy or CSV.';

function createPro({ vscode, context, prefix = 'tensorViewer', productName = 'NumPy Viewer', license, testKey }) {
  const emitter = new vscode.EventEmitter();
  const opts = { vscode, context, sku: SKU, name: 'Zovo', commandPrefix: prefix };
  if (testKey) { opts.publicKey = testKey; opts.allowTest = true; } // gate builds only
  const lic = license || createLicense(opts);
  const setContext = () => vscode.commands.executeCommand('setContext', `${prefix}.proLicensed`, lic.isPro(SKU));
  lic.onDidChange(() => { setContext(); emitter.fire(lic.isPro(SKU)); });
  Promise.resolve(lic.ready).then(setContext, () => {});

  const trialUsed = f => Number((context.globalState.get(TRIAL_KEY) || {})[f]) || 0;
  const trialLeft = f => (FEATURES[f] ? Math.max(0, TRIAL_USES - trialUsed(f)) : 0);
  async function useTrial(f) {
    if (lic.isPro(SKU)) return;
    const m = { ...(context.globalState.get(TRIAL_KEY) || {}) };
    m[f] = trialUsed(f) + 1;
    await context.globalState.update(TRIAL_KEY, m);
    const left = trialLeft(f);
    vscode.window.setStatusBarMessage(`${productName}: free try of ${FEATURES[f].title.toLowerCase()}, ${left} left on this computer.`, 6000);
    emitter.fire(false);
  }

  /** One modal, only after the user clicked a Pro feature with no free tries left. True when it got unlocked. */
  async function explain(f) {
    const x = FEATURES[f];
    const tried = trialUsed(f) >= TRIAL_USES ? `You have used the ${TRIAL_USES} free tries of this feature on this computer.\n\n` : '';
    return lic.requirePro(x.title, `${tried}${x.detail}\n\n${PRICE_LINE}\n\n${FREE_LINE}`);
  }

  /** Licensed or a free try left: the caller picks files first and calls useTrial() when the work starts. */
  async function allowed(f) {
    if (lic.isPro(SKU)) { lic.refresh(); return true; }
    if (trialLeft(f) > 0) return true;
    return explain(f);
  }

  context.subscriptions.push(emitter);
  return {
    get licensed() { return lic.isPro(SKU); },
    ready: lic.ready,
    allowed, explain, trialLeft, useTrial,
    trials: () => Object.fromEntries(Object.keys(FEATURES).map(f => [f, trialLeft(f)])),
    handleUri: uri => lic.handleUri(uri),
    onDidChange: emitter.event,
    features: FEATURES,
    license: lic,
  };
}

module.exports = { createPro, FEATURES, SKU, TRIAL_USES, TRIAL_KEY, PRICE_LINE };
