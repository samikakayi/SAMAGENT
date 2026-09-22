"use strict";
// Renders the real frontend/panels.js TradingView panel against a payload the
// backend itself produced, and prints the text a user would actually see.
// Driven by tests/test_new_api.py; no browser, no invented panel logic.
const fs = require("node:fs");
const vm = require("node:vm");

const [panelPath, payloadPath] = process.argv.slice(2);
const payload = JSON.parse(fs.readFileSync(payloadPath, "utf-8"));

const nodes = new Map();
function makeNode() {
  const classes = [];
  return { textContent: "", className: "", classList: { add: (name) => classes.push(name), names: classes } };
}

const document = {
  getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, makeNode());
    return nodes.get(id);
  },
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: () => makeNode(),
  addEventListener: () => {},  // the page never loads; the refresh is driven directly
};

const requested = [];
async function fetch(path) {
  requested.push(path);
  // A scenario file may script each endpoint separately, including failures.
  if (payload.responses) {
    const reply = payload.responses[path];
    if (!reply) return { ok: false, status: 503, json: async () => ({ detail: `no stub for ${path}` }) };
    if (reply.hang) return new Promise(() => {});  // a stalled broker never answers
    const status = reply.status || 200;
    return { ok: status < 400, status, json: async () => reply.body };
  }
  // Otherwise the file is one `/api/trading/status` payload, which carries the
  // same observation under `tradingview`, so either endpoint answers truthfully.
  const body = path === "/api/tradingview/state" ? payload.tradingview : payload;
  return { ok: true, status: 200, json: async () => body };
}

const context = { document, fetch, console, setInterval: () => {}, setTimeout };
context.window = context;
context.globalThis = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(panelPath, "utf-8"), context);

(async () => {
  await Promise.race([
    context.window.SAMTrading.tradingView(),
    new Promise((resolve) => setTimeout(resolve, Number(process.env.PANEL_TIMEOUT_MS || 1500))),
  ]);
  const rendered = {};
  for (const [id, node] of nodes) rendered[id] = node.textContent;
  console.log(JSON.stringify({ requested, rendered }));
})().catch((error) => { console.error(error.stack || String(error)); process.exit(1); });
