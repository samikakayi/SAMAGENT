"use strict";
// Renders the real frontend/panels.js TradingView panel against a payload the
// backend itself produced, and prints the text a user would actually see.
// Driven by tests/test_new_api.py; no browser, no invented panel logic.
//
// Two scenario shapes:
//   one `/api/trading/status` payload           -> a single refresh, real timers
//   { polls: [{responses, advanceMs}, ...] }    -> several refreshes on a virtual
//     clock, each poll fired like the page's interval would, without waiting
//     the real seconds between them.
// A scripted response may be {body, status}, {hang: true} or {delayMs, body}.
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

// --- Virtual clock (multi-poll mode only) ------------------------------------
let now = 0;
let timers = [];
let nextTimer = 1;
const clock = {
  setTimeout(fn, ms = 0) { const id = nextTimer++; timers.push({ id, at: now + ms, fn }); return id; },
  clearTimeout(id) { timers = timers.filter((timer) => timer.id !== id); },
  async advance(ms) {
    const until = now + ms;
    for (;;) {
      timers.sort((a, b) => a.at - b.at);
      const due = timers[0];
      if (!due || due.at > until) break;
      timers.shift();
      now = due.at;
      due.fn();
      await flush();
    }
    now = until;
    await flush();
  },
};
async function flush(rounds = 8) {
  for (let i = 0; i < rounds; i++) await new Promise((resolve) => setImmediate(resolve));
}

// --- Scripted fetch ----------------------------------------------------------
let responses = payload.responses || null;
const requested = [];
const inFlight = {};
const maxInFlight = {};
function track(path, delta) {
  inFlight[path] = (inFlight[path] || 0) + delta;
  maxInFlight[path] = Math.max(maxInFlight[path] || 0, inFlight[path]);
}

function reply(path, script) {
  const status = script.status || 200;
  return { ok: status < 400, status, json: async () => script.body };
}

async function fetch(path, options = {}) {
  requested.push(path);
  if (!responses) {
    // The file is one `/api/trading/status` payload, which carries the same
    // observation under `tradingview`, so either endpoint answers truthfully.
    const body = path === "/api/tradingview/state" ? payload.tradingview : payload;
    return { ok: true, status: 200, json: async () => body };
  }
  const script = responses[path];
  if (!script) return { ok: false, status: 503, json: async () => ({ detail: `no stub for ${path}` }) };
  track(path, +1);
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (fn, value) => { if (settled) return; settled = true; track(path, -1); fn(value); };
    if (options.signal) {
      const abort = () => finish(reject, Object.assign(new Error("The operation was aborted."), { name: "AbortError" }));
      if (options.signal.aborted) abort(); else options.signal.addEventListener("abort", abort);
    }
    if (script.hang) return;                                   // a stalled broker never answers
    if (script.delayMs) return void clock.setTimeout(() => finish(resolve, reply(path, script)), script.delayMs);
    finish(resolve, reply(path, script));
  });
}

const context = {
  document, fetch, console, AbortController,
  setInterval: () => {},
  setTimeout: payload.polls ? clock.setTimeout : setTimeout,
  clearTimeout: payload.polls ? clock.clearTimeout : clearTimeout,
};
context.window = context;
context.globalThis = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(panelPath, "utf-8"), context);

const errors = [];
const originalError = console.error;
console.error = (...args) => { errors.push(args.map(String).join(" ")); };

function snapshot(from) {
  const rendered = {};
  for (const [id, node] of nodes) rendered[id] = node.textContent;
  return { requested: requested.slice(from), inFlight: { ...inFlight }, rendered };
}

(async () => {
  if (!payload.polls) {
    await Promise.race([
      context.window.SAMTrading.tradingView(),
      new Promise((resolve) => setTimeout(resolve, Number(process.env.PANEL_TIMEOUT_MS || 1500))),
    ]);
    const { rendered } = snapshot(0);
    console.log(JSON.stringify({ requested, rendered }));
    return;
  }
  const polls = [];
  for (const poll of payload.polls) {
    if (poll.responses) responses = poll.responses;
    const from = requested.length;
    context.window.SAMTrading.tradingView().catch((error) => errors.push(String(error)));
    await flush();
    const fired = snapshot(from);
    await clock.advance(poll.advanceMs ?? 10_000);
    const afterInterval = snapshot(from);
    polls.push({ fired, afterInterval });
  }
  console.log(JSON.stringify({ polls, maxInFlight, inFlight, errors, requested, rendered: snapshot(0).rendered }));
})().catch((error) => { originalError(error.stack || String(error)); process.exit(1); });
