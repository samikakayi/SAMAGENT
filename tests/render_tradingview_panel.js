"use strict";
// Drives the real frontend/panels.js TradingView card the way the page does --
// SAMTrading entry points fired per poll on a virtual clock -- against scripted
// endpoint replies, and prints what a user would see. Used by tests/test_new_api.py.
//
// Scenario: { polls: [{ responses, fire, advanceMs }, ...] }
//   responses  path -> {body, status} | {hang: true} | {delayMs, body}; kept
//              from the previous poll when omitted
//   fire       SAMTrading functions to run this poll (default ["tradingView"])
//   advanceMs  virtual time to let pass afterwards (default 10s, the page's tick)
const fs = require("node:fs");
const vm = require("node:vm");

const [panelPath, scenarioPath] = process.argv.slice(2);
const { polls } = JSON.parse(fs.readFileSync(scenarioPath, "utf-8"));

const nodes = new Map();
const node = () => ({ textContent: "", className: "", classList: { add() {} } });
const document = {
  getElementById: (id) => nodes.get(id) || nodes.set(id, node()).get(id),
  querySelector: () => null, querySelectorAll: () => [], createElement: node,
  addEventListener() {},  // the page never loads; entry points are fired directly
};

let now = 0, timers = [], nextTimer = 1;
const setTimer = (fn, ms = 0) => { timers.push({ id: nextTimer, at: now + ms, fn }); return nextTimer++; };
const clearTimer = (id) => { timers = timers.filter((timer) => timer.id !== id); };
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(setImmediate); };
async function advance(ms) {
  const until = now + ms;
  for (timers.sort((a, b) => a.at - b.at); timers.length && timers[0].at <= until; timers.sort((a, b) => a.at - b.at)) {
    const due = timers.shift();
    now = due.at;
    due.fn();
    await flush();
  }
  now = until;
  await flush();
}

let responses = {};
const requested = [], inFlight = {}, maxInFlight = {}, errors = [];
function track(path, delta) {
  inFlight[path] = (inFlight[path] || 0) + delta;
  maxInFlight[path] = Math.max(maxInFlight[path] || 0, inFlight[path]);
}
function fetch(path, options = {}) {
  requested.push(path);
  const script = responses[path];
  if (!script) return Promise.resolve({ ok: false, status: 503, json: async () => ({ detail: `no stub for ${path}` }) });
  track(path, +1);
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (fn, value) => { if (!settled) { settled = true; track(path, -1); fn(value); } };
    options.signal?.addEventListener("abort", () => finish(reject, Object.assign(new Error("aborted"), { name: "AbortError" })));
    const status = script.status || 200;
    const reply = { ok: status < 400, status, json: async () => script.body };
    if (script.hang) return;  // a stalled broker never answers
    if (script.delayMs) setTimer(() => finish(resolve, reply), script.delayMs);
    else finish(resolve, reply);
  });
}

const context = {
  document, fetch, AbortController, setTimeout: setTimer, clearTimeout: clearTimer, setInterval() {},
  console: { ...console, error: (...args) => errors.push(args.join(" ")) },
};
context.window = context.globalThis = context;
vm.runInContext(fs.readFileSync(panelPath, "utf-8"), vm.createContext(context));

const rendered = () => Object.fromEntries([...nodes].map(([id, n]) => [id, n.textContent]));
(async () => {
  const seen = [];
  for (const poll of polls) {
    responses = poll.responses || responses;
    const from = requested.length;
    for (const entry of poll.fire || ["tradingView"]) context.window.SAMTrading[entry]().catch((error) => errors.push(String(error)));
    await advance(poll.advanceMs ?? 10_000);
    seen.push({ requested: requested.slice(from), rendered: rendered() });
  }
  console.log(JSON.stringify({ polls: seen, requested, maxInFlight, inFlight, errors, rendered: rendered() }));
})();
