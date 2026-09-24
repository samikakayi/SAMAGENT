"use strict";
// Renders the real frontend/hands-free.js status line, with the real
// frontend/i18n.js, for each /api/voice/handsfree payload in a scenario, and
// prints what a user would see. Used by tests/test_hands_free_panel.py.
//
// Scenario: { locale, states: [payload, ...] }
const fs = require("node:fs");
const vm = require("node:vm");

const [i18nPath, panelPath, scenarioPath] = process.argv.slice(2);
const { locale, states } = JSON.parse(fs.readFileSync(scenarioPath, "utf-8"));

const nodes = new Map();
const node = () => ({ textContent: "", title: "", hidden: true, disabled: false, style: {}, addEventListener() {} });
const document = {
  getElementById: (id) => nodes.get(id) || nodes.set(id, node()).get(id),
  querySelector: () => null, querySelectorAll: () => [],
  addEventListener() {},  // the page never loads; render() is called directly
};
const context = vm.createContext({
  document, console,
  setInterval: () => 0, clearInterval() {},
  fetch: () => Promise.reject(new Error("no requests in this test")),
});
context.window = context;
vm.runInContext(fs.readFileSync(i18nPath, "utf-8"), context, { filename: i18nPath });
context.SAM_I18N.locale = context.SAM_I18N.resolve(locale);
vm.runInContext(fs.readFileSync(panelPath, "utf-8"), context, { filename: panelPath });

const seen = states.map((state) => {
  context.SAMHandsFree.render(state);
  const get = (id) => nodes.get(id);
  return {
    text: get("hands-free-status").textContent,
    tooltip: get("hands-free-status").title,
    shown: !get("hands-free-bar").hidden,
    button: get("hands-free-toggle").textContent,
    dot: get("hands-free-dot").style.opacity,
  };
});
process.stdout.write(JSON.stringify(seen));
