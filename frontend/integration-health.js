(() => {
  "use strict";

  // Integration Health: whether SAM's outside connections will still work
  // tomorrow. Its own module for the same reason routing-panel.js is --
  // panels.js should not grow another section.
  //
  // Nothing here computes a verdict. The backend decides what is healthy,
  // what is expiring and what is overprivileged; this file only renders it.
  // It also never asks for a probe on its own: the states shown are cached,
  // and Refresh is the only thing that re-checks a provider.

  const $ = (id) => document.getElementById(id);

  const STATE_CLASS = {
    HEALTHY: "badge-success",
    NOT_CONFIGURED: "badge-subtle",
    RATE_LIMITED: "badge-subtle",
    MODEL_UNAVAILABLE: "badge-subtle",
    AUTH_ERROR: "badge-danger",
    QUOTA: "badge-danger",
    UNREACHABLE: "badge-danger",
    UNKNOWN: "badge-subtle",
  };

  const EXPIRY_LABEL = {
    OK: (facts) => `expires ${facts.expires_at}`,
    EXPIRING_SOON: (facts) => `expires ${facts.expires_at} (${facts.days_until_expiry} days)`,
    EXPIRED: (facts) => `expired ${facts.expires_at}`,
    UNKNOWN_EXPIRY: () => "no expiry published",
  };

  const PRIVILEGE_LABEL = {
    LEAST_PRIVILEGE: "least privilege",
    OVERPRIVILEGED: "broader than required",
    UNKNOWN: "scope not recorded",
  };

  function element(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = String(content);
    return node;
  }

  async function json(path) {
    const response = await fetch(path, { cache: "no-store", headers: { Accept: "application/json" } });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload?.detail?.error || payload?.detail || `Request failed (${response.status})`);
    return payload;
  }

  function renderIntegration(item) {
    const card = element("div", "level-row");
    const label = element("span", "level-label", item.integration);
    const badge = element("span", `badge ${STATE_CLASS[item.state] || "badge-subtle"}`, item.state);
    const facts = item.credential || {};
    const bits = [];
    if (facts.configured) {
      bits.push((EXPIRY_LABEL[facts.expiry] || EXPIRY_LABEL.UNKNOWN_EXPIRY)(facts));
      if (facts.privilege && facts.privilege !== "UNKNOWN") bits.push(PRIVILEGE_LABEL[facts.privilege]);
      else if (item.integration === "n8n") bits.push(PRIVILEGE_LABEL.UNKNOWN);
    } else {
      bits.push("no credential stored");
    }
    if (item.latency_ms) bits.push(`${Math.round(item.latency_ms)} ms`);
    const detail = element("span", "level-value", bits.join(" · "));
    card.append(label, badge, detail);
    return card;
  }

  function render(payload) {
    const root = $("integration-list");
    if (!root) return;
    root.replaceChildren();
    for (const item of payload.integrations || []) {
      root.appendChild(renderIntegration(item));
      for (const warning of item.warnings || []) {
        root.appendChild(element("p", "p-note", `⚠ ${item.integration}: ${warning}`));
      }
      if (item.recommended_action) {
        root.appendChild(element("p", "p-note", `→ ${item.recommended_action}`));
      }
    }
    const attention = payload.needs_attention || [];
    const summary = $("integration-summary");
    if (summary) {
      summary.textContent = attention.length
        ? `${attention.length} integration(s) need attention: ${attention.join(", ")}.`
        : "Every configured integration answered and nothing needs attention.";
    }
    const note = $("integration-note");
    if (note) note.textContent = `${payload.note || ""} Last checked ${payload.checked_at || "—"}.`;
  }

  async function refresh(force) {
    const root = $("integration-list");
    if (root) root.replaceChildren(element("p", "p-note", force ? "Re-checking…" : "Loading…"));
    try {
      render(await json(`/api/integrations${force ? "?refresh=true" : ""}`));
    } catch (error) {
      if (root) root.replaceChildren(element("p", "p-note", error.message));
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("integration-refresh-btn")?.addEventListener("click", () => refresh(true));
    refresh(false);
  });

  window.SAMIntegrations = { refresh, render, renderIntegration };
})();
