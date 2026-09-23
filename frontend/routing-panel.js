(() => {
  "use strict";

  // The routing profile and its free-tier candidate list. Kept out of
  // panels.js so the settings area does not grow another few hundred lines,
  // and so this feature's UI can be read in one sitting.
  //
  // The backend decides what "free" means. This panel only shows the verdict
  // it is given: which candidates were accepted, which were refused as paid,
  // and whether a paid fallback is currently permitted.

  const $ = (id) => document.getElementById(id);

  const EXPLAINERS = {
    PREMIUM: "Premium uses the model chain you already configured. Adding Groq or Gemini does not change it.",
    BALANCED: "Balanced tries your free-tier candidates first, then falls back to the configured chain.",
    FREE: "Free uses free-tier-eligible candidates only. If none are available SAM stops rather than using a paid model.",
  };

  let candidates = [];

  async function json(path, options = {}) {
    const response = await fetch(path, {
      cache: "no-store",
      headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}) },
      ...options,
    });
    let payload = null;
    try { payload = await response.json(); } catch { payload = {}; }
    if (!response.ok) throw new Error(payload?.detail || payload?.error || `Request failed (${response.status})`);
    return payload;
  }

  function note(id, message) {
    const node = $(id);
    if (node) node.textContent = message;
  }

  function badge(id, value) {
    const node = $(id);
    if (!node) return;
    node.textContent = String(value || "UNKNOWN").toUpperCase();
    node.className = "badge badge-subtle";
    if (value === "ALLOWED") node.classList.add("badge-success");
    else if (value === "REFUSED") node.classList.add("badge-danger");
  }

  function renderCandidates() {
    const root = $("routing-candidates");
    if (!root) return;
    root.replaceChildren();
    if (!candidates.length) {
      const empty = document.createElement("p");
      empty.className = "p-note";
      empty.textContent = "No free-tier candidates configured. Free mode has nothing to call until one is added.";
      root.appendChild(empty);
      return;
    }
    candidates.forEach((candidate, index) => {
      const row = document.createElement("div");
      row.className = "level-row";
      const label = document.createElement("span");
      label.className = "level-label";
      // Eligibility is the backend's word, not a guess made here.
      const mark = candidate.eligibility === "free" ? " · free-tier"
        : candidate.eligibility === "paid" ? " · paid" : " · unverified";
      label.textContent = `${index + 1}. ${candidate.reference}${mark}`;
      const actions = document.createElement("span");
      actions.className = "level-value";
      actions.appendChild(button("↑", () => move(index, -1)));
      actions.appendChild(button("↓", () => move(index, 1)));
      actions.appendChild(button("Remove", () => {
        candidates.splice(index, 1);
        renderCandidates();
      }));
      row.append(label, actions);
      root.appendChild(row);
    });
  }

  function button(text, onClick) {
    const node = document.createElement("button");
    node.type = "button";
    node.className = "secondary-button";
    node.textContent = text;
    node.addEventListener("click", onClick);
    return node;
  }

  function move(index, delta) {
    const target = index + delta;
    if (target < 0 || target >= candidates.length) return;
    [candidates[index], candidates[target]] = [candidates[target], candidates[index]];
    renderCandidates();
  }

  function render(routing) {
    if (!routing) return;
    const select = $("routing-profile-select");
    if (select) select.value = routing.routing_profile || "PREMIUM";
    note("routing-explainer", EXPLAINERS[routing.routing_profile] || EXPLAINERS.PREMIUM);
    badge("routing-paid-fallback", routing.paid_fallback_allowed ? "ALLOWED" : "REFUSED");
    candidates = (routing.free_candidates || []).slice();
    // A refused candidate is stated, not quietly dropped: the operator asked
    // for it and is entitled to know why Free will not use it.
    const refused = routing.refused_candidates || [];
    note("routing-refused", refused.length
      ? `Refused: ${refused.map((item) => `${item.reference} (${item.reason})`).join("; ")}`
      : "");
    renderCandidates();
  }

  function describeEffective(routing, resolution) {
    if (!routing) return "—";
    if (routing.routing_profile !== "FREE") {
      const chosen = resolution?.provider && resolution?.model
        ? `${resolution.provider} / ${resolution.model}` : "configured chain";
      return `${routing.routing_profile} · ${chosen}`;
    }
    const first = (routing.free_candidates || [])[0];
    return first ? `FREE · ${first.reference}` : "FREE · unavailable — no free-tier candidate is configured";
  }

  async function refresh() {
    try {
      const status = await json("/api/providers/status");
      render(status.routing);
      let resolution = null;
      try {
        resolution = (await json("/api/providers/resolution")).resolution;
      } catch {
        resolution = null;  // resolution is a nicety here; routing is the point
      }
      note("routing-effective", describeEffective(status.routing, resolution));
      return status.routing;
    } catch (error) {
      note("routing-explainer", error.message);
      return null;
    }
  }

  async function save() {
    const profile = $("routing-profile-select")?.value || "PREMIUM";
    try {
      await json("/api/settings", {
        method: "PUT",
        body: JSON.stringify({
          routing_profile: profile,
          free_candidates: candidates.map((item) => item.reference),
        }),
      });
      note("routing-explainer", `${EXPLAINERS[profile]} Saved.`);
      await refresh();
    } catch (error) {
      note("routing-explainer", `Not saved: ${error.message}`);
    }
  }

  function addCandidate() {
    const input = $("routing-candidate-input");
    const reference = (input?.value || "").trim();
    if (!reference.includes("/")) {
      note("routing-explainer", "A candidate looks like provider/model, for example groq/llama-3.3-70b-versatile.");
      return;
    }
    const [provider, ...rest] = reference.split("/");
    candidates.push({
      provider: provider.toLowerCase(),
      model: rest.join("/"),
      reference: `${provider.toLowerCase()}/${rest.join("/")}`,
      eligibility: "unverified",  // the backend replaces this on the next refresh
    });
    if (input) input.value = "";
    renderCandidates();
  }

  async function saveFreeKeys() {
    const pairs = [["groq-key-input", "groq_api_key"], ["gemini-key-input", "gemini_api_key"]];
    for (const [id, name] of pairs) {
      const input = $(id);
      const value = (input?.value || "").trim();
      if (!value) continue;
      try {
        await json("/api/providers/credentials", { method: "POST", body: JSON.stringify({ name, value }) });
        if (input) input.value = "";  // the key does not linger in the page
      } catch (error) {
        note("routing-explainer", `${name}: ${error.message}`);
        return;
      }
    }
    await refresh();
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("routing-refresh-btn")?.addEventListener("click", refresh);
    $("routing-save-btn")?.addEventListener("click", save);
    $("routing-candidate-add")?.addEventListener("click", addCandidate);
    $("free-keys-save-btn")?.addEventListener("click", saveFreeKeys);
    $("routing-profile-select")?.addEventListener("change", (event) => {
      note("routing-explainer", EXPLAINERS[event.target.value] || EXPLAINERS.PREMIUM);
    });
    refresh();
  });

  window.SAMRouting = { refresh, save, addCandidate, candidates: () => candidates, render };
})();
