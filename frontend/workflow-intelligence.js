(() => {
  "use strict";

  // Workflow Intelligence: find an automation, read what it actually does,
  // then decide. Its own module so panels.js does not grow another section,
  // following the split routing-panel.js established.
  //
  // Every judgement shown here is the backend's. This file renders risk, it
  // does not compute it: a classifier in the page could be talked out of its
  // answer by the workflow it is describing.

  const $ = (id) => document.getElementById(id);
  const RISK_CLASS = { LOW: "badge-success", MEDIUM: "badge-subtle", HIGH: "badge-danger", CRITICAL: "badge-danger" };

  let prepared = null;

  async function json(path, options = {}) {
    const response = await fetch(path, {
      cache: "no-store",
      headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}) },
      ...options,
    });
    let payload = null;
    try { payload = await response.json(); } catch { payload = {}; }
    if (!response.ok) throw new Error(payload?.detail?.error || payload?.detail || `Request failed (${response.status})`);
    return payload;
  }

  const text = (id, value) => { const n = $(id); if (n) n.textContent = value ?? "—"; };

  function badge(id, value, cls) {
    const node = $(id);
    if (!node) return;
    node.textContent = String(value || "UNKNOWN");
    node.className = `badge ${cls || "badge-subtle"}`;
  }

  function element(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = String(content);
    return node;
  }

  // -- status ---------------------------------------------------------------
  async function refreshStatus() {
    try {
      const status = await json("/api/workflows/status");
      const library = status.library || {};
      badge("wf-library-state", (library.state || "").replace("LIBRARY_", ""),
        library.state === "LIBRARY_AVAILABLE" ? "badge-success" : "badge-subtle");
      text("wf-library-detail", library.indexed_workflows
        ? `${library.indexed_workflows} workflows indexed from ${library.source_repository}.`
        : "The workflow library has not been read yet.");
      const n8n = status.n8n || {};
      badge("wf-n8n-state", n8n.status, n8n.status === "CONNECTED" ? "badge-success"
        : n8n.status === "NOT_CONFIGURED" ? "badge-subtle" : "badge-danger");
      text("wf-n8n-detail", n8n.detail || "");
    } catch (error) {
      text("wf-library-detail", error.message);
    }
  }

  // -- search ---------------------------------------------------------------
  async function search() {
    const query = $("wf-query")?.value?.trim() || "";
    const params = new URLSearchParams({ query, limit: "8" });
    const trigger = $("wf-trigger")?.value || "";
    if (trigger) params.set("trigger", trigger);
    const root = $("wf-results");
    if (root) root.replaceChildren(element("p", "p-note", "Searching…"));
    try {
      const payload = await json(`/api/workflows/search?${params}`);
      renderResults(payload);
    } catch (error) {
      if (root) root.replaceChildren(element("p", "p-note", error.message));
    }
  }

  function renderResults(payload) {
    const root = $("wf-results");
    if (!root) return;
    root.replaceChildren();
    const results = payload.results || [];
    if (!results.length) {
      root.appendChild(element("p", "p-note", "No workflow in the library matched that."));
      return;
    }
    if (payload.library_state === "LIBRARY_STALE_CACHE") {
      // Said, not hidden: these results are remembered, not current.
      root.appendChild(element("p", "p-note", "Showing a cached index; the library could not be reached."));
    }
    for (const item of results) {
      const card = element("div", "level-row");
      const label = element("span", "level-label",
        `${item.title} · ${item.trigger} · ${item.complexity}`);
      const why = element("span", "level-value", item.match_reason || item.category || "");
      const inspect = element("button", "secondary-button", "Inspect");
      inspect.type = "button";
      inspect.addEventListener("click", () => inspectWorkflow(item.workflow_id));
      card.append(label, why, inspect);
      root.appendChild(card);
    }
  }

  // -- inspect --------------------------------------------------------------
  async function inspectWorkflow(workflowId) {
    const root = $("wf-inspection");
    if (root) root.replaceChildren(element("p", "p-note", "Reading the workflow…"));
    try {
      const payload = await json(`/api/workflows/${encodeURIComponent(workflowId)}/inspect`);
      renderInspection(workflowId, payload);
    } catch (error) {
      if (root) root.replaceChildren(element("p", "p-note", error.message));
    }
  }

  function renderInspection(workflowId, payload) {
    const root = $("wf-inspection");
    if (!root) return;
    const report = payload.inspection || {};
    const risk = report.risk || {};
    root.replaceChildren();
    root.appendChild(element("h4", null, payload.name || workflowId));

    const riskRow = element("div", "level-row");
    riskRow.append(element("span", "level-label", "Risk"),
      Object.assign(element("span", `badge ${RISK_CLASS[risk.level] || "badge-subtle"}`, risk.level || "?"), {}));
    root.appendChild(riskRow);

    const rows = [
      ["Nodes", report.node_count],
      ["Integrations", (report.services || []).join(", ") || "none"],
      ["Triggers", (report.triggers || []).join(", ") || "none"],
      ["Credentials required", (report.credentials || []).map((c) => c.credential_type).join(", ") || "none"],
      ["External destinations", (report.http_destinations || []).join(", ") || "none"],
      ["Risk flags", (risk.flags || []).join(", ")],
      ["Source", payload.provenance?.source_path || ""],
    ];
    for (const [label, value] of rows) {
      const row = element("div", "level-row");
      row.append(element("span", "level-label", label), element("span", "level-value", value || "—"));
      root.appendChild(row);
    }
    for (const reason of (risk.reasons || []).slice(0, 6)) {
      root.appendChild(element("p", "p-note", reason));
    }
    // Code is shown, never run, and never hidden from whoever approves it.
    for (const preview of report.code_previews || []) {
      root.appendChild(element("p", "p-note", `${preview.node} (${preview.language}) — not executed:`));
      root.appendChild(element("pre", "code-preview", preview.code));
    }
    if (risk.incomplete) {
      root.appendChild(element("p", "p-note",
        "A referenced subworkflow could not be read, so this risk is a floor, not a verdict."));
    }
    const prepare = element("button", "primary-button", "Prepare with SAM");
    prepare.type = "button";
    prepare.addEventListener("click", () => prepareWorkflow(workflowId));
    root.appendChild(prepare);
  }

  // -- prepare --------------------------------------------------------------
  async function prepareWorkflow(workflowId) {
    const root = $("wf-prepared");
    if (root) root.replaceChildren(element("p", "p-note", "Preparing…"));
    try {
      prepared = await json("/api/workflows/prepare", {
        method: "POST",
        body: JSON.stringify({ workflow_id: workflowId, name: $("wf-name")?.value?.trim() || undefined }),
      });
      renderPrepared(prepared);
    } catch (error) {
      if (root) root.replaceChildren(element("p", "p-note", error.message));
    }
  }

  function renderPrepared(artifact) {
    const root = $("wf-prepared");
    if (!root) return;
    root.replaceChildren();
    root.appendChild(element("h4", null, artifact.name));

    const rows = [
      ["Target instance", artifact.target_instance],
      ["Workflow hash", artifact.sha256],
      ["Risk", artifact.inspection?.risk?.level],
      ["Valid", artifact.validation?.ok ? "yes" : "no"],
    ];
    for (const [label, value] of rows) {
      const row = element("div", "level-row");
      row.append(element("span", "level-label", label), element("span", "level-value", value ?? "—"));
      root.appendChild(row);
    }
    for (const line of artifact.diff?.summary || []) {
      root.appendChild(element("p", "p-note", `• ${line}`));
    }
    for (const error of artifact.validation?.errors || []) {
      root.appendChild(element("p", "p-note", `Blocked: ${error}`));
    }
    for (const item of artifact.unresolved_credentials || []) {
      root.appendChild(element("p", "p-note",
        `Needs a ${item.credential_type} credential that already exists in your n8n.`));
    }
    for (const note of artifact.notes || []) {
      root.appendChild(element("p", "p-note", note));
    }

    const importButton = element("button", "primary-button",
      artifact.importable ? "Import to n8n (inactive)" : "Cannot import yet");
    importButton.type = "button";
    importButton.disabled = !artifact.importable;
    importButton.addEventListener("click", () => importWorkflow(artifact.sha256));
    root.appendChild(importButton);
  }

  async function importWorkflow(sha256) {
    const root = $("wf-prepared");
    try {
      const result = await json("/api/workflows/import", {
        method: "POST", body: JSON.stringify({ workflow_sha256: sha256 }),
      });
      if (result.approval_required) {
        root?.appendChild(element("p", "p-note",
          `Approval required (${result.risk_level}): ${result.reason} Target: ${result.target_instance}`));
        return;
      }
      root?.appendChild(element("p", "p-note",
        `Imported as ${result.workflow_id}, inactive. Activate it separately when you are ready.`));
      await refreshRuns();
    } catch (error) {
      root?.appendChild(element("p", "p-note", error.message));
    }
  }

  // -- create from goal -----------------------------------------------------
  // One request carries the whole read half of the loop: search, rank,
  // inspect, adapt or generate, validate, risk, hash. What comes back is a
  // plan, never a change -- importing is still the separate approved step it
  // always was, and this panel cannot skip it.
  async function planGoal() {
    const goal = $("wf-goal")?.value?.trim() || "";
    const root = $("wf-goal-plan");
    if (!goal) {
      root?.replaceChildren(element("p", "p-note", "Describe what you want automated first."));
      return;
    }
    if (root) root.replaceChildren(element("p", "p-note", "Reading the library and planning…"));
    try {
      const plan = await json("/api/workflows/goal", {
        method: "POST",
        body: JSON.stringify({ goal, name: $("wf-name")?.value?.trim() || undefined }),
      });
      renderPlan(plan);
    } catch (error) {
      if (root) root.replaceChildren(element("p", "p-note", error.message));
    }
  }

  // "Prepared" until n8n has it, "Imported — Inactive" once it does, and
  // "Active" only after a separate activation. Nothing is ever called
  // "Running" here: only an execution record could justify that word.
  function renderPlan(plan) {
    const root = $("wf-goal-plan");
    if (!root) return;
    root.replaceChildren();

    if (plan.next_action === "clarify") {
      root.appendChild(element("p", "p-note", plan.blockers?.[0] || "That goal needs more detail."));
      return;
    }

    const prepared = plan.prepared || {};
    const risk = prepared.inspection?.risk || {};
    root.appendChild(element("h4", null,
      plan.origin === "library" ? "Best match from the library" : "Written by SAM for this goal"));
    root.appendChild(element("p", "p-note", plan.selection_reason || ""));

    const rows = [
      ["Status", "Prepared"],
      ["Workflow", prepared.name],
      ["Risk", risk.level],
      ["Nodes", prepared.inspection?.node_count],
      ["Credentials needed",
        (prepared.credentials || []).map((c) => c.credential_type).join(", ") || "none"],
      ["Valid", prepared.validation?.ok ? "yes" : "no"],
      ["Workflow hash", prepared.sha256],
      ["Target instance", plan.target_instance],
    ];
    for (const [label, value] of rows) {
      const row = element("div", "level-row");
      const cell = label === "Risk"
        ? element("span", `badge ${RISK_CLASS[value] || "badge-subtle"}`, value || "?")
        : element("span", "level-value", value ?? "—");
      row.append(element("span", "level-label", label), cell);
      root.appendChild(row);
    }

    for (const line of plan.adaptations || []) root.appendChild(element("p", "p-note", line));
    for (const line of prepared.diff?.summary || []) root.appendChild(element("p", "p-note", `• ${line}`));
    for (const line of plan.blockers || []) root.appendChild(element("p", "p-note", `Blocked: ${line}`));
    for (const note of prepared.notes || []) root.appendChild(element("p", "p-note", note));

    // Why "Activate" is not offered yet, said before it is missed.
    if (plan.activation?.reason) {
      root.appendChild(element("p", "p-note", plan.activation.reason));
    }

    if (plan.candidates?.length > 1) {
      root.appendChild(element("p", "p-note", "Also considered: " + plan.candidates.slice(1, 4)
        .map((c) => `${c.title} (${c.score})`).join(", ")));
    }

    // Two deliberate clicks, because the backend wants two calls: the first
    // raises a real approval record and answers with what is about to happen,
    // the second claims that exact approval. The button never sends both at
    // once -- the point of the gate is that somebody read the risk and the
    // target instance in between.
    let approval = null;
    const importButton = element("button", "primary-button",
      plan.importable ? "Import to n8n (inactive)" : "Cannot import yet");
    importButton.type = "button";
    importButton.disabled = !plan.importable;
    importButton.addEventListener("click", async () => {
      importButton.disabled = true;
      try {
        const result = await json("/api/workflows/import", {
          method: "POST",
          body: JSON.stringify(approval
            ? { workflow_sha256: prepared.sha256, approval_id: approval.approval_id }
            : { workflow_sha256: prepared.sha256 }),
        });
        if (result.approval_required) {
          approval = result;
          root.appendChild(element("p", "p-note",
            `Approval required (${result.risk_level}): ${result.reason} ` +
            `This will create ${result.name} in ${result.target_instance}, inactive.`));
          importButton.textContent = "Confirm import (inactive)";
          importButton.disabled = false;
          return;
        }
        root.appendChild(element("p", "p-note",
          `Imported as ${result.workflow_id} — Inactive. Activation is a separate decision.`));
        await refreshRuns();
      } catch (error) {
        root.appendChild(element("p", "p-note", error.message));
        importButton.disabled = false;
      }
    });
    root.appendChild(importButton);
  }

  // -- runs -----------------------------------------------------------------
  async function refreshRuns() {
    const root = $("wf-runs");
    if (!root) return;
    try {
      const payload = await json("/api/workflows/runs?limit=5");
      root.replaceChildren();
      const runs = payload.executions || [];
      if (!runs.length) {
        root.appendChild(element("p", "p-note", "No executions recorded in the configured n8n instance."));
      }
      for (const run of runs) {
        const row = element("div", "level-row");
        row.append(
          element("span", "level-label", `${run.workflow_id} · ${run.execution_id}`),
          element("span", "level-value",
            `${run.status}${run.failed_node ? ` · failed at ${run.failed_node}` : ""}`));
        root.appendChild(row);
      }
      if (payload.note) root.appendChild(element("p", "p-note", payload.note));
    } catch (error) {
      root.replaceChildren(element("p", "p-note", error.message));
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("wf-goal-btn")?.addEventListener("click", planGoal);
    $("wf-goal")?.addEventListener("keydown", (event) => { if (event.key === "Enter") planGoal(); });
    $("wf-search-btn")?.addEventListener("click", search);
    $("wf-query")?.addEventListener("keydown", (event) => { if (event.key === "Enter") search(); });
    $("wf-refresh-btn")?.addEventListener("click", () => { refreshStatus(); refreshRuns(); });
    refreshStatus();
  });

  window.SAMWorkflows = { search, inspectWorkflow, prepareWorkflow, planGoal, refreshStatus,
    refreshRuns, prepared: () => prepared, renderResults, renderInspection, renderPrepared,
    renderPlan };
})();
