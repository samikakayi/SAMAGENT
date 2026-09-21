/*
 * Agent workspace: the live view of an autonomous run.
 *
 * Self-contained on purpose. It injects its own tab, panel and styles into the
 * existing context pane and owns its own websocket, so it adds the autonomous
 * surface without reaching into app.js's state or risking its behaviour.
 *
 * What the user sees is deliberately staged: state first (what is happening
 * now), then the plan (where this is going), then the timeline (what was
 * actually done). Technical detail is available but folded away, because an
 * activity feed that shows everything at once shows nothing.
 */
(function () {
  "use strict";

  const POLL_INTERVAL_MS = 2500;
  const MAX_RENDERED_EVENTS = 300;

  // The backend owns the state names; these derive presentation from them
  // rather than mirroring the enum, so a new state needs no change here.
  const WARN_STATES = new Set(["FIXING", "RETRYING", "WAITING_FOR_APPROVAL"]);
  const BAD_STATES = new Set(["FAILED", "CANCELLED"]);
  const stateTone = (s) =>
    s === "IDLE" ? "idle" : s === "COMPLETED" ? "ok" : BAD_STATES.has(s) ? "bad" : WARN_STATES.has(s) ? "warn" : "busy";
  const stateLabel = (s) => {
    const words = String(s || "IDLE").toLowerCase().replace(/_/g, " ");
    return words.charAt(0).toUpperCase() + words.slice(1);
  };

  const EVENT_META = {
    state: { icon: "●", tone: "muted", label: "State" },
    thought: { icon: "○", tone: "muted", label: "Thinking" },
    plan: { icon: "≡", tone: "accent", label: "Plan" },
    tool: { icon: "▸", tone: "accent", label: "Tool" },
    result: { icon: "✓", tone: "ok", label: "Result" },
    success: { icon: "✓", tone: "ok", label: "Done" },
    error: { icon: "✕", tone: "bad", label: "Error" },
    fix: { icon: "↻", tone: "warn", label: "Self-correction" },
    fallback: { icon: "⇄", tone: "warn", label: "Model fallback" },
    approval: { icon: "⚠", tone: "warn", label: "Approval needed" },
  };

  const state = {
    taskId: null,
    task: null,
    socket: null,
    pollTimer: null,
    reconnectDelay: 1000,
    pendingApproval: null,
  };

  const escapeHtml = (value) =>
    String(value ?? "").replace(/[&<>"']/g, (character) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]
    ));

  const relativeTime = (seconds) => {
    if (!seconds) return "";
    const delta = Math.max(0, Date.now() / 1000 - seconds);
    if (delta < 60) return `${Math.floor(delta)}s ago`;
    if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
    return `${Math.floor(delta / 3600)}h ago`;
  };

  function injectStyles() {
    if (document.getElementById("sam-agent-styles")) return;
    const style = document.createElement("style");
    style.id = "sam-agent-styles";
    style.textContent = `
      .agent-panel { display: flex; flex-direction: column; gap: 14px; padding: 4px 0 18px; }
      .agent-compose { display: flex; flex-direction: column; gap: 8px; }
      .agent-compose textarea {
        width: 100%; min-height: 72px; resize: vertical; padding: 10px 12px;
        background: var(--panel-raised); color: var(--text); font: inherit; font-size: 13px;
        border: 1px solid var(--line-strong); border-radius: var(--radius-md); outline: none;
        transition: border-color .18s var(--ease), box-shadow .18s var(--ease);
      }
      .agent-compose textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
      .agent-run {
        display: inline-flex; align-items: center; justify-content: center; gap: 8px;
        padding: 9px 14px; border: 0; border-radius: var(--radius-md); cursor: pointer;
        background: var(--accent); color: #fff; font: inherit; font-size: 13px; font-weight: 600;
        transition: background .18s var(--ease), transform .18s var(--ease);
      }
      .agent-run:hover:not(:disabled) { background: var(--accent-strong); }
      .agent-run:active:not(:disabled) { transform: translateY(1px); }
      .agent-run:disabled { opacity: .5; cursor: not-allowed; }
      .agent-actions { display: flex; gap: 8px; }
      .agent-actions .agent-run { flex: 1; }
      .agent-stop {
        padding: 9px 16px; border-radius: var(--radius-md); cursor: pointer;
        background: transparent; border: 1px solid var(--danger); color: var(--danger);
        font: inherit; font-size: 13px; font-weight: 600;
        transition: background .18s var(--ease);
      }
      .agent-stop:hover:not(:disabled) { background: rgba(239, 127, 136, .12); }
      .agent-stop:disabled { opacity: .5; cursor: not-allowed; }
      .agent-after { display: flex; gap: 7px; flex-wrap: wrap; }
      .agent-after button {
        padding: 7px 11px; border-radius: var(--radius-sm); border: 1px solid var(--line-strong);
        background: var(--panel-raised); color: var(--text); font: inherit; font-size: 12px;
        font-weight: 600; cursor: pointer; transition: background .18s var(--ease);
      }
      .agent-after button:hover:not(:disabled) { background: var(--panel-hover); }
      .agent-after button:disabled { opacity: .5; cursor: not-allowed; }
      .agent-after button.danger { border-color: var(--danger); color: var(--danger); }
      .agent-diff-file { margin-bottom: 10px; }
      .agent-diff-file header {
        display: flex; align-items: center; gap: 8px; font-size: 11.5px;
        font-family: ui-monospace, monospace; color: var(--text-soft); margin-bottom: 4px;
      }
      .agent-diff-file .tag { font-size: 10px; padding: 1px 6px; border-radius: 999px; background: var(--accent-soft); color: var(--accent-strong); font-family: inherit; }
      .agent-diff-file .tag.user { background: rgba(244, 184, 90, .15); color: var(--amber); }
      .agent-diff pre {
        margin: 0; padding: 8px; background: var(--bg-soft); border-radius: var(--radius-sm);
        font-size: 10.5px; line-height: 1.5; overflow-x: auto; max-height: 260px; white-space: pre;
      }
      .agent-diff .add { color: var(--mint); }
      .agent-diff .del { color: var(--danger); }
      .agent-diff .hunk { color: var(--accent-strong); }
      .agent-state {
        display: flex; align-items: center; gap: 10px; padding: 10px 12px;
        background: var(--panel-raised); border: 1px solid var(--line); border-radius: var(--radius-md);
      }
      .agent-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--muted); flex: none; }
      .agent-dot.busy { background: var(--accent); animation: agent-pulse 1.4s var(--ease) infinite; }
      .agent-dot.ok { background: var(--mint); }
      .agent-dot.warn { background: var(--amber); animation: agent-pulse 1.4s var(--ease) infinite; }
      .agent-dot.bad { background: var(--danger); }
      @keyframes agent-pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .45; transform: scale(.82); } }
      @media (prefers-reduced-motion: reduce) { .agent-dot { animation: none !important; } }
      .agent-state-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
      .agent-state-text strong { font-size: 13px; font-weight: 600; }
      .agent-state-text span { font-size: 11px; color: var(--muted); overflow-wrap: anywhere; }
      .agent-section > h4 {
        margin: 0 0 8px; font-size: 11px; font-weight: 600; letter-spacing: .08em;
        text-transform: uppercase; color: var(--muted);
      }
      .agent-steps { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 5px; }
      .agent-step { display: flex; gap: 8px; font-size: 12px; line-height: 1.45; color: var(--text-soft); }
      .agent-step .mark { flex: none; width: 15px; text-align: center; color: var(--muted); }
      .agent-step.done .mark { color: var(--mint); }
      .agent-step.active { color: var(--text); }
      .agent-step.active .mark { color: var(--accent); }
      .agent-step.failed .mark { color: var(--danger); }
      .agent-timeline { display: flex; flex-direction: column; gap: 2px; max-height: 340px; overflow-y: auto; }
      .agent-event { display: flex; gap: 8px; padding: 5px 6px; border-radius: var(--radius-sm); font-size: 12px; line-height: 1.45; }
      .agent-event:hover { background: var(--panel-hover); }
      .agent-event .icon { flex: none; width: 14px; text-align: center; font-size: 10px; margin-top: 2px; }
      .agent-event .body { min-width: 0; flex: 1; }
      .agent-event .msg { color: var(--text-soft); overflow-wrap: anywhere; }
      .agent-event.tone-accent .icon { color: var(--accent); }
      .agent-event.tone-ok .icon { color: var(--mint); }
      .agent-event.tone-warn .icon { color: var(--amber); }
      .agent-event.tone-bad .icon { color: var(--danger); }
      .agent-event.tone-bad .msg { color: var(--danger); }
      .agent-event.tone-muted .icon { color: var(--faint); }
      .agent-event .when { font-size: 10px; color: var(--faint); }
      .agent-event details { margin-top: 3px; }
      .agent-event summary { cursor: pointer; font-size: 10px; color: var(--muted); }
      .agent-event pre {
        margin: 4px 0 0; padding: 7px 8px; background: var(--bg-soft); border-radius: var(--radius-sm);
        font-size: 10.5px; line-height: 1.5; color: var(--text-soft); overflow-x: auto; max-height: 190px;
      }
      .agent-approval {
        padding: 11px 12px; background: rgba(244, 184, 90, .09);
        border: 1px solid rgba(244, 184, 90, .32); border-radius: var(--radius-md);
        display: flex; flex-direction: column; gap: 9px;
      }
      .agent-approval p { margin: 0; font-size: 12px; line-height: 1.5; color: var(--text-soft); }
      .agent-approval .row { display: flex; gap: 7px; }
      .agent-approval button {
        flex: 1; padding: 7px 10px; border-radius: var(--radius-sm); border: 1px solid var(--line-strong);
        background: var(--panel-raised); color: var(--text); font: inherit; font-size: 12px;
        font-weight: 600; cursor: pointer; transition: background .18s var(--ease);
      }
      .agent-approval button.approve { background: var(--mint); border-color: var(--mint); color: #08150f; }
      .agent-approval button.deny:hover { background: var(--panel-hover); }
      .agent-files { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 3px; }
      .agent-files li { font-size: 11.5px; color: var(--text-soft); font-family: ui-monospace, monospace; overflow-wrap: anywhere; }
      .agent-history { display: flex; flex-direction: column; gap: 4px; }
      .agent-history button {
        text-align: left; padding: 7px 9px; border-radius: var(--radius-sm); cursor: pointer;
        background: transparent; border: 1px solid var(--line); color: var(--text-soft);
        font: inherit; font-size: 11.5px; display: flex; justify-content: space-between; gap: 8px;
      }
      .agent-history button:hover { background: var(--panel-hover); }
      .agent-history .badge { font-size: 10px; color: var(--muted); flex: none; }
      .agent-empty { font-size: 12px; color: var(--muted); margin: 0; line-height: 1.5; }
      .agent-model {
        display: flex; align-items: flex-start; gap: 9px; padding: 9px 11px;
        background: var(--panel-raised); border: 1px solid var(--line); border-radius: var(--radius-md);
        font-size: 12px; line-height: 1.5; color: var(--text-soft);
      }
      .agent-model .icon { flex: none; font-size: 13px; line-height: 1.4; }
      .agent-model .body { flex: 1; min-width: 0; }
      .agent-model .body strong { color: var(--text); font-family: ui-monospace, monospace; font-size: 11.5px; overflow-wrap: anywhere; }
      .agent-model .why { display: block; margin-top: 2px; }
      .agent-model .tag { font-size: 10px; color: var(--muted); margin-left: 6px; }
      .agent-model.tone-ok { border-color: rgba(78, 205, 150, .35); }
      .agent-model.tone-ok .icon { color: var(--mint); }
      .agent-model.tone-warn { border-color: rgba(244, 184, 90, .35); }
      .agent-model.tone-warn .icon { color: var(--amber); }
      .agent-model.tone-bad { border-color: rgba(240, 96, 96, .35); }
      .agent-model.tone-bad .icon, .agent-model.tone-bad .why { color: var(--danger); }
      .agent-model button {
        flex: none; padding: 3px 8px; border-radius: var(--radius-sm); border: 1px solid var(--line);
        background: transparent; color: var(--muted); font: inherit; font-size: 10.5px; cursor: pointer;
      }
      .agent-model button:hover { background: var(--panel-hover); color: var(--text); }
      .agent-summary {
        font-size: 12px; line-height: 1.55; color: var(--text-soft); white-space: pre-wrap;
        padding: 10px 11px; background: var(--panel-raised); border: 1px solid var(--line);
        border-radius: var(--radius-md); overflow-wrap: anywhere;
      }
    `;
    document.head.appendChild(style);
  }

  function mount() {
    const tablist = document.querySelector('.context-tabs[role="tablist"]');
    const pane = document.getElementById("context-pane");
    // The host page already owns the id "panel-agent" for its chat stage, so
    // this panel uses its own namespace to avoid colliding with it.
    if (!tablist || !pane || document.getElementById("panel-autopilot")) return false;

    const tab = document.createElement("button");
    tab.setAttribute("role", "tab");
    tab.dataset.panel = "autopilot";
    tab.setAttribute("aria-selected", "false");
    tab.tabIndex = -1;
    tab.textContent = "Autopilot";
    tablist.insertBefore(tab, tablist.firstChild);

    const panel = document.createElement("div");
    panel.className = "context-panel";
    panel.id = "panel-autopilot";
    panel.setAttribute("role", "tabpanel");
    panel.hidden = true;
    panel.innerHTML = `
      <div class="agent-panel">
        <div class="agent-compose">
          <textarea id="agent-goal" rows="3" placeholder="Describe a task. SAM will inspect the project, plan, edit files, run the tests and fix what fails."></textarea>
          <div class="agent-actions">
            <button class="agent-run" id="agent-run" type="button">Run autonomously</button>
            <button class="agent-stop" id="agent-stop" type="button" hidden>Stop</button>
          </div>
        </div>
        <div class="agent-model tone-muted" id="agent-model">
          <span class="icon">○</span>
          <span class="body">Checking which model will run…</span>
          <button type="button" id="agent-model-refresh" title="Re-check the provider (costs no tokens)">Re-check</button>
        </div>
        <div class="agent-state">
          <span class="agent-dot" id="agent-dot"></span>
          <span class="agent-state-text">
            <strong id="agent-state-label">Idle</strong>
            <span id="agent-state-detail">No task is running.</span>
          </span>
        </div>
        <div id="agent-approval-slot"></div>
        <div class="agent-after" id="agent-after" hidden>
          <button type="button" id="agent-view-diff">View diff</button>
          <button type="button" id="agent-rollback" class="danger">Rollback</button>
          <button type="button" id="agent-retry">Retry</button>
        </div>
        <div class="agent-section agent-diff" id="agent-diff-section" hidden>
          <h4>Changes made by this run</h4>
          <div id="agent-diff"></div>
        </div>
        <div class="agent-section" id="agent-plan-section" hidden>
          <h4>Plan</h4>
          <ol class="agent-steps" id="agent-steps"></ol>
        </div>
        <div class="agent-section" id="agent-files-section" hidden>
          <h4>Files changed</h4>
          <ul class="agent-files" id="agent-files"></ul>
        </div>
        <div class="agent-section" id="agent-summary-section" hidden>
          <h4>Result</h4>
          <div class="agent-summary" id="agent-summary"></div>
        </div>
        <div class="agent-section">
          <h4>Activity</h4>
          <div class="agent-timeline" id="agent-timeline">
            <p class="agent-empty">Activity appears here as SAM works.</p>
          </div>
        </div>
        <div class="agent-section">
          <h4>Recent tasks</h4>
          <div class="agent-history" id="agent-history"></div>
        </div>
      </div>`;
    pane.appendChild(panel);

    // The host page drives tab switching by data-panel; clicking still needs a
    // local handler because the panel was added after its listeners bound.
    tablist.querySelectorAll('[role="tab"]').forEach((button) => {
      button.addEventListener("click", () => {
        tablist.querySelectorAll('[role="tab"]').forEach((other) => {
          const selected = other === button;
          other.setAttribute("aria-selected", String(selected));
          other.tabIndex = selected ? 0 : -1;
        });
        pane.querySelectorAll(".context-panel").forEach((element) => {
          element.hidden = element.id !== `panel-${button.dataset.panel}`;
        });
      });
    });

    tab.addEventListener("click", () => loadModelResolution(false));
    document.getElementById("agent-model-refresh").addEventListener("click", () => loadModelResolution(true));
    document.getElementById("agent-run").addEventListener("click", startTask);
    document.getElementById("agent-stop").addEventListener("click", stopTask);
    document.getElementById("agent-view-diff").addEventListener("click", toggleDiff);
    document.getElementById("agent-rollback").addEventListener("click", rollbackTask);
    document.getElementById("agent-retry").addEventListener("click", retryTask);
    document.getElementById("agent-goal").addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") startTask();
    });
    return true;
  }

  async function api(path, options) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`${response.status} ${detail.slice(0, 300)}`);
    }
    return response.json();
  }

  // -- which model will actually run ----------------------------------------
  async function loadModelResolution(refresh) {
    const node = document.getElementById("agent-model");
    if (!node) return;
    if (refresh) node.querySelector(".body").textContent = "Re-checking the provider…";
    try {
      const payload = await api(`/api/providers/resolution${refresh ? "?refresh=true" : ""}`);
      renderModelResolution(payload.resolution);
    } catch (error) {
      renderModelResolution(null, error.message);
    }
  }

  const describeModel = (capability) => {
    if (!capability) return "";
    const tags = [capability.cost_class !== "unknown" ? capability.cost_class : "", capability.supports_tools ? "tools" : ""]
      .filter(Boolean).join(" · ");
    return `<strong>${escapeHtml(capability.model)}</strong>${tags ? `<span class="tag">${escapeHtml(tags)}</span>` : ""}`;
  };

  function renderModelResolution(resolution, failure) {
    const node = document.getElementById("agent-model");
    if (!node) return;
    // The backend composes the reason text; the panel only chooses a tone.
    let tone = "muted";
    let icon = "○";
    let body;
    if (!resolution) {
      tone = "bad"; icon = "✕";
      body = `Could not check the provider.<span class="why">${escapeHtml(failure || "")}</span>`;
    } else if (resolution.blocked) {
      tone = "bad"; icon = "✕";
      body = `Runs are blocked.<span class="why">${escapeHtml(resolution.fallback_reason)}</span>`;
    } else if (resolution.fallback_engaged) {
      tone = "warn"; icon = "⇄";
      body = `Will run on fallback ${describeModel(resolution.active)}<span class="why">${escapeHtml(resolution.fallback_reason)}</span>`;
    } else if (resolution.active.availability === "UNKNOWN") {
      body = `Model: ${describeModel(resolution.active)}<span class="why">${escapeHtml(resolution.active.reason)}</span>`;
    } else {
      tone = "ok"; icon = "✓";
      body = `Model: ${describeModel(resolution.active)}`;
    }
    node.className = `agent-model tone-${tone}`;
    node.querySelector(".icon").textContent = icon;
    node.querySelector(".body").innerHTML = body;
  }

  async function startTask() {
    const input = document.getElementById("agent-goal");
    const button = document.getElementById("agent-run");
    const goal = input.value.trim();
    if (!goal) {
      input.focus();
      return;
    }
    button.disabled = true;
    button.textContent = "Starting...";
    try {
      const payload = await api("/api/tasks", { method: "POST", body: JSON.stringify({ goal }) });
      state.taskId = payload.task_id;
      state.task = payload.task;
      input.value = "";
      render();
      startPolling();
    } catch (error) {
      setStatus("FAILED", `Could not start the task: ${error.message}`);
    } finally {
      button.disabled = false;
      button.textContent = "Run autonomously";
    }
  }

  async function stopTask() {
    if (!state.taskId) return;
    const button = document.getElementById("agent-stop");
    button.disabled = true;
    button.textContent = "Stopping...";
    try {
      const payload = await api(`/api/tasks/${state.taskId}/cancel`, { method: "POST" });
      if (!payload.cancelled) setStatus(state.task?.state || "IDLE", payload.reason || "");
      // The run notices the stop at its next checkpoint, so keep polling
      // until it actually reports a terminal state rather than assuming.
      startPolling();
      refreshTask();
    } catch (error) {
      setStatus("FAILED", `Could not stop the task: ${error.message}`);
    } finally {
      button.disabled = false;
      button.textContent = "Stop";
    }
  }

  function renderControls(task) {
    const stop = document.getElementById("agent-stop");
    if (!stop) return;
    // Stoppable means "running or paused": a task waiting for approval is
    // still holding resources and must be abandonable.
    stop.hidden = !task || task.terminal;

    // After-run controls only make sense once nothing is still driving.
    const after = document.getElementById("agent-after");
    const touched = Boolean(task?.checkpoints?.length);
    after.hidden = !task || !task.terminal;
    document.getElementById("agent-view-diff").hidden = !touched;
    document.getElementById("agent-rollback").hidden = !touched || Boolean(task?.rolled_back);
    if (!touched) document.getElementById("agent-diff-section").hidden = true;
  }

  async function toggleDiff() {
    const section = document.getElementById("agent-diff-section");
    if (!section.hidden) {
      section.hidden = true;
      return;
    }
    const container = document.getElementById("agent-diff");
    container.innerHTML = '<p class="agent-empty">Loading diff...</p>';
    section.hidden = false;
    try {
      const payload = await api(`/api/tasks/${state.taskId}/diff`);
      renderDiff(payload);
    } catch (error) {
      container.innerHTML = `<p class="agent-empty">Could not load the diff: ${escapeHtml(error.message)}</p>`;
    }
  }

  function renderDiff(payload) {
    const container = document.getElementById("agent-diff");
    if (!payload.files.length) {
      container.innerHTML = '<p class="agent-empty">This run did not change any files.</p>';
      return;
    }
    const colour = (line) => {
      const safe = escapeHtml(line);
      if (line.startsWith("+++") || line.startsWith("---")) return safe;
      if (line.startsWith("@@")) return `<span class="hunk">${safe}</span>`;
      if (line.startsWith("+")) return `<span class="add">${safe}</span>`;
      if (line.startsWith("-")) return `<span class="del">${safe}</span>`;
      return safe;
    };
    container.innerHTML = payload.files
      .map((file) => `<div class="agent-diff-file">
        <header>
          <span>${escapeHtml(file.path)}</span>
          <span class="tag">${escapeHtml(file.status)}</span>
          ${file.had_user_changes ? '<span class="tag user" title="You already had uncommitted changes in this file before the run">had your changes</span>' : ""}
        </header>
        <pre>${file.diff ? file.diff.split(String.fromCharCode(10)).map(colour).join(String.fromCharCode(10)) : "(no textual change)"}</pre>
      </div>`)
      .join("");
    if (payload.rolled_back) {
      container.insertAdjacentHTML("afterbegin", '<p class="agent-empty">These changes have been rolled back.</p>');
    }
  }

  async function rollbackTask() {
    if (!state.taskId) return;
    const files = state.task?.checkpoints?.length || 0;
    if (!window.confirm(`Restore ${files} file(s) to how they were before this run?`)) return;
    const button = document.getElementById("agent-rollback");
    button.disabled = true;
    try {
      const payload = await api(`/api/tasks/${state.taskId}/rollback`, { method: "POST" });
      if (!payload.rolled_back) setStatus(state.task?.state || "IDLE", payload.reason || "");
      await refreshTask();
      if (!document.getElementById("agent-diff-section").hidden) toggleDiff().then(toggleDiff);
    } catch (error) {
      setStatus(state.task?.state || "FAILED", `Rollback failed: ${error.message}`);
    } finally {
      button.disabled = false;
    }
  }

  async function retryTask() {
    if (!state.taskId) return;
    const button = document.getElementById("agent-retry");
    button.disabled = true;
    try {
      const payload = await api(`/api/tasks/${state.taskId}/retry`, { method: "POST" });
      state.taskId = payload.task_id;
      state.task = payload.task;
      document.getElementById("agent-diff-section").hidden = true;
      render();
      startPolling();
    } catch (error) {
      setStatus("FAILED", `Could not retry: ${error.message}`);
    } finally {
      button.disabled = false;
    }
  }

  async function refreshTask() {
    if (!state.taskId) return;
    try {
      const payload = await api(`/api/tasks/${state.taskId}`);
      state.task = payload.task;
      render();
      if (state.task.terminal) stopPolling();
    } catch {
      /* a transient read failure is not worth interrupting the user over */
    }
  }

  async function refreshHistory() {
    try {
      const payload = await api("/api/tasks?limit=8");
      renderHistory(payload.tasks || []);
    } catch {
      /* history is supplementary */
    }
  }

  function startPolling() {
    stopPolling();
    // The websocket is the primary channel; this is the safety net for a
    // dropped connection, so it can stay slow.
    state.pollTimer = setInterval(refreshTask, POLL_INTERVAL_MS);
  }

  function stopPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = null;
    refreshHistory();
  }

  async function decideApproval(approvalId, decision) {
    const slot = document.getElementById("agent-approval-slot");
    slot.innerHTML = `<p class="agent-empty">Sending your ${escapeHtml(decision)} decision...</p>`;
    try {
      await api(`/api/tasks/${state.taskId}/approvals`, {
        method: "POST",
        body: JSON.stringify({ approval_id: approvalId, decision }),
      });
      state.pendingApproval = null;
      startPolling();
      refreshTask();
    } catch (error) {
      slot.innerHTML = `<p class="agent-empty">That decision could not be sent: ${escapeHtml(error.message)}</p>`;
    }
  }

  // -- rendering -----------------------------------------------------------
  function setStatus(stateName, detail) {
    const dot = document.getElementById("agent-dot");
    const label = document.getElementById("agent-state-label");
    const detailNode = document.getElementById("agent-state-detail");
    if (!dot) return;
    dot.className = `agent-dot ${stateTone(stateName)}`;
    label.textContent = stateLabel(stateName);
    if (detail !== undefined) detailNode.textContent = detail;
  }

  function render() {
    const task = state.task;
    if (!task) return;

    const done = (task.plan || []).filter((step) => step.status === "done").length;
    const total = (task.plan || []).length;
    const progress = total ? ` · step ${Math.min(done + 1, total)}/${total}` : "";
    setStatus(task.state, `${task.goal}${task.terminal ? "" : progress}`);

    renderPlan(task.plan || []);
    renderTimeline(task.events || []);
    renderFiles(task.modified_files || []);
    renderSummary(task);
    renderApproval(task);
    renderControls(task);
  }

  function renderPlan(plan) {
    const section = document.getElementById("agent-plan-section");
    const list = document.getElementById("agent-steps");
    section.hidden = plan.length === 0;
    list.innerHTML = plan
      .map((step) => {
        const mark = { done: "✓", active: "▸", failed: "✕", skipped: "–" }[step.status] || "○";
        return `<li class="agent-step ${escapeHtml(step.status)}">
          <span class="mark">${mark}</span>
          <span>${escapeHtml(step.text)}</span>
        </li>`;
      })
      .join("");
  }

  function renderTimeline(events) {
    const container = document.getElementById("agent-timeline");
    if (!events.length) {
      container.innerHTML = '<p class="agent-empty">Activity appears here as SAM works.</p>';
      return;
    }
    const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 60;
    container.innerHTML = events
      .slice(-MAX_RENDERED_EVENTS)
      .map((event) => {
        const meta = EVENT_META[event.kind] || EVENT_META.state;
        const detail = event.detail && Object.keys(event.detail).length
          ? `<details><summary>details</summary><pre>${escapeHtml(JSON.stringify(event.detail, null, 2))}</pre></details>`
          : "";
        return `<div class="agent-event tone-${meta.tone}">
          <span class="icon" title="${escapeHtml(meta.label)}">${meta.icon}</span>
          <span class="body">
            <span class="msg">${escapeHtml(event.message)}</span>
            <span class="when"> ${escapeHtml(relativeTime(event.at))}</span>
            ${detail}
          </span>
        </div>`;
      })
      .join("");
    // Follow the feed only if the user has not scrolled up to read something.
    if (nearBottom) container.scrollTop = container.scrollHeight;
  }

  function renderFiles(files) {
    const section = document.getElementById("agent-files-section");
    section.hidden = files.length === 0;
    document.getElementById("agent-files").innerHTML = files
      .map((path) => `<li>${escapeHtml(path)}</li>`)
      .join("");
  }

  function renderSummary(task) {
    const section = document.getElementById("agent-summary-section");
    const visible = Boolean(task.summary) && task.terminal;
    section.hidden = !visible;
    if (visible) document.getElementById("agent-summary").textContent = task.summary;
  }

  function renderApproval(task) {
    const slot = document.getElementById("agent-approval-slot");
    if (task.state !== "WAITING_FOR_APPROVAL") {
      slot.innerHTML = "";
      return;
    }
    const approvalEvent = [...(task.events || [])].reverse().find((event) => event.kind === "approval" && event.detail?.approval_id);
    if (!approvalEvent) {
      slot.innerHTML = "";
      return;
    }
    const detail = approvalEvent.detail || {};
    if (state.pendingApproval === detail.approval_id && slot.innerHTML) return;
    state.pendingApproval = detail.approval_id;
    slot.innerHTML = `<div class="agent-approval">
      <p><strong>${escapeHtml(detail.tool || "An action")}</strong> needs your approval.</p>
      <p>${escapeHtml(approvalEvent.message)}</p>
      <details><summary>Arguments</summary><pre>${escapeHtml(JSON.stringify(detail.arguments || {}, null, 2))}</pre></details>
      <div class="row">
        <button class="approve" type="button" data-decision="approved">Approve</button>
        <button class="deny" type="button" data-decision="denied">Deny</button>
      </div>
    </div>`;
    slot.querySelectorAll("button[data-decision]").forEach((button) => {
      button.addEventListener("click", () => decideApproval(detail.approval_id, button.dataset.decision));
    });
  }

  function renderHistory(tasks) {
    const container = document.getElementById("agent-history");
    if (!container) return;
    if (!tasks.length) {
      container.innerHTML = '<p class="agent-empty">Completed runs will be listed here.</p>';
      return;
    }
    container.innerHTML = tasks
      .map((task) => `<button type="button" data-task="${escapeHtml(task.id)}">
        <span>${escapeHtml(task.goal.slice(0, 70))}</span>
        <span class="badge">${escapeHtml(stateLabel(task.state))}</span>
      </button>`)
      .join("");
    container.querySelectorAll("button[data-task]").forEach((button) => {
      button.addEventListener("click", () => {
        state.taskId = button.dataset.task;
        refreshTask();
      });
    });
  }

  // -- live stream ---------------------------------------------------------
  function connect() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    let socket;
    try {
      socket = new WebSocket(`${protocol}//${location.host}/ws/live`);
    } catch {
      return;
    }
    state.socket = socket;

    socket.addEventListener("open", () => {
      state.reconnectDelay = 1000;
    });
    socket.addEventListener("message", (message) => {
      let payload;
      try {
        payload = JSON.parse(message.data);
      } catch {
        return;
      }
      if (payload.type !== "task_event" && payload.type !== "task_state") return;
      // Adopt a run started from elsewhere (another tab, or voice) so the
      // panel always reflects what the agent is actually doing.
      if (!state.taskId) state.taskId = payload.task_id;
      if (payload.task_id !== state.taskId) return;
      if (payload.type === "task_state") {
        setStatus(payload.state, payload.message);
        if (["COMPLETED", "FAILED", "CANCELLED", "WAITING_FOR_APPROVAL"].includes(payload.state)) refreshTask();
      }
      if (payload.type === "task_event") refreshTask();
    });
    const reconnect = () => {
      state.socket = null;
      // Back off so a downed backend is not hammered, but stay responsive.
      setTimeout(connect, state.reconnectDelay);
      state.reconnectDelay = Math.min(state.reconnectDelay * 2, 15000);
    };
    socket.addEventListener("close", reconnect);
    socket.addEventListener("error", () => socket.close());
  }

  function boot() {
    injectStyles();
    if (!mount()) return;
    refreshHistory();
    loadModelResolution(false);
    connect();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
