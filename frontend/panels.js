(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const text = (id, value, fallback = "—") => {
    const element = document.getElementById(id);
    if (element) element.textContent = value === null || value === undefined || value === "" ? fallback : String(value);
  };
  const fmt = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
  const title = (value) => String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

  let latestAnalysis = null;
  let latestSetupId = null;

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

  function setStatus(id, status) {
    const element = document.getElementById(id);
    if (!element) return;
    element.textContent = String(status || "UNKNOWN").toUpperCase();
    element.className = "badge badge-subtle";
    const normalized = String(status || "").toUpperCase();
    if (["SUCCESS", "AVAILABLE", "CONNECTED", "RUNNING"].includes(normalized)) element.classList.add("badge-success");
    else if (["FAILED", "UNAVAILABLE", "OFFLINE"].includes(normalized)) element.classList.add("badge-danger");
    else element.classList.add("badge-warning");
  }

  function element(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = String(content);
    return node;
  }

  function toolStatus(id, label, ok = true) {
    const node = document.getElementById(id);
    if (!node) return;
    node.textContent = label;
    node.className = `tool-avail ${ok ? "ok" : "warn"}`;
  }

  async function loadToolStatus() {
    try {
      const [manifestPayload, health, desktop, trading] = await Promise.all([
        json("/api/tools/manifests"), json("/api/health"), json("/api/desktop/status"), requestBrokerStatus(),
      ]);
      const names = new Set((manifestPayload.tools || []).map((item) => item.name));
      toolStatus("tool-status-files", names.has("read_file") && names.has("write_file") ? "● Available" : "Unavailable", names.has("read_file"));
      toolStatus("tool-status-terminal", names.has("run_terminal") ? "● Guarded" : "Unavailable", names.has("run_terminal"));
      toolStatus("tool-status-python", names.has("run_python") ? "● Approval-gated" : "Unavailable", names.has("run_python"));
      toolStatus("tool-status-browser", names.has("browser_automate") ? "● Approval-gated" : "Unavailable", names.has("browser_automate"));
      const mt5State = trading?.market_data?.metatrader5?.state;  // null when the broker did not answer
      toolStatus("tool-status-mt5", mt5State === "AVAILABLE" ? "● Connected read-only" : title(mt5State), mt5State === "AVAILABLE");
      toolStatus("tool-status-tradingview", desktop.tradingview?.interactive ? "● Interactive window" : desktop.tradingview?.running ? "Process only" : "Offline", Boolean(desktop.tradingview?.interactive));
      toolStatus("tool-status-desktop", health.computer_control ? "● Control ON" : "Control OFF", Boolean(health.computer_control));
      toolStatus("tool-status-market", names.has("analyze_market") ? "● Deterministic engine" : "Unavailable", names.has("analyze_market"));
    } catch (error) {
      for (const id of ["tool-status-files", "tool-status-terminal", "tool-status-python", "tool-status-browser", "tool-status-mt5", "tool-status-tradingview", "tool-status-desktop", "tool-status-market"]) {
        toolStatus(id, `Unavailable · ${error.message}`, false);
      }
    }
  }

  function selectedTimeframes() {
    const checked = $$(".sam-timeframe-strip input:checked").map((item) => item.value);
    return checked.length ? checked : ["H1", "M15", "M5", "M1"];
  }

  async function loadMarketSnapshot() {
    text("trading-action-status", "Refreshing verified market data…");
    try {
      const symbol = ($("#trading-symbol")?.value || "XAUUSD").trim().toUpperCase();
      const result = await json("/api/trading/snapshot", {
        method: "POST",
        body: JSON.stringify({ symbol, timeframes: selectedTimeframes() }),
      });
      const data = result.data || {};
      setStatus("trading-data-state", result.status);
      text("trading-provider-verification", result.verified ? "Verified from provider metadata" : "Partial / stale / missing timeframe");
      text("trading-feed", data.feed || "Provider feed not reported");
      text("mi-price", fmt(data.bid));
      text("mi-bid", fmt(data.bid));
      text("mi-ask", fmt(data.ask));
      text("mi-spread", Number.isFinite(Number(data.spread)) ? fmt(data.spread, 3) : "—");
      text("mi-session", (data.session?.active || []).join(", ") || "CLOSED / TRANSITION");
      text("mi-updated", new Date().toLocaleTimeString());
      text("topbar-xauusd-price", fmt(data.bid));
      $("#topbar-xauusd")?.classList.toggle("loaded", Number.isFinite(Number(data.bid)));
      renderOhlc(data.timeframes || {});
      text("trading-action-status", result.error || `${result.status}: snapshot refreshed`);
      return result;
    } catch (error) {
      setStatus("trading-data-state", "UNAVAILABLE");
      text("trading-provider-verification", error.message);
      text("trading-action-status", `Market data unavailable: ${error.message}`);
      throw error;
    }
  }

  async function loadMt5() {
    try {
      const data = await json("/api/mt5");
      setStatus("mt5-connection-status", data.connected ? "CONNECTED" : "OFFLINE");
      if (!data.connected) {
        text("mt5-verification", data.mt5_error || "No provider connection");
        return;
      }
      text("mt5-bid", fmt(data.bid));
      text("mt5-ask", fmt(data.ask));
      text("mt5-spread", Number.isFinite(Number(data.spread)) ? `${fmt(data.spread, 1)} pts` : "—");
      text("mt5-symbol", data.symbol);
      text("mt5-broker", data.broker);
      text("mt5-terminal", data.provider_metadata?.terminal_path || "Connected terminal (path not reported)");
      text("mt5-scope", "Read-only OHLCV and ticks; no order API");
      text("mt5-verification", data.verified ? "Verified" : "Partial");
      renderOhlc(data.ohlc || {});
      text("topbar-xauusd-price", fmt(data.bid));
      $("#topbar-xauusd")?.classList.add("loaded");
    } catch (error) {
      setStatus("mt5-connection-status", "OFFLINE");
      text("mt5-verification", error.message);
    }
  }

  function renderOhlc(timeframes) {
    const container = $("#mt5-ohlc-table");
    if (!container) return;
    container.replaceChildren();
    const rows = Object.entries(timeframes);
    if (!rows.length) {
      container.appendChild(element("p", "compatibility-note", "No verified OHLC candles were returned."));
      return;
    }
    for (const [timeframe, raw] of rows) {
      const candle = raw?.metadata ? raw : {
        open: raw?.o, high: raw?.h, low: raw?.l, close: raw?.c, time: raw?.time,
      };
      const row = element("div", "status-row");
      row.appendChild(element("span", "label", timeframe));
      row.appendChild(element("span", "value", `O ${fmt(candle.open)}  H ${fmt(candle.high)}  L ${fmt(candle.low)}  C ${fmt(candle.close)}`));
      container.appendChild(row);
    }
  }

  async function loadTradingResources() {
    const [capabilityPayload, skillsPayload, theoriesPayload] = await Promise.allSettled([
      json(`/api/trading/capabilities?symbol=${encodeURIComponent($("#trading-symbol")?.value || "XAUUSD")}`),
      json("/api/trading/skills"),
      json("/api/trading/theories"),
    ]);
    const capabilityRoot = $("#market-capabilities");
    if (capabilityRoot) {
      capabilityRoot.replaceChildren();
      if (capabilityPayload.status === "fulfilled") {
        const caps = capabilityPayload.value;
        for (const [key, value] of Object.entries(caps)) {
          if (typeof value === "object" || key === "metadata") continue;
          const row = element("div", "status-row");
          row.appendChild(element("span", "label", title(key)));
          row.appendChild(element("span", "value", value));
          capabilityRoot.appendChild(row);
        }
      } else capabilityRoot.appendChild(element("p", "compatibility-note", capabilityPayload.reason?.message || "Unavailable"));
    }
    const skillsRoot = $("#trading-skills-health");
    if (skillsRoot) {
      skillsRoot.replaceChildren();
      const skills = skillsPayload.status === "fulfilled" ? skillsPayload.value.skills || [] : [];
      for (const skill of skills) {
        const row = element("div", "status-row");
        row.appendChild(element("span", "label", skill.name || skill.id));
        row.appendChild(element("span", "value", skill.health || skill.state || "UNKNOWN"));
        skillsRoot.appendChild(row);
      }
      if (!skills.length) skillsRoot.appendChild(element("p", "compatibility-note", "Trading skills unavailable."));
    }
    if (theoriesPayload.status === "fulfilled") {
      const select = $("#trading-theory");
      const selected = select?.value;
      if (select) {
        select.replaceChildren();
        for (const theory of theoriesPayload.value.built_in || []) {
          const option = element("option", "", `${theory.name} · ${theory.health}`);
          option.value = theory.id;
          select.appendChild(option);
        }
        if ([...select.options].some((item) => item.value === selected)) select.value = selected;
      }
    }
  }

  async function analyzeMarket(compare = false) {
    const button = $("#run-market-analysis-btn");
    if (button) button.disabled = true;
    text("trading-action-status", "Running deterministic multi-timeframe analysis…");
    try {
      const symbol = ($("#trading-symbol")?.value || "XAUUSD").trim().toUpperCase();
      const selected = $("#trading-theory")?.value || "default";
      const theories = compare ? [...new Set([selected, "snr", "smc", "ict", "wyckoff"])] : [selected];
      const result = await json("/api/trading/analyze", {
        method: "POST",
        body: JSON.stringify({ symbol, timeframes: selectedTimeframes(), theories, count: 600 }),
      });
      latestAnalysis = result.data || null;
      latestSetupId = result.setup_id || null;
      renderAnalysis(result);
      text("trading-action-status", result.error || `${result.status}: analysis completed in ${result.duration_ms || "?"} ms`);
    } catch (error) {
      setStatus("analysis-badge", "FAILED");
      text("analysis-conclusion", `Analysis failed: ${error.message}`);
      text("trading-action-status", error.message);
    } finally {
      if (button) button.disabled = false;
    }
  }

  function renderAnalysis(result) {
    const report = result.data || {};
    const decision = report.decision || "NO RESULT";
    setStatus("analysis-badge", decision);
    text("analysis-htf-bias", report.htf_bias);
    text("analysis-decision", decision);
    text("analysis-setup-state", report.setup_state);
    text("analysis-confidence", Number.isFinite(Number(report.confidence)) ? `${Math.round(Number(report.confidence) * 100)}%` : "—");
    text("analysis-missing", (report.missing_confirmation || []).join("; ") || "None reported");
    text("analysis-self-check", report.self_check?.passed ? "PASSED" : `BLOCKED: ${(report.self_check?.critical_failures || []).join("; ") || "verification gap"}`);
    text("analysis-entry", fmt(report.entry));
    text("analysis-invalidation", fmt(report.invalidation));
    text("analysis-stop", fmt(report.stop));
    text("analysis-tp1", fmt(report.tp1));
    text("analysis-tp2", fmt(report.tp2));
    text("analysis-tp3", fmt(report.tp3));
    text("analysis-rr", Number.isFinite(Number(report.rr)) ? fmt(report.rr, 2) : "—");
    text("analysis-conclusion", report.spoken_summary_ckb || `${decision}. ${(report.no_trade_reason || []).join(" ")}`);
    renderLevels(report);
    renderTheories(report.theories || {});
    const monitor = $("#monitor-setup-btn");
    if (monitor) monitor.disabled = !latestAnalysis;
    const draw = $("#draw-analysis-btn");
    if (draw) draw.disabled = true;
    text("drawing-capability", window.t ? window.t("drawing.blocked", "Drawing stays blocked until chart calibration is verified") : "Drawing stays blocked until chart calibration is verified");
  }

  function renderLevels(report) {
    const root = $("#analysis-levels");
    if (!root) return;
    root.replaceChildren();
    const levels = [
      ...(report.support || []).map((item) => ({ kind: "Support", ...item })),
      ...(report.resistance || []).map((item) => ({ kind: "Resistance", ...item })),
    ];
    for (const level of levels) {
      const row = element("div", "status-row");
      row.appendChild(element("span", "label", `${level.kind} · ${level.timeframe || "multi-TF"}`));
      row.appendChild(element("span", "value", `${fmt(level.price)} · score ${fmt(level.score, 2)}`));
      root.appendChild(row);
    }
    if (!levels.length) root.appendChild(element("p", "compatibility-note", "No verified scored levels."));
  }

  function renderTheories(theories) {
    const root = $("#analysis-theories");
    if (!root) return;
    root.replaceChildren();
    for (const [id, output] of Object.entries(theories)) {
      const row = element("div", "status-row");
      row.appendChild(element("span", "label", output?.theory?.name || id));
      const direction = output?.interpretation?.direction;
      row.appendChild(element("span", "value", `${output?.status || "UNKNOWN"}${direction ? ` · ${direction}` : ""}`));
      root.appendChild(row);
    }
    if (!Object.keys(theories).length) root.appendChild(element("p", "compatibility-note", "No theory output."));
  }

  function drawingAvailability(drawing) {
    // The drawing engine decides this; the panel only reports its answer, and
    // says so plainly when it never received one rather than inventing a verdict.
    if (!drawing) return "Unavailable; the drawing engine could not be reached";
    if (drawing.verified_price_drawing) return "Verified price drawing available";
    if (drawing.calibrated) return "Chart calibrated; desktop control permission still required";
    return "Unavailable until verified chart calibration";
  }

  // /api/trading/status reaches the broker feed. Every reader of it on this page
  // goes through here: one request at a time, shared while pending, given up
  // before the next tick. A stalled broker never accumulates unanswered requests,
  // an old answer never lands late, and it cannot stall or erase the TradingView
  // observation, which needs no broker and is asked for on its own.
  const BROKER_TIMEOUT_MS = 8_000;
  let brokerStatus = null;

  function requestBrokerStatus() {
    if (brokerStatus) return brokerStatus;
    const abort = new AbortController();
    const timer = window.setTimeout(() => abort.abort(), BROKER_TIMEOUT_MS);
    brokerStatus = json("/api/trading/status", { signal: abort.signal })
      .catch(() => null)  // timeout or error: no answer, no claim
      .finally(() => { window.clearTimeout(timer); brokerStatus = null; });
    return brokerStatus;
  }

  async function refreshTradingView() {
    const status = requestBrokerStatus();
    try {
      const state = await json("/api/tradingview/state");
      setStatus("tradingview-status", state.running ? "RUNNING" : "OFFLINE");
      text("tv-process-state", state.window_handle ? `Visible window · ${state.process_ids?.length || 0} process(es)` : state.running ? "Process found; no visible chart window" : "Not running");
      text("tv-observed-symbol", state.symbol || "Not exposed by title");
      text("tv-observed-price", fmt(state.current_price));
      text("tv-capture-state", state.interactive ? "Interactive session detected; permission still required" : "No interactive window");
      text("tv-disclosure", (state.observations || []).join(" ") || "TradingView state observed from the native Windows window.");
      // Last, so nothing above it waits on the broker.
      text("tv-drawing-state", drawingAvailability((await status)?.drawing));
    } catch (error) {
      setStatus("tradingview-status", "UNAVAILABLE");
      text("tv-process-state", error.message);
      // Without an observation, a verdict from a previous tick is no longer about
      // any chart we can see; it must not be left standing.
      text("tv-drawing-state", drawingAvailability(null));
    }
  }

  async function tradingViewAction(action, extra = {}) {
    text("tv-action-result", `Running ${action}…`);
    try {
      const result = await json("/api/tradingview/action", { method: "POST", body: JSON.stringify({ action, ...extra }) });
      text("tv-action-result", `${result.status} · ${result.verified ? "verified" : "not independently verified"}${result.error ? ` · ${result.error}` : ""}`);
      await refreshTradingView();
      return result;
    } catch (error) {
      text("tv-action-result", `FAILED · ${error.message}`);
      return null;
    }
  }

  async function loadDesktop() {
    try {
      const [status, health] = await Promise.all([json("/api/desktop/status"), json("/api/health")]);
      const root = $("#desktop-process-list");
      if (root) {
        root.replaceChildren();
        for (const process of status.processes || []) {
          const row = element("div", "proc-row");
          row.appendChild(element("span", "sdot g"));
          row.appendChild(element("span", "proc-name", process.name));
          row.appendChild(element("span", "proc-pid", `PID ${process.pid} · ${process.status || "unknown"}`));
          root.appendChild(row);
        }
        if (!(status.processes || []).length) root.appendChild(element("p", "compatibility-note", "No matching processes observed."));
      }
      text("desktop-workspace", health.workspace);
      text("desktop-uptime", `${Math.round(Number(health.uptime_seconds || 0))} seconds`);
      text("desktop-elevated", health.running_elevated ? "Yes (SAM should refuse startup)" : "No · normal user");
      text("desktop-control-permission", health.computer_control ? "ON" : "OFF");
      text("desktop-screen-permission", health.screen_access ? "ON" : "OFF");
      text("desktop-os", navigator.userAgentData?.platform || navigator.platform || "Windows");
      setStatus("desktop-capture-status", health.screen_access ? "AVAILABLE" : "OFF");
      text("desktop-screen-message", health.screen_access ? "Screen Access enabled. Use Capture Screen to create an audited snapshot." : "Screen Access is OFF. Enable it explicitly in Settings > Safety.");
    } catch (error) {
      text("desktop-screen-message", `Desktop status unavailable: ${error.message}`);
    }
  }

  async function emergencyStop() {
    window.speechSynthesis?.cancel?.();
    try {
      const result = await json("/api/abort", { method: "POST" });
      text("trading-action-status", `Emergency stop: ${result.cancelled_count || 0} task(s), ${result.stopped_monitors || 0} monitor(s)`);
    } catch (error) {
      text("trading-action-status", `Emergency stop failed: ${error.message}`);
    }
  }

  // --- Providers ------------------------------------------------------------
  // Everything below renders values the backend computed. No statistic, health
  // verdict, or trade result is derived in this file.

  let backtestController = null;

  async function loadProviders() {
    try {
      const data = await json("/api/providers/status");
      setStatus("provider-openrouter", data.openrouter?.status);
      setStatus("provider-ollama", data.ollama?.status);
      setStatus("provider-litellm", data.litellm?.status);
      const parts = [data.openrouter?.detail, data.ollama?.detail].filter(Boolean);
      text("provider-detail", parts.join(" ") || "—");
      renderSorani(data.sorani);
      return data;
    } catch (error) {
      text("provider-detail", error.message);
      return null;
    }
  }

  function renderSorani(sorani) {
    if (!sorani) return;
    const stt = sorani.stt?.primary || {};
    const tts = sorani.tts?.primary || {};
    setStatus("sorani-stt-status", stt.status || "UNKNOWN");
    setStatus("sorani-tts-status", tts.status || "UNKNOWN");
    setStatus("sorani-google-status", sorani.stt?.fallback?.status || "UNCONFIGURED");
    text("sorani-speaker", tts.selected_speaker || "—");
    // The provider writes its detail in Sorani, which is the language the user
    // reads, so it is shown as-is rather than replaced with English.
    const detail = [stt.detail, tts.detail].filter(Boolean).join(" ");
    text("sorani-detail", detail || "—");
  }

  async function loadSoraniSpeakers() {
    const select = $("#sorani-speaker-select");
    if (!select) return;
    try {
      const data = await json("/api/voice/sorani/speakers");
      if (data.status !== "CONNECTED") { text("sorani-detail", data.detail || data.status); return; }
      const current = select.value;
      select.innerHTML = '<option value="">Provider default voice</option>';
      for (const speaker of data.speakers || []) {
        const option = document.createElement("option");
        option.value = speaker.id ?? speaker.speaker_id ?? "";
        option.textContent = speaker.name || speaker.label || option.value;
        select.appendChild(option);
      }
      select.value = current || data.selected || "";
    } catch (error) {
      text("sorani-detail", error.message);
    }
  }

  async function saveSoraniKeys() {
    const fields = [
      ["#sorani-stt-key-input", "kurdishtts_stt_api_key"],
      ["#sorani-tts-key-input", "kurdishtts_tts_api_key"],
    ];
    const entered = fields.filter(([selector]) => ($(selector)?.value || "").trim());
    if (!entered.length) { text("sorani-detail", "Enter at least one KurdishTTS key first."); return; }
    text("sorani-detail", "Storing the key on the backend…");
    for (const [selector, name] of entered) {
      const input = $(selector);
      const value = (input?.value || "").trim();
      try {
        await json("/api/providers/credentials", {
          method: "POST",
          body: JSON.stringify({ name, value }),
        });
      } catch (error) {
        if (input) input.value = "";
        text("sorani-detail", error.message);
        return;
      } finally {
        // Clear immediately: a key must not linger in the DOM.
        if (input) input.value = "";
      }
    }
    await loadProviders();
    await loadSoraniSpeakers();
  }

  async function testSorani() {
    const speaker = $("#sorani-speaker-select")?.value || "";
    if (speaker) {
      try {
        await json("/api/settings", { method: "PUT", body: JSON.stringify({ sorani_speaker_id: speaker }) });
      } catch (error) { /* the reply is still worth attempting */ }
    }
    text("sorani-detail", "Speaking a Sorani test phrase…");
    try {
      const result = await json("/api/voice/speak", {
        method: "POST",
        body: JSON.stringify({ text: "سڵاو، من سامم. ئامادەم بۆ یارمەتیدانت.", language: "ckb" }),
      });
      text("sorani-detail", result.error
        ? result.error
        : `Spoken through ${result.engine}${result.speaker_id ? ` (${result.speaker_id})` : ""}.`);
    } catch (error) {
      text("sorani-detail", error.message);
    }
  }

  async function saveOpenRouterKey() {
    const input = $("#openrouter-key-input");
    const value = (input?.value || "").trim();
    if (!value) { text("provider-detail", "Enter a key first."); return; }
    text("provider-detail", "Storing the key on the backend…");
    try {
      const result = await json("/api/providers/credentials", {
        method: "POST",
        body: JSON.stringify({ name: "openrouter_api_key", value }),
      });
      // Clear immediately: the key must not linger in the DOM.
      if (input) input.value = "";
      const health = result.health || {};
      setStatus("provider-openrouter", health.status || "UNKNOWN");
      text("provider-detail", `Stored (fingerprint ${result.fingerprint}). ${health.detail || ""}`.trim());
      await loadProviders();
    } catch (error) {
      if (input) input.value = "";
      text("provider-detail", error.message);
    }
  }

  async function startOllama() {
    setStatus("provider-ollama", "STARTING");
    text("provider-detail", "Starting the local Ollama daemon…");
    try {
      const result = await json("/api/providers/ollama/start", { method: "POST" });
      setStatus("provider-ollama", result.status);
      text("provider-detail", result.detail || result.reason || "");
    } catch (error) {
      setStatus("provider-ollama", "ERROR");
      text("provider-detail", error.message);
    }
  }

  // --- Entry triggers -------------------------------------------------------

  async function loadTriggers() {
    const container = $("#trigger-list");
    try {
      const payload = await json("/api/trading/triggers");
      const triggers = payload.data?.triggers || [];
      text("trigger-count", triggers.length);
      for (const id of ["bt-trigger", "sc-trigger"]) {
        const select = document.getElementById(id);
        if (!select) continue;
        const first = select.options[0];
        select.innerHTML = "";
        if (first) select.appendChild(first);
        for (const trigger of triggers) {
          const option = document.createElement("option");
          option.value = trigger.id;
          option.textContent = `${trigger.name} (${trigger.direction})`;
          select.appendChild(option);
        }
      }
      if (!container) return;
      container.innerHTML = "";
      if (!triggers.length) { container.innerHTML = '<div class="p-empty"><p>No triggers registered.</p></div>'; return; }
      for (const trigger of triggers) {
        const row = element("div", "trade-card-body");
        row.style.padding = "8px 0";
        row.innerHTML = `<div class="level-row"><span class="level-label">${trigger.name}</span>` +
          `<span class="badge badge-subtle">${trigger.direction}</span></div>` +
          `<p class="p-note"><strong>Confirmation:</strong> ${trigger.confirmation}</p>` +
          `<p class="p-note"><strong>Invalidation:</strong> ${trigger.invalidation}</p>` +
          `<p class="p-note"><strong>Requires:</strong> ${(trigger.requirements || []).join("; ")}</p>` +
          `<p class="p-note"><strong>Timeframes:</strong> ${(trigger.preferred_timeframes || []).join(", ")} · ` +
          `<strong>Theories:</strong> ${(trigger.compatible_theories || []).join(", ")}</p>`;
        container.appendChild(row);
      }
    } catch (error) {
      if (container) container.innerHTML = `<div class="p-empty"><p>${error.message}</p></div>`;
    }
  }

  // --- Backtest -------------------------------------------------------------

  function statRow(label, value) {
    return `<div class="level-row"><span class="level-label">${label}</span><span class="level-value">${value}</span></div>`;
  }

  function equityCurve(trades) {
    if (!trades.length) return "";
    let equity = 0;
    const points = trades.map((trade, index) => {
      equity += Number(trade.r_multiple) || 0;
      return { x: index, y: equity };
    });
    const values = points.map((point) => point.y);
    const min = Math.min(0, ...values);
    const max = Math.max(0, ...values);
    const range = max - min || 1;
    const width = 320;
    const height = 90;
    const path = points.map((point, index) => {
      const x = (index / Math.max(1, points.length - 1)) * width;
      const y = height - ((point.y - min) / range) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const zero = height - ((0 - min) / range) * height;
    return `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img" aria-label="Cumulative R curve">` +
      `<line x1="0" y1="${zero.toFixed(1)}" x2="${width}" y2="${zero.toFixed(1)}" stroke="currentColor" stroke-opacity=".25" stroke-dasharray="4 4"/>` +
      `<path d="${path}" fill="none" stroke="currentColor" stroke-width="2"/></svg>`;
  }

  function renderBacktest(payload) {
    const container = $("#backtest-results");
    if (!container) return;
    const data = payload.data || {};
    // Comparison mode returns a ranking; single-trigger mode returns one result.
    if (data.ranking) {
      const rows = data.ranking.map((row) => {
        const result = data.results[row.trigger] || {};
        return `<tr><td>${row.trigger}</td><td>${result.total_setups ?? 0}</td>` +
          `<td>${((result.win_rate ?? 0) * 100).toFixed(1)}%</td>` +
          `<td>${(result.average_r ?? 0).toFixed(3)}</td>` +
          `<td>${result.profit_factor ?? "—"}</td>` +
          `<td>${(result.max_drawdown_r ?? 0).toFixed(2)}</td></tr>`;
      }).join("");
      container.innerHTML = `<p class="p-note">${data.note || ""}</p>` +
        `<table class="p-table"><thead><tr><th>Trigger</th><th>Setups</th><th>Win%</th><th>Avg R</th><th>PF</th><th>Max DD</th></tr></thead><tbody>${rows}</tbody></table>`;
      return;
    }
    const trades = data.trades || [];
    const held = trades.length ? trades.reduce((total, trade) => total + (trade.bars_held || 0), 0) / trades.length : 0;
    const sessions = Object.entries(data.by_session || {});
    const ranked = sessions.slice().sort((a, b) => (b[1].total_r ?? 0) - (a[1].total_r ?? 0));
    const stats =
      statRow("Total setups", data.total_setups ?? 0) +
      statRow("Wins", data.wins ?? 0) +
      statRow("Losses", data.losses ?? 0) +
      statRow("Breakeven", data.breakeven ?? 0) +
      statRow("Expired", data.expired ?? 0) +
      statRow("Win rate", `${((data.win_rate ?? 0) * 100).toFixed(1)}%`) +
      statRow("Loss rate", `${(100 - (data.win_rate ?? 0) * 100).toFixed(1)}%`) +
      statRow("Average R", (data.average_r ?? 0).toFixed(3)) +
      statRow("Expectancy", (data.expectancy ?? 0).toFixed(3)) +
      statRow("Total R", (data.total_r ?? 0).toFixed(2)) +
      statRow("Profit factor", data.profit_factor ?? "—") +
      statRow("Max drawdown (R)", (data.max_drawdown_r ?? 0).toFixed(2)) +
      statRow("Max consecutive wins", data.max_consecutive_wins ?? 0) +
      statRow("Max consecutive losses", data.max_consecutive_losses ?? 0) +
      statRow("Average hold (bars)", held.toFixed(1)) +
      statRow("Best session", ranked.length ? `${ranked[0][0]} (${ranked[0][1].total_r}R)` : "—") +
      statRow("Worst session", ranked.length ? `${ranked[ranked.length - 1][0]} (${ranked[ranked.length - 1][1].total_r}R)` : "—");
    const rows = trades.slice().reverse().map((trade) => `<tr>` +
      `<td>${(trade.time || "").slice(0, 16)}</td><td>${trade.direction}</td>` +
      `<td>${fmt(trade.entry)}</td><td>${fmt(trade.stop)}</td><td>${fmt(trade.target)}</td>` +
      `<td>${trade.outcome}</td><td>${(trade.r_multiple ?? 0).toFixed(2)}</td><td>${trade.bars_held}</td></tr>`).join("");
    container.innerHTML = stats +
      `<p class="p-note">${data.note || ""}</p>` + equityCurve(trades) +
      `<table class="p-table"><thead><tr><th>Time</th><th>Dir</th><th>Entry</th><th>Stop</th><th>Target</th><th>Result</th><th>R</th><th>Bars</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  async function runBacktest() {
    const container = $("#backtest-results");
    setStatus("backtest-status", "RUNNING");
    $("#bt-run-btn")?.setAttribute("disabled", "true");
    $("#bt-cancel-btn")?.removeAttribute("disabled");
    if (container) container.innerHTML = '<div class="p-empty"><p>Running on the backend engine…</p></div>';
    backtestController = new AbortController();
    const body = {
      symbol: $("#bt-symbol")?.value || "XAUUSD",
      timeframe: $("#bt-timeframe")?.value || "M15",
      count: Number($("#bt-count")?.value || 3000),
      stop_atr_multiple: Number($("#bt-stop")?.value || 1.5),
      reward_multiple: Number($("#bt-reward")?.value || 2.0),
      max_bars: Number($("#bt-maxbars")?.value || 60),
    };
    const trigger = $("#bt-trigger")?.value;
    if (trigger) body.trigger = trigger;
    try {
      const payload = await json("/api/trading/backtest", {
        method: "POST", body: JSON.stringify(body), signal: backtestController.signal,
      });
      setStatus("backtest-status", payload.status || "SUCCESS");
      renderBacktest(payload);
    } catch (error) {
      if (error.name === "AbortError") {
        setStatus("backtest-status", "CANCELLED");
        if (container) container.innerHTML = '<div class="p-empty"><p>Backtest cancelled.</p></div>';
      } else {
        setStatus("backtest-status", "FAILED");
        if (container) container.innerHTML = `<div class="p-empty"><p>${error.message}</p></div>`;
      }
    } finally {
      backtestController = null;
      $("#bt-run-btn")?.removeAttribute("disabled");
      $("#bt-cancel-btn")?.setAttribute("disabled", "true");
    }
  }

  // --- Strategy composer ----------------------------------------------------

  async function loadStrategies() {
    const container = $("#strategy-list");
    if (!container) return;
    try {
      const payload = await json("/api/trading/strategies");
      const strategies = payload.strategies || [];
      container.innerHTML = strategies.length ? "" : '<div class="p-empty"><p>No saved strategies yet.</p></div>';
      for (const strategy of strategies) {
        const row = element("div", "level-row");
        row.innerHTML = `<span class="level-label">${strategy.name} v${strategy.version}</span>` +
          `<span class="level-value">${(strategy.definition?.description || "").slice(0, 60)}</span>`;
        container.appendChild(row);
      }
    } catch (error) {
      container.innerHTML = `<div class="p-empty"><p>${error.message}</p></div>`;
    }
  }

  async function saveStrategy() {
    const list = (id) => ($(`#${id}`)?.value || "").split(",").map((item) => item.trim()).filter(Boolean);
    const body = {
      name: $("#sc-name")?.value || "",
      context: $("#sc-context")?.value || "",
      setup: $("#sc-setup")?.value || "",
      confirmation: $("#sc-confirmation")?.value || "",
      entry_trigger: $("#sc-trigger")?.value || "",
      invalidation: $("#sc-invalidation")?.value || "",
      stop: $("#sc-stop")?.value || "",
      targets: list("sc-targets"),
      timeframes: list("sc-timeframes"),
      minimum_rr: Number($("#sc-rr")?.value || 1.5),
      direction: $("#sc-direction")?.value || "BOTH",
    };
    if (!body.name) { text("composer-status", "Give the strategy a name first."); return; }
    try {
      const saved = await json("/api/trading/strategies", { method: "POST", body: JSON.stringify(body) });
      text("composer-status", `Saved ${saved.name} as version ${saved.version}.`);
      await loadStrategies();
    } catch (error) {
      text("composer-status", error.message);
    }
  }

  // --- Journal and research -------------------------------------------------

  async function loadJournal() {
    const container = $("#journal-list");
    const setups = $("#setup-list");
    const query = $("#journal-query")?.value || "";
    try {
      const [journal, saved] = await Promise.all([
        json(`/api/trading/journal?query=${encodeURIComponent(query)}`),
        json("/api/trading/setups"),
      ]);
      const entries = journal.entries || journal.journal || [];
      if (container) {
        container.innerHTML = entries.length ? "" : '<div class="p-empty"><p>No journal entries match.</p></div>';
        for (const entry of entries) {
          const row = element("div", "level-row");
          row.innerHTML = `<span class="level-label">${(entry.created_at || "").slice(0, 16)} ${entry.symbol}</span>` +
            `<span class="level-value">${entry.theory} · ${entry.payload?.result || entry.payload?.setup_state || "—"}</span>`;
          container.appendChild(row);
        }
      }
      if (setups) {
        const rows = saved.setups || [];
        setups.innerHTML = rows.length ? "" : "";
        for (const setup of rows.slice(0, 12)) {
          const row = element("div", "level-row");
          row.innerHTML = `<span class="level-label">${setup.symbol} · ${setup.theory}</span>` +
            `<span class="badge badge-subtle">${setup.state}</span>`;
          setups.appendChild(row);
        }
      }
    } catch (error) {
      if (container) container.innerHTML = `<div class="p-empty"><p>${error.message}</p></div>`;
    }
  }

  function wireExtendedPanels() {
    // Panels are hidden until their view is opened, so refresh on reveal
    // rather than showing whatever was loaded at page start.
    window.addEventListener("sam:view-changed", (event) => {
      if (event.detail?.view === "market") {
        Promise.allSettled([loadProviders(), loadSoraniSpeakers(), loadTriggers(), loadStrategies(), loadJournal()]);
      }
    });
    $("#providers-refresh-btn")?.addEventListener("click", loadProviders);
    $("#openrouter-save-btn")?.addEventListener("click", saveOpenRouterKey);
    $("#ollama-start-btn")?.addEventListener("click", startOllama);
    $("#sorani-refresh-btn")?.addEventListener("click", () => Promise.allSettled([loadProviders(), loadSoraniSpeakers()]));
    $("#sorani-save-btn")?.addEventListener("click", saveSoraniKeys);
    $("#sorani-test-btn")?.addEventListener("click", testSorani);
    $("#bt-run-btn")?.addEventListener("click", runBacktest);
    $("#bt-cancel-btn")?.addEventListener("click", () => backtestController?.abort());
    $("#sc-save-btn")?.addEventListener("click", saveStrategy);
    $("#sc-reload-btn")?.addEventListener("click", loadStrategies);
    $("#journal-refresh-btn")?.addEventListener("click", loadJournal);
    $("#journal-query")?.addEventListener("change", loadJournal);
  }

  function wire() {
    $("#fetch-mt5-btn")?.addEventListener("click", () => Promise.allSettled([loadMarketSnapshot(), loadMt5()]));
    $("#mt5-refresh-btn")?.addEventListener("click", loadMt5);
    $("#run-market-analysis-btn")?.addEventListener("click", () => analyzeMarket(false));
    $("#compare-theories-btn")?.addEventListener("click", () => analyzeMarket(true));
    $("#tv-focus-btn")?.addEventListener("click", () => tradingViewAction("focus"));
    $("#tv-symbol-btn")?.addEventListener("click", () => tradingViewAction("set_symbol", { symbol: $("#tv-symbol-input")?.value || "XAUUSD" }));
    $("#tv-timeframe-btn")?.addEventListener("click", () => tradingViewAction("set_timeframe", { timeframe: $("#tv-timeframe-input")?.value || "M1" }));
    $("#tv-capture-btn")?.addEventListener("click", () => tradingViewAction("capture"));
    $("#desktop-focus-tv-btn")?.addEventListener("click", () => tradingViewAction("focus"));
    $("#desktop-capture-btn")?.addEventListener("click", () => tradingViewAction("capture"));
    $("#emergency-stop-btn")?.addEventListener("click", emergencyStop);
    $("#monitor-setup-btn")?.addEventListener("click", async () => {
      if (!latestAnalysis) return;
      if (!latestSetupId) {
        const created = await json("/api/trading/setups", { method: "POST", body: JSON.stringify({ theory: $("#trading-theory")?.value || "default" }) });
        latestSetupId = created.setup?.id || null;
      }
      if (!latestSetupId) return;
      const result = await json(`/api/trading/setups/${encodeURIComponent(latestSetupId)}/monitor`, { method: "POST", body: JSON.stringify({ enabled: true }) });
      text("trading-action-status", `Monitoring ${result.setup?.id || latestSetupId}`);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    wire();
    wireExtendedPanels();
    Promise.allSettled([loadMarketSnapshot(), loadMt5(), loadTradingResources(), refreshTradingView(), loadDesktop(), loadToolStatus(),
      loadProviders(), loadTriggers(), loadStrategies(), loadJournal()]);
    window.setInterval(() => Promise.allSettled([refreshTradingView(), loadDesktop()]), 10_000);
  });

  window.SAMTrading = { refresh: loadMarketSnapshot, analyze: analyzeMarket, tradingViewAction, latest: () => latestAnalysis,
    tradingView: refreshTradingView, toolStatus: loadToolStatus,
    providers: loadProviders, triggers: loadTriggers, backtest: runBacktest, strategies: loadStrategies, journal: loadJournal };
})();
