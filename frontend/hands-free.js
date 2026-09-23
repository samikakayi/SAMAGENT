(() => {
  "use strict";

  // Hands-free voice: one status line and one button. Its own module so
  // panels.js does not grow another section, following routing-panel.js.
  //
  // The page never touches the microphone for wake detection -- that happens
  // in the backend, on this machine. This file only asks what state the loop
  // is in and renders it, so a stale page cannot make SAM listen or stop.

  const $ = (id) => document.getElementById(id);

  // What each state says to somebody who is not reading the code.
  const WORDING = {
    OFF: "Hands-free is off",
    WAKE_LISTENING: "Waiting for the wake phrase…",
    WAKE_DETECTED: "Yes?",
    LISTENING: "Listening…",
    TRANSCRIBING: "Getting that down…",
    THINKING: "Thinking…",
    SPEAKING: "Speaking…",
    ERROR: "Voice unavailable",
  };
  const LIVE = new Set(["WAKE_LISTENING", "WAKE_DETECTED", "LISTENING", "TRANSCRIBING",
    "THINKING", "SPEAKING"]);

  let timer = null;
  let current = "OFF";

  async function json(path, options = {}) {
    const response = await fetch(path, {
      cache: "no-store",
      headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}) },
      ...options,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload?.detail?.error || payload?.detail || `Request failed (${response.status})`);
    return payload;
  }

  function render(state) {
    const bar = $("hands-free-bar");
    const label = $("hands-free-status");
    const toggle = $("hands-free-toggle");
    const dot = $("hands-free-dot");
    if (!bar || !label) return;

    current = state?.state || "OFF";
    const live = LIVE.has(current);
    // The indicator is required whenever the microphone is open: nobody
    // should have to wonder whether SAM is listening.
    bar.hidden = !(live || current === "ERROR");
    const phrase = state?.wake_phrase || "Hey SAM";
    label.textContent = current === "WAKE_LISTENING"
      ? `Waiting for ${phrase}…`
      : (state?.detail || WORDING[current] || current);
    if (dot) dot.style.opacity = live ? "1" : "0.35";
    if (toggle) {
      toggle.textContent = live ? "Stop" : "Start";
      toggle.disabled = !state?.enabled && !live;
    }
  }

  async function refresh() {
    try {
      render(await json("/api/voice/handsfree"));
    } catch {
      // A failed poll says nothing about the microphone, so the badge is
      // retired rather than left claiming SAM is still listening.
      render({ state: "ERROR", detail: "Hands-free state is unavailable." });
    }
  }

  async function toggle() {
    const button = $("hands-free-toggle");
    if (button) button.disabled = true;
    try {
      const live = LIVE.has(current);
      render(await json(`/api/voice/handsfree/${live ? "stop" : "start"}`, { method: "POST" }));
    } catch (error) {
      render({ state: "ERROR", detail: error.message, enabled: true });
    } finally {
      if (button) button.disabled = false;
    }
  }

  function start(intervalMs = 2000) {
    stop();
    refresh();
    timer = setInterval(refresh, intervalMs);
    return timer;
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("hands-free-toggle")?.addEventListener("click", toggle);
    const slider = $("settings-hands-free-continuation");
    const output = $("hands-free-continuation-output");
    if (slider && output) {
      const show = () => { output.textContent = `${slider.value}s`; };
      slider.addEventListener("input", show);
      show();
    }
    start();
  });

  window.SAMHandsFree = { refresh, render, toggle, start, stop, WORDING };
})();
