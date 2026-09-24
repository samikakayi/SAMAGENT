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

  // As in app.js: English stands in when i18n.js has not loaded.
  const t = (key, fallback) => (window.SAM_I18N ? window.SAM_I18N.t(key, fallback) : fallback);

  let timer = null;
  let current = "OFF";
  // Whether the button would stop something: a live state whose listener runs.
  let listening = false;

  // What stops the listener hearing the phrase, when "waiting for it" is claimed.
  //
  // The state is what the loop last decided; the wake block describes the
  // listener thread and its speech model as they are now. On the 03:16:58
  // start that thread died on a numpy import race seconds after it began and
  // the model never loaded, yet the state stayed WAKE_LISTENING and this line
  // said "Waiting for Hey SAM…" to a microphone nobody was reading, until a
  // restart. So when the two disagree, the wake block is believed.
  //
  // Only the waiting line speaks for the listener. During a turn the command
  // is captured by the voice service itself, so "Listening…" or "Thinking…"
  // stays true whatever happened to the listener; the turn ends back here.
  function listenerFault(state) {
    const wake = state?.wake;
    if (!wake || state?.state !== "WAKE_LISTENING") return null;
    const errors = [
      [t("handsFree.listenerError", "Listener"), wake.error],
      [t("handsFree.modelError", "Speech model"), wake.detector?.error],
    ].filter(([, error]) => String(error || "").trim())
      .map(([source, error]) => `${source}: ${String(error).trim()}`);
    const stopped = wake.running === false;
    return stopped || errors.length ? { stopped, errors } : null;
  }

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
    const fault = listenerFault(state);
    // A listener that has stopped has closed the microphone, whatever the
    // state says; the button then starts it again rather than stopping nothing.
    const live = LIVE.has(current) && !fault?.stopped;
    listening = live;
    // The indicator is required whenever the microphone is open: nobody
    // should have to wonder whether SAM is listening.
    bar.hidden = !(live || fault || current === "ERROR");
    const phrase = state?.wake_phrase || "Hey SAM";
    // The speech model takes a while to load the first time, and until it has,
    // saying the phrase does nothing. Claiming to be waiting for it would send
    // somebody off to repeat themselves at a microphone that cannot hear yet.
    const warming = current === "WAKE_LISTENING" && state?.wake && state.wake.ready === false;
    label.textContent = fault
      ? (fault.stopped
        ? t("handsFree.stopped", "Not listening for {phrase}: the listener stopped")
        : t("handsFree.modelFailed", "{phrase} may not be heard: the speech model reported an error")
      ).replace("{phrase}", phrase)
      : warming
        ? "Getting ready to listen…"
        : current === "WAKE_LISTENING"
          ? `Waiting for ${phrase}…`
          : (state?.detail || WORDING[current] || current);
    // The error itself is for whoever has to fix it, so it waits in the tooltip.
    label.title = fault
      ? [...fault.errors, fault.stopped
        ? t("handsFree.stoppedHint", "Press Start to listen again. If it stops again, restart SAM.")
        : t("handsFree.modelHint", "If this does not clear, restart SAM.")].join("\n")
      : "";
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
      render(await json(`/api/voice/handsfree/${listening ? "stop" : "start"}`, { method: "POST" }));
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
