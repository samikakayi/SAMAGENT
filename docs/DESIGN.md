# SAM 2 — design (source of truth for the build)

Date: 2026-09-24. Owner: a Sorani Kurdish speaker in Iraq (UTC+3), non-expert, Windows 11 Home laptop
(Ryzen AI 9 HX 370, 31 GB RAM, iGPU only, 2880x1800), A50 X USB headset, MetaTrader 5 (demo, XAUUSD),
TradingView Desktop 3.4.1 (MSIX), VS Code + Cursor installed.

## 0. What the user asked (verbatim intent)
"Delete SAM. Keep only its features and the trading strategies and theories that were written. Keep every
API/key that was set up. Build everything from scratch — very smart, not simple like Hamawmin. It must talk
with me naturally (Sorani), understand me easily, do everything on my computer, keep a memory of my trading
strategies and theories, analyse my TradingView DESKTOP charts intelligently and DRAW lines on them, and
monitor market, price, volume and more."
Reference product: "Hamawmin" (TikTok @ahmed.dev8): a top-centre floating pill with avatar + live caption,
always-on natural Kurdish voice, opens VS Code and builds a website by voice. SAM 2 must match that
experience and be much smarter and more professional.

Why v1 failed (measured, see reports/audit-latency.json): 13–20 s to first sound (serial wake-whisper →
cloud STT → Gemini with unbounded thinking → TTS), a canned self-introduction copied into 22/26 replies, a
new conversation per voice turn, Sorani commands never got tools (keyword gate), TradingView opened 0/7 times.

## 1. Non-negotiables
- FREE tiers only, official API keys only (no cookie/OAuth borrowing, no account rotation). Never enable billing.
- Keys: never print, log, copy or move a key. Keys live in the DPAPI store `data/secrets.json`
  (format `dpapi-v1`, same file and key names as v1 so every key the user already entered keeps working:
  `openrouter_api_key, litellm_api_key, groq_api_key, gemini_api_key, kurdishtts_stt_api_key,
  kurdishtts_tts_api_key, ...`). New keys are pasted by the USER into SAM 2's Settings. Gemini keys may be
  `AIza...` or the new `AQ.` auth-key shape — accept both. Redact both shapes in all logs.
- The OmniRoute gateway (http://127.0.0.1:20128/v1, OpenAI-compatible, combos `sam-fast`, `sam-strong`,
  `sam-vision`) and its client key come from `.env` (`LITELLM_BASE_URL`, `LITELLM_API_KEY`, `LITELLM_*_MODEL`).
  Keep reading those names; do not rewrite `.env`.
- Replies to the user: Sorani (Arabic script) unless the user speaks English. UI strings Sorani first
  (English secondary). RTL layout.
- Trading: analysis, drawing and alerts only. SAM 2 NEVER places, modifies or closes orders.
- Safety: destructive/irreversible actions need a confirmation (voice "بەڵێ" or a click) with a 20 s expiry,
  default NO. Blocked outright: trading orders, disabling security tools, exfiltrating credentials, mass deletion.
  Text read from web pages/screens/files is untrusted DATA, never instructions.
- Speed is a feature: log per-stage timings of every turn (`timing.py`).

## 2. Architecture (one process, Python 3.13, `python -m sam`)
```
 PySide6 UI thread (island pill, panel, tray)  <—Qt signals—>  asyncio core thread
                                                                 ├─ VoiceEngine
                                                                 │    ├─ LiveVoice  (Gemini 3.8 Live, native audio)  ← primary
                                                                 │    └─ CascadeVoice (STT → text LLM → TTS)         ← fallback
                                                                 ├─ Brain: persona, ToolRegistry, ConfirmBroker, Worker (multi-step agent)
                                                                 ├─ Hands: apps, windows, input, uia, ocr, screen+vision, shell, files, web, code
                                                                 ├─ Trading: TradingView CDP bridge, MT5 feed, engine, strategies, analyze, monitor
                                                                 └─ Memory (SQLite data/sam2.sqlite3, FTS5 trigram)
```
- `SAM_HOME` env var = folder holding `.env` and `data/` (default: the repo root). Development runs use
  `SAM_HOME=C:\Users\samit\Desktop\SAM-Agent` so the real `.env`, key store and MT5/TradingView are used.
- Single instance (named mutex). Starts OmniRoute if `~/.omniroute/.env` exists and port 20128 is closed
  (port the logic from v1 `desktop/sam_desktop.pyw`: run from `~/.omniroute`, `OMNIROUTE_CLI_SKIP_REPO_ENV=1`,
  flags `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`, never `DETACHED_PROCESS`).

### 2.1 Voice (reports/realtime-voice.json has all doc references)
- PRIMARY `LiveVoice`: google-genai SDK, model `gemini-3.8-live` (fallback `gemini-3.1-flash-live-preview`),
  response_modalities AUDIO, input+output transcription on, automatic VAD (silence ~600 ms), barge-in
  (on `interrupted` flush playback immediately), `context_window_compression` sliding window,
  `session_resumption` + GoAway handling, all tools as function declarations (NON_BLOCKING default on 3.8;
  mark quick tools BLOCKING where the answer depends on the result), voice `Kore` default (setting).
  Mic 16 kHz int16 20–40 ms chunks; speaker 24 kHz on ONE continuous OutputStream (never per-chunk play/wait —
  v1 measured that as choppy). Watchdog: no audio within 5 s of end-of-speech → speak via cascade and mark
  Live degraded for this session.
- Sorani in Live is UNPROVEN (Live lists only "Kurdish (ku)"). The system instruction must say
  "RESPOND IN CENTRAL KURDISH (SORANI), ARABIC SCRIPT". Settings has "Voice engine: Automatic / Live / Cascade".
  Automatic = Live if a Gemini key exists and the last self-test passed, else Cascade.
- FALLBACK `CascadeVoice`: local VAD endpointing (webrtcvad or energy+Silero-ONNX if cheap) → STT
  (KurdishTTS STT dialect=sorani, key in store; fallback Gemini Flash-Lite audio transcription) → streaming
  text LLM (Groq `openai/gpt-oss-20b` first for chat ~0.8 s, OmniRoute `sam-fast`, Gemini direct; set
  reasoning effort minimal/low and max tokens >= 4096) → first sentence to TTS immediately
  (`gemini-3.8-flash-lite-tts`, Central Kurdish, streaming L16 24 kHz; KurdishTTS TTS as alternative — its
  free plan is only 20k chars/month) → same continuous speaker. Same tools via the text model's tool calling.
- Triggers: global hotkey (RegisterHotKey, default Ctrl+Alt+Space, setting) toggles listening; clicking the
  island toggles; "conversation window": session stays open while talking and sleeps after N s of silence
  (default 45 s, setting "always listening" keeps it open). No always-on whisper wake word.
- Self-test (Settings button + first run after a Gemini key is saved): Gemini TTS synthesises 3 fixed Sorani
  sentences → fed to Live as audio → compare input transcription (normalised CER) and check the reply script
  is Arabic-script Sorani, not Kurmanji/Latin. Store result; drives "Automatic".

### 2.2 Brain
- Persona (`brain/persona.py`): SAM is a very capable, calm, professional assistant and trading analyst who
  speaks natural everyday Sorani, warm and respectful, with light humour only when the user jokes. Voice style:
  1–3 short sentences, no Markdown/lists/emojis, numbers spoken naturally, never introduces itself unless
  asked, never repeats an earlier answer or phrase, acknowledges an action in a few words and then does it,
  reports results honestly (never claims success without the tool result), asks one clarifying question only
  when truly needed. Includes: current local time (Asia/Baghdad), user facts from memory (short), the
  strategy-card index (id + one line each), what tools exist.
- `ToolRegistry`: decorator `@tool(name, description, params schema, risk="safe|confirm|blocked",
  confirm_text_ckb=..., blocking=bool)` → produces Gemini FunctionDeclarations AND OpenAI-style tool schemas
  from one definition; handler is async; returns a compact JSON-able dict `{ok, summary_ckb, data}`.
  ~25 tools total, no keyword gating, always sent.
- `ConfirmBroker`: `await confirm(question_ckb, detail)` → speaks/shows the question, resolves on voice
  yes/no (Sorani + English yes/no word lists from the Live input transcription) or UI button; 20 s → NO.
- `Worker` (`delegate_task(goal)`): multi-step agent loop on a text model with the same tools (plan → act →
  verify with screen/UIA/OCR evidence → next), max ~25 steps, progress events to the island, final Sorani
  summary spoken by Live (send as client content / tool response, scheduling WHEN_IDLE). Model ladder:
  OmniRoute `sam-strong` → Gemini direct `gemini-3.5-flash-lite` (≈500 RPD) → Groq `qwen/qwen3.8-27b`
  (vision) / `openai/gpt-oss-120b`. `gemini-3.8-flash` (~20 RPD) only for the hardest single calls.
  Model IDs are settings; verify availability at startup (list models) and count requests per day.
- Memory (`brain/memory.py`): tables `facts` (user facts/preferences, FTS5 trigram), `conversations` +
  `turns` (transcripts), `notes`. Tools `remember`, `recall`. After a conversation sleeps, one cheap LLM call
  extracts new durable facts (dedup). Retrieval measured: FTS5 trigram beats local embeddings on Sorani
  (69% vs 56% top-1) — use FTS5 (reports/trading-intelligence.json).

### 2.3 Hands (reports/computer-control.json)
Layered, fastest first: (1) skills — `open_app` (Start-menu AppsFolder index cached + Sorani alias table,
e.g. ترەیدینگ ڤیو/تریدینگ ڤیو/ترێدینگ → TradingView, کرۆم → Chrome, ڤی ئێس کۆد → VS Code, fuzzy match;
launch by AUMID), windows (port v1 `windows_control.py`: list/focus/min/max/close/snap with verification),
media/volume, open_url / web_search (Gemini grounding if key, else DuckDuckGo HTML), files in known folders,
`run_powershell` (read-only allowlist auto, rest confirm; port v1 `policy.py` classifier + secret masking),
`build_project` (Hamawmin demo: create folder under ~/SAM Projects, write files via the Worker, open in
VS Code `code` CLI — note `code` on PATH resolves to Cursor; prefer
`%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd` — run/preview in the browser);
(2) UI Automation via the `uiautomation` library (MIT): numbered snapshot of the foreground window's
interactive controls, click/type/scroll by number or name; type long/Unicode text via clipboard paste;
(3) Windows OCR find/click text (port v1 `trading/ocr.py` dedicated MTA worker; use modular `winrt-*`
packages on 3.13); (4) vision fallback `screen_act(goal)` — window crop ≤1440 px JPEG → Gemini Flash-Lite
(bounding boxes / computer-use) → click, ≤12 steps, daily budget counter; Groq qwen3.8-27b as second.
Per-monitor DPI aware. Blank MT5 account area / password fields before sending screenshots.

### 2.4 Trading (reports/trading-intelligence.json)
- VERIFIED TODAY on this PC: TradingView Desktop can be started with a local Chrome DevTools port through
  Windows app activation (IApplicationActivationManager::ActivateApplication with AUMID
  `TradingView.Desktop_n534cwy3pjxzj!TradingView.Desktop`, args `--remote-debugging-port=9222`; launching the
  WindowsApps exe directly exits immediately). The chart page target url contains `/chart/`.
  `window.TradingViewApi.activeChart()` exposes `symbol(), resolution(), setSymbol, setResolution,
  getVisibleRange, getAllShapes, getAllStudies, createShape, createMultipointShape, removeEntity, exportData,
  getShapeById(id).getPoints()`; bars via
  `TradingViewApi._activeChartWidgetWV.value()._chartWidget.model().mainSeries().bars()` (`size(), last(),
  valueAt(i)` → `[time, o, h, l, c, v]`). Measured: a horizontal line + trend line created in 180 ms at exact
  price/time and removed cleanly. Scripts: scratchpad `tvcdp/probe.py`, `tvcdp/draw_test.py`.
- `trading/tradingview.py` (CDP bridge, `websockets`): ensure_running_with_port (if TradingView runs WITHOUT
  the port: confirm, then restart it with the port), connect/reconnect, chart_state, set_symbol,
  set_timeframe, bars(n), draw(kind ∈ horizontal_line | horizontal_ray | trend_line | rectangle | fib_retracement
  | text | arrow_up | arrow_down | long_position | short_position, points, text, color, style), list/clear
  SAM-owned drawings (ids persisted in `drawings` table; never delete user drawings), chart screenshot via
  `Page.captureScreenshot` (no window focus needed). Bind 127.0.0.1 only; start the port only while SAM runs.
  Tell the user once (README + Settings note): this is an unofficial local automation of their own app.
- Data: MT5 (broker feed) for background monitoring and when TradingView is closed; the chart's own bars when
  analysing/drawing on the chart the user sees (levels then align exactly). MT5 fixes from the audit:
  broker offset = round((tick.time − time.time())/900)*900 per connection (UTC+3 now), ONE persistent
  `mt5.initialize()` session with reconnect, symbol map (XAUUSD / XAUUSD.m.e / TVC:GOLD / OANDA:XAUUSD …).
- Engine: PORT v1 `sam_backend/trading/{types,indicators,patterns,analysis,analyst,geometry,research,replay}.py`
  and their tests into `sam/trading/engine/`, with the four audit fixes (time offset; drawing plan reads real
  entry/stop/tp keys; monitor transitions are spoken; real order blocks returned) and proper Sorani summaries.
- Theories: keep v1's 40-theory catalogue (`registry.py builtin_theories`) as `trading/theories.py` knowledge
  (description, components, limitations) — the user's "theories that were written".
- Strategy cards (`trading/strategies.py`): the user pastes/dictates a strategy in Sorani or English → one LLM
  call with a JSON schema → card {title_ckb, title_en, markets, timeframes{bias,setup,entry}, sessions,
  rules[{kind, text_ckb, text_en, check:{predicate, params}|null}], risk, source_text verbatim, version,
  status draft|active} → SAM reads back a 2-sentence Sorani summary, asks only for missing pieces, user says
  yes → active. ~30 predicates on engine outputs (trend_is, swept, mss_or_bos, in_fvg, in_order_block, in_ote,
  near_level, rsi_divergence, volume_spike, session_is, candle_pattern, ema_cross, rr_at_least, …).
  Rules without a predicate are judged by the LLM with the chart screenshot.
- `analyze_market(symbol?, timeframes?, strategy?)`: engine numbers (chart bars or MT5) + card predicates +
  optional one vision call on the chart screenshot → verdict WAIT / NO_TRADE / SETUP (never "buy now" as an
  order) with levels → draws zones/levels/entry-stop-targets on the chart → short spoken Sorani summary,
  full text in the panel.
- Monitor (`trading/monitor.py`): background watcher (no LLM on the hot path): price crosses, zone touches,
  candle-close conditions, volume spikes (k× average of n bars; tick volume noted), strategy-state changes;
  alerts spoken (Live `send_client_content` or TTS) + Windows toast + panel list; persisted in `alerts`.
- Import from v1 (`migrate_v1.py`, read-only on `data/sam.sqlite3`): custom_theories → strategy cards with
  status `archived` and a note "imported from SAM v1 (created by automated tests)"; trading_setups/journal/
  context → `v1_archive` table. Nothing from v1 is lost; nothing test-made becomes active.

### 2.5 UI (PySide6)
- Island: frameless, translucent, always-on-top, tool window (no taskbar), shows without activating, top
  centre of the primary screen, DPI aware. Pill ~ 380×58 that grows to show a caption line. Left: animated orb
  avatar (SAM monogram; reacts to mic/speaker level), name "SAM", status word (ئامادە / گوێ دەگرم / بیردەکەمەوە /
  قسە دەکەم / کار دەکەم / هەڵە), live one-line caption (latest transcript, RTL, elided). Colours per state.
  Click = toggle listening; double-click / right-click menu = open panel, mute, settings, quit. Drag to move
  (position remembered). A small progress line for Worker tasks. Confirmation cards appear under the pill.
- Panel (normal window, dark, RTL, Sorani font e.g. "Vazirmatn"/"Noto Naskh Arabic"/"Segoe UI"): tabs
  گفتوگۆ (history + text input — typing works exactly like speaking, same brain), ستراتیژییەکان (cards:
  add by paste, view, activate, archive, versions), چاودێری (alerts/monitors), چالاکی (activity/timings log),
  ڕێکخستنەکان (keys write-only fields with status dots and "Test" buttons — Gemini with link
  https://aistudio.google.com/apikey, Groq, KurdishTTS STT/TTS, OmniRoute status; voice engine choice +
  self-test; voice name; hotkey; conversation timeout; TradingView connect; MT5 status; privacy note:
  Gemini free tier may use prompts to improve Google products).
- Tray icon (QSystemTrayIcon) with Open / Mute / Restart / Quit.

## 3. Repository layout (branch `sam2/rebuild`, worktree `C:\Users\samit\Desktop\SAM2-build`)
All v1 code is deleted in this branch (git history keeps it; tag `v1-final`). New tree:
```
sam/ __init__.py __main__.py app.py config.py secrets.py db.py events.py timing.py bridge.py(Qt↔asyncio)
sam/voice/ audio.py live.py cascade.py stt.py tts.py hotkey.py engine.py selftest.py
sam/brain/ persona.py tools.py confirm.py llm.py worker.py memory.py
sam/hands/ apps.py aliases.py windows.py input.py uia.py ocr.py screen.py vision.py shell.py policy.py files.py web.py code.py system.py tools.py
sam/trading/ tradingview.py mt5.py theories.py strategies.py predicates.py analyze.py monitor.py tools.py engine/*
sam/ui/ island.py panel.py tray.py theme.py strings.py
sam/migrate_v1.py
scripts/ install.ps1 uninstall.ps1 dev.ps1   SAM.pyw (launcher: pythonw, single instance, OmniRoute start)
tests/ (pytest, no network, no speakers, Qt offscreen)   acceptance/ (live scripts, run on this PC)
docs/DESIGN.md (this file)   README.md (Sorani first, then English)   requirements.txt   pyproject.toml   pytest.ini   .env.example
```
Kept untouched: `.env`, `data/` (gitignored), `.freebuff/`, `.gitignore` (update entries), `workspace/.gitkeep`.

## 4. Acceptance (phase 1 = this build)
1. `python -m pytest` green; Qt offscreen smoke test builds island + panel.
2. `python -m sam` starts < 3 s to island visible; RAM < 400 MB idle; no console window from SAM.pyw.
3. Typed Sorani "ترەیدینگ ڤیو بکەرەوە" → TradingView focused/opened with CDP port within 3 s of the model
   reply; "گۆڵد لەسەر ١٥ خولەک پیشان بدە" → symbol/timeframe set; "هێڵی پشتگیری و بەرگری بکێشە" → SAM
   drawings appear at engine levels and "بیانسڕەوە" removes only SAM's.
4. Analysis on live XAUUSD returns in < 5 s (engine < 1.5 s) with no false "stale data" (offset fix).
5. Alerts: a price-cross alert fires on a synthetic feed test and on live data; spoken text is Sorani.
6. open_app finds Chrome, Edge, TradingView, Telegram, MT5, VS Code, Notepad by English and Sorani names.
7. Voice: cascade path works end-to-end with the keys already stored (KurdishTTS/OmniRoute/Groq) on a recorded
   Sorani WAV (no speaker output in tests); Live path unit-tested with a fake session; with a Gemini key the
   self-test reports CER and time-to-first-audio. Target TTFA: Live ≤ 1.5 s, cascade ≤ 4.5 s (logged).
8. No key ever appears in logs, DB, UI, test output or exceptions (grep test over logs).
