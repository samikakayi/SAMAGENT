# SAM 2 — module contracts

Written by the foundation stage (2026-09-24). Seven builders work in parallel, each owning one
package. **This file is the interface.** If you need something that is not here, make the smallest
compatible addition, keep it backwards compatible, and report it. Design and rationale:
`docs/DESIGN.md`. Research with measurements: the lead's scratchpad `sam2/reports/*.json`.

---------------------------------------------------------------------------------------------------

## 0. Rules for every builder

**Ownership.** All builders share ONE worktree, so edit only the files your package owns
(section 3). Foundation-owned (import, never edit; report needed changes): `sam/__init__.py,
__main__.py, app.py, config.py, secrets.py, db.py, events.py, timing.py, bridge.py, textnorm.py,
winapp.py`, `sam/brain/__init__.py, tools.py, confirm.py, llm.py, llm_backends.py`,
`sam/trading/__init__.py, common.py`, `tests/conftest.py`, `tests/test_core_*.py`, `pytest.ini`,
`requirements.txt`, `.gitignore`, `docs/`. Put shared test helpers for your package in
`tests/<package>_helpers.py`. Need a new dependency? Install it into `.venv` and list it (with the
exact `==` version) in your final report; the lead adds it to `requirements.txt`. New settings keys
go through `app.config.register_defaults` in your `register`.

**Entry module.** `sam.app.PACKAGES` imports these modules in this order and calls `register(app)`,
then (on the core loop) `await start(app)`, and on shutdown `await stop(app)` in reverse order:

| order | module | owner | sets |
| --- | --- | --- | --- |
| 1 | `sam.brain.memory` | brain | `app.memory` |
| 2 | `sam.brain.persona` | brain | `app.persona` |
| 3 | `sam.hands` (package `__init__`) | hands | `app.hands` |
| 4 | `sam.trading.chart_tools` | chart bridge | `app.trading.tv` |
| 5 | `sam.trading.tools` | engine | `app.trading.mt5/.engine/.theories/.strategies/.monitor` |
| 6 | `sam.brain.worker` | brain | `app.worker` |
| 7 | `sam.brain.conversation` | brain | `app.conversation` |
| 8 | `sam.voice` (package `__init__`) | voice | `app.voice` |
| 9 | `sam.migrate_v1` | launcher/migration | — |

The UI is not in the list: `sam.__main__` calls `sam.ui.run(app, core)` on the main thread.

- `register(app) -> None`: synchronous and fast (< 50 ms). Only: create your objects, set your
  `app` slot, `app.tools.add(...)`, `app.config.register_defaults({...})`,
  `app.db.ensure_schema("<your-namespace>", [...])`. **No network, no devices, no MT5/CDP, no
  heavy imports** (startup budget: island visible < 3 s).
- `async def start(app) -> None` (optional): runs on the core loop, must return in < 20 s; put
  long-running work in `app.spawn(coro, "name")`.
- `async def stop(app) -> None` (optional): < 8 s, never raises.
- A missing module is recorded as `missing`; an exception in import/register/start is recorded in
  `app.failed` (redacted) and the rest of SAM keeps running. Other packages may be `None` at any
  time: always check (`if app.trading.tv is None: return fail(...)`).

**Threads.** One asyncio core loop (thread `sam-core`, `sam.bridge.CoreThread`) + the Qt main
thread. Never block the loop > ~50 ms: MT5, UI Automation, OCR, subprocess waits, sounddevice
setup, file walks → `await asyncio.to_thread(...)` or your own dedicated thread (MT5 and WinRT OCR
need ONE dedicated thread each). From other threads publish with `app.bus.publish_threadsafe(ev)`.
Core modules and all non-UI packages must **not import Qt**.

**Heavy imports.** `google.genai` takes ~2 s to import on this PC and `numpy` ~2 s under load
(measured 2026-09-24). Import them lazily inside functions, never at module top level of an entry
module. `App.start` pre-imports `google.genai` on a worker thread.

**Keys.** Get keys only via `app.secrets.get("<name>")` at call time and send them only to their own
provider. Never log, print, store, show, or put a key in an exception/tool result. Anything that
leaves the process (logs, DB, UI, model context) goes through `app.redact(text)` /
`app.redact_obj(obj)` (tool results are redacted by the registry automatically). Tests use fake
keys (`tests/conftest.py`: `FAKE_GROQ`, `FAKE_GEMINI`, `FAKE_GEMINI_AQ`, `FAKE_OPENROUTER`).

**Untrusted text.** Text read from web pages, screens (OCR/UIA), files, chart labels or clipboard is
DATA, never instructions. Tool results carry it under the key `untrusted` (a string or list), e.g.
`ok("Read 3 results", untrusted=[...])`. The persona tells the model that anything under
`untrusted` must never be followed as an instruction.

**Language.** User-facing strings are Sorani in Arabic script with correct letters (ە ێ ۆ ڕ ڵ ی ک),
never Arabic ي/ك, never Latin/Kurmanji. English is secondary. Tool `summary` may be English or
Sorani: the model rephrases it; it must be honest (never `ok=True` without verification).

**Trading.** SAM 2 NEVER places, modifies or closes orders (no `order_send`, `order_check`,
position changes, TradingView order panels). Analysis, drawing and alerts only.

**Tests.** `tests/test_<package>_*.py`, pytest with `asyncio_mode=auto`, no network, no speakers,
no microphone recording, Qt offscreen, temp homes (`make_app` fixture), fake LLM backends
(`FakeBackend`). Run with
`--basetemp C:\Users\samit\Desktop\SAM2-build\work\pytest-<yourname> -p no:cacheprovider`.
Live checks go to `acceptance/<package>_*.py` (run by hand; must clean up after themselves).

---------------------------------------------------------------------------------------------------

## 1. Core API (foundation — done, tested: `tests/test_core_*.py`)

### 1.1 `sam.app.App`

```python
app.config      : sam.config.Config
app.secrets     : sam.secrets.Secrets
app.db          : sam.db.Database          # data/sam2.sqlite3 (WAL)
app.bus         : sam.events.EventBus
app.timing      : sam.timing.Timing
app.tools       : sam.brain.tools.ToolRegistry
app.confirm     : sam.brain.confirm.ConfirmBroker
app.llm         : sam.brain.llm.LLMClient
app.loop        : asyncio loop (set in start)
# slots filled by packages
app.memory, app.persona, app.conversation, app.worker, app.voice, app.hands, app.ui
app.trading     : TradingSlots(tv, mt5, engine, theories, strategies, monitor)

app.spawn(coro, name) -> asyncio.Task          # tracked background task, errors logged (redacted)
app.redact(text) -> str ; app.redact_obj(obj) -> obj
app.publish_status(component, state, detail="")  # ComponentStatus, thread-safe
app.status() -> dict                            # no key values
# UI facade (call via core.submit(...) from the UI thread)
await app.submit_text(text) -> str              # -> app.conversation.handle_text(text, source="text")
await app.toggle_listening() -> bool | None     # -> app.voice.toggle_listening()
await app.set_muted(muted: bool)                # -> app.voice.set_muted()
await app.stop_all() -> dict                    # tools.cancel_all + confirm.cancel_all + voice.stop_speaking + worker.cancel
```

Core registers the tool **`stop_all`** `{}` (safe, BLOCKING).

### 1.2 `sam.config.Config`

```python
config.home, config.data_dir, config.db_path, config.v1_db_path, config.workspace_dir, config.log_dir
config.env_value(NAME) -> str | None      # process env, then .env (never copied into os.environ)
config.get(key, default=None) -> Any       # DB value (JSON) else DEFAULTS[key] else default; returns a copy
config.set(key, value)                     # persists + publishes SettingsChanged(key, value)
config.reset(key); config.all() -> dict
config.register_defaults({key: default})   # your package's keys (namespaced "<package>.<name>")
```

`SAM_HOME` = folder with `.env` and `data/` (dev: `C:\Users\samit\Desktop\SAM-Agent`). Logs:
`%LOCALAPPDATA%\SAM2\logs\sam2.log` (never inside SAM_HOME). Known keys and defaults are in
`sam/config.py::build_defaults` (llm.*, providers.*, confirm.*, voice.*, conversation.*, worker.*,
memory.*, hands.*, trading.*, ui.*, migrate.*). Use those names.

### 1.3 `sam.secrets`

```python
app.secrets.get(name) -> str | None     # env -> .env -> DPAPI store data/secrets.json (v1 file, dpapi-v1)
app.secrets.has(name) -> bool
app.secrets.set(name, value) -> {"name","stored","fingerprint"}   # ONLY from the Settings UI (user paste)
app.secrets.clear(name) -> bool
app.secrets.status() -> {name: {"configured", "source", "fingerprint"}}
sam.secrets.redact(text) ; redact_obj(obj) ; SUPPORTED_KEYS ; KEY_PATTERNS
```

Key names: `openrouter_api_key, openai_api_key, litellm_api_key, groq_api_key, gemini_api_key
(AIza… or AQ.…), n8n_api_key, kurdishtts_stt_api_key, kurdishtts_tts_api_key,
google_stt_credentials_path`. OmniRoute's client key comes from `.env` `LITELLM_API_KEY`.
**Current state of this PC (2026-09-24, presence only):** configured = openrouter, litellm (.env),
groq, kurdishtts_stt, kurdishtts_tts, n8n. **No Gemini key yet** → Live voice, Gemini TTS and
Gemini vision are unavailable until the user pastes one in Settings; everything must degrade
gracefully (Automatic voice = Cascade; vision ladder falls to Groq qwen / OmniRoute sam-vision).

### 1.4 `sam.db.Database`

```python
db.execute(sql, params) ; db.query(sql, params) -> list[dict] ; db.query_one(...) -> dict | None
db.scalar(...) ; db.insert(table, row_dict, or_ignore=False) -> rowid   # dict/list values stored as JSON
with db.transaction(): ...                    # BEGIN IMMEDIATE ... COMMIT
await db.run(fn, *args)                       # fn(db, *args) in a worker thread
db.ensure_schema(namespace, [(version, sql_or_callable), ...]) -> version
db.log_activity(kind, name, ok=, summary=, detail=, duration_ms=, source=)   # pass redacted text
db.bump_usage(provider, model, kind="text", requests=1, errors=0, rate_limited=0, tokens_in=0, tokens_out=0, units=0.0)
db.usage_for(provider, model, kind=None) -> dict   # today (Gemini: Pacific day; others: Iraq day)
sam.db.fts_match_expr(text) -> str | None          # safe MATCH for trigram FTS (terms >= 3 chars)
```

Thread-safe (one connection + RLock); small queries may run directly on the loop. All timestamps are
REAL unix seconds UTC. Core tables (namespace `core`, v1) — **column lists are the contract**:

| table | writer | columns |
| --- | --- | --- |
| settings | config | key PK, value JSON, updated_at |
| facts (+ `facts_fts` trigram on text_norm, tags; triggers keep it in sync) | memory | id, text, text_norm (=normalize_ckb(text), UNIQUE where deleted=0), kind fact/preference/person/project/trading, tags, source user/extracted/import, confidence, created_at, updated_at, last_used_at, use_count, deleted |
| conversations | conversation | id, started_at, ended_at, source voice/text/mixed, title, summary, facts_extracted |
| turns | conversation (single writer) | id, conversation_id, at, role user/assistant/system/tool, text, source live/cascade/text/worker/system, meta JSON |
| notes (+ `notes_fts` on title, body_norm, tags) | memory | id, title, body, body_norm, kind note/theory/journal/doc, tags, source, created_at, updated_at |
| strategy_cards (+ `strategy_fts` on search_text) | engine | id TEXT slug PK, title_ckb, title_en, status draft/active/archived, version, card JSON, summary_ckb, source_text (verbatim), search_text (normalize_ckb of titles+summary+rules), note, created_at, updated_at |
| strategy_card_versions | engine | card_id, version (PK both), card JSON, reason, created_at |
| alerts | engine monitor | id, kind price_cross/zone_touch/candle_close/volume_spike/strategy_state, symbol, timeframe, params JSON, note, strategy_id, status active/fired/cancelled/expired, repeat, created_at, expires_at, fired_at, fire_count, last_value, last_text_ckb |
| drawings | chart bridge | id, tv_id (TradingView entity id), kind, symbol, timeframe, points JSON `[{time, price}]`, text, tag, created_at, removed_at |
| v1_archive | migrate_v1 | id, source_table, source_id (UNIQUE pair), payload JSON, note, imported_at |
| usage_counters | llm, voice, hands.vision | day, provider, model, kind text/vision/live/tts/stt/computer_use, requests, errors, rate_limited, tokens_in, tokens_out, units (chars/seconds) |
| timings | timing | id, at, turn_id, kind, stage, ms, extra JSON |
| activity | tools registry, anyone | id, at, kind tool/confirm/alert/worker/error/system/voice, name, ok, summary, detail JSON, duration_ms, source |

Need another table (e.g. an analysis journal)? `app.db.ensure_schema("trading", [(1, "CREATE TABLE
IF NOT EXISTS analyses (...)")])` in your `register`.

### 1.5 `sam.events` (frozen dataclasses; every event has `at: float`)

| event | fields | published by | consumed by |
| --- | --- | --- | --- |
| `VoiceState` | state idle/listening/thinking/speaking/working/error/sleeping/muted, engine live/cascade/"", detail | voice (conversation/worker may publish thinking/working when typed) | ui, conversation (sleeping → on_sleep) |
| `Caption` | text, role user/assistant/system, final | voice, conversation | ui island |
| `Transcript` | role, text, source live/cascade/text/worker/system, turn_id, conversation_id | voice (spoken turns), conversation (typed turns), worker (final summary) | **conversation persists turns (single writer)**, ui panel |
| `UserText` | text | (optional) ui | — (UI normally calls `app.submit_text`) |
| `ToolStarted` | call_id, name, args (redacted), source | tools registry | ui activity/island |
| `ToolFinished` | call_id, name, ok, summary, duration_ms, source | tools registry | ui |
| `ConfirmRequest` | confirm_id, question_ckb, detail, tool_name, expires_at | confirm broker | ui (card with بەڵێ / نەخێر), voice (make the question heard) |
| `ConfirmResult` | confirm_id, approved, via voice/click/timeout/cancel | confirm broker | ui (hide card), voice |
| `SpeakRequest` | text_ckb, source alert/worker/confirm/system, interrupt | monitor, worker, anyone | voice |
| `Alert` | alert_id, kind, symbol, text_ckb, price, timeframe | engine monitor | ui (panel list + tray balloon), voice (via SpeakRequest) |
| `WorkerProgress` | task_id, step, max_steps, text_ckb, done, ok | worker, long tools (`ctx.progress`) | ui island progress line |
| `Error` | where, message_ckb, detail (redacted) | anyone | ui |
| `LevelMeter` | source mic/speaker, level 0..1 | voice (≤ 25 Hz; UiAdapter throttles) | ui orb |
| `ComponentStatus` | component, state ok/degraded/down/unconfigured/unknown, detail | anyone (llm publishes provider failures) | ui settings dots, tray |
| `SettingsChanged` | key, value | config.set | anyone |

`bus.subscribe(EventType | (T1, T2) | None, callback) -> unsubscribe`; callback sync or async, runs
on the core loop; `bus.publish(ev)` on the loop; `bus.publish_threadsafe(ev)` elsewhere;
`await bus.wait_for(EventType, predicate, timeout)`.

### 1.6 `sam.timing`

```python
turn = app.timing.turn("live" | "cascade" | "text" | "worker" | "analysis")   # context manager
turn.mark("first_audio")          # ms since turn start
turn.add("stt", 420.0)            # measured duration
with turn.stage("tts_first_audio"): ...
turn.finish() -> {stage: ms}      # persists all stages + "total"
app.timing.record(stage, ms, kind=..., turn_id=..., **extra) ; with app.timing.measure(stage): ...
```

Stage names: `end_of_speech, stt, llm_first_token, llm_total, tts_first_audio, first_audio (TTFA),
live_connect, tool:<name>, confirm_wait, worker_step, analysis_engine, analysis_total, tv_cdp,
mt5_fetch, startup:<phase>`. Pass `turn=` to `app.llm.chat/stream` to get llm stages recorded.

### 1.7 `sam.brain.tools` — ToolRegistry

```python
from sam.brain.tools import tool, ToolContext, ok, fail

@tool(name, *, description: str, params: dict | None = None, risk="safe"|"confirm"|"blocked",
      description_ckb="", confirm_text_ckb: str | Callable[[dict], str] | None = None,
      blocking: bool = True, timeout_s: float = 60.0,
      classify: Callable[[dict], Risk | tuple[Risk, str | None]] | None = None,
      examples_ckb: Iterable[str] = ())
async def handler(ctx: ToolContext, **params) -> dict: ...      # return ok(summary, **data) / fail(summary, **data)

app.tools.add(handler, owner="<package>")        # or app.tools.add_from(module, owner=...)
app.tools.openai_tools(names=None) -> list[dict]                   # {"type":"function","function":{name, description, parameters}}
app.tools.gemini_declarations(names=None, live=True) -> list[types.FunctionDeclaration]
app.tools.gemini_tools(names=None, live=True) -> list[types.Tool]
app.tools.describe_for_prompt(names=None) -> str
await app.tools.dispatch(name, args: dict | json_str, *, source="live"|"cascade"|"text"|"worker"|"ui", call_id=None) -> {"ok", "summary", "data"}
app.tools.cancel_all(except_call=None) -> int ; app.tools.running() -> list[dict]
ctx.app, ctx.call_id, ctx.source, ctx.cancel (asyncio.Event), ctx.cancelled
await ctx.confirm(question_ckb, detail="") -> bool       # mid-tool confirmation (e.g. screen_act danger click)
ctx.progress(step, max_steps, text_ckb, done=False, ok_=None)   # WorkerProgress for the island
```

- `params` is a JSON schema `{"type":"object","properties":{...},"required":[...]}` (plain JSON
  schema: type/description/enum/items/properties/required; keep it simple — Gemini and Groq both
  accept it). Models often send numbers as strings: the registry coerces basic types.
- `blocking=True` → Live `Behavior.BLOCKING` (the model waits for the answer: quick tools whose
  result it must speak). `blocking=False` → `NON_BLOCKING` (slow tools: the voice engine answers
  with `scheduling="WHEN_IDLE"`). `behavior` is only sent to the Live API (`live=True`); the text
  API rejects it.
- Risk is decided by code: `risk` static or `classify(args)` (its verdict replaces the static one;
  exceptions fail safe to `confirm`). `confirm` → the registry asks the ConfirmBroker **before**
  calling the handler (question: `confirm_text_ckb` formatted with the args, or the classifier's
  text); declined/expired → `{"ok": false, "summary": "The user did not approve…", "data":
  {"declined": true}}`. `blocked` → never runs, `data: {"blocked": true}`.
- dispatch never raises: unknown tool, bad args, exceptions (`"<tool> failed: Type: msg"`,
  redacted), timeout (`data.timeout`), stop_all (`data.cancelled`) all become `ok=false` results.
  Results are redacted, `summary` ≤ 600 chars, `data` JSON ≤ 6000 chars (else
  `{"truncated": true, "preview": ...}`). Every call records `timings` `tool:<name>`, an `activity`
  row, and `ToolStarted`/`ToolFinished` events.

### 1.8 `sam.brain.confirm.ConfirmBroker`

```python
await app.confirm.confirm(question_ckb, detail="", *, tool_name="", timeout_s=None) -> bool   # 20 s -> False
app.confirm.offer_transcript(text) -> bool     # voice: call with EVERY final user transcript first; True = consumed as yes/no
app.confirm.resolve(confirm_id | None, approved, via="click") -> bool   # UI click; thread-safe; None = latest
app.confirm.pending() -> list[dict] ; app.confirm.has_pending ; app.confirm.cancel_all() -> int
sam.brain.confirm.classify_answer(text) -> True | False | None     # Sorani+English yes/no; "no" wins; > 8 words = None
```

There is deliberately **no model tool that approves** a pending confirmation (prompt-injection
risk). Only the user's own transcript or a click can say yes.

### 1.9 `sam.brain.llm.LLMClient` (text/vision models; Live voice is the voice package's own client)

```python
resp = await app.llm.chat(messages, *, ladder="chat"|"strong"|"vision"|"extract"|"hard"|"sorani"|"provider:model"|[refs],
                          tools=app.tools.openai_tools(...), tool_choice=None, max_tokens=None (>= 4096),
                          temperature=None, reasoning=None ("low" default; minimal|low|medium|high),
                          json_schema=None, timeout_s=None, turn=None) -> LLMResponse
async for chunk in app.llm.stream(messages, ...same minus json_schema...):   # LLMChunk
    chunk.kind == "text" -> chunk.text ; "tool_call" -> chunk.tool_call ; "done" -> chunk.response
resp.text, resp.tool_calls: list[ToolCall(id, name, arguments: dict)], resp.model_ref ("groq:openai/gpt-oss-20b"),
resp.usage {"tokens_in","tokens_out"}, resp.ttft_ms, resp.total_ms, resp.assistant_message(), resp.json()
await app.llm.list_models(provider) ; await app.llm.verify_models() ; await app.llm.test_provider(provider) -> {"ok","status","latency_ms",...}
LLMError(kind, message, provider, model, status, retry_after, attempts)   # kind "exhausted" when every rung failed
```

- Messages are OpenAI chat format. Images: content parts
  `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`. Tool loop: append
  `resp.assistant_message()` then one `{"role": "tool", "tool_call_id": call.id, "name": call.name,
  "content": json.dumps(result, ensure_ascii=False)}` per call. **Always append
  `assistant_message()` unchanged** — for Gemini it carries `_gemini_content` (thought signatures
  required by Gemini 3 for multi-turn tool use); `_`-prefixed keys are stripped for other providers.
- Ladders (settings `llm.ladder.*`, refs `provider:model`, providers omniroute/groq/openrouter/gemini):
  `chat` = groq:openai/gpt-oss-20b → omniroute:sam-fast → gemini:gemini-3.5-flash-lite;
  `strong` = omniroute:sam-strong → gemini:gemini-3.5-flash-lite → groq:openai/gpt-oss-120b;
  `vision` = gemini:gemini-3.5-flash-lite → groq:qwen/qwen3.8-27b → omniroute:sam-vision;
  `extract` = gemini:gemini-3.5-flash-lite → omniroute:sam-fast → groq:openai/gpt-oss-20b;
  `hard` = gemini:gemini-3.8-flash → omniroute:sam-strong → gemini:gemini-3.5-flash-lite;
  `sorani` = omniroute:sam-fast → gemini:gemini-3.5-flash-lite → groq:openai/gpt-oss-120b.
  Unconfigured providers are skipped; 429 → rung cools down (Retry-After or 60 s) and the next rung
  runs at once; 5xx/timeout/network → ONE retry on the same rung; 401/403 → provider cools 10 min;
  a 400 about reasoning/response_format → one retry without it. Daily caps (`llm.daily_caps`) skip a
  rung before Google refuses it. Every request bumps `usage_counters`.
- **Measured live on this PC today** (Sorani "ترەیدینگ ڤیو بکەرەوە" + an `open_app` tool): Groq
  gpt-oss-20b streaming → `open_app({"name": "TradingView"})` in 1.41 s; OmniRoute sam-fast →
  the same call in 5.83 s (201 output tokens even with reasoning_effort low). Full two-step loop
  (tool call → tool result → reply): Groq answered in 0.57 s but with broken Sorani ("بەکارهێنانی
  Trading View بەسەرە…"); OmniRoute sam-fast answered in 5.0 s with natural Sorani ("تریدینگ ڤیو
  بە سەرکەوتوویی کرایەوە…"). Hence the extra ladder `sorani` = omniroute:sam-fast →
  gemini:gemini-3.5-flash-lite → groq:openai/gpt-oss-120b for user-facing wording. One sample
  each: the brain builder must measure (speed vs Sorani quality) before choosing per step. All
  ladder model ids except Gemini's (no key yet) are listed by their providers (groq:
  openai/gpt-oss-20b, openai/gpt-oss-120b, qwen/qwen3.8-27b; omniroute: sam-fast, sam-strong,
  sam-vision). OpenAI-style tool calls keep provider extras (e.g. Gemini `extra_content` thought
  signatures through OmniRoute) inside `assistant_message()`; a `"name"` on tool messages is
  optional (used by the Gemini backend, stripped for OpenAI-style APIs).

### 1.10 Shared helpers

- `sam.textnorm.normalize_ckb(text, lower=True, strip_punct=False)`, `words(text)`,
  `is_arabic_script(text)` — use before comparing/searching Sorani text.
- `sam.winapp.activate_aumid(aumid, arguments="") -> pid` (IApplicationActivationManager; call via
  `asyncio.to_thread`), `acquire_single_instance()`, `signal_show()`,
  `watch_show_requests(callback)`, `set_dpi_awareness()`.
- `sam.trading.common`: `Bar` (time UTC s, open, high, low, close, volume), `ChartPoint` (time |
  bars_ago, price), `DRAWING_KINDS`, `DRAWING_POINTS`, `COLORS` (support, resistance, entry, stop,
  target, zone, info, liquidity), `TIMEFRAMES` (M1 M5 M15 M30 H1 H4 D1 W1 MN1),
  `normalize_timeframe("١٥ خولەک") == "M15"`, `to_tv_resolution`, `from_tv_resolution`,
  `canonical_symbol("زێڕ" | "OANDA:XAUUSD" | "TVC:GOLD") == "XAUUSD"`, `SYMBOL_ALIASES`.
- `sam.bridge.CoreThread` (`start`, `submit(coro) -> Future`, `run_sync(coro, timeout)`,
  `call_soon(fn, *a)`, `stop`) and `UiAdapter(bus, sink, types=None).attach()`.

---------------------------------------------------------------------------------------------------

## 2. Tool catalogue (no keyword gating; the text/cascade path sends a core tier + more_tools, see section 6)

| tool | owner | params (JSON schema properties; * = required) | risk | blocking |
| --- | --- | --- | --- | --- |
| stop_all | core | — | safe | yes |
| remember | brain.memory | text* string (the fact in the user's words), kind enum fact/preference/person/project/trading | safe | yes |
| recall | brain.memory | query* string, limit integer 1–10 | safe | yes |
| delegate_task | brain.worker | goal* string (complete goal with all context), context string | safe | **no** |
| open_app | hands | name* string (English or Sorani, e.g. "ترەیدینگ ڤیو"), args string | safe | yes |
| window_control | hands | action* enum list/focus/minimize/maximize/restore/close/snap_left/snap_right, target string (title or app) | classify: close → confirm, else safe | yes |
| type_text | hands | text* string, target string (number from last screen_look or control name), press_enter boolean | classify: press_enter in a messaging/email app → confirm | yes |
| press_keys | hands | keys* string ("ctrl+s", "alt+tab", "volume_up", "play_pause", "next_track"), repeat integer | classify: alt+f4, ctrl+w, shift+delete, ctrl+shift+delete, win+l → confirm | yes |
| click | hands | target* string (number from last screen_look, control name, or visible text), button enum left/right/double, window string | classify: label matches danger words (Delete, Send, Buy, Sell, Pay, Submit, Uninstall, Remove, سڕینەوە, بیسڕەوە, ناردن, بنێرە, کڕین, فرۆشتن, پارەدان) → confirm | yes |
| screen_look | hands | window string, mode enum controls/text/describe (default controls), query string | safe | yes |
| screen_act | hands | goal* string, window string, max_steps integer ≤ 12 | safe (each risky step → `ctx.confirm`) | **no** |
| run_powershell | hands | command* string, timeout_s integer | classify: read-only allowlist → safe; credential dumping, disabling security tools, mass deletion, format, registry hive export → blocked; else confirm | yes |
| files | hands | action* enum list/read/write/append/copy/move/delete/open/search/reveal, path* string, content string, dest string, pattern string | classify: list/read/open/search/reveal → safe; write/append/copy/move inside `hands.projects_dir` or `workspace/` → safe, elsewhere → confirm; delete → confirm; recursive/mass delete, system dirs, credential files → blocked | yes |
| open_url | hands | url* string (http/https only; other schemes blocked) | safe | yes |
| web_search | hands | query* string, open_in_browser boolean | safe | yes |
| build_project | hands | description* string, name string, kind enum website/python/other | safe | **no** |
| tv_open | trading chart | — | safe (restart of a port-less TradingView → `ctx.confirm`) | yes |
| tv_set_chart | trading chart | symbol string (any alias), timeframe string (any form) | safe | yes |
| chart_state | trading chart | — | safe | yes |
| draw_on_chart | trading chart | items* array of {kind* enum DRAWING_KINDS, points* array of {price* number, time integer, bars_ago integer}, text string, color string (hex or semantic name)}, tag string | safe | yes |
| clear_my_drawings | trading chart | tag string | safe | yes |
| get_price | trading engine | symbol string (default setting trading.default_symbol) | safe | yes |
| analyze_market | trading engine | symbol string, timeframes array of string, strategy_id string, draw enum none/levels/full (default full), vision boolean (default true) | safe | **no** |
| set_alert | trading engine | kind* enum price_cross/zone_touch/candle_close/volume_spike/strategy_state, symbol string, level number, low number, high number, direction enum up/down/any, timeframe string, k number, n integer, strategy_id string, repeat boolean, note string | safe | yes |
| list_alerts | trading engine | status enum active/fired/all | safe | yes |
| cancel_alert | trading engine | alert_id* string (an id or "all") | safe | yes |
| strategy_save | trading engine | text* string (the strategy verbatim, Sorani or English), strategy_id string (update), status enum draft/active | safe | yes (timeout 45 s) |
| strategy_list | trading engine | status enum draft/active/archived/all | safe | yes |
| strategy_get | trading engine | strategy_id* string | safe | yes |

Give every tool 1–3 `examples_ckb` (natural spoken Sorani), e.g. open_app: "کرۆم بکەرەوە",
"ترەیدینگ ڤیو بکەرەوە"; tv_set_chart: "گۆڵد لەسەر ١٥ خولەک پیشان بدە"; analyze_market:
"زێڕ شی بکەرەوە بە ستراتیژییەکەم"; draw_on_chart: "هێڵی پشتگیری و بەرگری بکێشە";
clear_my_drawings: "هێڵەکانت بسڕەوە"; set_alert: "ئەگەر زێڕ گەیشتە ٢٧٠٠ ئاگادارم بکەرەوە".

---------------------------------------------------------------------------------------------------

## 3. Package contracts

### 3.1 Voice — `sam/voice/` (audio.py live.py cascade.py stt.py tts.py hotkey.py engine.py selftest.py, `__init__.py`)

`sam/voice/__init__.py`: `register(app)` sets `app.voice = VoiceEngine(app)`; `start(app)` →
`await app.voice.start()`; `stop(app)` → `await app.voice.stop()`.

```python
class VoiceEngine:
    def __init__(self, app) -> None
    async def start(self) -> None          # hotkey (RegisterHotKey, setting voice.hotkey) + subscriptions; mic NOT opened until listening
    async def stop(self) -> None
    state: str                              # VoiceState names
    engine_name: str                        # "live" | "cascade" | ""
    async def toggle_listening(self) -> bool
    async def start_listening(self) -> None
    async def stop_listening(self) -> None  # conversation window closes -> VoiceState("sleeping")
    async def set_muted(self, muted: bool) -> None   # mic closed; alerts are still spoken
    async def speak(self, text_ckb: str, *, interrupt: bool = False, source: str = "system") -> None
    async def stop_speaking(self) -> None   # flush playback at once (barge-in, stop_all)
    async def run_selftest(self) -> dict    # {"ok","cer","script_ok","ttfa_ms","reply_sample","at"}; also stored in setting voice.selftest
    def status(self) -> dict                # engine, state, live_degraded, devices, last TTFA
```

- Engine choice (see section 6: auto = Live only after a PASSING self-test): setting `voice.engine` auto/live/cascade; auto = Live if
  `app.secrets.has("gemini_api_key")` and `voice.selftest.ok` else Cascade. **No Gemini key exists
  today → Cascade is the working default; Live must be unit-tested with a fake session.**
- Live: `google-genai` `client.aio.live.connect(model=voice.live_model, fallback
  voice.live_fallback_model)`, AUDIO modality, input+output transcription, automatic VAD
  (`voice.silence_ms`), barge-in (on `interrupted` flush), `context_window_compression`
  sliding window, `session_resumption` + GoAway, system instruction
  `app.persona.system_instruction("voice")`, tools `app.tools.gemini_tools(live=True)`. Tool calls:
  `result = await app.tools.dispatch(fc.name, fc.args or {}, source="live", call_id=fc.id)` then
  `FunctionResponse(id=fc.id, name=fc.name, response=result)`; for NON_BLOCKING tools add
  `scheduling=WHEN_IDLE`. Watchdog: no audio `voice.watchdog_s` (5 s) after end of speech →
  speak via cascade and mark Live degraded for the session. Declarations carry
  `parameters_json_schema` (+ `behavior`); the SDK's Live converter passes them through unchanged
  (google-genai 2.25.0 `_live_converters.py`), but this is unverified against the live service
  (no Gemini key yet): if a Live setup is rejected over the schema, report it — the foundation will
  add a `types.Schema` conversion in `gemini_declarations`.
- Cascade: local VAD (webrtcvad) endpointing → STT (`voice.stt_provider`: KurdishTTS dialect
  sorani with `kurdishtts_stt_api_key`; fallback Gemini `voice.stt_fallback_model` if key) →
  `async for delta in app.conversation.respond_stream(text, source="cascade", turn=turn)` →
  first sentence to TTS immediately (`voice.tts_provider`: gemini `voice.tts_model` if key, else
  KurdishTTS — its free plan is 20k chars/month: count `units` in usage_counters and warn) →
  one continuous 24 kHz OutputStream (never per-chunk play/wait: v1 measured that as choppy).
- Every final user transcript: first `if app.confirm.offer_transcript(text): <treat as answer>`;
  then publish `Transcript(role="user", source=engine)`. Assistant final text →
  `Transcript(role="assistant")`. Partials → `Caption`. Levels → `LevelMeter` (≤ 25 Hz).
- Consumes `SpeakRequest` (speak; `interrupt` only for urgent alerts), `ConfirmRequest` (make the
  question heard: TTS in cascade; in Live send a client-content text turn asking the model to say
  exactly the question, or TTS when a turn is running), `SettingsChanged` for `voice.*`.
- Conversation window: stay open while talking; after `voice.conversation_timeout_s` (45 s) of
  silence → `stop_listening()` → `VoiceState("sleeping")` (unless `voice.always_listening`).
- Timings per turn via `app.timing.turn("live"|"cascade")`: end_of_speech, stt, llm_first_token,
  tts_first_audio, first_audio (TTFA targets: Live ≤ 1.5 s, cascade ≤ 4.5 s). Usage: Live seconds
  (`kind="live"`), TTS chars (`"tts"`), STT seconds (`"stt"`).
- Tests: injectable audio sink/source (no speakers, no recording), fake Live session, fake STT/TTS.
  Opening an input device briefly to list it is fine; never save audio.
- Owns settings `voice.*` (defaults already in config). Registers no tools.

### 3.2 Brain — `sam/brain/memory.py persona.py conversation.py worker.py` (core `tools/confirm/llm` are foundation's)

**memory.py** — `register(app)`: `app.memory = Memory(app)`; tools `remember`, `recall`.

```python
class Memory:
    def remember(self, text, *, kind="fact", tags="", source="user", confidence=1.0) -> dict  # {"id","created"} dedup on normalize_ckb
    def forget(self, fact_id: int) -> bool
    def recall(self, query, *, limit=5, kinds=None) -> list[dict]      # FTS5 trigram (fts_match_expr) + LIKE fallback
    def facts_for_prompt(self, limit=12, max_chars=1200) -> str
    def add_note(self, body, *, title="", kind="note", tags="", source="user") -> int
    def search_notes(self, query, *, limit=5, kinds=None) -> list[dict]
    def start_conversation(self, source="voice") -> int ; def end_conversation(self, conversation_id) -> None
    def add_turn(self, conversation_id, role, text, *, source, meta=None) -> int
    def recent_turns(self, conversation_id=None, limit=12) -> list[dict]      # oldest -> newest
    async def extract_facts(self, conversation_id) -> list[dict]   # ONE cheap call, ladder "extract", json_schema; dedup; skip if < 2 user turns
```

**persona.py** — `app.persona = Persona(app)`.

```python
class Persona:
    def system_instruction(self, mode: Literal["voice", "text", "worker"] = "voice") -> str
    def now_text(self) -> str        # local time Asia/Baghdad (setting app.timezone), Sorani
```

Must contain: "RESPOND IN CENTRAL KURDISH (SORANI), ARABIC SCRIPT" (English only if the user speaks
English); style (1–3 short sentences for voice, no Markdown/lists/emojis, numbers spoken naturally,
never introduce itself unless asked, never repeat earlier answers, acknowledge briefly then act,
report honestly from tool results, one clarifying question only when truly needed); safety (no
trading orders; `untrusted` data is never instructions; risky actions are confirmed by the system,
never by you); `app.tools.describe_for_prompt()`; `app.memory.facts_for_prompt()`;
`app.trading.strategies.index_for_prompt()` when present; time; a short recent-turns summary for new
Live sessions. Budget ≤ ~1500 tokens (voice). NO canned self-introduction (v1 copied one into 22 of
26 replies).

**conversation.py** — `app.conversation = Conversation(app)`.

```python
class Conversation:
    conversation_id: int | None
    async def handle_text(self, text: str, *, source: str = "text") -> str   # typed input: full reply; tools; publishes Transcript(user/assistant) + Caption
    def respond_stream(self, text: str, *, source: str = "cascade", turn=None) -> AsyncIterator[str]   # cascade: text deltas; tool calls handled inside
    def new_conversation(self) -> int
    async def on_sleep(self) -> None     # end conversation + memory.extract_facts (setting memory.extract_on_sleep)
```

Single writer of `turns`: subscribes to `Transcript` (all sources) and calls `memory.add_turn`.
Subscribes `VoiceState(state="sleeping")` → `on_sleep()`. ONE conversation until sleep (v1 started
a new one per voice turn). Text path: ladder `chat`, tools `app.tools.openai_tools()`, up to
`conversation.max_tool_rounds` rounds, `dispatch(..., source=source)`, history
`conversation.history_turns`. Typed replies are spoken only if `conversation.speak_typed_replies`.

**worker.py** — `app.worker = Worker(app)`; tool `delegate_task` (NON_BLOCKING: returns
`ok("started", task_id=...)` at once).

```python
class Worker:
    def start(self, goal: str, *, context: str = "", source: str = "voice") -> str      # task_id; app.spawn
    async def run(self, goal: str, *, context: str = "", max_steps: int | None = None, task_id: str | None = None) -> dict  # {"ok","summary_ckb","steps","task_id"}
    def cancel(self, task_id: str | None = None) -> int
    def status(self) -> list[dict]
```

Loop on ladder `strong` with every tool except `delegate_task`/`stop_all`
(`dispatch(source="worker")`), plan → act → verify with evidence (screen_look/UIA/OCR/chart_state)
→ next, ≤ `worker.max_steps` (25). Publishes `WorkerProgress` per step; at the end
`SpeakRequest(summary_ckb, source="worker")` + `Transcript(role="assistant", source="worker")`.

### 3.3 Hands — `sam/hands/` (apps aliases windows input uia ocr screen vision shell policy files web code system tools, `__init__.py`)

`register(app)`: `app.hands = Hands(app)`, tools per section 2. `start(app)`: build/refresh the
Start-menu app index in the background (`app.spawn`, ~1.4 s measured).

```python
class Hands:                     # app.hands
    apps: AppIndex               # await resolve(name) -> AppEntry | None ; await launch(name_or_entry, args="") -> dict ; entries() -> list
    windows: Windows             # await list() -> list[WindowInfo] ; await find(query) ; await focus(query) -> bool ; minimize/maximize/restore/close/snap
    uia: Uia                     # await snapshot(hwnd=None) -> list[Control] (numbered) ; await click(number|name) ; await type(number|name, text)
    ocr: Ocr                     # await read(hwnd=None, region=None) -> list[{"text","rect"}] ; await find_text(text, hwnd=None)
    screen: Screen               # await capture(hwnd=None, *, max_side=1440, fmt="jpeg", blank_rects=()) -> bytes
    vision: Vision               # await describe(image: bytes, question: str) -> str ; await act(goal, hwnd=None, max_steps=12) -> dict (budgeted)
    policy: Policy               # classify_powershell(cmd) -> (risk, reason) ; classify_path(path, action) -> (risk, reason)
```

- `AppEntry(name, aumid, path, aliases)`; `WindowInfo(hwnd, title, process, pid, rect, minimized,
  visible)`; `Control(number, name, role, rect, enabled)`.
- open_app: Start-menu AppsFolder index (cached) + Sorani alias table (ترەیدینگ ڤیو / تریدینگ ڤیو /
  ترێدینگ → TradingView, کرۆم → Chrome, ئێج → Edge, تێلێگرام → Telegram, مێتاترەیدەر → MT5,
  ڤی ئێس کۆد → VS Code, نۆتپاد → Notepad) + rapidfuzz; launch via `sam.winapp.activate_aumid`.
  **If the resolved AUMID equals setting `trading.tv_aumid` and `app.trading.tv` exists, call
  `await app.trading.tv.ensure_running()` instead** (TradingView must get its CDP port). Acceptance
  6: finds Chrome, Edge, TradingView, Telegram, MT5, VS Code, Notepad by English and Sorani names.
- Verify every action cheaply (window title, UIA re-read, OCR) before `ok=True`.
- Unicode/long text: clipboard paste. Keyboard shortcuts must work under a Kurdish/Arabic layout
  (port v1 `trading/desktop_input.py` virtual-key approach). Per-monitor DPI aware coordinates.
- run_powershell / files: port v1 `policy.py` classifier + secret masking; child processes get the
  normal user environment (do NOT strip APPDATA etc.) but never the `.env` values (they are not in
  `os.environ`). Output → `app.redact`.
- Screenshots sent to a model: window crop ≤ 1440 px JPEG, blank password fields and the MT5
  account area; budget counter `hands.vision_daily_budget` (usage kind `computer_use`/`vision`).
  Vision ladder `vision`; Gemini computer use only with a Gemini key.
- build_project: folder under `hands.projects_dir` (~/SAM Projects), files written via the Worker,
  opened with `hands.vscode_path` (`code` on PATH is Cursor), previewed in the browser.
- web_search: Gemini grounding if key, else DuckDuckGo HTML; results under `untrusted`.
- Owns settings `hands.*`.

### 3.4 Trading chart bridge — `sam/trading/tradingview.py`, `sam/trading/chart_tools.py`

`chart_tools.register(app)`: `app.trading.tv = TradingViewBridge(app)`; tools tv_open, tv_set_chart,
chart_state, draw_on_chart, clear_my_drawings. `stop(app)`: `await app.trading.tv.close()` (closes
the CDP socket only; never closes TradingView).

```python
class TradingViewBridge:
    def __init__(self, app, *, port: int | None = None)          # port = setting trading.tv_port (9222)
    async def ensure_running(self, *, allow_restart: bool = False, confirm=None) -> dict
        # {"ok", "state": "connected"|"started"|"restarted"|"needs_restart"|"not_installed"|"failed", "detail"}
        # not running -> activate_aumid(trading.tv_aumid, "--remote-debugging-port=<port>") and wait for /json/list;
        # running WITHOUT the port -> needs_restart unless allow_restart (then confirm(...) first when given)
    async def connect(self) -> bool ; connected: bool
    async def chart_state(self) -> dict   # {"symbol","canonical","timeframe","resolution","visible_range":{"from","to"},
                                          #  "last_bar": Bar, "bar_count", "studies": [...], "my_drawings", "all_drawings"}
    async def set_symbol(self, symbol: str) -> dict        # any alias via canonical_symbol; verified by reading back
    async def set_timeframe(self, timeframe: str) -> dict  # normalize_timeframe -> to_tv_resolution; verified
    async def bars(self, count: int = 500) -> list[Bar]    # the chart's own bars, oldest -> newest, UTC
    async def draw(self, kind, points: list[ChartPoint], *, text="", color=None, style=None, tag="") -> dict  # {"ok","id","db_id"}
    async def draw_many(self, items: list[dict], *, tag="") -> dict   # {"ok","drawn","ids","errors"}
    async def my_drawings(self, *, symbol: str | None = None) -> list[dict]
    async def clear_my_drawings(self, *, tag: str | None = None) -> int
    async def screenshot(self, *, fmt="jpeg", max_width=1440) -> bytes    # Page.captureScreenshot, no focus needed
    async def evaluate(self, js: str, *, await_promise=False, timeout_s=10) -> Any
    async def close(self) -> None
```

- Verified API (lead, 2026-09-24; scripts `scratchpad/tvcdp/probe.py`, `draw_test.py`): chart page
  target url contains `/chart/`; `window.TradingViewApi.activeChart()` → `symbol(), resolution(),
  setSymbol, setResolution, getVisibleRange, getAllShapes, getAllStudies, createShape,
  createMultipointShape, removeEntity, getShapeById(id).getPoints()`; bars via
  `TradingViewApi._activeChartWidgetWV.value()._chartWidget.model().mainSeries().bars()`
  (`size(), last(), valueAt(i)` → `[time, o, h, l, c, v]`). Horizontal + trend line drawn in 180 ms
  and removed cleanly.
- `points[].time` missing → last bar time; `bars_ago` → time of that bar. Colors: `COLORS` names or
  hex. Persist every created id in `drawings`; `clear_my_drawings` removes ONLY ids from that table
  (never user drawings) and sets `removed_at`.
- 127.0.0.1 only; reconnect on socket loss; `ComponentStatus("tradingview", ...)`; timings `tv_cdp`.
- Live checks may draw but must remove every drawing they create and restore symbol/timeframe.
- Tell the user once (Settings note/README, launcher builder): unofficial local automation of
  their own app.

### 3.5 Trading engine / strategies / monitor — `sam/trading/tools.py` (entry), `mt5.py theories.py strategies.py predicates.py analyze.py monitor.py engine/*`

`tools.register(app)`: sets `app.trading.mt5 = MT5Feed(app)`, `.engine = Engine(app)`, `.theories =
THEORIES`, `.strategies = StrategyStore(app)`, `.monitor = Monitor(app)`; tools per section 2.
`start(app)`: `app.spawn(mt5.connect())` then `monitor.start()`. `stop(app)`: monitor.stop, mt5.close.

```python
class MT5Feed:                       # ONE persistent mt5.initialize() on a dedicated thread, reconnect on error
    async def connect(self) -> bool  # computes broker_offset_s = round((tick.time - time.time()) / 900) * 900
    async def status(self) -> dict   # {"connected", "broker_offset_s", "server_time_ok", "symbols": [...]} (no account numbers)
    async def resolve_symbol(self, symbol: str) -> str | None       # canonical/alias/TV -> broker name (XAUUSD, XAUUSD.m.e, ...)
    async def bars(self, symbol: str, timeframe: str, count: int = 500) -> list[Bar]   # UTC (offset removed)
    async def tick(self, symbol: str) -> dict   # {"symbol","bid","ask","last","spread","time"(UTC),"volume"}
    async def close(self) -> None
class Engine:                        # port of v1 trading/{types,indicators,patterns,analysis,analyst,geometry,research,replay}
    async def analyze(self, symbol: str, timeframes: list[str], *, bars_by_tf: dict[str, list[Bar]] | None = None,
                      strategy: dict | None = None) -> dict       # report (below); CPU work in a thread
async def analyze_market(app, symbol=None, timeframes=None, strategy_id=None, draw="full", vision=True) -> dict  # analyze.py
class StrategyStore:
    async def ingest(self, text: str, *, strategy_id: str | None = None) -> dict  # ONE llm call (ladder "extract", json_schema) -> draft card + "readback_ckb" + "missing"
    def save(self, card: dict, *, reason: str = "") -> dict      # new version row each save
    def get(self, strategy_id) -> dict | None ; def list(self, status=None) -> list[dict]
    def set_status(self, strategy_id, status) -> dict ; def versions(self, strategy_id) -> list[dict]
    def index_for_prompt(self, max_cards: int = 50) -> str        # "id — title_ckb: one line" for ACTIVE cards
    def search(self, query: str, limit: int = 5) -> list[dict]
class Monitor:                       # no LLM on the hot path; interval trading.monitor_interval_s
    async def start(self) -> None ; async def stop(self) -> None
    def add(self, spec: dict) -> dict ; def list(self, status: str = "active") -> list[dict] ; def cancel(self, alert_id: int | str) -> int
```

- Report (analyze / analyze_market) keys: `symbol, timeframes, price, data_source
  ("tradingview"|"mt5"), verdict ("WAIT"|"NO_TRADE"|"SETUP" — never "buy now"), direction
  ("long"|"short"|None), entry, stop, targets [..], rr, levels [{price, kind support|resistance, tf,
  strength}], zones [{low, high, kind fvg|order_block|supply|demand|ote, tf}], order_blocks [...]
  (real ones, audit fix), trend {tf: up|down|range}, strategy {id, rules [{text_ckb, passed, how
  predicate|vision|llm}]}, checks [...], summary_ckb (proper Sorani), engine_ms, total_ms, drawn
  (ids)`. Audit fixes are mandatory: MT5 time offset (no false "stale data"), drawing plan reads
  real entry/stop/tp keys, monitor transitions are spoken, real order blocks.
- Data: chart bars (`app.trading.tv.bars()`) when analysing/drawing what the user sees (levels align
  exactly); MT5 for background monitoring and when TradingView is closed. Timings:
  `analysis_engine` (target < 1.5 s), `analysis_total` (< 5 s), `mt5_fetch`.
- draw: `levels` = S/R only (fast path for "هێڵی پشتگیری و بەرگری بکێشە"); `full` = levels +
  zones + entry/stop/targets (`long_position`/`short_position` or lines), tag `analysis:<n>`.
- Vision (optional): one `app.llm.chat(ladder="vision")` on `tv.screenshot()` for rules without a
  predicate.
- Strategy card JSON: `{id, title_ckb, title_en, markets [..], timeframes {bias, setup, entry},
  sessions [..], rules [{id, kind bias|setup|trigger|entry|stop|target|risk|filter|manage, text_ckb,
  text_en, check {predicate, params} | null}], risk {max_risk_pct, max_losses_per_day}, management,
  source_text (verbatim), version, status draft|active|archived, summary_ckb}`. ~30 predicates in
  predicates.py (trend_is, swept, mss_or_bos, in_fvg, in_order_block, in_ote, near_level,
  rsi_divergence, volume_spike, session_is, candle_pattern, ema_cross, usdx_trend, rr_at_least, …).
- Theories: v1's 40-theory catalogue (`registry.py builtin_theories`) as knowledge in theories.py.
- Monitor: alert kinds as in section 2 (volume_spike = k × average of n bars; note tick volume).
  On fire: update `alerts`, publish `Alert(...)` and `SpeakRequest(text_ckb, source="alert")`,
  `db.log_activity("alert", ...)`. The UI shows the tray balloon (no extra toast dependency).
- MT5 is READ-ONLY: never `order_send`, `order_check`, `positions_*` changes. The terminal is running.

### 3.6 UI — `sam/ui/` (island.py panel.py tray.py theme.py strings.py, `__init__.py`)

```python
def run(app, core: CoreThread) -> int      # sam/ui/__init__.py: QApplication (main thread), island, panel, tray; blocks; returns exit code
```

- Sets `app.ui` (a small object with `show_panel()`), attaches
  `UiAdapter(app.bus, qt_bridge.event.emit).attach()` (Qt queued signal → GUI thread).
- Island: frameless, translucent, always-on-top, tool window, shows without activating, top centre
  of the primary screen, DPI aware, ~380×58 growing for the caption; orb reacting to `LevelMeter`;
  name "SAM"; status word per `VoiceState` (ئامادە / گوێ دەگرم / بیردەکەمەوە / قسە دەکەم / کار دەکەم /
  هەڵە); one-line RTL elided caption from `Caption`; progress line from `WorkerProgress`;
  confirmation cards under the pill on `ConfirmRequest` with بەڵێ / نەخێر →
  `app.confirm.resolve(id, True|False, "click")` (thread-safe), hidden on `ConfirmResult`. Click →
  `core.submit(app.toggle_listening())`; double-click/right-click menu: panel, mute, settings, quit.
  Drag to move, position in setting `ui.island_pos`.
- Panel (dark, RTL, font setting `ui.font_family` with fallbacks "Noto Naskh Arabic", "Segoe UI"),
  tabs: گفتوگۆ (history from `Transcript` + `app.memory.recent_turns`, text input →
  `core.submit(app.submit_text(text))`), ستراتیژییەکان (`app.trading.strategies`: add by paste →
  `ingest`, view, activate, archive, versions), چاودێری (`app.trading.monitor.list/cancel`, `Alert`
  events), چالاکی (`activity` + `timings` tables), ڕێکخستنەکان (write-only key fields →
  `app.secrets.set(name, value)` then clear the field; status dots from `app.secrets.status()` and
  `ComponentStatus`; Test buttons → `core.submit(app.llm.test_provider(p))`; Gemini link
  https://aistudio.google.com/apikey; voice engine choice + `app.voice.run_selftest()`; voice name;
  hotkey; conversation timeout; TradingView connect → `app.trading.tv.ensure_running(allow_restart=True)`;
  MT5 status; privacy note: Gemini free tier may use prompts to improve Google products; the
  TradingView automation note).
- Tray (QSystemTrayIcon): Open / Mute / Restart (spawn `pythonw -m sam --after-pid <pid>` then quit)
  / Quit; balloon on `Alert`. Second launch → `sam.winapp.watch_show_requests(callback)` shows the
  panel.
- Every call into core objects goes through `core.submit(...)` / `core.call_soon(...)` except the
  explicitly thread-safe ones (`app.confirm.resolve`, `app.config.get/set`, `app.secrets.*`,
  `app.db.query`). Tests: Qt offscreen smoke test builds island + panel with a `make_app` App.
- Strings in `strings.py` (Sorani first, English secondary). Owns settings `ui.*`.

### 3.7 Launcher / migration — `SAM.pyw`, `scripts/{install,uninstall,dev}.ps1`, `sam/omniroute.py`, `sam/migrate_v1.py`, `README.md`, `.env.example`, `pyproject.toml`, `acceptance/`

- `SAM.pyw` (repo root): `pythonw` entry, no console; handles `sys.stdout is None`; calls
  `sam.__main__.main(argv)` (supports `--home`). Shortcuts pass `--home <SAM_HOME>`.
- `sam/omniroute.py`: `def is_running(url="http://127.0.0.1:20128") -> bool`;
  `async def ensure_running(app) -> str` ("already_running" | "started" | "not_installed" |
  "failed"): only when `~/.omniroute/.env` exists and the port is closed; port v1
  `desktop/sam_desktop.pyw` logic (run from `~/.omniroute`, `OMNIROUTE_CLI_SKIP_REPO_ENV=1`, flags
  `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`, never `DETACHED_PROCESS`, log to
  `%LOCALAPPDATA%\SAM2\logs\omniroute.log`). Never stops OmniRoute. `App.start` already calls it in
  the background when the module exists.
- `sam/migrate_v1.py`: `register(app)` (no tools); `async def start(app)`: if not
  `migrate.v1_done` and `config.v1_db_path` exists → in a thread, open v1 DB **read-only**
  (`sqlite3.connect("file:...?mode=ro", uri=True)`), `custom_theories` (13 rows: id, name,
  version, definition_json, archived, …) → `strategy_cards` with status `archived`, id `v1-<id>`,
  note "imported from SAM v1 (created by automated tests)"; `trading_setups` (8), `setup_events`
  (8), `trading_journal` (8), `trading_context` (6), `memories` (0) → `v1_archive` (source_table,
  source_id, payload). Idempotent; then `config.set("migrate.v1_done", True)`. Also
  `def run_migration(app) -> dict` for tests. Never writes the v1 DB.
- `scripts/install.ps1` (venv + requirements + Start-menu and Startup shortcuts to
  `pythonw SAM.pyw --home ...`), `uninstall.ps1`, `dev.ps1` (runs with
  `SAM_HOME=C:\Users\samit\Desktop\SAM-Agent`). Do NOT run install during the build (v1 still
  autostarts on this PC; the lead switches over).
- README.md Sorani first then English (what SAM 2 does, keys pasted in Settings, privacy note,
  TradingView automation note, how to run tests). `.env.example` with the names SAM 2 reads (no
  values). `pyproject.toml` (metadata, python >= 3.13, pytest config stays in pytest.ini).
- `acceptance/` live scripts for design section 4 (run by hand; clean up drawings, restore chart).

---------------------------------------------------------------------------------------------------

## 4. Environment facts measured by the foundation (2026-09-24)

- Python 3.13.15 venv at `.venv`; every dependency in `requirements.txt` imports on 3.13, including
  `MetaTrader5==5.0.6180` (cp313 wheel), `webrtcvad-wheels`, the modular `winrt-*` 3.2.1 packages,
  and `tzdata` (needed: Windows has no IANA zone data for `zoneinfo`). SQLite 3.50.4 with FTS5 trigram.
- `import google.genai` ≈ 2.0 s, `import numpy` ≈ 2.1 s, `import sam.app` ≈ 0.4 s on this busy PC.
- The v1 DPAPI store at `SAM-Agent\data\secrets.json` decrypts with SAM 2's SecretStore.
- Running `python -m sam --check --home C:\Users\samit\Desktop\SAM-Agent` created
  `data\sam2.sqlite3` there (allowed) and wrote nothing else.

---------------------------------------------------------------------------------------------------

## 5. Integration additions (integrator, 2026-09-24)

Compatible additions made while wiring the packages together; everything above still holds.

**Start-up / shutdown (`sam/__main__.py`).** Order: config → secrets → db → `load_packages()` (every
`register`, tools and slots exist) → core thread (`bus.bind_loop`) → `app.start()` submitted to the
core loop **without waiting** → `sam.ui.run(app, core, started=...)` shows the island at once. The
launcher stage measured the old order (UI after `app.start()`) at 2.1–5.3 s to the island because
voice start alone took 0.56–5.4 s. Late component states reach the UI through `sam.ui.status_seed`.
Shutdown (tray Quit, `--quit`, Ctrl+C headless): wait ≤ 25 s for `start()` to settle →
`app.stop()` (packages in reverse: voice stops listening and releases the hotkey, the monitor stops,
MT5 closes, the CDP socket closes) → loop stops → `app.close()` (SQLite checkpoints the WAL) →
`winapp.release_single_instance()`. Timings: `startup:registered`, `startup:island_visible` (after
the first Qt event-loop turn), `startup:core_ready`. `OPENBLAS_NUM_THREADS=2` is defaulted in
`sam/__main__.py` as in `SAM.pyw` (852 → 148 MB private bytes, launcher measurement).

**`python -m sam --quit` / `SAM.pyw --quit`** ask the running SAM 2 to shut down cleanly (same
path as tray Quit) and wait ≤ 30 s; exit 0 = stopped or not running. `sam.winapp` gains
`QUIT_EVENT_NAME`, `signal_quit()`, `watch_quit_requests(cb)`, `instance_running()` (never creates
the mutex) and `release_single_instance()`. `sam.ui.run(app, core, *, started=None)`; the UI
controller has a `quit_requested` signal.

**Tool catalogue (section 2) as registered: 33 tools.** Added by builders: `forget` (brain.memory:
query, fact_id), `fetch_page` (hands: url*; public http/https only; text under `untrusted`),
`system_control` (hands: action* info/set_volume/volume_up/volume_down/mute/unmute, level),
`theory_info` (trading: name). Parameter additions: `open_app.new_window`, `files` action `rename`,
`click.button` scroll_up/scroll_down, `analyze_market.theory`, `set_alert.expires_in_hours/source`,
`build_project` timeout 600 s (result adds timed_out, vscode_closed, problems). `validate_args`
drops arguments a tool does not declare (Gemini sent `{"reason": ...}` to the parameterless
`tv_open`).

**Package interfaces used across packages** (all verified by `tests/test_ui_integration.py` on the
real packages and by `acceptance/integration_smoke.py` on this PC):
`TradingViewBridge.ensure_running(*, allow_restart=False, confirm=None, focus=True)` (hands'
`open_app` passes `allow_restart=True, confirm=ctx.confirm`), `clear(tag=None) -> dict`,
`probe()`, `status()`, `TvError(code)`; `Worker.build_project(description, *, project_dir, kind,
name, progress, cancel, source)` (hands `build_project`); `VoiceEngine.send_text(text) -> bool` +
`live_session_open` (typed text into an open Live session), `test_key(name)`, `choose_engine()`;
`Conversation.respond_stream(text, *, source, turn)` (cascade), `context_for_prompt(mode, *,
max_chars)` (persona); `sam.ui.strings.learn_tool_labels(registry)`.

**Behaviour fixes from the live smoke** (free tiers saturated: OmniRoute is shared with SAM v1 and
Groq allows ~8k tokens a minute against ~5k-token requests):
- `LLMClient`: a 5xx/timeout that took ≥ `llm.slow_failure_s` (8 s) is not retried on the same rung
  and cools it for `llm.cooldown_slow_s` (180 s); measured OmniRoute 503s after 29–39 s and a 40 s
  timeout, and the old same-rung retry made one typed turn take 118 s.
- `Conversation`: each model request of a turn is capped at `conversation.llm_timeout_s` (15 s;
  good answers took 1–10.5 s); `groq:openai/gpt-oss-20b` is the quality ladder's last resort;
  requests answered with SORANI_NO_MODEL are left out of later prompts (`drop_abandoned`: a stale
  "هێڵەکانت بسڕەوە" was replayed for a price question); when no rung can word a tool result, the
  tool's own Sorani summary is said instead of a bare "تەواو بوو.".
- `clear_my_drawings`: a tag of none/all/null/"" or one that names no SAM group clears all of SAM's
  drawings (the model sent "none" and "levels" for "هێڵەکانت بسڕەوە"; 7 drawings stayed).

**Symbols.** `SYMBOL_ALIASES` maps the Latin transliterations of زێڕ (zar, zer, zeer, zêr, zir,
zhir) to XAUUSD: the voice live check saw Groq send `get_price("ZAR")`, which MT5 matched to EURZAR.

**Live smoke** (`acceptance/integration_smoke.py`, SAM_HOME = the user's SAM-Agent folder): phase A
starts the real `pythonw SAM.pyw --background` (island time, idle RAM, clean `--quit`); phase B runs
`sam.__main__.main` in-process with a null speaker stream and a file-fed microphone and drives the
typed-text path (a–e of the integration brief) plus the voice cascade on a recorded Sorani clip,
then restores the chart, removes SAM's drawings and deletes the smoke conversation.

---------------------------------------------------------------------------------------------------

## 6. Repair additions (repairer, 2026-09-24)

Compatible additions after the second review (three lenses: voice-brain-speed, safety-launcher,
trading-hands-ui). Section 2's "every tool is sent every turn" now means: Live and the worker get every
tool with its full description; the text/cascade path gets a core tier (below) plus `more_tools`.

**LLM client (`sam/brain/llm.py`).** `chat(..., rung_timeouts={ref|provider: s}, deadline_s=None,
retry_transient=True)`: a rung that misses its cap is not retried and rests like a slow failure; the
whole-ladder deadline never blames a rung. Rests grow per consecutive failure (429: 60 -> 180 -> 600 s;
slow 5xx/timeout: 180 -> 600 -> 1200 s; a fast 5xx that fails its retry: like a 429) and are stored in
table `llm_health` (namespace `llm`), so a restart remembers them. New: `cooling(ref)`, `strikes(ref)`,
`healthy_order(refs)`; `status()` adds `strikes`. `OpenAICompatBackend` keeps connections 120 s and has
`warm()` (a HEAD on the API host, no key).

**Conversation (`conversation.py` + `responder.py` + `ladders.py` + `outcome.py`).** Settings
`conversation.ladder.voice/text/reply` default to `"auto"`: picker = Gemini direct (3.5-flash-lite,
3.1-flash-lite, only with a key) -> Groq 20b/120b (voice) or 120b/20b (text) -> OmniRoute; wording =
Gemini direct -> OmniRoute -> Groq; every ladder is health-ordered. OmniRoute rungs are capped at 6 s,
Gemini direct at 7 s; rounds have deadlines (`conversation.picker_deadline_s` 14, `wording_deadline_s` 8,
`reword_deadline_s` 5); `conversation.reasoning` "minimal". Small talk a Groq rung answered is worded
again by the wording ladder (unless it is a short clean Sorani sentence that does not echo the user);
text that comes together with a tool call is not spoken. Tool tiers: `conversation.tool_tier` "core"
(default) sends `conversation.core_tools` (15) with compact descriptions plus `more_tools(tools, need)`,
which attaches the named tools to the next round only; "all" restores the old behaviour. When no model can
word a result, `outcome.tool_sentence(name, args, result)` speaks it (Sorani templates per tool).
Chunks yielded by `respond_stream` are `AckText` / `AnswerText` (str subclasses); timing stages
`first_answer` (text) and, in the cascade, `first_answer_audio`. `TurnTimer.mark_once(stage)`.
`textnorm.fix_letters(text)` (display/speech letter forms) is applied to every reply.

**Tools registry.** `@tool(..., private_args=("text",))`: those arguments reach events, the activity
table and conversation tool turns as `"<N chars>"` (type_text.text, files.content); the confirmation
card still shows them. `openai_tools(names, compact=False)`, `ToolSpec.model_description(compact=)`.
Taint (`sam/brain/taint.py`): `taint.begin(user_text, inherit=False)` opens a scope per user turn
(respond_stream, Live's `_finish_user`) and per worker task (inherits a tainted parent); a result with
`untrusted` data taints the scope, and then fetch_page/open_url to a host the user did not name,
web_search with words the user did not say, remember, strategy_save, files write/append/copy/move/rename,
type_text with Enter, delegate_task, build_project and screen_act ask first.

**Confirmations.** `classify_answer`: a yes must be the whole utterance (<= 3 words, fillers allowed);
«بکە», «ئا», «ئەها», «تەواو», «باشە» alone are not a yes; wait/later/enough/not/«بەسە»/«لێگەڕێ»/«پێویست
ناکات» are no. Questions never quote model or user text (it is on the card). The cascade ignores what the
mic hears while SAM reads a confirmation question and for 1 s after (`CascadeVoice.confirm_quiet_until`,
`VoiceEngine.confirm_quiet()`, also used by Live).

**Voice.** Automatic = Live only after a PASSING self-test (none/inconclusive/fail -> cascade); the
self-test starts ~5 s after a key exists, also while listening. `VoiceEngine.live_text_trusted()`. The
Live watchdog stalls only when Live transcribed words for the utterance (noise keeps Live, no STT);
Live's transcript replaces STT only after a passing self-test. Hybrid VAD: `audio_stream_end` at local
end of speech (`voice.live_hybrid_vad`). Barge-in over SAM's voice needs `voice.barge_in_ms` (400) voiced
ms; shorter sounds duck the speaker (`Speaker.gain`, `voice.duck_gain`) and are dropped without STT; a
backchannel that cut a reply starts no turn (`Utterance.cut_reply`). KurdishTTS STT+TTS share one
kept-alive pool (`sam/voice/kurdish_http.py`), warmed when listening starts. `TtsRouter.low_budget(share)`:
below `voice.tts_low_budget_share` only the first sentence of an answer is spoken. Decimals and
percentages are spoken as Sorani words (`sam/voice/numbers_ckb.py`, `voice.tts_verbalize_numbers`
decimals|all|off; KurdishTTS reads whole numbers itself, measured by a TTS->STT round trip). New modules to stay under ~700
lines: `engine_support.py` (VoiceEngine mixin), `live_calls.py` (LiveVoice mixin).

**Hands.** `sam/hands/guards.py` holds the screen-dependent classifiers: open_app arguments need a yes
(shell/interpreter arguments are classified like run_powershell, script hosts with arguments blocked;
`AppIndex.launch` re-checks the resolved program); typing + Enter into a console/Windows Terminal is
classified like run_powershell, into the Run box asks; in MetaTrader 5 / TradingView, clicks on
buy/sell/order/position/close/flatten/reverse/modify/lot controls and the order hotkeys (F9, Alt/Shift+B/S)
are blocked, and screen_act judges the element under the click point (an unidentifiable click there is
blocked). Policy: ForEach-Object member calls (`| % Delete`) ask, and are blocked after `-Recurse`;
wildcard paths that could match a key file and paths built from variables/pieces for file-reading
cmdlets are blocked / ask; DNS and connection probes ask. `windows.scrub_title(process, title)` masks
MT5 account digits everywhere a title leaves the process; screen_look drops MT5 balance lines.
`Uia.password_rects` raises on failure (the capture is then blanked entirely).

**Trading.** `sam/trading/symbols.py`: `resolve_instrument(text)` (aliases with Sorani suffixes, «گۆڵت»,
spelled XAUUSD, Latin transliterations, edit-distance-1 near misses; never 1-2 letters) used by every
engine tool; `same_instrument(a, b)`; `user_named(app, symbol, source)`. `tv_set_chart` never applies a
symbol the user did not name (with a timeframe it applies only the timeframe). `tv_symbol_for(...,
learned=)` + setting `trading.tv_learned_symbols` (the user's own feed per instrument, learned from
chart_state). Drawing ownership (`sam/trading/tv_owner.py`): missing ids are forgotten only after 3
misses over >= 6 s and never within 10 s of a symbol change or while loading; re-created drawings are
re-adopted by label + prices (JS bundle v2 adds `shapesDetailed`). `draw_on_chart` refuses prices more
than 15% from the chart price unless the user said the number. S/R: `snr_levels` ranks per side and adds
day/previous-day high/low and last swing anchors. Closed market / stale feed: levels and zones are drawn
(never entry/stop/targets). Draw plans merge overlapping same-side zones, draw one zone per side
(extended right, alternating label corners) and one line per 0.3 ATR. `compact_report.drawing` =
{plan, dry_run, reason}; a Sorani sentence says when nothing was drawn. Alerts: M1 wicks count only
for bars opened after creation / re-arm; armed/direction/fired_at/armed_at persist in `alerts.params`.
MT5 re-verifies the broker offset from live ticks (every 5 min, every 1 min while ticks look off).
`StrategyStore.get(id, fuzzy=True)`: fuzzy only over draft/active cards and only one strong hit;
`strategy_save` uses exact ids and never changes an archived card. Spoken prices are rounded
(`sorani.spoken_price`).

**Launcher.** `scripts/install.ps1` step 0 stops when SAM v1 (sam_desktop.pyw, sam_backend, LiteLLM)
still runs from the repository folder (`-StopV1` stops them); `.venv.old-*/` is gitignored.

---------------------------------------------------------------------------------------------------

## 7. Acceptance additions (acceptance stage, 2026-09-24 evening; the user's first real test)

Compatible additions; everything above still holds.

**Ladders.** `GEMINI_CAP_S` 5 s. The slow head of a round (Gemini direct, OmniRoute ahead of the first
healthy Groq rung) shares at most `ladders.head_budget(deadline)` = min(`HEAD_MAX_S` 6, deadline -
`FAST_RESERVE_S` 3); a slow rung is not started with less than `MIN_SLOW_RUNG_S` 2.5 s left; the rest
goes to Groq (`Responder._chat_reserving`, `ladders.split_head`). `LLMClient.chat`: a rung the round's
deadline cut after it used 3/4 of its own cap IS blamed (rests). `LLMClient.reset_provider(provider, *,
auth_only=False)` forgets a provider's rests and strikes (memory and `llm_health`); Settings calls it after
a key is saved, `test_provider` clears the provider-wide auth rest when listing works.

**Background budget (`sam/brain/budget.py`, `BackgroundBudget` at `app.conversation.budget`).**
Conversation summaries, fact extraction and rewording are optional model calls. Summaries and extraction
never run during a live exchange (a turn running: `Conversation.active_turns`; the voice engine
thinking/speaking/working; or speech/text within `brain.background.quiet_s` 40 s); they run in
`Conversation.background_tick()` (every 15 s from `idle_watch`), one request to one rung, with
`retry_transient=False`. Any of them is skipped while ANY rung of the live picker/wording ladders (or of
its own ladder) rests, or when the day's budget is low (`brain.background.daily_max` 30 summary+extraction
requests; `reword_daily_max` 60; a capped model past `cap_share` 0.6 of `llm.daily_caps`). Under pressure a
summary is folded without a model (extractive) and an extraction waits (retried in a later pause, dropped
after 24 h). Every decision is counted in `usage_counters` (provider `background`, model = purpose, kind =
ran | skipped | deferred | failed); `budget.counts_today()`. `Memory.extract_facts(conversation_id, *,
ladder=None)` takes the one-rung ladder.

**Confirmations.** `classify_answer(text, question="")`: the pending action's own verb is a yes when the
pending question contains its stem (`ACTION_VERBS`: «بەڵێ بینێرە», «بینێرە» for «... بنێرم؟»,
«بیسڕەوە», «دایبخە» ...); negative imperatives are a no. `ConfirmBroker.needs_clear_answer(text)`: a
confirmation waits and a short utterance is neither yes nor no -- the caller should answer `ASK_AGAIN_CKB`
(«بەڵێ یان نەخێر؟») instead of starting a turn (typed text does; the cascade should, voice package).

**Taint.** `taint.note(scope, name, result)` (called by `ToolRegistry.dispatch`) marks the scope and keeps
the scope's own web_search result links; `fetch_page` of exactly such a link needs no question (a changed
query string or another host still asks).

**Hands.** PowerShell: every ForEach-Object argument that is not a script block is a member call
(`policy.foreach_members`: quoted, -Mem/-MemberName:, variables, expressions, splatting, `.ForEach('x')`),
blocked after any recursion flag (-r/-Rec/-Depth) or on a listing of a drive/home/main folder; `powershell
-c "..."` is classified inside; `Remove-Item <main folder>\*`, `[IO.Directory]::Delete(x, $true)` and
`.Delete($true)` are mass deletion. `guards.click_windows(args)`: the click guard judges the window the
click goes to (`Windows.find_sync(window)`, and for a numbered target the window of the last screen_look);
'Algo Trading', AutoTrading, Expert Advisors and Ctrl+E are blocked in trading apps. Grounded web search
skips Gemini while its model rests (and for 5 min after it failed).

**Trading.** `symbols`: the edit-distance fallback never applies to inputs with digits, currency-pair shapes
or `NOT_NEAR_MISSES` (US500, UK100, UKOIL, GER40, USDT stay themselves). `tv_owner.match_shapes` re-adopts
only labelled rows of the same kind (`SHAPE_NAMES`). `tv_feeds.FeedMemory` (bridge `.feeds`): symbols SAM set
are kept in `trading.tv_sam_set_symbols` (30 days) and never learned as the user's feed unless the user
changes the chart to them. `analyze.feed_offset`: no chart-minus-MT5 offset when the MT5 quote is older
than the chart's newest bar by > 300 s or differs by > 2 %; `draw_report` does not draw a stale MT5 feed
on an open market (reason `stale_feed`, Sorani sentence in `NOT_DRAWN_CKB`).

**UI / launcher.** A normal `SAM.pyw` launch sets `SAM_SHOW_PANEL=1` (background launch: 0); the UI opens
the panel itself 30 ms after the island's first frame (no show-request polling). `Panel.show_and_raise`
brings the panel to the foreground (`win32.bring_to_front`: SetForegroundWindow, then AttachThreadInput when
Windows refuses; `Panel.in_front`); a second launch calls AllowSetForegroundWindow(ASFW_ANY) before
signalling. Settings controls carry unique Sorani accessible names and English descriptions (`a11y.*`
strings, `widgets.accessible`), key rows have objectNames `KeyRow_<secret>`; engine chips and nav items
act on `toggled`, so UI Automation's Toggle works without a click. SAM's own hands still never act on SAM's
own windows (its chat box accepts «بەڵێ» as a confirmation answer).

---------------------------------------------------------------------------------------------------

## 8. Listening additions (voice, 2026-09-24 night: deliberate, noise-robust, quota-safe listening)

Why: in the user's first real test SAM transcribed the TV and a family conversation as commands (149
KurdishTTS STT calls, every free model quota used up) and a Gemini TTS 429 plus the SDK's own ~27 s retry
left the island on «بیردەکەمەوە» for ~40 s. Compatible additions (everything above still holds):

**Listening windows (`sam/voice/listening.py`).** Default (`voice.always_listening` False) = push-to-talk
turns: `start_listening()` (click / hotkey / a confirmation question) opens ONE utterance window of
`voice.start_timeout_s` (8 s); after SAM's answer a follow-up window of `voice.followup_s` (6 s). An
automatic close publishes `VoiceState("idle", detail="no_speech"|"turn_end")` + `VoiceNotice(kind="closed")`
and does NOT end the conversation; `VoiceState("sleeping", detail="conversation_end")` follows after
`voice.conversation_timeout_s` with the mic closed (so `Conversation.on_sleep` / fact extraction run once
per conversation). `stop_listening()` by the user still publishes "sleeping" (contract 3.1). Always
listening: the cascade only, and the transcript must start with «سام»/SAM (`starts_with_name`) before any
model call -- except a yes/no to a pending confirmation or an utterance that began while SAM answered or in
the follow-up window.

**Near-field gate (`sam/voice/gate.py`)** before STT or Live: per 30 ms frame, webrtcvad AND level >=
clamp(max(p30 floor + `voice.gate_margin_db`, rejected-talker level + 4, user level - 8), `voice.gate_abs_min_db`,
ceiling) where ceiling = user level - 4 dB (or `voice.gate_ceiling_db`). The user level
(`voice.gate_user_level_db`) is measured by the enrollment and learned (EMA) from the first utterance after a
click or any voiceprint-verified one. Failing frames are silence for the endpointer. `Endpointer.frames_so_far()`.

**"Only my voice" (`sam/voice/voiceprint.py`, `enroll.py`).** sherpa-onnx 1.13.8 (+ `sherpa-onnx-core`
1.13.8, add to requirements.txt) CAM++ `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`
(Apache-2.0, 28.3 MB, SHA-256 checked, downloaded to `%LOCALAPPDATA%\SAM2\models` only when the user starts
the enrollment; `voice.speaker_model_path` overrides). `app.voice.speaker_check: SpeakerCheck`
(`enrolled`, `enabled`, `threshold(speech_ms)`, `await verify(pcm, speech_ms=) -> VerifyResult(ok, score,
threshold, ms, reason)`, `status()`). Voiceprint = one embedding, DPAPI-protected, table `voice_profile`
(namespace `voiceprint`, row id 1). Settings `voice.only_my_voice` (True: active once enrolled),
`voice.only_my_voice_sensitivity` low/normal/high = 0.40/0.50/0.60 (speech < 1 s: -0.10). A rejected
utterance costs no STT/model call (`VoiceNotice(kind="ignored")`, activity `not_my_voice`). New
`VoiceEngine` methods: `request_enrollment(source)`, `await enroll_begin() -> {ok, sentences}`,
`await enroll_record(index) -> {ok, reason, speech_ms, level_db}`, `await enroll_finish() -> {ok,
consistency, level_db, clips}`, `await enroll_cancel()`, `await voiceprint_delete()`, `voiceprint_status()`,
`admit_transcript(text, meta) -> text | None` (cascade hook), `listening_status()`; `status()` adds
`listening_window`, `gate`, `voiceprint`, `rests`.

**Live (`frames.py`, `live_config.py`).** Audio is sent only for accepted utterances (held until
`voice.gate_min_voiced_ms` of near-field speech and, with a voiceprint, a match on the first 1.2 s; pre-roll
included; `audio_stream_end` at local end of speech). `start_of_speech_sensitivity` =
`voice.live_start_sensitivity` ("low").

**Never stall on TTS/STT (`sam/voice/genai_client.py`, `quota.py`).** Voice Gemini clients never retry inside
the SDK (`HttpRetryOptions(attempts=1)` + the Interactions client's `retry_config = None`; an httpx mock
proves exactly one request per 429). Gemini TTS must deliver audio within `voice.tts_first_audio_s` (2.5 s);
429/timeout/5xx rest the provider (`quota.rests(app)`: daily 429 -> next Pacific midnight = 10:00 Iraq in
summer; per-minute -> retryDelay clamped 60-180 s; unlabelled 120 s, a second within 10 min -> daily;
timeout/5xx 60 s), stored in setting `voice.rests` (survives restarts). Gemini STT: `voice.stt_gemini_timeout_s`
(8 s) and the same rests. `voice.tts_prewarm` is False (lazy caching; an opted-in prewarm never uses Gemini);
the automatic self-test runs at most once per `voice.selftest_min_interval_s` (86400) and never while Gemini
TTS rests.

**Events (`sam/voice/notices.py`, compatible: subclasses of `sam.events.Event`, forwarded to the UI).**
`VoiceNotice(kind closed|ignored|quota|models|enroll, text_ckb, detail, until)` and `VoiceEnrollRequest(source)`.
The island (`sam/ui/island_hints.py`) shows every notice as its caption; a `models` notice (the brain's
SORANI_NO_MODEL reply) shows «سنووری ئەمڕۆ پڕە» as the idle status until the reset or a normal answer.
Settings has a card «گوێگرتن و دەنگی من» (`sam/ui/voice_profile.py`: follow-up seconds, how strictly far/quiet
speech is ignored, «تەنها دەنگی من», sensitivity, «ناساندنی دەنگی من», «سڕینەوەی دەنگی من»); the enrollment
dialog also opens when the user says «دەنگم بناسە». `CascadeVoice.submit_utterance(..., meta=)`,
`CascadeVoice.reply_active`.

---------------------------------------------------------------------------------------------------

## 10. Local brain, no-AI fast path, voice defaults (brain, 2026-09-25)

The user's decisions after the first real test: every free quota was used up in one evening, Gemini Live
heard «سڵاو سام چۆنی» as Korean (answered in English, then Italian), and an A/B listening test preferred
KurdishTTS's voice. Compatible additions; everything above still holds.

**Local brain = the implicit LAST rung of every ladder.** Provider `ollama` (in `llm.PROVIDERS`, never in a
ladder setting). `LLMClient.chat(..., local=None)` / `stream(..., local=None)`: the cloud ladder runs exactly as
before; only when it is exhausted (every rung resting, failing, over its cap, unconfigured, offline, or cut by
the round deadline) the local model answers, with its OWN timeout `llm.local.timeout_s` (150 s: the round's
deadline is spent and a cold answer takes ~1.5 min). `local=False` where a local answer is worse than none:
`Responder._reword`, the slow head rungs in `_chat_reserving`, conversation summaries, fact extraction.
Requests with images never go local (qwen3:8b has no vision). A local failure of kind network/not_found rests
`ollama:*` 120 s. Mixin `sam/brain/llm_local.py` (`LocalRung`): `brain_mode` ("cloud"|"local", who answered
last), `local_backend()`, `local_ready()`, `cloud_usable(refs)`, `await local_model()` (`llm.local.model`, else
the first installed `llm.local.fallback_models`), `prewarm_local(messages, tools)` (one task at a time),
`local_status()`, `note_cloud_answer(resp)`. Usage is counted as provider `ollama`.

**Backend** `sam/brain/llm_ollama.py` (`OllamaBackend`, native `/api/chat`: `think: false`, `keep_alive`,
`options.num_ctx`/`num_predict` (capped by `llm.local.max_tokens`), tools in OpenAI shape, NDJSON streaming,
`loaded()` = `/api/ps`, `warm(model, messages, tools)`; `aclose()` stops the server only if SAM started it).
`split_context(system)`: the persona now puts its stable text first (fixed rules + tool list) and the per-turn
part after `persona.CONTEXT_HEADING` ("Current context (changes every turn):" + time, user name, facts,
strategies, conversation); the backend moves that part in front of the LAST user message so the stable prefix
and the tool schemas stay in Ollama's prompt cache. **Server** `sam/brain/local_server.py` (`OllamaServer`):
started on demand only when nothing listens on `llm.local.host` (SAM v1's `ollama serve` on 11434 is shared,
never stopped), `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP` (never DETACHED_PROCESS), env OLLAMA_HOST /
OLLAMA_MODELS, never OLLAMA_IGPU_ENABLE, log `%LOCALAPPDATA%\SAM2\logs\ollama.log`; stopped (process tree) on
quit if SAM started it. Settings (`sam/config.py` `LOCAL_BRAIN_DEFAULTS`): `llm.local.enabled` True,
`.model` "qwen3:8b", `.fallback_models` ["qwen3.5:4b"], `.host` "127.0.0.1:11434", `.ollama_exe` "" (=
SAM_HOME/tools/ollama*/ollama.exe, then %LOCALAPPDATA%\Programs\Ollama, PATH), `.models_dir` "" (=
<data>/ollama-models when it exists), `.keep_alive` "5m", `.num_ctx` 8192, `.max_tokens` 1024, `.timeout_s` 150,
`.temperature` 0.3, `.think` False, `.vision` False, `.stop_on_quit` True, `.prewarm` True.

Measured on this PC (2026-09-24/25, CPU; ~2 GB RAM free while other work ran), SAM's real voice prompt and the
compact core tools (4.1-4.6k prompt tokens), 14 Sorani commands (lead scratchpad `sam2/localbrain`):

| model | load | generate | first prompt | next prompts | per command (warm) | right tool |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3:8b | 8.3 s | 8.4 tok/s | 86 s (48 tok/s) | 1-2.7 s (prefix cache) | 2.6-9.2 s | 12/14 |
| qwen3.5:4b | 6.5 s | 14-16 tok/s | 52 s | 12-16 s (no prefix reuse) | 13-19 s | 13/14 |

GPU: Ollama 0.33.1 drops the Radeon 890M (integrated) by default; with `OLLAMA_IGPU_ENABLE=1` llama-server
crashed fitting either model to Vulkan memory (exit 0xe06d7363, "AMD driver is too old"), so CPU only.
Default qwen3:8b (prefix cache 3-5x faster per turn, natural «سڵاو، فەرموو. چۆنی؟»; its two misses were
«نۆتپاد/کرۆم بکەرەوە» answered as text, which the fast path answers first). Live end to end with SAM's own
code (`acceptance/brain_local.py`, no cloud key): SAM started `ollama serve` on 11436 hidden, the warm-up took
100 s, then warm turns 3.7-12.8 s (tool turns) with the right tool 5/6, small talk 5.8 s, a free question
22.7 s (weak Sorani); quitting stopped the server (port closed). Known limit: qwen3:8b may claim an action it
did not call a tool for («نۆتپاد ئامادەیە» with no tool); common commands never reach it (fast path).

**Conversation on the local brain** (`responder.py`, `conversation.py`): after a LOCAL tool call the tool's own
Sorani result is the answer (`outcome.own_sentence`; no second 3-10 s local round; list_alerts has a fixed
sentence `outcome.alerts_sentence`; read-type tools are still worded by the model). Voice: when no cloud rung
can answer and the model is not loaded, the turn first says `LOCAL_ACK_CKB` «یەک چرکە، بە مێشکی ناوخۆیی
بیری لێ دەکەمەوە.». `Conversation.prewarm_local_brain(mode)` runs on `VoiceState("listening")` when no cloud
rung is usable (loads the model and reads the stable prompt while the user speaks).
**Who answers** is published once per switch: `ComponentStatus("brain", "degraded", "local: ollama:<model>")` /
`("brain", "ok", "cloud: <ref>")` and `VoiceNotice(kind="local"|"cloud", text_ckb=...)`. UI (small additions):
`island_hints` shows «مێشکی ناوخۆیی» as the idle/sleeping/thinking status word while local; the panel sidebar
has a «مێشک» row (`COMPONENT_ROWS["brain"]`, label «مێشکی ناوخۆیی» when local); `status_seed` reports it.

**No-AI fast path** (`sam/brain/intents.py` matcher, `sam/brain/fastpath.py` runner; setting
`brain.fastpath.enabled` True). `Responder.respond_stream` (typed text and the cascade) first asks
`fastpath.intent_for(app, text)`; a match runs ONE tool through `app.tools.dispatch` (risk, confirmations,
timings, taint as usual) and answers with the tool's own Sorani result: no model, no quota, milliseconds.
Intents: price (get_price), open_tradingview (tv_open), open_app (known aliases from `hands.aliases`, generic
ones excluded), set_chart (tv_set_chart symbol/timeframe), analyze (analyze_market draw full, vision False),
draw_levels (analyze_market draw levels on the chart's symbol), clear_drawings, list_alerts, cancel_alerts
(all / a number), stop (stop_all). Precision rule: every word must be consumed by one intent's grammar;
questions, negations, past tense, conditions, «و»/"and" joining actions, orders and unknown words go to the
model. Tolerates KurdishTTS STT spellings («نەخنەشکی زێڕ چەندە»), Arabic letters, word-final ه, clitics
(«ترەیدینگ ڤیوم بۆ بکەرەوە»). Corpus `tests/fastpath_corpus.py`: 290 labelled rows (146 commands, 144
non-commands): precision 1.00, recall 0.99; the 67 held-out rows written before tuning scored precision 1.00,
recall 0.87 on the first run (0.95 after three fixes). Usage counted as provider `fastpath`, model = intent.
Test helper `brain_app(..., fastpath=False)`: model-loop tests keep the fast path off.
`outcome.tool_sentence` fix: `data.cancelled` means "stopped" only when it is `True` (cancel_alert's count made
«هەموو ئاگادارکردنەوەکان هەڵبوەشێنەوە» answer «ڕاگیرا.»).

**Voice defaults.** `voice.tts_provider` = "kurdishtts" (Gemini TTS is the fallback; `TtsRouter.order`).
"Automatic" never picks Live: `voice.auto_live` False (True restores "Live after a passing self-test");
`VoiceEngine.live_wanted()`; the automatic self-test (3 Gemini TTS requests) runs only when Live can be used.
