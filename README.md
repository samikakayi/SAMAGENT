# SAM — local desktop AI agent for Windows

SAM is a local-first, realtime Windows AI agent with a polished browser-based
desktop UI. It can chat in Sorani Kurdish or English, remember approved context,
plan work, edit a selected workspace, run PowerShell and Python, automate an
isolated browser, inspect Windows, control a verified TradingView Desktop
window, and analyze read-only MetaTrader 5 data. All computer actions pass
through a deterministic policy and audit layer.

AUTO routing prefers an available local Ollama model and can fall back only to
cloud providers that you explicitly configure. LiteLLM, OpenRouter, and direct
OpenAI adapters are included; no credential is bundled or returned by the API.

## What is included

- Conversation history and durable memory in local SQLite.
- Workspace-aware listing, searching, reading, atomic writes, exact text edits,
  and deletion.
- PowerShell/terminal and Python execution with timeouts, bounded output, a
  reduced environment, and approval gates.
- Browser navigation plus optional isolated browser automation.
- Windows application launching.
- Structured planning and a bounded multi-step tool loop.
- **Autonomous runs**: give SAM a goal and it inspects the project, plans,
  edits files, runs the project's own tests, diagnoses failures, re-plans, and
  retries until the work is verified or it can honestly say it is stuck. Live
  progress appears in the **Autopilot** panel.
- Project intelligence: a cached map of structure, dependencies, commands, API
  routes, test suites and git state.
- Verification engine that runs a project's declared test/build/lint checks and
  parses the result, so "done" is proven rather than asserted.
- Capability discovery for installed developer tools and reachable model
  providers.
- Approval cards that show the exact operation, risk, reason, and arguments.
- Redacted, hash-linked audit records.
- Responsive UI, keyboard navigation, light/dark themes, and optional browser
  speech recognition and text-to-speech.
- Cost-aware AUTO/LOCAL_ONLY/CLOUD_ONLY/MANUAL routing through Ollama,
  LiteLLM, OpenRouter, and direct OpenAI adapters.
- Realtime microphone input with interim transcripts, browser VAD, barge-in,
  sentence-chunk speech output, and four listening modes.
- Read-only MT5 OHLCV/ticks, deterministic multi-timeframe analysis, theory
  comparison, setup state, journal, and monitor.
- TradingView process/window/symbol/price observation plus guarded controls.
- Setup, launch, diagnostics, and automated tests.

## Safety by default

SAM intentionally does **not** have unrestricted administrator access.

- The API listens on `127.0.0.1` by default.
- SAM should be run as a standard Windows user, never as Administrator.
- Deletes, overwrites, arbitrary Python, unknown app launches, writes outside
  the workspace, package installs, credential access, system changes, external
  messages/uploads, and irreversible actions require a specific approval.
- Elevation, opaque/encoded execution, destructive disk commands, credential
  dumping, and disabling security controls are blocked in safe mode.
- Approval is single-use and bound to one normalized tool request. Chat text is
  never treated as approval.
- Secret-looking values are redacted from logs and refused as durable memory.

Read [Security model](docs/SECURITY.md) before enabling broad control. An
approved arbitrary process still runs with your Windows account's permissions;
use Windows Sandbox or a VM for untrusted code.

## Quick start (Windows PowerShell)

### 1. Prerequisites

- Windows 10 or 11.
- Python 3.11 or newer.
- Ollama for fully local chat. Install it from
  [ollama.com/download/windows](https://ollama.com/download/windows), then leave
  it running in the background.

### 2. Install SAM

Open PowerShell in this folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

If Ollama is not installed, the setup script can install it through `winget`
only when you explicitly request that extra action:

```powershell
.\setup.ps1 -InstallOllama -PullRecommendedModel
```

The recommended default is `qwen3.5:4b` (about 3.4 GB). It was selected after
local Sorani and tool-calling smoke tests. On a smaller machine, use
`qwen3.5:2b`:

```powershell
ollama pull qwen3.5:2b
```

Then set `SAM_MODEL=qwen3.5:2b` in `.env`.

If `winget` is unavailable, SAM also recognizes the official standalone
Windows build at `tools\ollama-v<version>\ollama.exe`. The bundled start script
starts and stops that portable service with SAM; it never requires
Administrator rights. Portable identity files and model blobs stay under
`data\ollama-user` and `data\ollama-models`.

### 3. Start SAM

```powershell
.\start.ps1
```

Open <http://127.0.0.1:8765>. Press `Ctrl+C` in PowerShell to stop SAM.

To start without opening a browser:

```powershell
.\start.ps1 -NoBrowser
```

## First conversation

Try:

> Inspect this workspace, create a short plan, and tell me what you would do.

Then try a bounded edit:

> Create `hello.txt` in the workspace with a short greeting.

SAM may create a new workspace file directly. Overwriting or deleting an
existing file creates an approval card. Review the exact path and arguments,
then approve or deny it from the Approvals panel.

## How to talk to SAM now

1. Start SAM with `.\start.ps1` and open <http://127.0.0.1:8765>.
2. Click the microphone beside the message box.
3. Allow microphone access when Edge or Chrome asks.
4. Speak in Sorani Kurdish or English. In the default **Push to talk** mode,
   click the microphone again (or pause) and the recognized sentence is sent.
5. Turn on **Read replies aloud** in **Settings → Voice** for spoken answers.

The microphone can transcribe immediately, but normal open-ended AI replies
also require one available chat model. If the sidebar says **No chat model
available**, run `.\setup.ps1 -InstallOllama -PullRecommendedModel`, restart
SAM, and refresh the page. Deterministic market-data status and analysis remain
available without a chat model.

Voice modes are **Push to talk**, **Conversation**, **Always listening**, and
**Wake word**. The defaults are recognition locale `ckb-IQ` and wake word
`SAM`. Speaking while SAM reads a reply triggers barge-in: speech output and
the cancellable active task stop before the next utterance is accepted.

Browser recognition may use the browser vendor's service. Optional
faster-whisper, Silero VAD, and Piper packages can be installed with
`.\setup.ps1 -InstallLocalVoice`, but stay reported as **unconfigured** until
their models and voice are configured.

## Model configuration

Edit `.env`, then restart SAM.

### AUTO routing with Ollama (recommended local-first default)

```dotenv
SAM_PROVIDER=auto
SAM_MODEL_MODE=AUTO
SAM_MODEL=qwen3.5:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

List downloaded models with `ollama list`. SAM also discovers them through
Ollama's local `/api/tags` endpoint.

### OpenAI (optional cloud boundary)

```dotenv
SAM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

The OpenAI adapter uses the Responses API with custom function tools and
`store=false`. Keep `.env` private. For long-term use, prefer injecting the key
from Windows Credential Manager rather than storing it in a file.

When a cloud provider is active, chat context and non-sensitive tool results may
leave the machine. SAM blocks known secret paths from cloud context by default,
but no classifier is perfect—do not ask a cloud model to inspect secrets.

### LiteLLM gateway (optional)

Install and start the local gateway in a second PowerShell:

```powershell
.\setup.ps1 -InstallLiteLLM
.\start-litellm.ps1
```

`litellm-config.yaml` defines `sam-fast`, `sam-strong`, and `sam-vision`
aliases backed by local Ollama by default. Change the underlying routes only
after securely configuring the corresponding provider credentials.

### OpenRouter (optional cloud boundary)

Inject `OPENROUTER_API_KEY` into the SAM process environment. Keep
`SAM_PROVIDER=auto` for fallback routing or select `openrouter` manually. The
adapter uses the OpenAI-compatible chat endpoint, tool calling, app attribution
headers, and returned usage/cost data. Daily and monthly cloud budgets are
visible in Settings.

## Trading brain and desktop control

The **Market** panel reads exact provider candles and quotes from the installed
MetaTrader 5 terminal. It separates facts, observations, theory interpretation,
setup/confirmation, invalidation, and risk/reward. Missing requirements produce
`WAIT`, `WATCH`, or `NO_TRADE`; the UI never fills entry, stop, targets,
confidence, or sweep labels with placeholders.

Deterministic modules cover indicators, swing/market structure, HH/HL/LH/LL,
BOS/CHOCH/MSS, scored S/R, liquidity/equal highs/equal lows/sweeps, FVG,
supply/demand candidates, candlestick observations, volume profile, VWAP,
sessions/DST, statistics, multi-timeframe entry gating, theory registry,
versioned custom theories, setup monitoring, self-check, and a journal.
Order-flow, footprint, CVD, DOM, tape, macro calendar, rates, and sentiment are
capability-gated and never inferred from ordinary OHLC candles.

The **TradingView** panel observes the native window handle, title, symbol,
quote, geometry, monitor, and foreground state. Computer Control and Screen
Access are separate switches and are OFF by default. Symbol changes require
native-title confirmation. The installed TradingView build does not expose the
interval through its title/accessibility surface, so timeframe commands return
`PARTIAL` until independently verified. Drawing stays partial until chart
price-to-screen calibration is verified.

The MetaTrader integration is **market-data read-only**. Live broker order
placement is not implemented.

## Workspace

The default controlled workspace is `workspace/` inside this project. Change it
in `.env` with an absolute path:

```dotenv
SAM_WORKSPACE=D:\Projects\MyProject
```

Choose a narrow project directory. Do not use a drive root, your entire user
profile, `C:\Windows`, or `Program Files`. Changes outside the configured root
require approval, and protected operations remain blocked.

## Voice

Use the microphone button in the composer and enable spoken replies in
Settings. Voice uses browser Web Speech plus a local audio analyser for
activity/silence detection. Availability and where recognition is processed
depend on the installed browser; typed chat always works.

## Browser automation

Opening an ordinary `http` or `https` page is a normal action. Interactions that
can create external side effects—clicking, typing, submitting, uploading, or
sending—require approval. Browser automation uses a fresh isolated profile and
does not inherit the primary browser's cookies, password manager, or sign-in.

## Permissions and approvals

Each approval contains:

- Tool and normalized arguments.
- Working directory or destination.
- Risk level and policy reason.
- Expiration time and one-time request fingerprint.

Approve only when the operation exactly matches your intent. A changed,
expired, replayed, or cross-session request is rejected. There is deliberately
no global “approve everything” switch.

Permission presets never remove hard blocks or approval for destructive work:

- **Guarded** (default): approve terminal commands and high-risk actions; allow
  ordinary reads and new workspace files.
- **Strict**: also approve new files, navigation, and read-only browser runs.
- **Trusted workspace**: allow commands classified as safe inside the selected
  workspace; overwrites, deletes, installs, system changes, external side
  effects, and opaque commands still require approval or remain blocked.

## Audit and memory data

Runtime data is stored in `data/sam.sqlite3`. The UI exposes a read-only audit
viewer and memory controls. Do not edit the database while SAM is running.

Back up the database by stopping SAM and copying the file. To start with a fresh
state, stop SAM and move the database elsewhere; keeping the old file makes the
operation recoverable.

## Tests

```powershell
.\run-tests.ps1
```

The suite checks workspace containment, dangerous command decisions, approval
binding/replay protection, secret redaction, memory/database behavior, routing
and budgets, trading types/indicators/analysis/capability guards, custom theory
versioning, voice/trading API declarations, and API health.

## Troubleshooting

**The UI says Ollama is offline**

Start Ollama, then verify:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

Download at least one tool-capable model:

```powershell
ollama pull qwen3.5:4b
```

**Port 8765 is already in use**

```powershell
.\start.ps1 -Port 8877
```

**Microphone input is unavailable**

Use current Microsoft Edge or Chrome, grant microphone permission to the local
page, and check Windows **Settings → Privacy & security → Microphone**.

**A command was blocked**

Open the audit panel. Opaque, encoded, elevated, credential-dumping, or
disk/security-destructive commands cannot be approved in safe mode. Rewrite the
request as a narrow, transparent action. Do not disable the safety layer to run
untrusted code.

## Development

```powershell
.\.venv\Scripts\Activate.ps1
python -m sam_backend --host 127.0.0.1 --port 8765
```

The frontend has no build step and is served from `frontend/`. API endpoints are
documented by FastAPI at <http://127.0.0.1:8765/api/docs> while SAM is running.

See [Architecture](docs/ARCHITECTURE.md) for the component map and extension
rules.
