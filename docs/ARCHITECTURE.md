# SAM architecture

SAM is a local-first Windows agent. The browser UI is only a control surface;
conversation data, memory, approvals, and audit events are stored locally in
SQLite.

```text
Browser UI (127.0.0.1)  <--- live activity over /ws/live
        |
        v
Local API
        |---- Agent loop (chat turns) ------.
        |                                    |
        |---- Autonomous orchestrator -------+--> ToolExecutor --> Policy
        |         (multi-step goals)              (one owner of        |
        |                                          tool safety)        v
        |                                                        Approval queue
        |         |
        |         |-- Project scanner --- cached project map
        |         |-- Planner ----------- plan / re-plan
        |         |-- Context engine ---- bounded per-step prompt
        |         |-- Task store -------- durable state + timeline
        |         `-- Verifier ---------- runs the project's own checks
        |
        |---- Model router ----- Ollama / LiteLLM / OpenRouter / OpenAI
        |
        `---- Tool broker ---- Policy engine ---- Approval queue
                                  |                    |
                                  v                    v
                          workspace / shell /       human decision
                          Python / browser / Windows
                          TradingView / MT5 data
                                  |
                                  v
                         SQLite + audit chain
```

## Trust boundaries

- Model output is untrusted. Models request structured tools; they do not call
  operating-system APIs directly.
- File and web content is untrusted data. Instructions found in those sources
  cannot alter the policy or approve a pending action.
- The tool broker normalizes arguments, determines risk, records the decision,
  and either executes, queues one exact approval, or blocks the request.
- SAM runs as the signed-in standard Windows user. It never performs automatic
  UAC elevation.
- The service binds to loopback by default. Do not expose it to a LAN or the
  public internet.
- AUTO is local-first when Ollama is available. Selecting or falling back to a
  configured cloud provider is an explicit privacy boundary because message
  context is sent to that provider.

## Main components

- `sam_backend/app.py`: local HTTP API, service wiring and static UI hosting.
- `sam_backend/agent_api.py`: HTTP surface for autonomous runs (tasks, project
  map, environment) plus the live-event fan-out and the background-run
  registry. Wired into `create_app` as a router.
- `sam_backend/execution.py`: the single owner of "safely run a tool call" --
  policy, approvals, execution and audit.
- `sam_backend/agent.py`: bounded model/tool loop for conversational turns.
- `sam_backend/autonomy/`: the autonomous runtime.
  - `orchestrator.py` -- the understand -> scan -> plan -> execute -> validate
    -> fix -> retry state machine, plus the lifecycle around it (single-driver
    ownership, cancellation, approvals, restart handling).
  - `checkpoints.py` -- workspace snapshot / diff / restore mechanics.
- `sam_backend/tools/`: the tool layer.
  - `registry.py` -- the engine: dispatch, workspace containment, process
    execution and every handler.
  - `catalogue.py` -- the data: model-facing schemas and per-tool permission
    and manifest metadata.
- `sam_backend/planner.py`: model-driven planning and re-planning, with a
  deterministic fallback so planning never hard-fails.
- `sam_backend/tasks.py`: durable task state, a validated state machine, and
  the activity timeline the UI renders.
- `sam_backend/project_map.py`: cached project intelligence (structure,
  dependencies, commands, routes, tests, git state).
- `sam_backend/verification.py`: runs the project's own checks and parses them
  into a structured verdict.
- `sam_backend/context_engine.py`: assembles a bounded, relevant prompt per
  step instead of replaying the whole conversation.
- `sam_backend/capabilities.py`: probes which developer tools and model
  providers this machine can actually use.
- `sam_backend/models.py` and `routing.py`: Ollama, LiteLLM, OpenRouter, and
  OpenAI adapters, capability routing, fallback, and cost budgets.
- `sam_backend/trading/`: read-only MT5 data, deterministic indicators,
  structure/theory analysis, setup state, and TradingView observation/control.
- `sam_backend/windows_control.py`: process, native-window, installed-app,
  clipboard, screenshot, and optional UI Automation capabilities.
- `sam_backend/tools.py`: structured file, search, shell, Python, browser, app,
  plan, and memory tools.
- `sam_backend/policy.py`: path containment, secret-path detection, command
  classification, approvals, and hard denies.
- `sam_backend/db.py`: conversations, memories, approvals, settings,
  and audit events.
- `frontend/`: dependency-free responsive UI with optional browser speech input
  and output.

## Runtime flow

1. The user sends a message in a conversation.
2. SAM builds a prompt from the conversation, relevant local memories, tool
   definitions, and its safety instructions.
3. The selected model returns text or a structured tool call.
4. The broker classifies every tool call. Allowed calls run; approval-required
   calls are frozen; prohibited calls are rejected.
5. Tool output is size-limited, treated as untrusted, audited, and returned to
   the model for the next bounded step.
6. The assistant response, tool events, and plan state are persisted locally.

## Extending SAM

Add a tool by defining a strict JSON schema and an executor, then registering a
policy rule before exposing it to a model. A tool without an explicit policy is
denied. Prefer narrow tools such as `read_file` and `apply_patch` over a shell
command. Never put API keys, cookies, passwords, or approval tokens in a tool
description, prompt, model-visible result, or audit record.


## Autonomous runs

A goal posted to `POST /api/tasks` starts a run that outlives the request.
The orchestrator moves it through a validated state machine
(`IDLE -> UNDERSTANDING -> SCANNING -> PLANNING -> EXECUTING -> OBSERVING ->
VALIDATING -> COMPLETED`, with `FIXING`/`RETRYING` on failure and
`WAITING_FOR_APPROVAL` when policy demands a human decision). Every
transition and event is persisted, so a run can be inspected afterwards,
survives a restart, and resumes after an approval.

Three independent bounds stop a run: tool steps, self-heal retries, and
re-plans. Exhausting one reports honestly rather than continuing silently.

Completion is earned. A task is only `completed_verified` when the
verification engine actually executed the project's declared checks and they
passed; when nothing could be verified the run reports
`completed_unverified` and says so in its summary.

### Secret handling in the new surfaces

- The project map records environment variable *names* only; no `.env` value
  is ever read into it, because the map is summarised into model prompts.
- `git_diff` output passes through `redact_secrets` before it reaches the
  model or the UI, so a tracked credential cannot leak through a diff.


## Structure of the autonomous stack

Both loops -- the conversational one in `agent.py` and the orchestrator in
`autonomy.py` -- call **one** `ToolExecutor` (`execution.py`) to run a tool.
It evaluates policy, queues an approval when one is required, and on
resolution enforces the three guarantees that make an approval meaningful:
single use (a second click cannot replay the tool), hash binding (the request
must still match the tool and arguments it was issued for), and re-evaluation
(policy runs again at execution time). It writes the tool and approval audit
entries and decides what counts as sensitive.

The loops never re-implement that sequence. They consume a `ToolOutcome` and
decide only how the result *reads* -- a chat message, or a timeline event and
a state transition. That split is why the two can no longer drift apart the
way they did when each carried its own copy.

Ownership, so later passes do not duplicate it again:

| Concern | Owner |
| --- | --- |
| Tool safety: policy, approvals, audit, sensitivity | `execution.ToolExecutor` |
| Run state, plan, timeline, checkpoints | `tasks.TaskStore` / `AgentTask` |
| Driving a run; one driver per task id | `autonomy.AutonomousOrchestrator` |
| HTTP shape of runs; background drivers; live fan-out | `agent_api.AgentApi` |
| Which checks exist and what they prove | `verification.VerificationEngine` |
| Service construction and wiring | `app.create_app` |
| Undoing a run's file changes | `autonomy.checkpoints.WorkspaceCheckpoints` |
| What tools exist and how they are classified | `tools.catalogue` |

Data flows one way: HTTP -> orchestrator -> executor -> tools, with results
travelling back as `ToolOutcome` and reaching the UI as task events. The API
layer holds no run state; the orchestrator holds no HTTP concerns.


### Why these two were split, and no further

`autonomy/` separates the *mechanics* of undoing work from the *orchestration*
around it: `WorkspaceCheckpoints` snapshots, diffs and restores files and knows
nothing about tasks, events, audit or approvals. The drive loop and the
lifecycle deliberately stay in one module -- `run -> drive -> execute ->
observe -> drive` is a single state machine with one owner, and separating it
would only produce two objects calling each other back.

`tools/` separates the declarative catalogue from the execution engine: adding
a tool's schema and changing how tools run are different jobs. The handlers
themselves stay together because they share the registry's containment and
process helpers; grouping them by domain would scatter that shared core
without making any one group easier to read.

## Frozen decisions

These were each reached by tracing the real code, and each is a decision not
to go further. They are recorded so the next reader knows the current shape is
deliberate rather than unfinished.

**One composition root.** `create_app` builds the services and registers route
groups; it implements no endpoint. `api/` holds those groups -- system,
conversations, oversight, settings, providers, voice, desktop, security, and a
`trading/` package split into market, chart, replay, strategies and record.
Every group is registered the same way and names its dependencies at the top
of its register function.

**One owner per fact.**

- `application.state.adapters` is the only answer to "which adapter registry is
  current". Routes read it rather than holding a snapshot, because the registry
  is replaced wholesale when a credential changes.
- `MarketAnalyst.latest` is the only latest analysis. The chart draws from it
  and a monitored setup is created from it; creating one before any analysis is
  a 409, and that ordering is why those route groups must share one service.
- The persisted calibration row is the only calibration. Manual anchors and OCR
  both write it under the same viewport key, so either can unblock a drawing.
- `TaskStore.save` is the only place that decides whether a task may claim it
  was verified, and the exemption for older rows is read from the stored row,
  never from the object in hand.

**Single owners that were deliberately not split.** `AutonomousOrchestrator`
remains the one owner of the run state machine; what is left in it after
`observations.py` and `control.py` came out is lifecycle decisions, and
splitting those would produce objects calling each other back. `TradingService`
stays a facade because twenty-two of its twenty-four methods are exactly that,
and the two that were not are now `MarketAnalyst`. `DrawingEngine` is one
drawing engine: its pixel verification is drawing-domain knowledge, not a
generic utility. `TradingViewController` is one controller for one window.

**Desktop access has three styles, on purpose for now.** `DrawingEngine` takes
an injected `DesktopInput`; `TradingViewController` reaches the OS through its
`_modules()` seam; a few raw `ctypes`/clipboard calls sit below both. All three
are correct and testable enough that the behaviour above them is pinned.
Unifying them is a real option, but it should wait for a concrete need -- a
feature or a test that cannot be written otherwise -- rather than being built
speculatively.

**Four methods are not unit-executed, and that is accepted.**
`_send_unicode`, the two clipboard calls and `_open_symbol_search` sit directly
on `SendInput`, `win32clipboard`, and screenshot-plus-OCR. Driving them under
test would exercise a mock of Windows rather than any decision SAM makes. What
protects them is the layer above: every gate, refusal and state assembly that
decides *whether* to call them is covered, and a failure inside them surfaces
as the fail-closed error the caller already asserts. If a future change needs
their internals verified, that is the moment to introduce one shared desktop
adapter -- not before.

**History is kept as it was recorded.** Task rows written before the
verification invariant existed still say `completed_verified` with no evidence.
They load, they serve, and they can be rolled back; they are not rewritten.

## What rollback guarantees

Rollback restores files from copies saved before the run touched them. The
copies live outside the workspace and can go missing independently of the task
record -- a tidied data directory, a pruned backup, a half-copied profile --
so the record can outlive the evidence it refers to.

**Guaranteed.** Every entry is checked before anything is written. A file the
run created needs no saved copy, because undoing it means deleting it; a file
that already existed needs its copy still present and still openable. If any
required copy is missing, unreadable, or recorded as `existed` with no copy at
all, the whole rollback is refused and *no file is touched* -- not even the
ones that could still be restored. The refusal names the workspace file, never
the internal copy's path, and reaches the caller as the same
`{"rolled_back": false, "reason": ...}` shape used for "stop the run first".
A refused rollback writes no event, no audit entry and no `rolled_back` flag,
so nothing in the record claims it happened. The diff view applies the same
rule: a file whose original is gone reports `original_available: false` and an
empty diff rather than diffing against nothing, which would paint every line
as something the run added.

**Not guaranteed.** This is not a filesystem transaction. Preflight rules out
what is knowable in advance; it cannot rule out a failure that begins after the
checks pass. If the disk fills, a file is locked, or permissions change between
the check and the write, the restore stops partway: files already written stay
written, and because `shutil.copyfile` truncates its target before copying, the
file being written when the failure hits can be left truncated. The caller is
told the rollback failed, but not which files reached which state.

That residual case is reliability debt, not a security or data-integrity
boundary. It needs a filesystem to fail in the window between a successful
check and the write that follows; every path involved is a local file under the
user's own account, nothing is remote or concurrent, and the failure is loud --
an OS error, not silent corruption. There is no evidence it has occurred here.

Closing it properly means choosing a guarantee and building for it. Staging
each replacement to a temporary file and finishing with `os.replace` makes any
*single* file's swap atomic, which removes the truncated-file case but still
leaves a multi-file restore able to stop halfway. Making the whole set
all-or-nothing needs more: keeping the pre-restore contents so a failure can be
compensated back, or staging every file in memory first, which only works while
the files are small enough to hold. A journal with compensating undo is the
general answer and the heaviest. The honest label for the first two is
*best-effort compensated multi-file rollback*; only the last earns the word
*transactional*, and none of it should be built before something demonstrates
the current behaviour is actually costing someone a file.

## Known debt

Real, none blocking. Each is here because it is worth knowing, not because it
is scheduled.

1. **`AutonomousOrchestrator` is ~787 lines.** What remains is the state
   machine. Reducing it further means modelling stages as data rather than
   methods, which is a design change and needs its own decision, not another
   extraction pass.
2. **Four TradingView methods are not unit-executed** (see above). Accepted
   boundary, not a gap to close with mocks.
3. **Three desktop-access styles in `trading/`** (see above). Correct today;
   unify only when something concrete requires it.
4. **A repeated decision on an already-decided approval answers 200.** It
   executes nothing and the approval keeps its original outcome, so it fails
   closed; the response is simply more optimistic than the truth. Predates this
   work.
5. **`DirectToolRequest` is a schema with no route.** There is no direct
   tool-execution endpoint at all, which is why it is dead rather than
   dangerous.
6. **A few unused imports** in `trading/analysis.py`, `trading/replay.py`,
   `windows_control.py`, `tools/registry.py` and `api/desktop.py`. Inert; left
   alone rather than swept up in a stabilisation pass.
7. **Rollback is preflighted, not transactional.** An unexpected I/O failure
   during the write phase -- disk full, a lock taken after the check, a
   permission change -- can still leave a partial restore, and the file being
   written at the time can be left truncated. Known-invalid checkpoint
   evidence is caught before any mutation; an unpredictable filesystem is not.
   See the section above for what a future fix would have to promise.
8. **`TradingViewState` declares fields `observe()` never assigns**:
   `chart_type`, `visible_price_range`, `visible_time_range`,
   `price_scale_geometry`, `time_scale_geometry`, `visible_indicators` and
   `layout`. They serialise as null and read like observations that were made.
   A sibling field, `chart_geometry`, caused a real defect this way -- the
   panel gated drawing availability on it and contradicted a calibrated chart
   -- so that one was removed. The rest are left until something needs them,
   but they carry the same trap.
