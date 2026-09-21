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
