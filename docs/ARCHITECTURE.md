# SAM architecture

SAM is a local-first Windows agent. The browser UI is only a control surface;
conversation data, memory, approvals, and audit events are stored locally in
SQLite.

```text
Browser UI (127.0.0.1)
        |
        v
Local API / Agent loop
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

- `sam_backend/app.py`: local HTTP API and static UI hosting.
- `sam_backend/agent.py`: bounded model/tool loop and planning context.
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
