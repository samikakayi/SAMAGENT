"""SAM 2 brain: tool registry, confirmation broker, text LLM client (core,
written by the foundation stage) plus persona, memory, conversation and worker
(brain builder). Each brain module that needs app wiring exposes its own
``register(app)``; see ``sam.app.PACKAGES`` and docs/CONTRACTS.md."""
