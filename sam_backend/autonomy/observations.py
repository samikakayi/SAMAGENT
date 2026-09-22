"""What the model is told about a tool call it just made.

This is the agent's entire view of what happened: the orchestrator decides
whether a run continues, but these few lines decide what it continues on.
Getting them wrong is expensive and quiet -- summarising a file read down to
its path once left the model re-reading the same file instead of acting on
text it had already asked for, with nothing in the timeline to show why.

Two rules hold throughout. Output is bounded, because one read must not
crowd the next prompt. And a result marked sensitive never appears here at
all: the agent is told the call happened and nothing more.
"""

from __future__ import annotations

from typing import Any

# How much of a file's text an observation may carry back to the model. Large
# enough to act on, small enough that one read cannot swamp the next prompt.
OBSERVATION_CONTENT_LIMIT = 4000

SENSITIVE_SUCCESS = "succeeded; its output is sensitive and was withheld from context"
SENSITIVE_FAILURE = "the call failed; its details are sensitive and were withheld"
UNEXPLAINED_FAILURE = "The tool reported a failure without a message."


def summarise_result(name: str, result: Any) -> str:
    """The text form of a successful tool result, bounded by kind."""
    output = result.output
    if isinstance(output, dict):
        if name == "run_tests":
            return str(output.get("summary") or "checks finished")[:400]
        if name == "project_map":
            return f"{output.get('file_count', '?')} files, commands {output.get('commands', {})}"
        if "content" in output:
            # The point of reading a file is the text inside it. Summarising
            # to a path told the model nothing and left it re-reading the
            # same file instead of acting on what it had asked for.
            content = str(output.get("content") or "")
            return f"{output.get('path')}:\n{content[:OBSERVATION_CONTENT_LIMIT]}"
        if "path" in output:
            return f"{output.get('path')} ({output.get('bytes', '?')} bytes)"
        if "exit_code" in output:
            return f"exit {output.get('exit_code')}"
        if "commits" in output:
            return f"{len(output['commits'])} commit(s)"
        if "changed" in output:
            return f"{len(output['changed'])} changed path(s) on {output.get('branch', '?')}"
    text = result.model_text() if hasattr(result, "model_text") else str(output)
    return text.strip()[:400] or "done"


def describe_success(outcome: Any) -> str:
    """What a call that worked is allowed to tell the model."""
    if outcome.sensitive:
        return SENSITIVE_SUCCESS
    return summarise_result(outcome.call.name, outcome.result)


def describe_failure(outcome: Any) -> str:
    """What a call that failed is allowed to tell the model."""
    if outcome.sensitive:
        return SENSITIVE_FAILURE
    return outcome.result.error or UNEXPLAINED_FAILURE
