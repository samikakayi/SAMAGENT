"""Context assembly: give the model what it needs, and little else.

Replaying an entire conversation into every request is the easy approach and
the wrong one. It grows without bound, pushes the important facts to the
middle where they are attended to least, and costs money on every turn.

This builds a focused prompt instead: the goal, the plan with its current
step, what has actually been observed, the errors that matter, and a compact
project description. Everything is budgeted, and the newest information wins
when the budget runs out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .project_map import ProjectMap
    from .tasks import AgentTask, TaskStep

# Roughly four characters per token. Deliberately conservative: overshooting
# the window costs a failed request, undershooting only costs a little detail.
CHARS_PER_TOKEN = 4
DEFAULT_BUDGET_TOKENS = 6_000
MAX_OBSERVATIONS = 12
MAX_ERRORS = 5


class ContextEngine:
    """Builds bounded, relevant message lists for the executor."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build(
        self,
        *,
        task: "AgentTask",
        step: "TaskStep",
        system_prompt: str,
        project_map: "ProjectMap | None" = None,
        capabilities_summary: str = "",
    ) -> list[dict[str, Any]]:
        sections: list[str] = [f"GOAL: {task.goal}"]
        if task.constraints:
            sections.append("CONSTRAINTS:\n" + "\n".join(f"- {item}" for item in task.constraints))
        if project_map is not None:
            sections.append("PROJECT:\n" + project_map.summary_text(max_routes=15))
        if capabilities_summary:
            sections.append(capabilities_summary)
        sections.append(self._plan_section(task))
        observations = self._observations_section(task)
        if observations:
            sections.append(observations)
        errors = self._errors_section(task)
        if errors:
            sections.append(errors)
        sections.append(
            f"CURRENT STEP ({step.index}/{len(task.plan)}, kind={step.kind}):\n{step.text}\n\n"
            "Call exactly one tool that advances this step, or reply with a short summary if it is already done."
        )

        user_content = self._fit("\n\n".join(sections))
        return [
            {"role": "system", "content": system_prompt + f"\nWorkspace: {self.settings.workspace_root}"},
            {"role": "user", "content": user_content},
        ]

    # -- sections ----------------------------------------------------------
    @staticmethod
    def _plan_section(task: "AgentTask") -> str:
        lines = ["PLAN:"]
        for item in task.plan:
            marker = {"done": "x", "active": ">", "failed": "!", "skipped": "-"}.get(item.status, " ")
            detail = f" -- {item.detail[:120]}" if item.detail and item.status in {"done", "failed"} else ""
            lines.append(f"[{marker}] {item.index}. {item.text}{detail}")
        return "\n".join(lines)

    @staticmethod
    def _observations_section(task: "AgentTask") -> str:
        if not task.observations:
            return ""
        # Newest last: the model reads the most recent facts closest to the
        # instruction that follows them.
        recent = task.observations[-MAX_OBSERVATIONS:]
        return "OBSERVED SO FAR:\n" + "\n".join(f"- {item[:400]}" for item in recent)

    @staticmethod
    def _errors_section(task: "AgentTask") -> str:
        if not task.errors:
            return ""
        recent = task.errors[-MAX_ERRORS:]
        return (
            "RECENT FAILURES (address the cause, do not repeat the same call):\n"
            + "\n".join(f"- {item[:500]}" for item in recent)
        )

    def _fit(self, text: str) -> str:
        """Trim from the middle, which is where the least decision-relevant
        material sits: the header states the goal, the tail states the task."""
        budget_chars = DEFAULT_BUDGET_TOKENS * CHARS_PER_TOKEN
        if len(text) <= budget_chars:
            return text
        head = int(budget_chars * 0.45)
        tail = budget_chars - head - 80
        return text[:head] + "\n\n...[context trimmed to fit the model window]...\n\n" + text[-tail:]
