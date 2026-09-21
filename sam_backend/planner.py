"""Planning: turning a goal plus a project into an ordered, checkable plan.

The model proposes the plan, but the plan is not trusted blindly. It is
parsed, validated, bounded and normalised into TaskStep records, and if the
model returns nothing usable a deterministic fallback plan is produced from
the project map instead. Planning must never be the thing that fails a run.

Re-planning exists because reality diverges from plans: when validation fails
the planner is given the failure and asked for a corrected plan rather than
the orchestrator blindly repeating the same steps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .project_map import ProjectMap
from .tasks import TaskStep

MAX_STEPS = 14
MIN_STEPS = 1

PLANNER_SYSTEM_PROMPT = """You are the planning stage of an autonomous software engineering agent.
Given a goal and a factual description of the project, produce a concrete, ordered plan.

Rules:
- Return ONLY a JSON object: {"understanding": "...", "steps": [{"text": "...", "kind": "..."}]}
- kind is one of: inspect, edit, command, test, verify, browser, report
- Between 2 and 12 steps. Each step is one concrete action, not a vague intention.
- Inspect before editing. Always include at least one verification step (test/verify) when code changes.
- Prefer the project's own declared commands over invented ones.
- Do not invent files, routes or commands that were not described to you."""

REPLAN_SYSTEM_PROMPT = """You are the re-planning stage of an autonomous software engineering agent.
A previous attempt failed. Given the goal, what was already tried, and the exact failure, produce a corrected plan.

Rules:
- Return ONLY a JSON object: {"diagnosis": "...", "steps": [{"text": "...", "kind": "..."}]}
- kind is one of: inspect, edit, command, test, verify, browser, report
- Address the actual root cause named in the failure. Do not simply repeat the failed steps.
- Between 1 and 10 steps."""

VALID_KINDS = {"inspect", "edit", "command", "test", "verify", "browser", "report"}


@dataclass(slots=True)
class PlanResult:
    steps: list[TaskStep]
    understanding: str = ""
    source: str = "model"  # model | fallback
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.as_dict() for step in self.steps],
            "understanding": self.understanding,
            "source": self.source,
            "note": self.note,
        }


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model reply.

    Models wrap JSON in prose or fenced blocks often enough that requiring a
    clean response would make planning fail for cosmetic reasons.
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    if start != -1:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def steps_from_payload(payload: dict[str, Any]) -> list[TaskStep]:
    """Normalise whatever the model returned into bounded TaskStep records."""
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return []
    steps: list[TaskStep] = []
    for item in raw_steps:
        if isinstance(item, str):
            text, kind = item.strip(), "action"
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("step") or item.get("description") or "").strip()
            kind = str(item.get("kind") or item.get("type") or "action").strip().lower()
        else:
            continue
        if not text:
            continue
        if kind not in VALID_KINDS:
            kind = "action"
        steps.append(TaskStep(index=len(steps) + 1, text=text[:400], kind=kind))
        if len(steps) >= MAX_STEPS:
            break
    return steps


def fallback_plan(goal: str, project_map: ProjectMap | None) -> PlanResult:
    """A useful plan when the model is unavailable or unparseable.

    This is deliberately generic but never empty: an agent that cannot plan
    should still inspect, act and verify rather than refuse to start.
    """
    steps = [
        TaskStep(index=1, text="Inspect the project structure and locate the files relevant to the goal", kind="inspect"),
        TaskStep(index=2, text=f"Implement the change required by: {goal[:200]}", kind="edit"),
    ]
    test_command = (project_map.commands.get("test") if project_map else None)
    if test_command:
        steps.append(TaskStep(index=3, text=f"Run the project's checks ({test_command})", kind="test"))
    else:
        steps.append(TaskStep(index=3, text="Verify the change by inspecting the result", kind="verify"))
    steps.append(TaskStep(index=len(steps) + 1, text="Report exactly what changed and what was verified", kind="report"))
    return PlanResult(
        steps=steps,
        understanding=f"Goal: {goal[:300]}",
        source="fallback",
        note="The model did not return a usable plan, so a structural plan was derived from the project map.",
    )


class Planner:
    """Produces and revises plans. Model-driven, with a guaranteed floor."""

    def __init__(self, router: Any) -> None:
        self.router = router

    async def _ask(
        self, system_prompt: str, user_prompt: str, *, task_id: str | None,
        provider: str | None = None, model: str | None = None,
    ) -> str:
        turn, _choice, _fallbacks = await self.router.complete(
            message=user_prompt,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Planning is pure reasoning: offering tools here invites the model
            # to start acting before a plan exists.
            tools=[],
            provider=provider,
            model=model,
            conversation_id=None,
            task_id=task_id,
        )
        return turn.content or ""

    async def plan(
        self,
        goal: str,
        *,
        project_map: ProjectMap | None = None,
        capabilities_summary: str = "",
        constraints: list[str] | None = None,
        task_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> PlanResult:
        context_lines = [f"GOAL: {goal}"]
        if project_map:
            context_lines.append("\nPROJECT:\n" + project_map.summary_text())
        if capabilities_summary:
            context_lines.append("\nENVIRONMENT:\n" + capabilities_summary)
        if constraints:
            context_lines.append("\nCONSTRAINTS:\n" + "\n".join(f"- {item}" for item in constraints))
        prompt = "\n".join(context_lines)

        try:
            reply = await self._ask(PLANNER_SYSTEM_PROMPT, prompt, task_id=task_id,
                                    provider=provider, model=model)
        except Exception as exc:  # noqa: BLE001 - any provider failure must still yield a plan
            result = fallback_plan(goal, project_map)
            result.note = f"Planning model unavailable ({type(exc).__name__}); used a structural plan."
            return result

        payload = extract_json_object(reply)
        steps = steps_from_payload(payload) if payload else []
        if len(steps) < MIN_STEPS:
            return fallback_plan(goal, project_map)
        return PlanResult(
            steps=steps,
            understanding=str((payload or {}).get("understanding", ""))[:1000],
            source="model",
        )

    async def replan(
        self,
        goal: str,
        *,
        failure: str,
        attempted: list[TaskStep],
        project_map: ProjectMap | None = None,
        task_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> PlanResult:
        attempted_text = "\n".join(
            f"- [{step.status}] {step.text}" + (f" ({step.detail[:200]})" if step.detail else "")
            for step in attempted
        ) or "- (nothing completed)"
        prompt = (
            f"GOAL: {goal}\n\nALREADY ATTEMPTED:\n{attempted_text}\n\n"
            f"FAILURE:\n{failure[:4000]}\n"
        )
        if project_map:
            prompt += "\nPROJECT:\n" + project_map.summary_text()

        try:
            reply = await self._ask(REPLAN_SYSTEM_PROMPT, prompt, task_id=task_id,
                                    provider=provider, model=model)
        except Exception as exc:  # noqa: BLE001
            return PlanResult(
                steps=[
                    TaskStep(index=1, text=f"Diagnose the failure: {failure[:200]}", kind="inspect"),
                    TaskStep(index=2, text="Apply the smallest fix that addresses the root cause", kind="edit"),
                    TaskStep(index=3, text="Re-run the checks", kind="test"),
                ],
                understanding="",
                source="fallback",
                note=f"Re-planning model unavailable ({type(exc).__name__}).",
            )

        payload = extract_json_object(reply)
        steps = steps_from_payload(payload) if payload else []
        if not steps:
            return PlanResult(
                steps=[
                    TaskStep(index=1, text=f"Diagnose the failure: {failure[:200]}", kind="inspect"),
                    TaskStep(index=2, text="Apply the smallest fix that addresses the root cause", kind="edit"),
                    TaskStep(index=3, text="Re-run the checks", kind="test"),
                ],
                source="fallback",
                note="The model did not return a usable revised plan.",
            )
        return PlanResult(steps=steps, understanding=str((payload or {}).get("diagnosis", ""))[:1000], source="model")
