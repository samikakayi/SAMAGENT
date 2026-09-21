"""Visual review: a model looks at the rendered page and says what is wrong.

The screenshot goes to whichever provider the router picks for vision work.
The only outcome that counts is a parsed verdict the model committed to. A
provider that is unreachable, a model that cannot see the image, or a reply
that is not a verdict all become SKIPPED with a reason -- a review that did
not happen is never reported as a pass.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .planner import extract_json_object
from .verification import CheckOutcome, CheckResult

REVIEW_SYSTEM_PROMPT = """You are the visual review stage of an autonomous software engineering agent.
You are shown a screenshot of a web page the agent just changed, plus the goal it was working toward.

Judge only what is visible: broken layout, overflow or clipping, unreadable contrast, missing or
placeholder content, error banners, overlapping elements, an empty page where content was expected.
Return ONLY a JSON object:
{"verdict": "pass" | "fail", "findings": ["concrete problem", ...]}
If no image is attached or you cannot see one, return exactly {"verdict": "no_image"}."""

MAX_FINDINGS = 10


@dataclass(slots=True)
class ReviewResult:
    outcome: CheckOutcome
    findings: list[str] = field(default_factory=list)
    reason: str = ""
    provider: str = ""

    @property
    def check(self) -> CheckResult:
        if self.outcome is CheckOutcome.SKIPPED:
            return CheckResult(kind="ui_review", command="", outcome=self.outcome, reason=self.reason)
        return CheckResult(
            kind="ui_review", command=f"visual review via {self.provider or 'model'}",
            outcome=self.outcome, exit_code=0 if self.outcome is CheckOutcome.PASSED else 1,
            failed_count=len(self.findings) or None, failures=self.findings,
            summary=("The screenshot passed visual review" if self.outcome is CheckOutcome.PASSED
                     else f"Visual review found {len(self.findings)} problem(s)"),
            stderr_tail="\n".join(self.findings),
        )


def _image_message(text: str, png: bytes) -> dict[str, Any]:
    """One user message carrying the image in the two shapes the existing
    adapters pass through unchanged: an OpenAI-style content array, and
    Ollama's top-level `images` list."""
    encoded = base64.b64encode(png).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ],
        "images": [encoded],
    }


async def review_screenshot(
    router: Any, *, screenshot: Path, goal: str, console_errors: list[str], task_id: str | None = None,
) -> ReviewResult:
    try:
        png = screenshot.read_bytes()
    except OSError as exc:
        return ReviewResult(CheckOutcome.SKIPPED, reason=f"The screenshot could not be read: {exc}")

    prompt = f"GOAL: {goal[:600]}"
    if console_errors:
        prompt += "\nBROWSER CONSOLE ALREADY REPORTED:\n" + "\n".join(f"- {item[:200]}" for item in console_errors[:10])
    prompt += "\nReview the attached screenshot."

    try:
        turn, choice, _fallbacks = await router.complete(
            # The word "screenshot" is what the router's profile keys vision on.
            message="review this screenshot of the page",
            messages=[{"role": "system", "content": REVIEW_SYSTEM_PROMPT}, _image_message(prompt, png)],
            tools=[],
            provider=None,
            model=None,
            conversation_id=None,
            task_id=task_id,
        )
    except Exception as exc:  # noqa: BLE001 - any provider failure is a skip, never a pass
        return ReviewResult(CheckOutcome.SKIPPED, reason=f"no vision-capable model was reachable ({type(exc).__name__}: {exc})"[:300])

    provider = str(getattr(choice, "provider", "") or "")
    payload = extract_json_object(turn.content or "")
    verdict = str((payload or {}).get("verdict", "")).strip().lower()
    if verdict == "no_image":
        return ReviewResult(CheckOutcome.SKIPPED, reason=f"the {provider or 'selected'} model reported it could not see the screenshot", provider=provider)
    if verdict not in {"pass", "fail"}:
        return ReviewResult(CheckOutcome.SKIPPED, reason=f"the {provider or 'selected'} model did not return a verdict", provider=provider)
    findings = [str(item).strip()[:300] for item in (payload or {}).get("findings") or [] if str(item).strip()][:MAX_FINDINGS]
    if verdict == "fail" and not findings:
        findings = ["The reviewer judged the page as failing without naming a specific problem."]
    return ReviewResult(
        CheckOutcome.FAILED if verdict == "fail" else CheckOutcome.PASSED,
        findings=findings, provider=provider,
    )
