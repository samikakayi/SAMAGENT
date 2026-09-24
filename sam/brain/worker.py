"""Background worker: multi-step jobs the voice turn should not wait for.

``delegate_task(goal)`` returns at once (NON_BLOCKING in Live: the model says
"started" and keeps talking); the worker then loops on the ``strong`` ladder
with the same tools as the conversation:

    plan -> act (tool calls) -> check the evidence -> next step

until the model calls the worker-only ``finish_task(ok, summary_ckb,
evidence)`` or ``worker.max_steps`` (25) run out. Honesty is enforced in code,
not only by prompt: a ``finish_task(ok=true)`` right after a failed action is
sent back once ("your last action failed"), because v1-style "done!" without
a result was the behaviour the user complained about. The final Sorani summary
is spoken (``SpeakRequest``) and shown/stored (``Transcript``, source worker).

``stop_all`` cancels the task (``Worker.cancel``) and any tool it is running
(``ToolRegistry.cancel_all``). One task at a time: two desktop-driving tasks
would fight over the mouse and keyboard.

``build_project`` (used by the hands ``build_project`` tool) lives in
``project_builder.py`` and is exposed here as ``Worker.build_project``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..events import Caption, SpeakRequest, Transcript, WorkerProgress, new_id
from ..textnorm import is_arabic_script
from .conversation import clean_tool_args
from . import taint
from .llm import LLMError
from .tools import ToolContext, fail, ok, tool

log = logging.getLogger("sam.worker")

FINISH = "finish_task"
EXCLUDED_TOOLS = frozenset({"delegate_task", "stop_all", "more_tools"})  # the worker gets every other tool
# Tools that only look; a finish right after one of these is fine even if an
# earlier action failed and was then re-checked.
READ_ONLY_TOOLS = frozenset({"screen_look", "chart_state", "recall", "strategy_get", "strategy_list",
                             "list_alerts", "get_price", "web_search"})
KEEP_FULL_TOOL_RESULTS = 6
OLD_RESULT_CHARS = 400

FINISH_SCHEMA: dict[str, Any] = {"type": "function", "function": {
    "name": FINISH,
    "description": "End the task. Call exactly once, when the goal is reached or cannot be reached.",
    "parameters": {"type": "object", "properties": {
        "ok": {"type": "boolean", "description": "true only if tool results show the goal was reached"},
        "summary_ckb": {"type": "string",
                        "description": "1-2 short spoken Sorani sentences (Arabic script) for the user: what was "
                                       "done, or what failed and what is needed."},
        "evidence": {"type": "string", "description": "Which tool results prove it (short, English)."}},
        "required": ["ok", "summary_ckb"]}}}

SORANI_STARTED = "دەستم پێکرد"
SORANI_CANCELLED = "کارەکە ڕاگیرا."
SORANI_OUT_OF_STEPS = "ببورە، کارەکە لە ماوەی دیاریکراودا تەواو نەبوو."
SORANI_NO_MODEL = "ببورە، نەمتوانی کارەکە بکەم چونکە هیچ مۆدێلێک وەڵامی نەدایەوە."
SORANI_DONE = "کارەکە تەواو بوو."
SORANI_FAILED = "ببورە، کارەکە سەرکەوتوو نەبوو."

DEFAULTS: dict[str, Any] = {
    "worker.ladder": "strong",
    "worker.build_ladder": "strong",
    "worker.reasoning": "low",
}


@dataclass
class WorkerTask:
    task_id: str
    goal: str
    context: str = ""
    source: str = "voice"
    started_at: float = field(default_factory=time.time)
    max_steps: int = 25
    step: int = 0
    state: str = "running"          # running|done|failed|cancelled
    summary_ckb: str = ""
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[Any] | None = None

    def public(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "goal": self.goal[:200], "state": self.state, "step": self.step,
                "max_steps": self.max_steps, "started_at": self.started_at, "summary_ckb": self.summary_ckb}


class Worker:
    """``app.worker`` (docs/CONTRACTS.md 3.2)."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._tasks: dict[str, WorkerTask] = {}

    # -- bookkeeping -------------------------------------------------------------------------
    def busy(self) -> WorkerTask | None:
        return next((t for t in self._tasks.values() if t.state == "running"), None)

    def status(self) -> list[dict[str, Any]]:
        return [t.public() for t in sorted(self._tasks.values(), key=lambda t: t.started_at, reverse=True)][:20]

    def _record(self, goal: str, context: str, source: str, task_id: str | None, max_steps: int) -> WorkerTask:
        task_id = task_id or new_id()
        record = self._tasks.get(task_id)
        if record is None:
            record = WorkerTask(task_id=task_id, goal=goal, context=context, source=source, max_steps=max_steps)
            self._tasks[task_id] = record
        record.max_steps = max_steps
        # Keep the list short (the island/panel only shows recent tasks).
        while len(self._tasks) > 50:
            oldest = min((t for t in self._tasks.values() if t.state != "running"),
                         key=lambda t: t.started_at, default=None)
            if oldest is None:
                break
            self._tasks.pop(oldest.task_id, None)
        return record

    def start(self, goal: str, *, context: str = "", source: str = "voice") -> str:
        """Run ``goal`` in the background; returns the task id at once."""
        max_steps = int(self.app.config.get("worker.max_steps", 25) or 25)
        record = self._record(goal, context, source, None, max_steps)
        record.task = self.app.spawn(self.run(goal, context=context, task_id=record.task_id), f"worker:{record.task_id}")
        return record.task_id

    def cancel(self, task_id: str | None = None) -> int:
        """Stop running task(s) (stop_all). Returns how many were running."""
        count = 0
        for record in list(self._tasks.values()):
            if record.state != "running" or (task_id is not None and record.task_id != task_id):
                continue
            record.cancel.set()
            if record.task is not None and not record.task.done() and record.task is not asyncio.current_task():
                record.task.cancel()
            count += 1
        return count

    def _progress(self, record: WorkerTask, text_ckb: str, *, done: bool = False, ok_: bool | None = None) -> None:
        self.app.bus.publish(WorkerProgress(task_id=record.task_id, step=record.step, max_steps=record.max_steps,
                                            text_ckb=text_ckb, done=done, ok=ok_))

    # -- the loop --------------------------------------------------------------------------------
    def _tool_schemas(self) -> list[dict[str, Any]]:
        names = [n for n in self.app.tools.names() if n not in EXCLUDED_TOOLS]
        return self.app.tools.openai_tools(names) + [FINISH_SCHEMA]

    def _step_text(self, name: str) -> str:
        spec = self.app.tools.get(name)
        return (spec.description_ckb if spec is not None and spec.description_ckb else name)[:80]

    @staticmethod
    def _compact(messages: list[dict[str, Any]]) -> None:
        """Keep the newest tool results whole; shorten older ones so 25 steps
        of screen text do not overflow the context window."""
        tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        for index in tool_indexes[:-KEEP_FULL_TOOL_RESULTS]:
            content = str(messages[index].get("content", ""))
            if len(content) > OLD_RESULT_CHARS:
                messages[index]["content"] = content[:OLD_RESULT_CHARS] + "…(older result shortened)"

    async def run(self, goal: str, *, context: str = "", max_steps: int | None = None,
                  task_id: str | None = None) -> dict[str, Any]:
        """Execute the loop; returns {"ok", "summary_ckb", "steps", "task_id"}
        (+ "cancelled" / "evidence")."""
        max_steps = int(max_steps or self.app.config.get("worker.max_steps", 25) or 25)
        # Its own taint scope; a task delegated from a turn that read untrusted
        # text starts tainted (sam/brain/taint.py).
        taint.begin(goal, inherit=True)
        record = self._record(goal, context, "worker", task_id, max_steps)
        turn = self.app.timing.turn("worker", turn_id=record.task_id)
        self._progress(record, SORANI_STARTED)
        system = self.app.persona.system_instruction("worker") if self.app.persona is not None else ""
        user = f"Goal: {goal}" + (f"\nContext from the conversation: {context}" if context else "") + (
            f"\nYou have at most {max_steps} steps. Work now; do not ask questions.")
        messages: list[dict[str, Any]] = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}]
        tools = self._tool_schemas()
        ladder = str(self.app.config.get("worker.ladder", "strong") or "strong")
        reasoning = str(self.app.config.get("worker.reasoning", "low") or "low")
        actions: list[dict[str, Any]] = []
        nudged = rejected_finish = False
        try:
            while record.step < max_steps:
                if record.cancel.is_set():
                    raise asyncio.CancelledError
                record.step += 1
                self._compact(messages)
                began = time.perf_counter()
                try:
                    response = await self.app.llm.chat(messages, ladder=ladder, tools=tools, reasoning=reasoning,
                                                       turn=turn)
                except LLMError as err:
                    log.info("worker %s: model failed (%s)", record.task_id, err.kind)
                    return await self._finish(record, False, SORANI_NO_MODEL, actions, evidence=err.kind)
                turn.add("worker_step", (time.perf_counter() - began) * 1000.0, step=record.step)
                messages.append(response.assistant_message())
                if not response.tool_calls:
                    if not nudged:
                        nudged = True
                        messages.append({"role": "user", "content": "Continue with tool calls, or call finish_task "
                                                                    "with ok and a Sorani summary_ckb."})
                        continue
                    text = (response.text or "").strip()
                    return await self._finish(record, self._last_ok(actions), text, actions)
                finish_args: dict[str, Any] | None = None
                for call in response.tool_calls:
                    if call.name == FINISH:
                        verdict = self._check_finish(call.arguments, actions, rejected_finish)
                        if verdict is None:
                            finish_args = call.arguments
                            result: dict[str, Any] = {"ok": True, "summary": "Task finished.", "data": None}
                        else:
                            rejected_finish = True
                            result = {"ok": False, "summary": verdict, "data": None}
                    elif call.name in EXCLUDED_TOOLS:
                        result = fail(f"{call.name} is not available inside a background task.")
                    else:
                        self._progress(record, self._step_text(call.name))
                        result = await self.app.tools.dispatch(
                            call.name, clean_tool_args(self.app.tools, call.name, call.arguments),
                            source="worker", call_id=call.id)
                        actions.append({"step": record.step, "name": call.name, "ok": bool(result.get("ok")),
                                        "summary": str(result.get("summary", ""))[:200],
                                        "args": call.arguments})
                        if result.get("data") and isinstance(result["data"], dict) and result["data"].get("cancelled"):
                            raise asyncio.CancelledError
                    messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                     "content": json.dumps(result, ensure_ascii=False, default=str)})
                if finish_args is not None:
                    return await self._finish(record, bool(finish_args.get("ok")),
                                              str(finish_args.get("summary_ckb") or ""), actions,
                                              evidence=str(finish_args.get("evidence") or ""))
            return await self._finish(record, False, SORANI_OUT_OF_STEPS, actions)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and hasattr(current, "uncancel"):
                current.uncancel()
            record.state = "cancelled"
            record.summary_ckb = SORANI_CANCELLED
            self._progress(record, SORANI_CANCELLED, done=True, ok_=False)
            self.app.db.log_activity("worker", "delegate_task", ok=False, summary="cancelled: " + goal[:200],
                                     source=record.source)
            return {"ok": False, "cancelled": True, "summary_ckb": SORANI_CANCELLED, "steps": record.step,
                    "task_id": record.task_id}
        finally:
            turn.finish(steps=record.step, state=record.state)

    @staticmethod
    def _last_ok(actions: list[dict[str, Any]]) -> bool:
        changing = [a for a in actions if a["name"] not in READ_ONLY_TOOLS]
        return bool(changing[-1]["ok"]) if changing else bool(actions and actions[-1]["ok"])

    @staticmethod
    def _check_finish(args: dict[str, Any], actions: list[dict[str, Any]], already_rejected: bool) -> str | None:
        """None = accept; else the reason sent back to the model (once)."""
        if already_rejected or not bool(args.get("ok")):
            return None
        changing = [a for a in actions if a["name"] not in READ_ONLY_TOOLS]
        if changing and not changing[-1]["ok"]:
            last = changing[-1]
            return (f"Not accepted: your last action {last['name']} failed ({last['summary']}). Fix it and verify, "
                    "or call finish_task with ok=false and say what failed.")
        return None

    async def _sorani(self, text: str, ok_: bool) -> str:
        """Make sure the summary is Sorani in Arabic script (one cheap rewrite
        on the 'sorani' ladder when the model answered in English)."""
        text = " ".join((text or "").split())[:400]
        if text and is_arabic_script(text):
            return text
        if text:
            try:
                response = await self.app.llm.chat(
                    [{"role": "system", "content": "Rewrite as 1-2 short, natural spoken Central Kurdish (Sorani) "
                                                   "sentences in Arabic script. Output only the sentences."},
                     {"role": "user", "content": text}], ladder="sorani", reasoning="low", timeout_s=20)
                rewritten = " ".join((response.text or "").split())[:400]
                if rewritten and is_arabic_script(rewritten):
                    return rewritten
            except LLMError:
                pass
        return SORANI_DONE if ok_ else SORANI_FAILED

    async def _finish(self, record: WorkerTask, ok_: bool, summary: str, actions: list[dict[str, Any]], *,
                      evidence: str = "") -> dict[str, Any]:
        summary_ckb = await self._sorani(summary, ok_)
        record.state = "done" if ok_ else "failed"
        record.summary_ckb = summary_ckb
        bus = self.app.bus
        self._progress(record, summary_ckb, done=True, ok_=ok_)
        bus.publish(Transcript(role="assistant", text=summary_ckb, source="worker", turn_id=record.task_id))
        bus.publish(Caption(text=summary_ckb, role="assistant", final=True))
        bus.publish(SpeakRequest(text_ckb=summary_ckb, source="worker"))
        self.app.db.log_activity("worker", "delegate_task", ok=ok_, summary=self.app.redact(summary_ckb)[:300],
                                 detail={"goal": self.app.redact(record.goal)[:500], "steps": record.step,
                                         "evidence": self.app.redact(evidence)[:300],
                                         "actions": [{k: a[k] for k in ("name", "ok")} for a in actions[-25:]]},
                                 source=record.source)
        return {"ok": ok_, "summary_ckb": summary_ckb, "steps": record.step, "task_id": record.task_id,
                "evidence": evidence}

    # -- project generation (hands build_project) -----------------------------------------------------
    async def build_project(self, description: str, *, project_dir: str | Path, kind: str = "website",
                            name: str = "", progress: Callable[..., Any] | None = None,
                            cancel: asyncio.Event | None = None, source: str = "worker") -> dict[str, Any]:
        """Generate a multi-file project into ``project_dir`` (see
        project_builder.build_project for the result keys)."""
        from .project_builder import build_project

        return await build_project(self.app, description, project_dir=project_dir, kind=kind, name=name,
                                   progress=progress, cancel=cancel, source=source)


@tool("delegate_task",
      description="Hand a multi-step job to SAM's background worker: building a website or app, a long desktop "
                  "procedure, research across several pages. Returns at once; the worker reports the result to "
                  "the user in Sorani when done. Put every detail the user gave into 'goal'.",
      description_ckb="کارێکی چەند هەنگاوی لە پاشبنەمادا",
      params={"type": "object", "properties": {
          "goal": {"type": "string", "description": "The complete goal with all details the user gave."},
          "context": {"type": "string", "description": "Useful facts from the conversation (optional)."}},
          "required": ["goal"]},
      risk="safe", blocking=False, timeout_s=10,
      examples_ckb=("ماڵپەڕێکی سادە بۆ دوکانەکەم دروست بکە", "سێ سەرچاوە لەسەر هەواڵی زێڕ بدۆزەوە و کورتی بکەرەوە"))
async def delegate_task(ctx: ToolContext, goal: str, context: str = "") -> dict[str, Any]:
    worker = getattr(ctx.app, "worker", None)
    if worker is None:
        return fail("The background worker is not available.")
    running = worker.busy()
    if running is not None:
        return fail(f"Another background task is still running: {running.goal[:120]}. Ask the user whether to "
                    "wait or stop it.", running_task_id=running.task_id)
    task_id = worker.start(goal, context=context, source=ctx.source)
    return ok("Started in the background; the user will hear the result when it is done. Tell the user in a few "
              "words that you have started.", task_id=task_id)


def register(app: Any) -> None:
    app.config.register_defaults(DEFAULTS)
    app.worker = Worker(app)
    app.tools.add(delegate_task, owner="brain.worker")


async def stop(app: Any) -> None:
    if app.worker is not None:
        app.worker.cancel()


__all__ = ["Worker", "WorkerTask", "register", "stop", "delegate_task", "FINISH_SCHEMA", "DEFAULTS"]
