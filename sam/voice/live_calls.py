"""Tool calls and queued [SAM] text turns of ``LiveVoice`` (split out of
live.py to keep it under ~700 lines). ``LiveCalls`` is a mixin: the state it
uses (``_calls``, ``_say_queue``, ``_generation``...) is created by
``LiveVoice.__init__``.

Tool calls: ``app.tools.dispatch(..., source="live")`` runs concurrently; BLOCKING
calls of one message are answered together, NON_BLOCKING ones each as soon as
they finish (with scheduling WHEN_IDLE on models that support async tools). A
call the server cancels while it waits for the user's yes/no is DETACHED: it
still resolves on the user's own words and its result is told to the model as
a [SAM] text turn.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from dataclasses import dataclass
from typing import Any

from ..events import ToolStarted
from .live_config import RESULT_NOTE, declared_args, supports_async_tools

log = logging.getLogger("sam.voice.live")


@dataclass
class _Call:
    id: str
    name: str
    args: dict[str, Any]
    blocking: bool
    generation: int
    task: "asyncio.Future[Any] | None" = None
    started: bool = False
    detached: bool = False


class LiveCalls:
    """Mixin for ``LiveVoice``."""

    app: Any
    hooks: Any
    model: str
    _calls: dict[str, _Call]
    _generation: int
    _model_active: bool
    _say_queue: collections.deque[tuple[str, str, str, float]]
    _say_timer: asyncio.TimerHandle | None

    # -- tools ------------------------------------------------------------------------------------------
    def _on_tool_started(self, event: ToolStarted) -> None:
        call = self._calls.get(event.call_id)
        if call is not None:
            call.started = True

    def _on_tool_call(self, tool_call: Any) -> None:
        self._mark_responded()
        self._finish_user()
        self._model_active = True
        calls: list[_Call] = []
        for fc in getattr(tool_call, "function_calls", None) or []:
            if not getattr(fc, "id", None):
                log.warning("live tool call without id ignored: %s", getattr(fc, "name", "?"))
                continue
            spec = self.app.tools.get(fc.name or "")
            call = _Call(id=fc.id, name=fc.name or "", args=declared_args(spec, dict(fc.args or {})),
                         blocking=spec.blocking if spec is not None else True, generation=self._generation)
            call.task = asyncio.ensure_future(
                self.app.tools.dispatch(call.name, call.args, source="live", call_id=call.id))
            self._calls[call.id] = call
            calls.append(call)
        if not calls:
            return
        self.hooks.set_state("working", ", ".join(c.name for c in calls))
        async_ok = supports_async_tools(self.model)
        blocking = [c for c in calls if c.blocking or not async_ok]
        if blocking:
            asyncio.ensure_future(self._respond(blocking))
        for call in calls:
            if call not in blocking:
                asyncio.ensure_future(self._respond([call]))

    async def _respond(self, calls: list[_Call]) -> None:
        """BLOCKING calls of one message are answered together (docs send one
        FunctionResponse list); NON_BLOCKING ones each as soon as they finish."""
        results = await asyncio.gather(*(c.task for c in calls if c.task is not None), return_exceptions=True)
        types = self._types()
        responses = []
        for call, result in zip(calls, results):
            self._calls.pop(call.id, None)
            if isinstance(result, asyncio.CancelledError):
                continue  # cancelled at the server's request before it ran
            if isinstance(result, BaseException):
                result = {"ok": False, "summary": f"{call.name} failed: {type(result).__name__}", "data": None}
            if call.detached or call.generation != self._generation or not self.ready:
                self._queue_note(RESULT_NOTE.format(name=call.name, summary=str(result.get("summary", ""))[:300]))
                continue
            kwargs: dict[str, Any] = {"id": call.id, "name": call.name, "response": result}
            if not call.blocking and supports_async_tools(self.model):
                kwargs["scheduling"] = types.FunctionResponseScheduling.WHEN_IDLE
            responses.append(types.FunctionResponse(**kwargs))
        if responses:
            await self._send(lambda s: s.send_tool_response(function_responses=responses))
        if not self._calls:
            self.hooks.set_state("speaking" if self.speaker.playing else "thinking")

    def _on_cancellation(self, ids: list[str]) -> None:
        pending_tools = {p.get("tool_name") for p in self.app.confirm.pending()}
        for call_id in ids:
            call = self._calls.get(call_id)
            if call is None or call.task is None:
                continue
            if call.started or call.name in pending_tools:
                call.detached = True  # running, or waiting for the user's own yes/no: finish and report
            else:
                call.task.cancel()

    # -- queued [SAM] text turns ------------------------------------------------------------------------
    def _queue_note(self, prompt: str) -> None:
        """A late tool result for the model: sent at the next idle moment, or
        after 10 s anyway (a turn that never reports IDLE must not swallow it)."""
        self._say_queue.append((prompt, "", "result", time.monotonic() + 10.0))
        if not self._model_active:
            self._drain_say_queue()
        else:
            self._arm_say_timer()

    def _drain_say_queue(self) -> None:
        if not self._say_queue or self._model_active or not self.ready:
            return
        prompt, text, source, _deadline = self._say_queue.popleft()
        asyncio.ensure_future(self._send_text(prompt, say_turn=bool(text)))

    def _arm_say_timer(self) -> None:
        if self._say_timer is None:
            self._say_timer = asyncio.get_running_loop().call_later(0.5, self._say_tick)

    def _say_tick(self) -> None:
        self._say_timer = None
        now = time.monotonic()
        keep: collections.deque[tuple[str, str, str, float]] = collections.deque()
        for item in self._say_queue:
            prompt, text, source, deadline = item
            if now < deadline:
                keep.append(item)
            elif text:
                asyncio.ensure_future(self.hooks.speak_fallback(text, source))  # waited too long: use TTS
            else:
                asyncio.ensure_future(self._send_text(prompt, say_turn=False))  # result note: send anyway
        self._say_queue = keep
        if self._say_queue:
            self._arm_say_timer()


__all__ = ["LiveCalls"]
