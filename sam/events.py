"""Typed events and the asyncio EventBus that carries them.

Every event is a frozen dataclass. Subscribers are sync or async callables;
they run on the core loop. A failing subscriber is logged and never breaks the
publisher or other subscribers. Other threads (UI, audio callbacks, MT5
worker) publish with ``publish_threadsafe``.

Voice states (``VoiceState.state``) and their Sorani island words (ui.strings):
  idle      ئامادە        listening  گوێ دەگرم      thinking  بیردەکەمەوە
  speaking  قسە دەکەم     working    کار دەکەم      error     هەڵە
  sleeping  (idle, conversation window closed)     muted     (mic off)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

log = logging.getLogger("sam.events")

VoiceStateName = Literal["idle", "listening", "thinking", "speaking", "working", "error", "sleeping", "muted"]
Role = Literal["user", "assistant", "system"]


def new_id() -> str:
    """Short random id for turns, tool calls, confirmations, tasks."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True, slots=True)
class Event:
    """Base class. ``at`` is unix seconds."""

    at: float = field(default_factory=time.time, kw_only=True)


@dataclass(frozen=True, slots=True)
class VoiceState(Event):
    state: VoiceStateName
    engine: str = ""            # "live" | "cascade" | ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Caption(Event):
    """One live line for the island (partial or final, RTL text)."""

    text: str
    role: Role = "assistant"
    final: bool = False


@dataclass(frozen=True, slots=True)
class Transcript(Event):
    """A finished utterance. The single source for persisting turns: the
    conversation module subscribes and writes ``turns`` rows."""

    role: Role
    text: str
    source: str = "live"        # live|cascade|text|worker|system
    turn_id: str = ""
    conversation_id: int | None = None


@dataclass(frozen=True, slots=True)
class UserText(Event):
    """Typed input from the panel (handled exactly like speech)."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolStarted(Event):
    call_id: str
    name: str
    args: dict[str, Any]        # already redacted
    source: str = ""            # live|cascade|text|worker|ui


@dataclass(frozen=True, slots=True)
class ToolFinished(Event):
    call_id: str
    name: str
    ok: bool
    summary: str                # redacted, short
    duration_ms: float = 0.0
    source: str = ""


@dataclass(frozen=True, slots=True)
class ConfirmRequest(Event):
    confirm_id: str
    question_ckb: str
    detail: str = ""
    tool_name: str = ""
    expires_at: float = 0.0


@dataclass(frozen=True, slots=True)
class ConfirmResult(Event):
    confirm_id: str
    approved: bool
    via: str = ""               # voice|click|timeout|cancel


@dataclass(frozen=True, slots=True)
class SpeakRequest(Event):
    """Ask the voice engine to say something not produced by a model turn
    (alerts, worker summaries, confirmation questions in cascade mode)."""

    text_ckb: str
    source: str = "system"      # alert|worker|confirm|system
    interrupt: bool = False     # True: cut current speech (urgent alerts only)


@dataclass(frozen=True, slots=True)
class Alert(Event):
    alert_id: int
    kind: str
    symbol: str
    text_ckb: str
    price: float | None = None
    timeframe: str = ""


@dataclass(frozen=True, slots=True)
class WorkerProgress(Event):
    task_id: str
    step: int
    max_steps: int
    text_ckb: str
    done: bool = False
    ok: bool | None = None


@dataclass(frozen=True, slots=True)
class Error(Event):
    where: str
    message_ckb: str
    detail: str = ""            # redacted English detail


@dataclass(frozen=True, slots=True)
class LevelMeter(Event):
    source: Literal["mic", "speaker"]
    level: float                # 0..1 RMS-ish, ~20 Hz max


@dataclass(frozen=True, slots=True)
class ComponentStatus(Event):
    """Status dot for Settings/tray: voice, live, cascade, omniroute, groq,
    gemini, kurdishtts, tradingview, mt5, ..."""

    component: str
    state: Literal["ok", "degraded", "down", "unconfigured", "unknown"]
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SettingsChanged(Event):
    key: str
    value: Any = None


Subscriber = Callable[[Any], "None | Awaitable[None]"]


class EventBus:
    """Publish/subscribe on the core asyncio loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop
        self._subs: list[tuple[type | None, Subscriber]] = []
        self._lock = threading.Lock()
        self._tasks: set[asyncio.Task[Any]] = set()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    def subscribe(self, event_type: type | tuple[type, ...] | None, callback: Subscriber) -> Callable[[], None]:
        """Subscribe to one type, a tuple of types, or everything (None).
        Returns an unsubscribe function."""
        types: list[type | None] = list(event_type) if isinstance(event_type, tuple) else [event_type]
        entries = [(t, callback) for t in types]
        with self._lock:
            self._subs.extend(entries)

        def unsubscribe() -> None:
            with self._lock:
                for entry in entries:
                    if entry in self._subs:
                        self._subs.remove(entry)
        return unsubscribe

    def publish(self, event: Event) -> None:
        """Deliver ``event`` now (sync subscribers) / as tasks (async ones).
        Must be called on the core loop thread (or with no loop in tests)."""
        with self._lock:
            targets = [cb for t, cb in self._subs if t is None or isinstance(event, t)]
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        for callback in targets:
            try:
                result = callback(event)
                if inspect.isawaitable(result):
                    if running is not None:
                        task = asyncio.ensure_future(result)
                        self._tasks.add(task)
                        task.add_done_callback(self._task_done)
                    elif self._loop is not None and not self._loop.is_closed():
                        # Published off-loop by mistake: run it on the core loop.
                        asyncio.run_coroutine_threadsafe(result, self._loop)  # type: ignore[arg-type]
                    elif inspect.iscoroutine(result):
                        result.close()
                        log.warning("async subscriber skipped: no event loop for %s", type(event).__name__)
            except Exception:  # noqa: BLE001
                log.exception("event subscriber failed for %s", type(event).__name__)

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("async event subscriber failed: %r", task.exception())

    def publish_threadsafe(self, event: Event) -> None:
        """Publish from any thread (UI, audio callback, worker threads)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self.publish(event)
        else:
            loop.call_soon_threadsafe(self.publish, event)

    async def wait_for(self, event_type: type, predicate: Callable[[Any], bool] | None = None,
                       timeout: float | None = None) -> Any:
        """Await the next event of ``event_type`` matching ``predicate``."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def on_event(event: Any) -> None:
            if not future.done() and (predicate is None or predicate(event)):
                future.set_result(event)

        unsubscribe = self.subscribe(event_type, on_event)
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            unsubscribe()


__all__ = [
    "Event", "VoiceState", "Caption", "Transcript", "UserText", "ToolStarted", "ToolFinished",
    "ConfirmRequest", "ConfirmResult", "SpeakRequest", "Alert", "WorkerProgress", "Error", "LevelMeter",
    "ComponentStatus", "SettingsChanged", "EventBus", "new_id", "VoiceStateName",
]
