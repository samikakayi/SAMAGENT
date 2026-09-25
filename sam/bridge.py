"""Run the asyncio core on its own thread; hand events to the UI thread.

Why a thread and not qasync: the core (Live audio, CDP, MT5, LLM streams)
must keep running while Qt is busy painting or a modal dialog is open, and
the core must not import Qt at all (tests run headless). Qt stays on the main
thread; the core loop lives in ``CoreThread``.

UI -> core:  ``core.submit(app.submit_text("..."))`` (returns a
             concurrent.futures.Future) or ``core.call_soon(fn, *args)``.
core -> UI:  ``UiAdapter(bus, sink)`` calls ``sink(event)`` on the core thread
             for every event; the UI passes ``some_qobject.signal.emit`` as the
             sink, and Qt queues the signal onto the GUI thread (AutoConnection
             across threads is a QueuedConnection -- thread-safe by design).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Awaitable, Callable, Coroutine, TypeVar

from .events import EventBus

log = logging.getLogger("sam.bridge")
T = TypeVar("T")


class CoreThread:
    """An asyncio event loop running forever on a daemon thread."""

    def __init__(self, name: str = "sam-core") -> None:
        self.name = name
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> asyncio.AbstractEventLoop:
        if self._thread is not None:
            return self.loop
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        self._ready.wait(5)
        return self.loop

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.set_exception_handler(self._on_loop_error)
        self.loop.call_soon(self._ready.set)
        try:
            self.loop.run_forever()
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(self.loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            finally:
                self.loop.close()

    @staticmethod
    def _on_loop_error(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        log.error("core loop error: %s", context.get("message"), exc_info=context.get("exception"))

    @property
    def is_core_thread(self) -> bool:
        return threading.current_thread() is self._thread

    def call_soon(self, fn: Callable[..., Any], *args: Any) -> None:
        """Schedule a plain callable on the core loop (thread-safe)."""
        self.loop.call_soon_threadsafe(fn, *args)

    def submit(self, coro: Coroutine[Any, Any, T]) -> "concurrent.futures.Future[T]":
        """Run a coroutine on the core loop from any thread."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run_sync(self, coro: Coroutine[Any, Any, T], timeout: float | None = 30.0) -> T:
        """Block the calling (non-core) thread until ``coro`` finishes."""
        if self.is_core_thread:
            raise RuntimeError("run_sync() would deadlock on the core thread; await instead")
        return self.submit(coro).result(timeout)

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout)
        self._thread = None


class UiAdapter:
    """Forward bus events to a thread-safe ``sink`` (e.g. a Qt signal emit).

    ``types`` limits which events are forwarded (default: all). LevelMeter
    events are throttled to ``max_meter_hz`` so the UI thread is not flooded.
    """

    def __init__(self, bus: EventBus, sink: Callable[[Any], None], types: tuple[type, ...] | None = None,
                 max_meter_hz: float = 25.0) -> None:
        self.bus = bus
        self.sink = sink
        self.types = types
        self._min_meter_gap = 1.0 / max_meter_hz if max_meter_hz > 0 else 0.0
        self._last_meter: dict[str, float] = {}
        self._unsubscribe: Callable[[], None] | None = None

    def attach(self) -> "UiAdapter":
        if self._unsubscribe is None:
            self._unsubscribe = self.bus.subscribe(self.types, self._forward)
        return self

    def detach(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def _forward(self, event: Any) -> None:
        from .events import LevelMeter

        if isinstance(event, LevelMeter) and self._min_meter_gap:
            last = self._last_meter.get(event.source, 0.0)
            if event.at - last < self._min_meter_gap:
                return
            self._last_meter[event.source] = event.at
        try:
            self.sink(event)
        except RuntimeError:
            # Qt raises RuntimeError once the receiving QObject is deleted
            # (window closed during shutdown): stop forwarding quietly.
            self.detach()


def run_blocking(loop: asyncio.AbstractEventLoop, fn: Callable[..., T], *args: Any) -> Awaitable[T]:
    """Await a blocking call in the default executor (helper for packages)."""
    return loop.run_in_executor(None, fn, *args)


__all__ = ["CoreThread", "UiAdapter", "run_blocking"]
