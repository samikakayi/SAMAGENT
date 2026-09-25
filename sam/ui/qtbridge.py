"""The Qt side of the core <-> UI bridge (``sam.bridge`` is the core side).

- Events in: ``UiAdapter(app.bus, bridge.event.emit)`` runs on the core thread
  and emits a Qt signal; the receiver lives on the GUI thread, so Qt queues the
  call (AutoConnection across threads == QueuedConnection). ``subscribe`` then
  fans events out to widgets by type, on the GUI thread.
- Commands out: ``call(coro)`` schedules a coroutine on the core loop;
  ``on_core(fn, *args)`` runs a quick sync function ON the core loop (objects
  owned by the core such as the monitor or the strategy store); ``run(fn)``
  runs a blocking function in a worker thread (``SecretStore.set`` shells out
  to icacls for up to 15 s, DPAPI status reads, DB scans). Results come back
  through ``on_ok`` / ``on_err`` on the GUI thread. Nothing here blocks the
  GUI thread.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
from typing import Any, Callable, Coroutine

from PySide6.QtCore import QObject, Signal, Slot

from ..bridge import UiAdapter

log = logging.getLogger("sam.ui.bridge")

Callback = Callable[[Any], None] | None


class QtBridge(QObject):
    """Owned by the GUI thread. ``core`` is a ``sam.bridge.CoreThread`` (or any
    object with ``submit(coro) -> concurrent.futures.Future``)."""

    event = Signal(object)                 # bus events, queued from the core thread
    _result = Signal(object, object, object, object)   # on_ok, on_err, value, error
    show_requested = Signal()              # second launch / other threads -> show panel

    def __init__(self, app: Any, core: Any, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.app = app
        self.core = core
        self._subs: list[tuple[type | tuple[type, ...] | None, Callable[[Any], None]]] = []
        self._adapter: UiAdapter | None = None
        self.event.connect(self._dispatch)
        self._result.connect(self._deliver)

    # -- events ------------------------------------------------------------------
    def attach(self) -> None:
        """Start receiving bus events (idempotent)."""
        if self._adapter is None and getattr(self.app, "bus", None) is not None:
            self._adapter = UiAdapter(self.app.bus, self.event.emit).attach()

    def detach(self) -> None:
        if self._adapter is not None:
            self._adapter.detach()
            self._adapter = None

    def subscribe(self, types: type | tuple[type, ...] | None, callback: Callable[[Any], None]) -> Callable[[], None]:
        """``callback(event)`` on the GUI thread for matching events."""
        entry = (types, callback)
        self._subs.append(entry)

        def unsubscribe() -> None:
            if entry in self._subs:
                self._subs.remove(entry)
        return unsubscribe

    @Slot(object)
    def _dispatch(self, event: Any) -> None:
        for types, callback in list(self._subs):
            if types is None or isinstance(event, types):
                try:
                    callback(event)
                except Exception:  # noqa: BLE001 - one broken widget must not stop the rest
                    log.exception("UI handler failed for %s", type(event).__name__)

    def deliver(self, event: Any) -> None:
        """Feed an event directly (tests, or UI-local events)."""
        self._dispatch(event)

    # -- commands ---------------------------------------------------------------------
    def call(self, coro: Coroutine[Any, Any, Any], on_ok: Callback = None, on_err: Callback = None
             ) -> concurrent.futures.Future[Any] | None:
        """Run ``coro`` on the core loop; callbacks run on the GUI thread."""
        if self.core is None or not hasattr(self.core, "submit"):
            if inspect.iscoroutine(coro):
                coro.close()
            self._fail(on_err, RuntimeError("core is not running"))
            return None
        try:
            future = self.core.submit(coro)
        except Exception as exc:  # noqa: BLE001 - loop closed during shutdown
            if inspect.iscoroutine(coro):
                coro.close()
            self._fail(on_err, exc)
            return None
        future.add_done_callback(lambda f: self._finished(f, on_ok, on_err))
        return future

    def on_core(self, fn: Callable[..., Any], *args: Any, on_ok: Callback = None, on_err: Callback = None,
                **kwargs: Any) -> concurrent.futures.Future[Any] | None:
        """Run a quick sync ``fn`` on the core loop thread (core-owned objects)."""
        async def _invoke() -> Any:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        return self.call(_invoke(), on_ok, on_err)

    def run(self, fn: Callable[..., Any], *args: Any, on_ok: Callback = None, on_err: Callback = None,
            **kwargs: Any) -> concurrent.futures.Future[Any] | None:
        """Run a blocking ``fn`` in a worker thread (via the core loop's executor)."""
        async def _invoke() -> Any:
            return await asyncio.to_thread(fn, *args, **kwargs)
        return self.call(_invoke(), on_ok, on_err)

    def _finished(self, future: concurrent.futures.Future[Any], on_ok: Callback, on_err: Callback) -> None:
        # Runs on the core thread: hop to the GUI thread through the signal.
        if on_ok is None and on_err is None:
            if not future.cancelled() and future.exception() is not None:
                log.warning("UI command failed: %s", self._safe_error(future.exception()))
            return
        if future.cancelled():
            value, error = None, asyncio.CancelledError()
        else:
            error = future.exception()
            value = None if error is not None else future.result()
        try:
            self._result.emit(on_ok, on_err, value, error)
        except RuntimeError:
            pass  # bridge deleted during shutdown

    def _fail(self, on_err: Callback, error: BaseException) -> None:
        if on_err is not None:
            try:
                on_err(error)
            except Exception:  # noqa: BLE001
                log.exception("UI error callback failed")
        else:
            log.warning("UI command not run: %s", self._safe_error(error))

    @Slot(object, object, object, object)
    def _deliver(self, on_ok: Callback, on_err: Callback, value: Any, error: Any) -> None:
        try:
            if error is not None:
                if on_err is not None:
                    on_err(error)
                else:
                    log.warning("UI command failed: %s", self._safe_error(error))
            elif on_ok is not None:
                on_ok(value)
        except Exception:  # noqa: BLE001
            log.exception("UI result callback failed")

    def _safe_error(self, error: BaseException | None) -> str:
        text = f"{type(error).__name__}: {error}" if error is not None else "unknown"
        redact = getattr(self.app, "redact", None)
        return redact(text) if callable(redact) else text


__all__ = ["QtBridge"]
