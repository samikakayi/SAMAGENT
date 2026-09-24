"""The App object: core services + slots every package fills in.

Lifecycle (``sam.__main__``)::

    app = App(home)                      # sync, fast: config, db, secrets, bus, tools, confirm, llm
    app.load_packages()                  # import PACKAGES, call each register(app)
    core.run_sync(app.start())           # on the core loop: each package's async start(app)
    sam.ui.run(app, core)                # Qt on the main thread (blocks)
    core.run_sync(app.stop()); app.close()

Packages are imported defensively: a missing or broken package is recorded
in ``app.failed`` and start-up continues (parallel builders; a broken voice
package must not take the chart tools down with it).
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import logging.handlers
import sys
import time
import traceback
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Coroutine

from .brain.confirm import ConfirmBroker
from .brain.llm import LLMClient
from .brain.tools import ToolContext, ToolRegistry, ok, tool
from .config import Config
from .db import Database
from .events import ComponentStatus, Error, EventBus, SettingsChanged
from .secrets import SecretStore, Secrets, install_log_redaction, redact, redact_obj
from .timing import Timing

log = logging.getLogger("sam.app")

# Import order = register/start order (stop runs in reverse). Each module
# exposes register(app) and optionally ``async def start(app)`` /
# ``async def stop(app)``. The UI is not listed: __main__ runs it on the main
# thread via sam.ui.run(app, core).
PACKAGES: tuple[str, ...] = (
    "sam.brain.memory",        # app.memory      (brain builder)
    "sam.brain.persona",       # app.persona     (brain builder)
    "sam.hands",               # app.hands       (hands builder)
    "sam.trading.chart_tools", # app.trading.tv  (chart bridge builder)
    "sam.trading.tools",       # app.trading.mt5/engine/theories/strategies/monitor (engine builder)
    "sam.brain.worker",        # app.worker      (brain builder)
    "sam.brain.conversation",  # app.conversation (brain builder)
    "sam.voice",               # app.voice       (voice builder)
    "sam.migrate_v1",          # one-time import from v1 (launcher/migration builder)
)
START_TIMEOUT_S = 20.0
STOP_TIMEOUT_S = 8.0


@dataclass
class TradingSlots:
    """``app.trading``: filled by the two trading entry modules."""

    tv: Any = None          # TradingViewBridge            (chart_tools)
    mt5: Any = None         # MT5Feed                      (tools)
    engine: Any = None      # engine facade / analyst      (tools)
    theories: Any = None    # dict[str, Theory]            (tools)
    strategies: Any = None  # StrategyStore                (tools)
    monitor: Any = None     # Monitor                      (tools)


@tool("stop_all",
      description="Stop everything SAM is doing right now: speech, running tools, background tasks and "
                  "pending confirmations. Use when the user says stop/enough/cancel everything.",
      description_ckb="ڕاگرتنی هەموو کارەکان",
      examples_ckb=("بوەستە", "هەمووی ڕابگرە", "بەسە"),
      risk="safe", blocking=True, timeout_s=10)
async def stop_all_tool(ctx: ToolContext) -> dict[str, Any]:
    result = await ctx.app.stop_all(except_call=ctx.call_id)
    return ok("Stopped.", **result)


class App:
    def __init__(self, home: Any = None, *, environ: dict[str, str] | None = None,
                 llm_backends: dict[str, Any] | None = None) -> None:
        self.started_at = time.time()
        self.config = Config(home, environ=environ)
        self.db = Database(self.config.db_path)
        self.bus = EventBus()
        self.config.attach_db(self.db, on_change=self._setting_changed)
        self.secrets = Secrets(SecretStore(self.config.data_dir), self.config.env_value)
        self.timing = Timing(self.db)
        self.confirm = ConfirmBroker(self.bus, timeout_s=float(self.config.get("confirm.timeout_s", 20)), db=self.db)
        self.tools = ToolRegistry(app=self, bus=self.bus, confirm=self.confirm, timing=self.timing, db=self.db,
                                  redact_obj=self.redact_obj)
        self.llm = LLMClient(self.config, self.secrets, db=self.db, timing=self.timing, bus=self.bus,
                             backends=llm_backends)
        # Slots filled by packages in register(app).
        self.memory: Any = None
        self.persona: Any = None
        self.conversation: Any = None
        self.worker: Any = None
        self.voice: Any = None
        self.hands: Any = None
        self.trading = TradingSlots()
        self.ui: Any = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.loaded: dict[str, ModuleType] = {}
        self.missing: list[str] = []
        self.failed: dict[str, str] = {}
        self._started: list[str] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        self._stopping = False
        self.tools.add(stop_all_tool, owner="core")

    # -- redaction helpers (use everywhere text leaves the process) --------------
    def redact(self, text: str) -> str:
        return self.secrets.redact(text)

    def redact_obj(self, obj: Any) -> Any:
        return redact_obj(obj, self.secrets.known_values())

    def _setting_changed(self, key: str, value: Any) -> None:
        self.bus.publish_threadsafe(SettingsChanged(key=key, value=value))

    # -- packages -------------------------------------------------------------------
    def load_packages(self, names: tuple[str, ...] | list[str] = PACKAGES) -> dict[str, str]:
        """Import each package and call ``register(app)``. Returns a status map
        {name: "ok" | "missing" | "failed: ..."}."""
        status: dict[str, str] = {}
        for name in names:
            started = time.perf_counter()
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError as exc:
                if exc.name and (name == exc.name or name.startswith(exc.name + ".")):
                    self.missing.append(name)
                    status[name] = "missing"
                    log.info("package %s not present yet", name)
                    continue
                self._fail(name, exc)
                status[name] = f"failed: {type(exc).__name__}"
                continue
            except Exception as exc:  # noqa: BLE001
                self._fail(name, exc)
                status[name] = f"failed: {type(exc).__name__}"
                continue
            register = getattr(module, "register", None)
            if register is not None:
                try:
                    register(self)
                except Exception as exc:  # noqa: BLE001
                    self._fail(name, exc)
                    status[name] = f"failed: {type(exc).__name__}"
                    continue
            self.loaded[name] = module
            status[name] = "ok"
            self.timing.record(f"startup:register:{name}", (time.perf_counter() - started) * 1000.0, kind="startup")
        return status

    def _fail(self, name: str, exc: BaseException) -> None:
        detail = self.redact("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-3000:])
        self.failed[name] = detail
        log.error("package %s failed: %s", name, detail)
        try:
            self.db.log_activity("error", name, ok=False, summary=self.redact(f"{type(exc).__name__}: {exc}")[:300],
                                 source="startup")
        except Exception:  # noqa: BLE001
            pass

    # -- lifecycle --------------------------------------------------------------------
    async def start(self) -> None:
        """Run on the core loop: bind the bus, start packages, warm up."""
        self.loop = asyncio.get_running_loop()
        self.bus.bind_loop(self.loop)
        # google-genai costs ~2 s to import on this PC (measured 2026-09-24):
        # import it off the loop so the first Gemini call does not stall voice.
        self.spawn(asyncio.to_thread(importlib.import_module, "google.genai"), "prewarm-genai")
        for name, module in self.loaded.items():
            start = getattr(module, "start", None)
            if start is None:
                continue
            began = time.perf_counter()
            try:
                result = start(self)
                if asyncio.iscoroutine(result):
                    await asyncio.wait_for(result, START_TIMEOUT_S)
                self._started.append(name)
            except Exception as exc:  # noqa: BLE001
                self._fail(name, exc)
                self.bus.publish(Error(where=name, message_ckb="بەشێک لە سام دەستی پێنەکرد.",
                                       detail=self.redact(f"{type(exc).__name__}: {exc}")[:300]))
            self.timing.record(f"startup:start:{name}", (time.perf_counter() - began) * 1000.0, kind="startup")
        self._start_background_services()
        self.timing.record("startup:app", (time.time() - self.started_at) * 1000.0, kind="startup")

    def _start_background_services(self) -> None:
        try:
            omniroute = importlib.import_module("sam.omniroute")
        except ModuleNotFoundError:
            omniroute = None
        if omniroute is not None and hasattr(omniroute, "ensure_running"):
            self.spawn(omniroute.ensure_running(self), "omniroute")
        if self.config.get("llm.verify_on_start", True) and any(b.configured() for b in self.llm.backends.values()):
            self.spawn(self._verify_models(), "verify-models")

    async def _verify_models(self) -> None:
        await asyncio.sleep(3.0)  # let OmniRoute come up first
        result = await self.llm.verify_models()
        missing = [ref for ref, present in result.items() if present is False]
        if missing:
            log.warning("ladder models not listed by their provider: %s", ", ".join(missing))
            self.db.log_activity("system", "verify_models", ok=False, summary="missing: " + ", ".join(missing))

    async def stop(self) -> None:
        """Stop packages in reverse order; never raises."""
        self._stopping = True
        self.confirm.cancel_all()
        self.tools.cancel_all()
        for name in reversed(self._started):
            stop = getattr(self.loaded.get(name), "stop", None)
            if stop is None:
                continue
            try:
                result = stop(self)
                if asyncio.iscoroutine(result):
                    await asyncio.wait_for(result, STOP_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001
                log.warning("stop of %s failed: %s", name, self.redact(str(exc)))
        self._started.clear()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.llm.aclose()

    def close(self) -> None:
        """Release the DB (after the core loop has stopped)."""
        self.db.close()

    def spawn(self, coro: Coroutine[Any, Any, Any] | Any, name: str = "") -> asyncio.Task[Any]:
        """Start a tracked background task on the core loop; errors are logged
        (redacted) instead of vanishing. Cancelled on stop()."""
        task = asyncio.ensure_future(coro)
        if name:
            try:
                task.set_name(f"sam:{name}")
            except AttributeError:
                pass
        self._tasks.add(task)

        def done(t: asyncio.Task[Any]) -> None:
            self._tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                exc = t.exception()
                log.error("background task %s failed: %s", name, self.redact(f"{type(exc).__name__}: {exc}"))
        task.add_done_callback(done)
        return task

    # -- UI facade (call through CoreThread.submit from the UI thread) ---------------
    async def submit_text(self, text: str) -> str:
        """Typed input: handled exactly like speech by the conversation."""
        if self.conversation is None:
            self.bus.publish(Error(where="conversation", message_ckb="بەشی گفتوگۆ ئامادە نییە."))
            return ""
        return await self.conversation.handle_text(text, source="text")

    async def toggle_listening(self) -> bool | None:
        if self.voice is None:
            return None
        return await self.voice.toggle_listening()

    async def set_muted(self, muted: bool) -> None:
        if self.voice is not None:
            await self.voice.set_muted(muted)

    async def stop_all(self, *, except_call: str | None = None) -> dict[str, Any]:
        """Emergency stop: speech, tools, worker tasks, pending confirmations."""
        stopped = {"tools": self.tools.cancel_all(except_call=except_call), "confirmations": self.confirm.cancel_all()}
        if self.voice is not None:
            try:
                await self.voice.stop_speaking()
            except Exception:  # noqa: BLE001
                log.exception("voice stop failed")
        if self.worker is not None:
            try:
                stopped["worker"] = self.worker.cancel()
            except Exception:  # noqa: BLE001
                log.exception("worker cancel failed")
        self.db.log_activity("system", "stop_all", ok=True, summary=str(stopped))
        return stopped

    def status(self) -> dict[str, Any]:
        """Component overview for Settings/diagnostics (never any key value)."""
        return {
            "home": str(self.config.home), "db": str(self.config.db_path),
            "packages": {"loaded": list(self.loaded), "missing": self.missing, "failed": list(self.failed)},
            "tools": self.tools.names(),
            "keys": {k: {"configured": v["configured"], "source": v["source"]} for k, v in self.secrets.status().items()},
            "llm": self.llm.status(),
            "slots": {name: getattr(self, name) is not None
                      for name in ("memory", "persona", "conversation", "worker", "voice", "hands")}
                     | {f"trading.{n}": getattr(self.trading, n) is not None
                        for n in ("tv", "mt5", "engine", "theories", "strategies", "monitor")},
        }

    def publish_status(self, component: str, state: str, detail: str = "") -> None:
        self.bus.publish_threadsafe(ComponentStatus(component=component, state=state, detail=detail))  # type: ignore[arg-type]


def setup_logging(app: App, *, console: bool = False, level: int = logging.INFO) -> None:
    """Rotating file log in %LOCALAPPDATA%\\SAM2\\logs, secrets redacted."""
    root = logging.getLogger()
    root.setLevel(level)
    try:
        app.config.log_dir.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            app.config.log_dir / "sam2.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    except OSError:
        pass
    if console and sys.stderr is not None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(stream)
    install_log_redaction(app.secrets)


__all__ = ["App", "PACKAGES", "TradingSlots", "setup_logging", "redact"]
