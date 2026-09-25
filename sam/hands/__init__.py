"""SAM 2 hands: everything SAM does on the computer (DESIGN 2.3).

Layers, fastest first (reports/computer-control.json):
1. skills: open_app (Start-menu index + Sorani aliases), windows, keys/media,
   files, PowerShell, web, build_project, system status;
2. UI Automation: numbered controls, click/type by number or name;
3. Windows OCR: find/click visible text (Electron/canvas apps such as
   TradingView expose almost nothing to UIA);
4. vision: screen_look(describe) and screen_act, budgeted per day.

``register`` only registers tools and settings; every component is built on
first use (importing all of them costs ~50 ms on this busy PC, the whole
register budget). ``start`` refreshes the app index in the background
(~1.5 s on a worker thread). No devices, COM or network in register.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Any

log = logging.getLogger("sam.hands")

# The Start-menu app index cache (apps.py). Kept here so register() does not
# have to import apps.py.
APPS_SCHEMA = [(1, """
CREATE TABLE IF NOT EXISTS hands_apps (
    app_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    target TEXT,
    target_args TEXT,
    updated_at REAL NOT NULL
)""")]

DEFAULTS: dict[str, Any] = {
    "hands.app_aliases": {},          # user aliases: {"spoken name": "App name"}
    "hands.launch_wait_s": 12,
    "hands.paste_settle_ms": 350,
    "hands.uia_max_controls": 80,
    "hands.uia_budget_s": 4.0,
    "hands.search_model": "gemini-3.5-flash-lite",
    "hands.vision_ladder": "vision",
    "hands.build_timeout_s": 540,     # build_project generation budget (the tool allows 600 s)
    "hands.backup_dir": None,         # None -> %LOCALAPPDATA%\SAM2\backups (never inside SAM_HOME)
}


class Hands:
    """``app.hands`` -- see CONTRACTS.md 3.3 for the public surface
    (apps, windows, uia, ocr, screen, vision, policy; plus input, files,
    web, code, system)."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._input_lock = asyncio.Lock()
        from . import _win

        removed = _win.scrub_host_environment()
        if removed:
            log.info("removed %d inherited Electron/VS Code host variables so launched apps start normally",
                     len(removed))

    @cached_property
    def windows(self) -> Any:
        from .windows import Windows

        return Windows()

    @cached_property
    def input(self) -> Any:
        from .input import Input

        settle = float(self.app.config.get("hands.paste_settle_ms", 350) or 350) / 1000.0
        return Input(paste_settle_s=settle)

    @cached_property
    def policy(self) -> Any:
        from .policy import Policy

        return Policy.from_app(self.app)

    @cached_property
    def apps(self) -> Any:
        from .apps import AppIndex

        return AppIndex(self.app, windows=self.windows)

    @cached_property
    def uia(self) -> Any:
        from .uia import Uia

        config = self.app.config
        return Uia(windows=self.windows, input_runner=self.run_input,
                   budget_s=float(config.get("hands.uia_budget_s", 4.0) or 4.0),
                   max_controls=int(config.get("hands.uia_max_controls", 80) or 80))

    @cached_property
    def screen(self) -> Any:
        from .screen import Screen, password_redactor

        screen = Screen(self.windows)
        screen.redactors.append(password_redactor(self.uia))
        return screen

    @cached_property
    def ocr(self) -> Any:
        from .ocr import Ocr

        ocr = Ocr(self.screen)
        self.screen.ocr = ocr  # MT5 redaction reads text locally before any upload
        return ocr

    @cached_property
    def vision(self) -> Any:
        from .vision import Vision

        _ = self.ocr  # wires screen.ocr for MT5 redaction
        return Vision(self.app, screen=self.screen, windows=self.windows, input_runner=self.run_input,
                      uia=self.uia, ocr=self.ocr)

    @cached_property
    def files(self) -> Any:
        from .files import Files

        return Files(self.policy, backup_dir=self._backup_dir())

    @cached_property
    def web(self) -> Any:
        from .web import Web

        return Web(self.app)

    @cached_property
    def code(self) -> Any:
        from .code import CodeBuilder

        return CodeBuilder(self.app, windows=self.windows)

    @cached_property
    def system(self) -> Any:
        from . import system

        return system

    def _backup_dir(self) -> Path:
        configured = self.app.config.get("hands.backup_dir")
        if configured:
            return Path(str(configured))
        return Path(self.app.config.log_dir).parent / "backups"

    def refresh_policy(self) -> None:
        """Rebuild path rules after hands.projects_dir changed."""
        self.__dict__.pop("policy", None)
        if "files" in self.__dict__:
            self.files.policy = self.policy

    async def run_input(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Run one ``Input`` method on a DPI-aware thread; one input action at
        a time so two tools never interleave keystrokes."""
        from . import _win

        def call() -> Any:
            with _win.dpi_aware():
                return getattr(self.input, method)(*args, **kwargs)

        async with self._input_lock:
            return await asyncio.to_thread(call)

    def status(self) -> dict[str, Any]:
        apps = self.__dict__.get("apps")
        return {"apps_indexed": len(apps._rows or []) if apps is not None else 0,
                "last_refresh": apps.last_refresh if apps is not None else {},
                "uia_last_ms": getattr(self.__dict__.get("uia"), "last_ms", None),
                "vision_used_today": self.vision.used_today(), "vision_budget": self.vision.budget()}


def register(app: Any) -> None:
    from .tools import register_tools

    app.config.register_defaults(DEFAULTS)
    app.db.ensure_schema("hands", APPS_SCHEMA)
    app.hands = Hands(app)
    register_tools(app)

    def on_setting(event: Any) -> None:
        if getattr(event, "key", "") == "hands.projects_dir" and app.hands is not None:
            app.hands.refresh_policy()

    try:
        from ..events import SettingsChanged

        app.bus.subscribe(SettingsChanged, on_setting)
    except Exception:  # noqa: BLE001
        log.debug("settings subscription failed", exc_info=True)


async def start(app: Any) -> None:
    if app.hands is None or os.name != "nt":
        return
    app.hands.apps.refresh_in_background()


async def stop(app: Any) -> None:
    hands = getattr(app, "hands", None)
    if hands is None:
        return
    for name in ("uia", "apps"):
        component = hands.__dict__.get(name)
        worker = getattr(component, "worker", None) or getattr(component, "_worker", None)
        try:
            if worker is not None:
                worker.shutdown()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["APPS_SCHEMA", "DEFAULTS", "Hands", "register", "start", "stop"]
