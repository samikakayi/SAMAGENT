"""SAM 2 desktop UI (PySide6): island pill, panel, tray.

Entry point (docs/CONTRACTS.md 3.6)::

    sam.ui.run(app, core) -> int     # main thread; blocks until Quit

The UI is the only place that imports Qt. It talks to the core through
``QtBridge`` (events in through a queued Qt signal, commands out through
``core.submit``) -- see ``qtbridge.py``. This module stays import-light: Qt is
imported inside the functions so ``import sam.ui.strings`` works headless.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

log = logging.getLogger("sam.ui")

UI_DEFAULTS = {
    "ui.island_pos": None,          # [centre_x, top_y] of the pill (logical px) after a drag
    "ui.font_family": "Vazirmatn",  # first choice; falls back to installed Sorani-capable fonts
    "ui.show_english": True,        # secondary English labels in Settings
    "ui.panel_on_start": False,     # open the panel at start-up (the island is always shown)
}


class UiHandle:
    """``app.ui``: what other packages may call. Thread-safe (signals)."""

    def __init__(self, controller: Any) -> None:
        self._controller = controller

    def show_panel(self, page: str = "") -> None:
        self._controller.panel_requested.emit(page or "")

    def show_caption(self, text: str, tone: str = "system") -> None:
        self._controller.caption_requested.emit(text, tone)


def _make_controller_class() -> type:
    from PySide6.QtCore import QObject, QTimer, Signal
    from PySide6.QtWidgets import QApplication

    from ..events import ComponentStatus, VoiceState
    from . import theme
    from .island import Island
    from .qtbridge import QtBridge
    from .status_seed import snapshot
    from .strings import learn_tool_labels, tr
    from .tray import Tray, spawn_restart

    class UiController(QObject):
        """Builds and wires island, panel and tray (no ``exec``: tests use it too)."""

        panel_requested = Signal(str)
        caption_requested = Signal(str, str)
        quit_requested = Signal()          # from other threads (quit event watcher)

        def __init__(self, app: Any, core: Any) -> None:
            super().__init__()
            self.app = app
            self.core = core
            self.qapp = QApplication.instance()
            try:
                app.config.register_defaults(UI_DEFAULTS)
            except Exception:  # noqa: BLE001
                pass
            if getattr(app, "tools", None) is not None:
                learn_tool_labels(app.tools)    # tools added later by other packages
            self.families = theme.ui_families(self._cfg("ui.font_family"))
            self.bridge = QtBridge(app, core)
            self.island = Island(getattr(app, "config", None))
            self.tray = Tray(self.families)
            self.panel: Any = None
            self._seen: set[str] = set()       # components that reported since the UI attached
            self._last: dict[str, Any] = {}    # latest state per component, replayed into a new panel
            self.bridge.subscribe((ComponentStatus, VoiceState), self._note_seen)
            self.bridge.subscribe(None, self.island.handle_event)
            self.bridge.subscribe(None, self.tray.handle_event)
            self.island.toggleListeningRequested.connect(self.toggle_listening)
            self.island.openPanelRequested.connect(lambda: self.show_panel())
            self.island.settingsRequested.connect(lambda: self.show_panel("settings"))
            self.island.quitRequested.connect(self.quit)
            self.island.muteRequested.connect(self.set_muted)
            self.island.stopAllRequested.connect(self.stop_all)
            self.island.confirmAnswered.connect(self.answer_confirm)
            self.tray.openRequested.connect(lambda: self.show_panel())
            self.tray.muteRequested.connect(self.set_muted)
            self.tray.restartRequested.connect(self.restart)
            self.tray.quitRequested.connect(self.quit)
            self.panel_requested.connect(self.show_panel)
            self.quit_requested.connect(self.quit)
            self.caption_requested.connect(lambda text, tone: self.island.set_caption(text, tone))
            self.bridge.show_requested.connect(lambda: self.show_panel())
            app.ui = UiHandle(self)

        def _cfg(self, key: str, default: Any = None) -> Any:
            try:
                return self.app.config.get(key, default)
            except Exception:  # noqa: BLE001
                return default

        def start(self, *, show: bool = True) -> None:
            self.bridge.attach()
            # States published during app.start() came before this attach: read them back once.
            self.bridge.call(snapshot(self.app), on_ok=self._apply_snapshot, on_err=lambda _e: None)
            if show:
                self.island.show()
                # Tray and window icons are drawn after the island's first frame
                # (start-up budget: island visible < 3 s; icons are ~9 pixmaps each).
                QTimer.singleShot(0, self.tray.show)
                QTimer.singleShot(0, self._set_window_icon)
            # The panel is built right after the island is on screen (start-up
            # budget: island visible < 3 s), so it already collects events.
            QTimer.singleShot(200, self.ensure_panel)
            # SAM.pyw sets SAM_BACKGROUND=1 for the sign-in start: island only,
            # even when the user asked for the panel at start-up.
            if self._cfg("ui.panel_on_start", False) and os.environ.get("SAM_BACKGROUND") != "1":
                QTimer.singleShot(250, lambda: self.show_panel())

        def _set_window_icon(self) -> None:
            from .orb import orb_icon

            if self.qapp is not None:
                self.qapp.setWindowIcon(orb_icon())

        def ensure_panel(self) -> Any:
            if self.panel is None:
                from .panel import Panel

                started = time.perf_counter()
                self.panel = Panel(self.app, self.bridge)
                for ev in list(self._last.values()):     # states that arrived before the panel existed
                    self.panel.handle_event(ev)
                try:
                    self.app.timing.record("startup:ui_panel", (time.perf_counter() - started) * 1000.0,
                                           kind="startup")
                except Exception:  # noqa: BLE001
                    pass
            return self.panel

        def show_panel(self, page: str = "") -> None:
            self.ensure_panel().show_and_raise(page or None)

        # -- start-up snapshot -----------------------------------------------------------------------
        @staticmethod
        def _key(ev: Any) -> str:
            return "voice-state" if isinstance(ev, VoiceState) else f"component:{ev.component}"

        def _note_seen(self, ev: Any) -> None:
            key = self._key(ev)
            self._seen.add(key)
            self._last[key] = ev

        def _apply_snapshot(self, events: Any) -> None:
            """Deliver snapshot events whose component has not reported since
            the UI attached (a real event is always newer than the snapshot)."""
            for ev in events or []:
                if self._key(ev) not in self._seen:
                    self.bridge.deliver(ev)

        # -- commands --------------------------------------------------------------------------------
        def toggle_listening(self) -> None:
            if getattr(self.app, "voice", None) is None:
                self.island.set_caption(tr("island.voice_missing"), "danger")
                return
            self.bridge.call(self.app.toggle_listening(), on_err=self._command_failed)

        def set_muted(self, muted: bool) -> None:
            self.island.muted = bool(muted)
            self.island.update()
            self.tray.set_muted(bool(muted))
            if getattr(self.app, "voice", None) is not None:
                self.bridge.call(self.app.set_muted(bool(muted)), on_err=self._command_failed)

        def stop_all(self) -> None:
            self.bridge.call(self.app.stop_all(), on_err=self._command_failed)

        def answer_confirm(self, confirm_id: str, approved: bool, via: str) -> None:
            # ConfirmBroker.resolve is thread-safe (contract): no core hop needed.
            try:
                self.app.confirm.resolve(confirm_id, bool(approved), via=via)
            except Exception:  # noqa: BLE001
                log.exception("confirm resolve failed")

        def _command_failed(self, _error: BaseException) -> None:
            self.island.set_caption(tr("common.error"), "danger")

        def restart(self) -> None:
            if spawn_restart(getattr(getattr(self.app, "config", None), "home", None)):
                self.quit()

        def quit(self) -> None:
            self.shutdown()
            if self.qapp is not None:
                self.qapp.quit()

        def shutdown(self) -> None:
            self.bridge.detach()
            self.tray.hide()
            self.island.hide()
            if self.panel is not None:
                self.panel.hide()

    return UiController


_CONTROLLER_CLASS: type | None = None


def build(app: Any, core: Any, *, show: bool = True) -> Any:
    """Create the QApplication if needed and a started ``UiController``."""
    global _CONTROLLER_CLASS
    from PySide6.QtWidgets import QApplication

    qapp = QApplication.instance()
    if qapp is None:
        qapp = QApplication(sys.argv[:1])
    _setup_qapp(qapp, app)
    if _CONTROLLER_CLASS is None:
        _CONTROLLER_CLASS = _make_controller_class()
    controller = _CONTROLLER_CLASS(app, core)
    controller.start(show=show)
    return controller


def _setup_qapp(qapp: Any, app: Any) -> None:
    from . import theme

    qapp.setApplicationName("SAM")
    qapp.setApplicationDisplayName("SAM")
    qapp.setQuitOnLastWindowClosed(False)
    try:
        family = app.config.get("ui.font_family", None)
    except Exception:  # noqa: BLE001
        family = None
    qapp.setFont(theme.ui_font(13, 400, family))


def run(app: Any, core: Any, *, started: float | None = None) -> int:
    """Main-thread entry: build the UI and run the Qt event loop.

    ``started`` is the ``time.perf_counter()`` of process start (``sam.__main__``):
    ``startup:island_visible`` is recorded after the first event-loop turn,
    i.e. once the island has actually been painted."""
    began = time.perf_counter()
    controller = build(app, core)
    try:
        app.timing.record("startup:ui_island", (time.perf_counter() - began) * 1000.0, kind="startup")
    except Exception:  # noqa: BLE001
        pass
    if started is not None:
        from PySide6.QtCore import QTimer

        def visible() -> None:
            try:
                app.timing.record("startup:island_visible", (time.perf_counter() - started) * 1000.0,
                                  kind="startup")
            except Exception:  # noqa: BLE001
                pass
        QTimer.singleShot(0, visible)
    try:
        from ..winapp import watch_quit_requests, watch_show_requests

        watch_show_requests(controller.bridge.show_requested.emit)
        # ``python -m sam --quit`` / SAM.pyw --quit: the same clean path as tray Quit.
        watch_quit_requests(controller.quit_requested.emit)
    except Exception:  # noqa: BLE001
        log.warning("second-launch watcher unavailable")
    code = int(controller.qapp.exec() or 0)
    controller.shutdown()
    return code


__all__ = ["run", "build", "UiHandle", "UI_DEFAULTS"]
