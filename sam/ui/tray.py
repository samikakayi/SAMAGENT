"""Tray icon: کردنەوە (open) / بێدەنگ (mute) / دەستپێکردنەوە (restart) / داخستن (quit).

The icon is generated (the SAM orb) and tinted by the voice state. Alerts pop a
balloon (``QSystemTrayIcon.showMessage``), which on Windows 11 becomes a
normal notification -- no extra toast dependency.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from ..events import Alert, VoiceState
from . import theme
from .orb import orb_icon
from .strings import tr

# State -> icon tint. Idle-like states share one icon so the tray does not flicker.
TRAY_STATES = {"listening": "listening", "speaking": "speaking", "thinking": "thinking", "working": "working",
               "error": "error", "muted": "muted"}


class Tray(QObject):
    openRequested = Signal()
    muteRequested = Signal(bool)
    restartRequested = Signal()
    quitRequested = Signal()

    def __init__(self, families: list[str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._icons: dict[str, QIcon] = {}
        self.muted = False
        # The icon pixmaps are drawn on show(), not here: the controller builds
        # the tray before the island appears, and the island comes first.
        self.icon = QSystemTrayIcon(parent)
        self.icon.setToolTip("SAM")
        self.menu = QMenu()
        self.menu.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.menu.setStyleSheet(theme.menu_stylesheet(families))
        self.open_action = QAction(tr("tray.open"), self.menu)
        self.open_action.triggered.connect(self.openRequested.emit)
        self.mute_action = QAction(tr("menu.mute"), self.menu)
        self.mute_action.setCheckable(True)
        self.mute_action.triggered.connect(lambda checked: self._mute(bool(checked)))
        self.restart_action = QAction(tr("tray.restart"), self.menu)
        self.restart_action.triggered.connect(self.restartRequested.emit)
        self.quit_action = QAction(tr("menu.quit"), self.menu)
        self.quit_action.triggered.connect(self.quitRequested.emit)
        self.menu.addAction(self.open_action)
        self.menu.addAction(self.mute_action)
        self.menu.addSeparator()
        self.menu.addAction(self.restart_action)
        self.menu.addAction(self.quit_action)
        self.icon.setContextMenu(self.menu)
        self.icon.activated.connect(self._activated)

    @property
    def actions(self) -> list[QAction]:
        return [a for a in self.menu.actions() if not a.isSeparator()]

    def _icon(self, state: str) -> QIcon:
        key = TRAY_STATES.get(state, "idle")
        if key not in self._icons:
            self._icons[key] = orb_icon(state=key)
        return self._icons[key]

    def show(self) -> bool:
        if self.icon.icon().isNull():
            self.icon.setIcon(self._icon("idle"))
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.icon.show()
            return True
        return False

    def hide(self) -> None:
        self.icon.hide()

    def _activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.openRequested.emit()

    def _mute(self, muted: bool) -> None:
        self.muted = muted
        self.muteRequested.emit(muted)

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        self.mute_action.setChecked(muted)

    def handle_event(self, ev: Any) -> None:
        if isinstance(ev, VoiceState):
            self.icon.setIcon(self._icon(ev.state))
            if ev.state == "muted":
                self.set_muted(True)
            elif ev.state == "listening":
                self.set_muted(False)
        elif isinstance(ev, Alert):
            self.icon.showMessage(tr("tray.alert_title"), ev.text_ckb or ev.symbol,
                                  QSystemTrayIcon.MessageIcon.Information, 8000)


def restart_command(home: str | os.PathLike[str] | None, pid: int | None = None,
                    launcher: Path | None = None) -> list[str]:
    """The replacement process, without a console window.

    Preferred: ``pythonw SAM.pyw --background --after-pid <pid> --home <home>``
    -- the launcher keeps its log and starts OmniRoute early, skips its own
    single-instance check because of ``--after-pid``, and ``--background``
    brings back the island only. Fallback (no SAM.pyw):
    ``pythonw -m sam --after-pid <pid> --home <home>``."""
    from ..config import REPO_ROOT

    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    python = str(pythonw if pythonw.exists() else exe)
    launcher = REPO_ROOT / "SAM.pyw" if launcher is None else launcher
    pid_args = ["--after-pid", str(pid or os.getpid())]
    if launcher.is_file():
        cmd = [python, str(launcher), "--background", *pid_args]
    else:
        cmd = [python, "-m", "sam", *pid_args]
    if home:
        cmd += ["--home", str(home)]
    return cmd


def spawn_restart(home: str | os.PathLike[str] | None) -> bool:
    """Start the replacement process; the caller then quits. The new process
    waits for this pid to exit before taking the single-instance mutex."""
    from ..config import REPO_ROOT

    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP   # never DETACHED_PROCESS
    try:
        subprocess.Popen(restart_command(home), cwd=str(REPO_ROOT), creationflags=flags, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False


__all__ = ["Tray", "restart_command", "spawn_restart"]
