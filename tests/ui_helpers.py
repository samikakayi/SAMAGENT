"""Test helpers for the UI package (Qt offscreen, no real desktop, no keys).

- ``qapp``: one QApplication per test session. The offscreen platform has no
  system fonts on Windows unless ``QT_QPA_FONTDIR`` points at them (measured:
  0 families without it, 279 with C:\\Windows\\Fonts), so it is set first.
- ``core``: a real ``CoreThread`` (the UI talks to the core only through it).
- ``ui_app``: a temp-home ``App`` whose bus is bound to that core loop, so the
  real ``UiAdapter`` forwards events through a queued Qt signal as in SAM.
- ``wait_until``: pump the Qt event loop until a condition holds.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))


def wait_until(predicate: Callable[[], Any], timeout: float = 5.0, step: float = 0.01) -> bool:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if app is not None:
            app.processEvents()
        if predicate():
            return True
        time.sleep(step)
    if app is not None:
        app.processEvents()
    return bool(predicate())


def pump(ms: int = 50) -> None:
    wait_until(lambda: False, timeout=ms / 1000.0)


@pytest.fixture(scope="session")
def qapp() -> Any:
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(["sam-ui-tests"])


@pytest.fixture
def core() -> Any:
    from sam.bridge import CoreThread

    thread = CoreThread(name="sam-core-test")
    thread.start()
    yield thread
    thread.stop()


@pytest.fixture
def ui_app(make_app: Any, core: Any, qapp: Any) -> Any:
    app = make_app()
    app.bus.bind_loop(core.loop)
    return app


@pytest.fixture
def controller(ui_app: Any, core: Any, qapp: Any) -> Any:
    import sam.ui as ui

    ctrl = ui.build(ui_app, core, show=False)
    yield ctrl
    ctrl.shutdown()
    if ctrl.panel is not None:
        ctrl.panel.deleteLater()
    ctrl.island.deleteLater()
    ctrl.tray.icon.deleteLater()
    pump(20)


def all_texts(root: Any) -> list[str]:
    """Every string a user could see or copy from a widget tree."""
    from PySide6.QtWidgets import (QAbstractButton, QComboBox, QLabel, QLineEdit, QPlainTextEdit, QTextEdit,
                                   QWidget)

    out: list[str] = []
    widgets = [root, *root.findChildren(QWidget)]
    for w in widgets:
        out += [w.toolTip(), w.windowTitle(), w.accessibleName(), w.statusTip()]
        if isinstance(w, QLabel):
            out.append(w.text())
        elif isinstance(w, QLineEdit):
            out += [w.text(), w.placeholderText(), w.displayText()]
        elif isinstance(w, QPlainTextEdit):
            out += [w.toPlainText(), w.placeholderText()]
        elif isinstance(w, QTextEdit):
            out.append(w.toPlainText())
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
    return [t for t in out if t]


class FakeVoice:
    """VoiceEngine stand-in: records calls, never touches devices."""

    def __init__(self, engine_name: str = "cascade", state: str = "idle") -> None:
        self.engine_name = engine_name
        self.state = state
        self.calls: list[tuple[str, Any]] = []

    async def toggle_listening(self) -> bool:
        self.calls.append(("toggle", None))
        return True

    async def set_muted(self, muted: bool) -> None:
        self.calls.append(("mute", muted))

    async def stop_speaking(self) -> None:
        self.calls.append(("stop_speaking", None))

    async def send_text(self, text: str) -> bool:
        self.calls.append(("send_text", text))
        return True

    async def run_selftest(self) -> dict[str, Any]:
        self.calls.append(("selftest", None))
        return {"ok": True, "cer": 0.08, "script_ok": True, "ttfa_ms": 1210.0}


class FakeConversation:
    """Publishes the same events as brain.conversation.handle_text."""

    def __init__(self, app: Any, reply: str = "باشە، کرایەوە.") -> None:
        self.app = app
        self.reply = reply
        self.texts: list[str] = []

    async def handle_text(self, text: str, *, source: str = "text") -> str:
        from sam.events import Caption, Transcript

        self.texts.append(text)
        bus = self.app.bus
        bus.publish(Caption(text=text, role="user", final=True))
        bus.publish(Transcript(role="user", text=text, source=source))
        bus.publish(Caption(text=self.reply[:4], role="assistant", final=False))
        bus.publish(Caption(text=self.reply, role="assistant", final=True))
        bus.publish(Transcript(role="assistant", text=self.reply, source=source))
        return self.reply


class FakeStrategies:
    """Same shapes as ``sam.trading.strategies.StrategyStore``: ``list()`` gives
    summary rows (``rules`` is a count, no card JSON); ``get()`` the whole card."""

    def __init__(self) -> None:
        self.cards = [
            {"id": "asia-fvg", "title_ckb": "ڕاماڵینی ئاسیا", "title_en": "Asia sweep", "status": "active",
             "version": 2, "summary_ckb": "کورتە", "markets": ["XAUUSD"],
             "timeframes": {"bias": "H4", "setup": "M15", "entry": "M5"}, "source_text": "Asia sweep then FVG",
             "rules": [{"kind": "bias", "text_ckb": "ئاراستەی H4 سەرەوە", "check": {"predicate": "trend_is"}}]},
            {"id": "ob", "title_ckb": "ئۆردەر بلۆک", "title_en": "OB", "status": "draft", "version": 1,
             "summary_ckb": "", "rules": []},
        ]
        self.status_calls: list[tuple[str, str]] = []
        self.get_calls: list[str] = []

    def list(self, status: Any = None) -> list[dict[str, Any]]:
        return [{"id": c["id"], "title_ckb": c["title_ckb"], "title_en": c["title_en"], "status": c["status"],
                 "version": c["version"], "summary_ckb": c["summary_ckb"], "rules": len(c["rules"]),
                 "markets": c.get("markets", []), "updated_at": time.time(), "note": ""} for c in self.cards]

    def get(self, strategy_id: str) -> dict[str, Any] | None:
        self.get_calls.append(strategy_id)
        card = next((c for c in self.cards if c["id"] == strategy_id), None)
        return dict(card) if card else None

    def set_status(self, strategy_id: str, status: str) -> dict[str, Any]:
        self.status_calls.append((strategy_id, status))
        for card in self.cards:
            if card["id"] == strategy_id:
                card["status"] = status
        return {"ok": True}

    def versions(self, strategy_id: str) -> list[dict[str, Any]]:
        return [{"version": 2, "created_at": time.time(), "reason": "edit"}]


class FakeMonitor:
    def __init__(self) -> None:
        self.cancelled: list[Any] = []

    def cancel(self, alert_id: Any) -> int:
        self.cancelled.append(alert_id)
        return 1


__all__ = ["wait_until", "pump", "qapp", "core", "ui_app", "controller", "all_texts", "FakeVoice",
           "FakeConversation", "FakeStrategies", "FakeMonitor"]
