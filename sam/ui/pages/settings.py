"""ڕێکخستنەکان — keys (write-only), voice, trading, privacy.

Key safety (DESIGN §1, CONTRACTS "Keys"):
- key fields are password fields; the pasted value is taken out of the field
  and the field cleared *before* the save is scheduled, it is passed only to
  ``app.secrets.set`` (in a worker thread: the store runs DPAPI + icacls, up to
  15 s) and is never put in a label, tooltip, log line or error message;
- after saving, the UI shows only configured / not set, the source and the
  store's non-reversible fingerprint (first 6 hex of a SHA-256);
- error messages are fixed Sorani text (the store's ValueError never contains
  the value either).
"""

from __future__ import annotations

import re
from typing import Any

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from ...events import ComponentStatus, Error, SettingsChanged
from .. import theme
from ..strings import GEMINI_VOICES, ckb_digits, en, tr, tr_or
from ..widgets import Card, StatusDot, ToggleSwitch, accessible
from . import SCROLL_GUTTER, Page, scroll_area

GEMINI_KEY_URL = "https://aistudio.google.com/apikey"
# (secret name, LLM provider for the Test button or None, "get a key" link or None)
KEY_ROWS: tuple[tuple[str, str | None, str | None], ...] = (
    ("gemini_api_key", "gemini", GEMINI_KEY_URL),
    ("groq_api_key", "groq", "https://console.groq.com/keys"),
    ("openrouter_api_key", "openrouter", None),
    ("kurdishtts_stt_api_key", None, None),
    ("kurdishtts_tts_api_key", None, None),
)
# Fallback check when the voice package is not loaded. The voice engine's own
# parser (sam.voice.hotkey.parse_hotkey) is the source of truth: RegisterHotKey
# needs at least one modifier, so a bare F-key is rejected by both.
HOTKEY_RE = re.compile(r"^((ctrl|alt|shift|win)\+){1,3}([a-z0-9]|space|f([1-9]|1[0-9]|2[0-4])|enter|tab)$")
STATUS_TO_STATE = {"connected": "ok", "auth_failed": "down", "rate_limited": "degraded", "unreachable": "down",
                   "unconfigured": "unconfigured", "error": "down"}


class KeyRow(QWidget):
    """One write-only key: status dot, name, fingerprint, paste field, Save, Test."""

    def __init__(self, page: "SettingsPage", name: str, provider: str | None, link: str | None) -> None:
        super().__init__()
        self.page = page
        self.name = name
        self.provider = provider
        # A unique AutomationId per row (Qt builds it from objectNames): "...Card.KeyRow_groq_api_key.QLineEdit".
        self.setObjectName(f"KeyRow_{name}")
        label = tr(f"set.key.{name}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 6)
        lay.setSpacing(8)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.dot = StatusDot("unknown")
        title = QLabel(tr(f"set.key.{name}"))
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        self.status = QLabel(tr("common.loading"))
        self.status.setObjectName("Faint")
        top.addWidget(self.dot)
        top.addWidget(title)
        if page.show_english and en(f"set.key.{name}") != tr(f"set.key.{name}"):
            sub = QLabel(en(f"set.key.{name}"))
            sub.setObjectName("Faint")
            top.addWidget(sub)
        top.addStretch(1)
        top.addWidget(self.status)
        if link:
            get = QPushButton(tr("set.key.get") + " ↗")
            get.setObjectName("Ghost")
            get.setToolTip(link)
            get.setCursor(Qt.CursorShape.PointingHandCursor)
            get.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(link)))
            accessible(get, "a11y.key.get", name=label)
            top.addWidget(get)
        lay.addLayout(top)
        line = QHBoxLayout()
        line.setSpacing(8)
        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit.setPlaceholderText(tr("set.paste"))
        self.edit.setLayoutDirection(Qt.LayoutDirection.LeftToRight)   # keys are Latin
        self.edit.setClearButtonEnabled(True)
        self.edit.returnPressed.connect(self.save)
        accessible(self.edit, "a11y.key.field", object_name="key", name=label)
        self.save_button = QPushButton(tr("set.save"))
        self.save_button.setObjectName("Primary")
        self.save_button.clicked.connect(self.save)
        accessible(self.save_button, "a11y.key.save", name=label)
        self.test_button = QPushButton(tr("set.test"))
        self.test_button.clicked.connect(self.test)
        accessible(self.test_button, "a11y.key.test", object_name="test", name=label)
        accessible(self.status, "a11y.key.status", name=label)
        line.addWidget(self.edit, 1)
        line.addWidget(self.save_button)
        line.addWidget(self.test_button)
        lay.addLayout(line)
        self.result = QLabel("")
        self.result.setObjectName("Muted")
        self.result.setWordWrap(True)
        self.result.hide()
        lay.addWidget(self.result)

    def set_status(self, info: dict[str, Any] | None) -> None:
        info = info or {}
        if info.get("configured"):
            fp = str(info.get("fingerprint") or "")[:6]
            src = tr("set.from_env") if info.get("source") == "environment" else ""
            text = " · ".join(x for x in (tr("set.configured"), src, f"#{fp}" if fp else "") if x)
            self.dot.set_state("ok")
        else:
            text = tr("set.not_set")
            self.dot.set_state("unconfigured")
        self.status.setText(text)

    def _show(self, text: str, state: str | None = None) -> None:
        self.result.setText(text)
        self.result.setVisible(bool(text))
        if state:
            self.dot.set_state(state)

    def save(self) -> None:
        value = self.edit.text().strip()
        # setText() (unlike clear()) also drops the undo history, so the key
        # cannot be brought back with Ctrl+Z after saving.
        self.edit.setText("")
        if not value:
            return
        self.save_button.setEnabled(False)
        self._show(tr("set.saving"))
        secrets = self.page.app.secrets
        self.page.bridge.run(secrets.set, self.name, value, on_ok=self._saved, on_err=self._save_failed)
        del value

    def _saved(self, _result: Any) -> None:
        self.save_button.setEnabled(True)
        self._show(tr("set.saved"), "ok")
        self.page.refresh_keys()
        llm = getattr(self.page.app, "llm", None)
        if self.provider and llm is not None and hasattr(llm, "reset_provider"):
            # A new key: forget the rests the old one earned (a wrong-key rest is
            # persisted for 10 minutes; verify review 2026-09-24).
            self.page.bridge.on_core(llm.reset_provider, self.provider, on_err=lambda _e: None)

    def _save_failed(self, error: BaseException) -> None:
        self.save_button.setEnabled(True)
        self._show(tr("set.bad_key") if isinstance(error, ValueError) else tr("set.save_failed"), "down")

    def test(self) -> None:
        app = self.page.app
        if self.provider is not None and getattr(app, "llm", None) is not None:
            self._show(tr("set.testing"))
            self.test_button.setEnabled(False)
            self.page.bridge.call(app.llm.test_provider(self.provider), on_ok=self._tested,
                                  on_err=lambda _e: self._tested({"ok": False, "status": "error"}))
            return
        voice = getattr(app, "voice", None)
        tester = getattr(voice, "test_key", None)
        if callable(tester):
            self._show(tr("set.testing"))
            self.test_button.setEnabled(False)
            self.page.bridge.call(tester(self.name), on_ok=self._tested,
                                  on_err=lambda _e: self._tested({"ok": False, "status": "error"}))
            return
        self._show(tr("set.test.presence"))

    def _tested(self, result: Any) -> None:
        self.test_button.setEnabled(True)
        self._show(*describe_test(result))


def describe_test(result: Any) -> tuple[str, str]:
    """Sorani line + dot state for a ``test_provider`` result."""
    result = result if isinstance(result, dict) else {}
    latency = result.get("latency_ms")
    if result.get("ok"):
        extra = f" · {ckb_digits(int(latency))} ms" if isinstance(latency, (int, float)) else ""
        return tr("set.test_ok") + extra, "ok"
    status = str(result.get("status") or "error")
    return tr_or(f"set.test.{status}", tr("set.test.error")), STATUS_TO_STATE.get(status, "down")


class SettingsPage(Page):
    key = "settings"

    def __init__(self, app: Any, bridge: Any, parent: QWidget | None = None) -> None:
        super().__init__(app, bridge, parent)
        self.show_english = bool(self.cfg("ui.show_english", True))
        self.add_header()
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(SCROLL_GUTTER, 0, 0, 0)           # RTL scroll bar is on the left
        col.setSpacing(16)
        col.addWidget(self._keys_card())
        col.addWidget(self._voice_card())
        # Listening windows + «تەنها دەنگی من» (sam/ui/voice_profile.py, voice package owner).
        from ..voice_profile import VoiceProfileCard
        self.voice_profile = VoiceProfileCard(app, bridge)
        col.addWidget(self.voice_profile)
        col.addWidget(self._trading_card())
        col.addWidget(self._privacy_card())
        col.addWidget(self._about_card())
        col.addStretch(1)
        self.root.addWidget(scroll_area(inner), 1)
        self._timeout_timer = QTimer(self)
        self._timeout_timer.setSingleShot(True)
        self._timeout_timer.setInterval(600)
        self._timeout_timer.timeout.connect(lambda: self._set("voice.conversation_timeout_s", self.timeout.value()))

    # -- cards ------------------------------------------------------------------------------------------
    def _keys_card(self) -> Card:
        card = Card(tr("set.keys"), tr("set.keys_sub"))
        self.key_rows: dict[str, KeyRow] = {}
        for name, provider, link in KEY_ROWS:
            row = KeyRow(self, name, provider, link)
            self.key_rows[name] = row
            card.body.addWidget(row)
        # OmniRoute: the local gateway; its client key lives in .env (never shown).
        omni = QHBoxLayout()
        self.omni_dot = StatusDot("unknown")
        title = QLabel(tr("set.omniroute"))
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        base = QLabel(str(self.cfg("providers.omniroute.base_url", "")))
        base.setObjectName("Faint")
        self.omni_status = QLabel("")
        self.omni_status.setObjectName("Muted")
        test = QPushButton(tr("set.test"))
        test.clicked.connect(self.test_omniroute)
        accessible(test, "a11y.omniroute.test", object_name="omniroute_test")
        omni.addWidget(self.omni_dot)
        omni.addWidget(title)
        omni.addWidget(base)
        omni.addStretch(1)
        omni.addWidget(self.omni_status)
        omni.addWidget(test)
        card.body.addSpacing(6)
        card.body.addLayout(omni)
        return card

    def _voice_card(self) -> Card:
        card = Card(tr("set.voice"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(14)
        grid.setColumnStretch(1, 1)
        r = 0
        engines = QHBoxLayout()
        engines.setSpacing(8)
        self.engine_group = QButtonGroup(self)
        current = str(self.cfg("voice.engine", "auto"))
        self.engine_buttons: dict[str, QPushButton] = {}
        for key in ("auto", "live", "cascade"):
            chip = QPushButton(tr(f"set.voice.{key}"))
            chip.setObjectName("Chip")
            chip.setCheckable(True)
            chip.setChecked(key == current)
            # toggled, not clicked: UI Automation's Toggle (screen readers, SAM, tests) checks
            # the chip without a click, and the choice must still be saved.
            chip.toggled.connect(lambda on, k=key: self._choose_engine(k, on))
            accessible(chip, "a11y.engine", choice=tr(f"set.voice.{key}"))
            self.engine_group.addButton(chip)
            self.engine_buttons[key] = chip
            engines.addWidget(chip)
        engines.addStretch(1)
        grid.addWidget(_key_label(tr("set.voice.engine")), r, 0)
        grid.addLayout(engines, r, 1)
        r += 1
        test_line = QHBoxLayout()
        self.selftest_button = QPushButton(tr("set.voice.selftest"))
        self.selftest_button.clicked.connect(self.run_selftest)
        accessible(self.selftest_button, "a11y.selftest", object_name="selftest")
        self.selftest_result = QLabel(self._selftest_text(self.cfg("voice.selftest")))
        self.selftest_result.setToolTip(self.selftest_details(self.cfg("voice.selftest")))
        self.selftest_result.setObjectName("Muted")
        self.selftest_result.setWordWrap(True)
        test_line.addWidget(self.selftest_button)
        test_line.addWidget(self.selftest_result, 1)
        grid.addWidget(_key_label(""), r, 0)
        grid.addLayout(test_line, r, 1)
        r += 1
        note = QLabel(tr("set.voice.selftest_note"))
        note.setObjectName("Faint")
        grid.addWidget(note, r, 1)
        r += 1
        self.voice_name = QComboBox()
        names = [n for n, _ in GEMINI_VOICES]
        for name, style in GEMINI_VOICES:
            self.voice_name.addItem(f"{name} — {style}", name)
        chosen = str(self.cfg("voice.voice_name", "Kore"))
        self.voice_name.setCurrentIndex(names.index(chosen) if chosen in names else 0)
        self.voice_name.currentIndexChanged.connect(
            lambda i: self._set("voice.voice_name", self.voice_name.itemData(i)))
        self.voice_name.setMaximumWidth(280)
        accessible(self.voice_name, "a11y.voice_name", object_name="voice_name")
        grid.addWidget(_key_label(tr("set.voice.name")), r, 0)
        grid.addWidget(self.voice_name, r, 1, Qt.AlignmentFlag.AlignLeading)
        r += 1
        self.hotkey = QLineEdit(str(self.cfg("voice.hotkey", "ctrl+alt+space")))
        self.hotkey.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.hotkey.setMaximumWidth(280)
        self.hotkey.editingFinished.connect(self._save_hotkey)
        accessible(self.hotkey, "a11y.hotkey", object_name="hotkey")
        self.hotkey_hint = QLabel("")
        self.hotkey_hint.setObjectName("Faint")
        hk = QHBoxLayout()
        hk.addWidget(self.hotkey)
        hk.addWidget(self.hotkey_hint, 1)
        grid.addWidget(_key_label(tr("set.voice.hotkey")), r, 0)
        grid.addLayout(hk, r, 1)
        r += 1
        self.timeout = QSpinBox()
        self.timeout.setRange(10, 600)
        self.timeout.setSingleStep(5)
        self.timeout.setSuffix(f"  {tr('set.voice.seconds')}")
        self.timeout.setValue(int(self.cfg("voice.conversation_timeout_s", 45) or 45))
        self.timeout.setMaximumWidth(160)
        self.timeout.valueChanged.connect(lambda _v: self._timeout_timer.start())
        accessible(self.timeout, "a11y.timeout", object_name="conversation_timeout")
        grid.addWidget(_key_label(tr("set.voice.timeout")), r, 0)
        grid.addWidget(self.timeout, r, 1, Qt.AlignmentFlag.AlignLeading)
        r += 1
        self.always = ToggleSwitch()
        self.always.setChecked(bool(self.cfg("voice.always_listening", False)))
        self.always.toggled.connect(lambda on: self._set("voice.always_listening", bool(on)))
        accessible(self.always, "a11y.always", object_name="always_listening")
        grid.addWidget(_key_label(tr("set.voice.always")), r, 0)
        grid.addWidget(self.always, r, 1, Qt.AlignmentFlag.AlignLeading)
        card.body.addLayout(grid)
        return card

    def _trading_card(self) -> Card:
        card = Card(tr("set.trading"))
        tv = QHBoxLayout()
        self.tv_dot = StatusDot("unknown")
        tv_title = QLabel(tr("set.tv"))
        tv_title.setStyleSheet("font-weight: 600; font-size: 14px;")
        self.tv_status = QLabel("")
        self.tv_status.setObjectName("Muted")
        self.tv_button = QPushButton(tr("set.tv.connect"))
        self.tv_button.setObjectName("Primary")
        self.tv_button.clicked.connect(self.connect_tradingview)
        accessible(self.tv_button, "a11y.tv.connect")
        tv.addWidget(self.tv_dot)
        tv.addWidget(tv_title)
        tv.addStretch(1)
        tv.addWidget(self.tv_status)
        tv.addWidget(self.tv_button)
        card.body.addLayout(tv)
        tv_note = QLabel(tr("set.tv.note"))
        tv_note.setObjectName("Faint")
        tv_note.setWordWrap(True)
        card.body.addWidget(tv_note)
        mt5 = QHBoxLayout()
        self.mt5_dot = StatusDot("unknown")
        mt5_title = QLabel(tr("set.mt5"))
        mt5_title.setStyleSheet("font-weight: 600; font-size: 14px;")
        self.mt5_status = QLabel("")
        self.mt5_status.setObjectName("Muted")
        self.mt5_button = QPushButton(tr("set.mt5.refresh"))
        self.mt5_button.clicked.connect(self.check_mt5)
        accessible(self.mt5_button, "a11y.mt5.check", object_name="mt5_check")
        mt5.addWidget(self.mt5_dot)
        mt5.addWidget(mt5_title)
        mt5.addStretch(1)
        mt5.addWidget(self.mt5_status)
        mt5.addWidget(self.mt5_button)
        card.body.addSpacing(4)
        card.body.addLayout(mt5)
        return card

    def _privacy_card(self) -> Card:
        card = Card(tr("set.privacy"))
        text = QLabel(tr("set.privacy.note"))
        text.setWordWrap(True)
        text.setStyleSheet(f"color: {theme.TEXT_MUTED}; line-height: 140%;")
        card.body.addWidget(text)
        return card

    def _about_card(self) -> Card:
        card = Card(tr("set.about"))
        config = getattr(self.app, "config", None)
        for key, value in (("set.home", getattr(config, "home", "")), ("set.logs", getattr(config, "log_dir", ""))):
            line = QHBoxLayout()
            line.addWidget(_key_label(tr(key)))
            path = QLabel(str(value))
            path.setObjectName("Faint")
            path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            line.addWidget(path, 1)
            card.body.addLayout(line)
        return card

    # -- behaviour -----------------------------------------------------------------------------------------------
    def on_shown(self) -> None:
        self.refresh_keys()
        self._component_defaults()
        self._load_voice_status()

    def refresh_keys(self) -> None:
        secrets = getattr(self.app, "secrets", None)
        if secrets is None:
            return
        self.bridge.run(secrets.status, on_ok=self._apply_key_status, on_err=lambda _e: None)

    def _apply_key_status(self, status: Any) -> None:
        status = status if isinstance(status, dict) else {}
        for name, row in self.key_rows.items():
            row.set_status(status.get(name))
        litellm = status.get("litellm_api_key") or {}
        if not self.omni_status.text():
            self.omni_status.setText(tr("set.configured") if litellm.get("configured") else tr("set.not_set"))

    def _component_defaults(self) -> None:
        trading = getattr(self.app, "trading", None)
        if getattr(trading, "tv", None) is None:
            self.tv_status.setText(tr("status.unavailable"))
            self.tv_button.setEnabled(False)
        if getattr(trading, "mt5", None) is None:
            self.mt5_status.setText(tr("status.unavailable"))
            self.mt5_button.setEnabled(False)
        if getattr(self.app, "voice", None) is None:
            self.selftest_button.setEnabled(False)
            self.selftest_button.setToolTip(tr("island.voice_missing"))

    def _choose_engine(self, key: str, on: bool) -> None:
        if on and str(self.cfg("voice.engine", "auto")) != key:
            self._set("voice.engine", key)

    def _set(self, key: str, value: Any) -> None:
        config = getattr(self.app, "config", None)
        if config is not None:
            self.bridge.run(config.set, key, value, on_err=lambda _e: None)

    def _save_hotkey(self) -> None:
        value = self.hotkey.text().strip().lower().replace(" ", "")
        if value == str(self.cfg("voice.hotkey", "") or ""):
            self.hotkey.setText(value)
            return
        if valid_hotkey(value):
            self.hotkey.setText(value)
            self.hotkey_hint.setText("")
            self._set("voice.hotkey", value)
        else:
            self.hotkey_hint.setText(tr("set.voice.hotkey_bad"))

    def _load_voice_status(self) -> None:
        """Show a hotkey the voice engine could not register (another program
        owns the chord) -- read once when the page opens."""
        voice = getattr(self.app, "voice", None)
        if voice is not None and callable(getattr(voice, "status", None)):
            self.bridge.on_core(voice.status, on_ok=self._voice_status, on_err=lambda _e: None)

    def _voice_status(self, status: Any) -> None:
        hotkey = (status or {}).get("hotkey") if isinstance(status, dict) else None
        if isinstance(hotkey, dict) and hotkey.get("error") and not hotkey.get("registered"):
            self.hotkey_hint.setText(tr("set.voice.hotkey_taken"))

    def test_omniroute(self) -> None:
        if getattr(self.app, "llm", None) is None:
            return
        self.omni_status.setText(tr("set.testing"))
        self.bridge.call(self.app.llm.test_provider("omniroute"), on_ok=self._omni_tested,
                         on_err=lambda _e: self._omni_tested({"ok": False, "status": "error"}))

    def _omni_tested(self, result: Any) -> None:
        text, state = describe_test(result)
        self.omni_status.setText(text)
        self.omni_dot.set_state(state)

    def run_selftest(self) -> None:
        voice = getattr(self.app, "voice", None)
        if voice is None or not hasattr(voice, "run_selftest"):
            self.selftest_result.setText(tr("island.voice_missing"))
            return
        self.selftest_button.setEnabled(False)
        self.selftest_result.setText(tr("set.testing"))
        self.bridge.call(voice.run_selftest(), on_ok=self._selftest_done,
                         on_err=lambda _e: self._selftest_done({"ok": False}))

    def _selftest_done(self, result: Any) -> None:
        self.selftest_button.setEnabled(True)
        self.selftest_result.setText(self._selftest_text(result, fresh=True))
        self.selftest_result.setToolTip(self.selftest_details(result))

    @staticmethod
    def _selftest_text(result: Any, fresh: bool = False) -> str:
        """Plain Sorani (good / weak / unfinished); the measurements are in
        ``selftest_details`` (tooltip) for whoever wants them."""
        if not isinstance(result, dict) or not result:
            return tr("set.voice.selftest_none") if not fresh else tr("set.voice.selftest_unfinished")
        if result.get("ok"):
            return tr("set.voice.selftest_good")
        measured = isinstance(result.get("cer"), (int, float)) or result.get("script_ok") is False
        return tr("set.voice.selftest_weak") if measured else tr("set.voice.selftest_unfinished")

    @staticmethod
    def selftest_details(result: Any) -> str:
        if not isinstance(result, dict):
            return ""
        parts = []
        if isinstance(result.get("cer"), (int, float)):
            parts.append(f"CER {result['cer'] * 100:.0f}%")
        if isinstance(result.get("ttfa_ms"), (int, float)):
            parts.append(f"TTFA {result['ttfa_ms']:.0f} ms")
        if result.get("script_ok") is False:
            parts.append("script ✗")
        return " · ".join(parts)

    def connect_tradingview(self) -> None:
        tv = getattr(getattr(self.app, "trading", None), "tv", None)
        if tv is None:
            return
        self.tv_button.setEnabled(False)
        self.tv_status.setText(tr("set.testing"))
        confirm = getattr(getattr(self.app, "confirm", None), "confirm", None)
        # A restart of a port-less TradingView is confirmed on the island first.
        self.bridge.call(tv.ensure_running(allow_restart=True, confirm=confirm), on_ok=self._tv_done,
                         on_err=lambda _e: self._tv_done({"ok": False, "state": "failed"}))

    def _tv_done(self, result: Any) -> None:
        self.tv_button.setEnabled(True)
        result = result if isinstance(result, dict) else {}
        state = str(result.get("state") or ("connected" if result.get("ok") else "failed"))
        self.tv_status.setText(tr_or(f"tv.state.{state}", state))
        self.tv_dot.set_state("ok" if result.get("ok") else "degraded" if state == "needs_restart" else "down")

    def check_mt5(self) -> None:
        mt5 = getattr(getattr(self.app, "trading", None), "mt5", None)
        if mt5 is None:
            return
        self.mt5_button.setEnabled(False)
        self.bridge.call(mt5.status(), on_ok=self._mt5_done, on_err=lambda _e: self._mt5_done({"connected": False}))

    def _mt5_done(self, status: Any) -> None:
        self.mt5_button.setEnabled(True)
        status = status if isinstance(status, dict) else {}
        if status.get("connected"):
            text = tr("status.connected")
            offset = status.get("broker_offset_s")
            if isinstance(offset, (int, float)):
                hours = offset / 3600.0
                text += f" · {tr('mt5.offset')} {'+' if hours >= 0 else '−'}{ckb_digits(f'{abs(hours):g}')} " \
                        f"{tr('mt5.hours')}"
            self.mt5_dot.set_state("ok")
        else:
            text = tr("status.not_connected")
            self.mt5_dot.set_state("down")
        self.mt5_status.setText(text)

    def handle_event(self, ev: Any) -> None:
        if isinstance(ev, ComponentStatus):
            if ev.component == "tradingview":
                self.tv_dot.set_state(ev.state)
                self.tv_status.setText(_link_state_text(ev.state))
            elif ev.component == "mt5":
                self.mt5_dot.set_state(ev.state)
                self.mt5_status.setText(_link_state_text(ev.state))
            elif ev.component == "omniroute":
                self.omni_dot.set_state(ev.state)
                self.omni_status.setText(tr_or(f"status.{ev.state}", ev.state))
            elif ev.component in ("gemini", "groq", "openrouter"):
                row = self.key_rows.get(f"{ev.component}_api_key")
                if row is not None and ev.state != "ok":
                    row.dot.set_state(ev.state)
        elif isinstance(ev, SettingsChanged):
            if ev.key == "voice.selftest":
                self.selftest_result.setText(self._selftest_text(ev.value))
                self.selftest_result.setToolTip(self.selftest_details(ev.value))
            elif ev.key == "voice.engine" and ev.value in self.engine_buttons:
                self.engine_buttons[ev.value].setChecked(True)
            elif ev.key == "voice.hotkey" and isinstance(ev.value, str) and not self.hotkey.hasFocus():
                # e.g. the voice engine moved to a free fallback chord
                self.hotkey.setText(ev.value)
        elif isinstance(ev, Error) and ev.where == "voice.hotkey" and ev.message_ckb:
            self.hotkey_hint.setText(ev.message_ckb)


def valid_hotkey(value: str) -> bool:
    """True when the voice engine will accept ``value`` as its hotkey."""
    try:
        from ...voice.hotkey import parse_hotkey
    except Exception:  # noqa: BLE001 - voice package missing: use the fallback pattern
        return bool(HOTKEY_RE.match(value))
    try:
        parse_hotkey(value)
    except ValueError:
        return False
    return True


def _link_state_text(state: str) -> str:
    """TradingView / MT5 are connections: "connected" reads better than "OK"."""
    return {"ok": tr("status.connected"), "down": tr("status.not_connected")}.get(
        state, tr_or(f"status.{state}", state))


def _key_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Muted")
    lab.setMinimumWidth(170)
    return lab


__all__ = ["SettingsPage", "KeyRow", "describe_test", "valid_hotkey", "KEY_ROWS", "HOTKEY_RE"]
