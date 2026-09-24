"""Settings card «گوێگرتن و دەنگی من» + the voice enrollment dialog.

- Listening: follow-up window seconds (``voice.followup_s``) and how strictly
  quiet/far speech is ignored (``voice.gate_margin_db``); "always listening"
  stays in the voice card above (it now also needs the word «سام»).
- «تەنها دەنگی من»: toggle (``voice.only_my_voice``), sensitivity
  (``voice.only_my_voice_sensitivity``), «ناساندنی دەنگی من» (opens the
  enrollment dialog) and «سڕینەوەی دەنگی من» (deletes the voiceprint).
- The dialog also opens when the user SAYS «دەنگم بناسە» (``VoiceEnrollRequest``).

Enrollment is the only time SAM records on purpose: the dialog opens on a
Start step (nothing is recorded until «دەست پێبکە» is clicked -- a spoken
«دەنگم بناسە» only opens it), then shows one Sorani sentence at a time,
records it (sam/voice/enroll.py; audio stays in memory), and at the end
stores only the voiceprint. When one clip does not match the others, only
that sentence is read again. The card also shows the learned speech level
with «ئاستی دەنگم لەبیر بکە», and says so when the voiceprint cannot run
(the check then lets every voice through). Every control has a Sorani
accessible name so UI Automation (and SAM itself) can operate it.
Everything reaches the core through ``bridge.call`` (never blocks the GUI).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from ..voice.notices import VoiceEnrollRequest, VoiceNotice
from .widgets import Card, ToggleSwitch

TEXT: dict[str, str] = {
    "card": "گوێگرتن و دەنگی من",
    "card_sub": "کرتەیەک بۆ یەک قسە. دوای ناساندنی دەنگت، {followup} چرکە دوای وەڵام بۆ قسەیەکی تر گوێ دەگرم. "
                "دەنگی تەلەفزیۆن و دوور گوێی پێ نادرێت.",
    "followup": "ماوەی قسەی دووەم دوای وەڵام",
    "seconds": "چرکە",
    "strict": "دەنگی دوور و نزم",
    "strict.low": "کەم فلتەر",
    "strict.normal": "ئاسایی",
    "strict.high": "توند",
    "only": "تەنها دەنگی من",
    "sens": "هەستیاری",
    "sens.low": "نزم (ئاسانتر قبووڵ دەکات)",
    "sens.normal": "ئاسایی",
    "sens.high": "بەرز (توندتر)",
    "enroll": "ناساندنی دەنگی من",
    "delete": "سڕینەوەی دەنگی من",
    "state.none": "دەنگی تۆ هێشتا نەناسراوە — تا ئەو کاتە تەنها دەنگی نزیک و بەرز قبووڵ دەکرێت.",
    "state.on": "دەنگی تۆ ناسراوە؛ تەنها دەنگی تۆ وەردەگیرێت.",
    "state.off": "دەنگی تۆ ناسراوە، بەڵام «تەنها دەنگی من» کوژاوەتەوە.",
    "state.unavailable": "ناسینەوەی دەنگ ئامادە نییە: ",
    "state.not_ready": "ناسینەوەی دەنگ ئامادە نییە — ئێستا هەموو دەنگێکی نزیک وەردەگیرێت. "
                       "«ناساندنی دەنگی من» دووبارە بکەرەوە.",
    "level": "ئاستی دەنگی تۆ",
    "level.none": "هێشتا نەپێوراوە",
    "level.value": "{db} dB",
    "level.reset": "ئاستی دەنگم لەبیر بکە",
    "privacy": "دەنگەکە تەنها لەم کۆمپیوتەرەدا دەمێنێتەوە (پارێزراو بە Windows DPAPI) و هەر کاتێک دەسڕدرێتەوە.",
    "dlg.title": "ناساندنی دەنگی من",
    "dlg.intro": "ئەم ڕستانە یەک بە یەک بە دەنگی ئاسایی خۆت بخوێنەوە. تەنها ناسنامەی دەنگەکەت هەڵدەگیرێت، نەک دەنگەکە خۆی.",
    "dlg.preparing": "ئامادە دەکرێت…",
    "dlg.ready": "کە ئامادە بوویت «دەست پێبکە» دابگرە. پێش ئەوە هیچ تۆمار ناکرێت.",
    "dlg.start": "دەست پێبکە",
    "dlg.read": "ئێستا بیخوێنەوە:",
    "dlg.step": "ڕستەی {n} لە {total}",
    "dlg.listening": "گوێ دەگرم…",
    "dlg.good": "باشە ✓",
    "dlg.saving": "دەنگەکەت پاشەکەوت دەکرێت…",
    "dlg.retry": "دووبارە",
    "dlg.cancel": "پاشگەزبوونەوە",
    "dlg.close": "داخستن",
    "dlg.failed": "سەرکەوتوو نەبوو.",
}
MARGINS = {"low": 10.0, "normal": 14.0, "high": 18.0}
_EASTERN = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
SENS = ("low", "normal", "high")


def t(key: str, **fmt: Any) -> str:
    text = TEXT.get(key, key)
    return text.format(**fmt) if fmt else text


def _named(widget: QWidget, key: str, name: str = "") -> QWidget:
    widget.setAccessibleName(t(key) if not name else name)
    widget.setObjectName(f"voice_{key.replace('.', '_')}")
    return widget


class EnrollDialog(QDialog):
    """One sentence at a time; auto-records each and moves on."""

    def __init__(self, app: Any, bridge: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app, self.bridge = app, bridge
        self.setWindowTitle(t("dlg.title"))
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(520)
        self.sentences: list[str] = []
        self.index = 0
        self.redo = False              # re-reading one sentence the check did not accept
        self.done_ok = False
        self._closing = False
        self._needs_begin = False      # a failed save ended the engine's enrollment: start again
        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        intro = QLabel(t("dlg.intro"))
        intro.setWordWrap(True)
        self.step = QLabel("")
        self.step.setObjectName("Muted")
        self.sentence = QLabel("")
        self.sentence.setWordWrap(True)
        self.sentence.setStyleSheet("font-size: 22px; font-weight: 600; padding: 10px 0;")
        self.sentence.setAccessibleName(t("dlg.read"))
        self.status = QLabel(t("dlg.ready"))
        self.status.setWordWrap(True)
        self.status.setAccessibleName("status")
        buttons = QHBoxLayout()
        self.start_button = _named(QPushButton(t("dlg.start")), "dlg.start")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.begin)
        buttons.addWidget(self.start_button)
        self.retry_button = _named(QPushButton(t("dlg.retry")), "dlg.retry")
        self.retry_button.clicked.connect(self._retry)
        self.retry_button.setVisible(False)
        self.cancel_button = _named(QPushButton(t("dlg.cancel")), "dlg.cancel")
        self.cancel_button.clicked.connect(self.cancel)
        buttons.addWidget(self.retry_button)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        for widget in (intro, self.step, self.sentence, self.status):
            lay.addWidget(widget)
        lay.addLayout(buttons)
        self._unsub = bridge.subscribe(VoiceNotice, self._on_notice) if hasattr(bridge, "subscribe") else None

    # -- flow ---------------------------------------------------------------------------------------
    def begin(self) -> None:
        self.start_button.setVisible(False)
        self.status.setText(t("dlg.preparing"))
        voice = getattr(self.app, "voice", None)
        if voice is None or not hasattr(voice, "enroll_begin"):
            self._fail({"message_ckb": t("state.unavailable")})
            return
        self.bridge.call(voice.enroll_begin(), on_ok=self._begun, on_err=lambda e: self._fail({}))

    def _begun(self, result: Any) -> None:
        if not isinstance(result, dict) or not result.get("ok"):
            self._fail(result if isinstance(result, dict) else {})
            return
        self.sentences = list(result.get("sentences") or [])
        self.index = 0
        self._record()

    def _retry(self) -> None:
        if self._needs_begin:
            self._needs_begin = False
            self.cancel_button.setText(t("dlg.cancel"))
            self.cancel_button.setAccessibleName(t("dlg.cancel"))
            self.begin()
        else:
            self._record()

    def _record(self) -> None:
        if self._closing or self.index >= len(self.sentences):
            return
        self.retry_button.setVisible(False)
        self.step.setText(t("dlg.step", n=str(self.index + 1).translate(_EASTERN),
                            total=str(len(self.sentences)).translate(_EASTERN)))
        self.sentence.setText(self.sentences[self.index])
        self.status.setText(t("dlg.read") + " " + t("dlg.listening"))
        self.bridge.call(self.app.voice.enroll_record(self.index), on_ok=self._recorded,
                         on_err=lambda e: self._retry_with({}))

    def _recorded(self, result: Any) -> None:
        if self._closing:
            return
        if not isinstance(result, dict) or not result.get("ok"):
            self._retry_with(result if isinstance(result, dict) else {})
            return
        self.status.setText(t("dlg.good"))
        self.index += 1
        if self.redo:
            self.redo = False
            self.index = len(self.sentences)
        if self.index < len(self.sentences):
            QTimer.singleShot(600, self._record)
        else:
            self.status.setText(t("dlg.saving"))
            self.bridge.call(self.app.voice.enroll_finish(), on_ok=self._finished, on_err=lambda e: self._fail({}))

    def _retry_with(self, result: dict[str, Any]) -> None:
        self.status.setText(str(result.get("message_ckb") or t("dlg.failed")))
        self.retry_button.setVisible(True)

    def _finished(self, result: Any) -> None:
        if isinstance(result, dict) and isinstance(result.get("retry_index"), int) \
                and 0 <= result["retry_index"] < len(self.sentences):
            # Only this sentence again (another voice was mixed in, or it was too quiet).
            self.index = int(result["retry_index"])
            self.redo = True
            self._retry_with(result)
            return
        ok = isinstance(result, dict) and bool(result.get("ok"))
        self.done_ok = ok
        self.status.setText(str((result or {}).get("message_ckb") or t("dlg.failed")))
        self.sentence.setText("")
        self.step.setText("")
        self.cancel_button.setText(t("dlg.close"))
        self.cancel_button.setAccessibleName(t("dlg.close"))
        if not ok:
            self.index = 0
            self._needs_begin = True
            self.retry_button.setVisible(bool(self.sentences))

    def _fail(self, result: dict[str, Any]) -> None:
        self.status.setText(str(result.get("message_ckb") or t("dlg.failed")))
        self.cancel_button.setText(t("dlg.close"))

    def _on_notice(self, event: Any) -> None:
        if getattr(event, "kind", "") == "enroll" and "download" in str(getattr(event, "detail", "")):
            self.status.setText(event.text_ckb)

    def cancel(self) -> None:
        self._closing = True
        voice = getattr(self.app, "voice", None)
        if voice is not None and hasattr(voice, "enroll_cancel") and not self.done_ok:
            self.bridge.call(voice.enroll_cancel())
        self.close()

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if not self._closing:
            self._closing = True
            voice = getattr(self.app, "voice", None)
            if voice is not None and hasattr(voice, "enroll_cancel") and not self.done_ok:
                self.bridge.call(voice.enroll_cancel())
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        super().closeEvent(event)


class VoiceProfileCard(Card):
    """The Settings card (added by pages/settings.py)."""

    def __init__(self, app: Any, bridge: Any, parent: QWidget | None = None) -> None:
        cfg = getattr(app, "config", None)
        followup = int(cfg.get("voice.followup_s", 6) or 6) if cfg is not None else 6
        super().__init__(t("card"), t("card_sub", followup=str(followup).translate(_EASTERN)), parent)
        self.app, self.bridge = app, bridge
        self.dialog: EnrollDialog | None = None
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)
        grid.setColumnStretch(1, 1)
        self.followup = _named(QSpinBox(), "followup")
        self.followup.setRange(2, 30)
        self.followup.setSuffix(f"  {t('seconds')}")
        self.followup.setValue(followup)
        self.followup.setMaximumWidth(160)
        self.followup.valueChanged.connect(lambda v: self._set("voice.followup_s", int(v)))
        self.strict = _named(QComboBox(), "strict")
        for key in SENS:
            self.strict.addItem(t(f"strict.{key}"), key)
        margin = float(self._cfg("voice.gate_margin_db", 14.0) or 14.0)
        self.strict.setCurrentIndex(min(range(3), key=lambda i: abs(MARGINS[SENS[i]] - margin)))
        self.strict.currentIndexChanged.connect(
            lambda i: self._set("voice.gate_margin_db", MARGINS[self.strict.itemData(i)]))
        self.strict.setMaximumWidth(220)
        self.only = _named(ToggleSwitch(), "only")
        self.only.setChecked(bool(self._cfg("voice.only_my_voice", True)))
        self.only.toggled.connect(lambda on: self._set("voice.only_my_voice", bool(on)))
        self.sens = _named(QComboBox(), "sens")
        for key in SENS:
            self.sens.addItem(t(f"sens.{key}"), key)
        current = str(self._cfg("voice.only_my_voice_sensitivity", "normal") or "normal")
        self.sens.setCurrentIndex(SENS.index(current) if current in SENS else 1)
        self.sens.currentIndexChanged.connect(
            lambda i: self._set("voice.only_my_voice_sensitivity", self.sens.itemData(i)))
        self.sens.setMaximumWidth(260)
        for r, (key, widget) in enumerate((("followup", self.followup), ("strict", self.strict),
                                           ("only", self.only), ("sens", self.sens))):
            label = QLabel(t(key))
            label.setObjectName("Muted")
            grid.addWidget(label, r, 0)
            grid.addWidget(widget, r, 1, Qt.AlignmentFlag.AlignLeading)
        self.body.addLayout(grid)
        buttons = QHBoxLayout()
        self.enroll_button = _named(QPushButton(t("enroll")), "enroll")
        self.enroll_button.setObjectName("Primary")
        self.enroll_button.clicked.connect(self.open_enrollment)
        self.delete_button = _named(QPushButton(t("delete")), "delete")
        self.delete_button.setObjectName("Danger")
        self.delete_button.clicked.connect(self.delete_voiceprint)
        buttons.addWidget(self.enroll_button)
        buttons.addWidget(self.delete_button)
        buttons.addStretch(1)
        self.body.addLayout(buttons)
        level_row = QHBoxLayout()
        level_label = QLabel(t("level"))
        level_label.setObjectName("Muted")
        self.level = QLabel("")
        self.level.setAccessibleName(t("level"))
        self.level_reset = _named(QPushButton(t("level.reset")), "level.reset")
        self.level_reset.clicked.connect(self.reset_level)
        level_row.addWidget(level_label)
        level_row.addWidget(self.level)
        level_row.addWidget(self.level_reset)
        level_row.addStretch(1)
        self.body.addLayout(level_row)
        self._show_level()
        self.state = QLabel(t("state.none"))
        self.state.setWordWrap(True)
        self.state.setAccessibleName(t("only"))
        privacy = QLabel(t("privacy"))
        privacy.setObjectName("Faint")
        privacy.setWordWrap(True)
        self.body.addWidget(self.state)
        self.body.addWidget(privacy)
        self._unsub = bridge.subscribe((VoiceEnrollRequest, VoiceNotice), self._on_event) \
            if hasattr(bridge, "subscribe") else None
        self.refresh()

    def _cfg(self, key: str, default: Any = None) -> Any:
        try:
            return self.app.config.get(key, default)
        except Exception:  # noqa: BLE001
            return default

    def _set(self, key: str, value: Any) -> None:
        try:
            self.app.config.set(key, value)   # thread-safe (contract 3.6)
        except Exception:  # noqa: BLE001
            pass
        if key == "voice.only_my_voice":
            self.refresh()

    def _show_level(self) -> None:
        level = self._cfg("voice.gate_user_level_db", None)
        known = isinstance(level, (int, float))
        # LTR mark: "-26 dB" must not become "dB 26-" inside the RTL layout.
        self.level.setText("\u200e" + t("level.value", db=f"{float(level):.0f}") if known else t("level.none"))
        self.level_reset.setEnabled(known)

    def reset_level(self) -> None:
        voice = getattr(self.app, "voice", None)
        if voice is not None and hasattr(voice, "reset_user_level"):
            self.bridge.on_core(voice.reset_user_level, on_ok=lambda _r: self._show_level(),
                                on_err=lambda _e: None)
        else:
            self._set("voice.gate_user_level_db", None)
            self._show_level()

    # -- status -----------------------------------------------------------------------------------
    def refresh(self) -> None:
        voice = getattr(self.app, "voice", None)
        if voice is not None and hasattr(voice, "voiceprint_status"):
            self.bridge.on_core(voice.voiceprint_status, on_ok=self.show_status, on_err=lambda _e: None)
        else:
            self.enroll_button.setEnabled(False)

    def show_status(self, status: Any) -> None:
        status = status if isinstance(status, dict) else {}
        enrolled = bool(status.get("enrolled"))
        self.delete_button.setEnabled(enrolled)
        self._show_level()
        if status.get("sherpa") is False:
            self.state.setText(t("state.unavailable") + "sherpa-onnx")
        elif enrolled and status.get("enabled") and (status.get("usable") is False or status.get("unavailable")):
            self.state.setText(t("state.not_ready"))
        elif not enrolled:
            self.state.setText(t("state.none"))
        else:
            self.state.setText(t("state.on") if status.get("enabled") else t("state.off"))

    # -- actions ------------------------------------------------------------------------------------
    def open_enrollment(self) -> EnrollDialog:
        if self.dialog is not None and self.dialog.isVisible():
            self.dialog.raise_()
            return self.dialog
        self.dialog = EnrollDialog(self.app, self.bridge, self.window())
        self.dialog.finished.connect(lambda _r: self.refresh())
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()   # the Start step: nothing is recorded before «دەست پێبکە»
        return self.dialog

    def delete_voiceprint(self) -> None:
        voice = getattr(self.app, "voice", None)
        if voice is not None and hasattr(voice, "voiceprint_delete"):
            self.bridge.call(voice.voiceprint_delete(), on_ok=lambda _r: self.refresh(), on_err=lambda _e: None)

    def _on_event(self, event: Any) -> None:
        if isinstance(event, VoiceEnrollRequest):
            self.open_enrollment()
        elif getattr(event, "kind", "") == "enroll" and getattr(event, "detail", "") in ("saved", "deleted"):
            self.refresh()
        elif getattr(event, "kind", "") == "voiceprint":
            self.refresh()


__all__ = ["VoiceProfileCard", "EnrollDialog", "TEXT"]
