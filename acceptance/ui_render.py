"""Render check for the UI (offscreen): island states + panel pages -> PNGs.

Usage:  .venv\\Scripts\\python.exe acceptance\\ui_render.py [out_dir]

Uses a temporary SAM_HOME with fake data (no keys, no network, no devices).
The island is rendered at 2x (the user's laptop runs 2880x1800 at 200%) and
composited onto a dark and a bright wallpaper-like background so contrast can
be judged on both. Nothing is shown on the real desktop.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")   # offscreen has no system fonts otherwise
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = Path(os.environ.get("SAM_UI_SHOTS", ROOT / "work" / "ui-shots"))


class FakeStore:
    """StrategyStore stand-in with two cards (list/get/set_status/versions)."""

    def __init__(self) -> None:
        now = time.time()
        self.cards = [
            {"id": "asia-sweep-fvg", "title_ckb": "ڕاماڵینی ئاسیا و FVG", "title_en": "Asia sweep + FVG",
             "status": "active", "version": 3, "updated_at": now - 3600,
             "summary_ckb": "لە کاتی لەندەن، دوای ڕاماڵینی بەرزی یان نزمیی ئاسیا و شکاندنی پێکهاتە لە M15، "
                            "لە ناو FVG دەچینە ژوورەوە بە مەترسیی ١٪.",
             "card": {"markets": ["XAUUSD"], "timeframes": {"bias": "H4", "setup": "M15", "entry": "M5"},
                      "sessions": ["London", "New York"], "risk": {"max_risk_pct": 1, "max_losses_per_day": 2},
                      "rules": [
                          {"kind": "bias", "text_ckb": "ئاراستەی H4 دەبێت سەرەوە بێت.",
                           "check": {"predicate": "trend_is", "params": {"tf": "H4", "dir": "up"}}},
                          {"kind": "setup", "text_ckb": "نزمیی ئاسیا ڕاماڵرابێت.",
                           "check": {"predicate": "swept", "params": {"level": "asia_low"}}},
                          {"kind": "entry", "text_ckb": "مۆمێکی بەهێزی سەوز لە ناو FVG دابخرێت.", "check": None}],
                      "source_text": "London session: after Asia low is swept and M15 shifts structure, "
                                     "enter inside the FVG, 1% risk."}},
            {"id": "ob-retest", "title_ckb": "گەڕانەوە بۆ ئۆردەر بلۆک", "title_en": "Order block retest",
             "status": "draft", "version": 1, "updated_at": now - 86400, "summary_ckb": "",
             "card": {"rules": []}},
            {"id": "v1-7", "title_ckb": "تیۆریی کۆن (SAM v1)", "title_en": "imported", "status": "archived",
             "version": 1, "updated_at": now - 9 * 86400, "summary_ckb": "", "card": {}},
        ]

    def list(self, status=None):
        # Same shape as sam.trading.strategies.StrategyStore.list(): summary rows,
        # "rules" is a count and the card JSON is not included.
        return [{"id": c["id"], "title_ckb": c["title_ckb"], "title_en": c["title_en"], "status": c["status"],
                 "version": c["version"], "summary_ckb": c["summary_ckb"],
                 "rules": len(c["card"].get("rules") or []), "markets": c["card"].get("markets") or [],
                 "updated_at": c["updated_at"], "note": ""}
                for c in self.cards if status in (None, "all", c["status"])]

    def get(self, strategy_id):
        card = next((c for c in self.cards if c["id"] == strategy_id), None)
        if card is None:
            return None
        return {**card["card"], **{k: v for k, v in card.items() if k != "card"}}

    def set_status(self, strategy_id, status):
        for card in self.cards:
            if card["id"] == strategy_id:
                card["status"] = status
        return self.get(strategy_id) or {}

    def versions(self, strategy_id):
        now = time.time()
        return [{"version": 3, "created_at": now - 3600, "reason": "rule added"},
                {"version": 2, "created_at": now - 7200, "reason": ""},
                {"version": 1, "created_at": now - 86400, "reason": "created"}]


class _Router:
    def __init__(self, ok: bool) -> None:
        self.ok = ok

    def configured(self) -> bool:
        return self.ok


class FakeVoice:
    """VoiceEngine stand-in with the attributes the UI reads (no devices)."""

    engine_name = "cascade"
    state = "idle"
    live_session_open = False

    def __init__(self) -> None:
        self.stt, self.tts = _Router(True), _Router(True)

    def status(self) -> dict:
        return {"engine": "cascade", "state": "idle", "live_degraded": False}

    async def toggle_listening(self) -> bool:
        return True

    async def set_muted(self, muted: bool) -> None:
        return None

    async def run_selftest(self) -> dict:
        return {"ok": True, "cer": 0.06, "script_ok": True, "ttfa_ms": 1180.0}


class FakeTv:
    def status(self) -> dict:
        return {"connected": True, "port": 9222}

    async def ensure_running(self, **_kw) -> dict:
        return {"ok": True, "state": "connected"}


class FakeMt5:
    connected = True
    offset_verified = True

    async def status(self) -> dict:
        return {"connected": True, "broker_offset_s": 10800, "server_time_ok": True}


def seed(app) -> None:
    now = time.time()
    db = app.db
    db.insert("alerts", {"kind": "price_cross", "symbol": "XAUUSD", "timeframe": "M15",
                         "params": {"level": 2687.5, "direction": "up"}, "note": "ئاگادارم بکەرەوە",
                         "status": "active", "created_at": now - 600})
    db.insert("alerts", {"kind": "zone_touch", "symbol": "XAUUSD", "timeframe": "H1",
                         "params": {"low": 2651.2, "high": 2656.8}, "status": "active", "repeat": 1,
                         "created_at": now - 1800})
    db.insert("alerts", {"kind": "volume_spike", "symbol": "XAUUSD", "timeframe": "M5",
                         "params": {"k": 2.5, "n": 20}, "status": "fired", "created_at": now - 7200,
                         "fired_at": now - 5400, "fire_count": 1,
                         "last_text_ckb": "قەبارەی زێڕ لە M5 دوو و نیو هێندەی تێکڕا بەرز بووەوە."})
    for i, (name, ok, ms, summary) in enumerate([
            ("open_app", True, 820, "Opened TradingView"), ("tv_set_chart", True, 310, "XAUUSD M15"),
            ("draw_on_chart", True, 180, "Drew 4 levels"), ("analyze_market", True, 2400, "SETUP long"),
            ("run_powershell", False, 45, "The user did not approve the action.")]):
        db.log_activity("tool", name, ok=ok, summary=summary, duration_ms=ms, source="live")
    for stage, ms in (("end_of_speech", 610), ("stt", 420), ("llm_first_token", 780), ("tts_first_audio", 520),
                      ("first_audio", 1380), ("total", 3900)):
        app.timing.record(stage, ms, kind="cascade", turn_id="demo-turn")
    for stage, ms in (("tool:open_app", 820), ("tool:draw_on_chart", 180), ("analysis_engine", 950),
                      ("live_connect", 640), ("tv_cdp", 45), ("mt5_fetch", 120)):
        app.timing.record(stage, ms, kind="tool")


def compose(pix, path: Path, bg: str) -> None:
    from PySide6.QtCore import QPointF, QRectF
    from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QRadialGradient

    dpr = pix.devicePixelRatio()
    w, h = int(pix.width() / dpr) + 120, int(pix.height() / dpr) + 60
    img = QImage(int(w * dpr), int(h * dpr), QImage.Format.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(dpr)
    p = QPainter(img)
    grad = QLinearGradient(0, 0, 0, h)
    if bg == "dark":
        grad.setColorAt(0, QColor("#1b2233"))
        grad.setColorAt(1, QColor("#3a2a2a"))
    else:
        grad.setColorAt(0, QColor("#dfe6ee"))
        grad.setColorAt(1, QColor("#f4efe6"))
    p.fillRect(QRectF(0, 0, w, h), grad)
    sun = QRadialGradient(QPointF(w * 0.72, h * 1.1), h * 0.9)
    sun.setColorAt(0, QColor(255, 90, 20, 200 if bg == "dark" else 110))
    sun.setColorAt(1, QColor(255, 90, 20, 0))
    p.fillRect(QRectF(0, 0, w, h), sun)
    p.drawPixmap(QPointF(60, 20), pix)
    p.end()
    img.save(str(path))


def render_widget(widget, dpr: float = 2.0):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap

    pix = QPixmap(int(widget.width() * dpr), int(widget.height() * dpr))
    pix.setDevicePixelRatio(dpr)
    pix.fill(Qt.GlobalColor.transparent)
    widget.render(pix)
    return pix


def main(out_dir: Path) -> list[Path]:
    from PySide6.QtWidgets import QApplication

    from sam.app import App
    from sam.bridge import CoreThread
    from sam.events import (Caption, ComponentStatus, ConfirmRequest, ToolFinished, ToolStarted, Transcript,
                            VoiceState, WorkerProgress)
    import sam.ui as ui

    out_dir.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="sam-ui-render-"))
    (home / "data").mkdir()
    app = App(home, environ={}, llm_backends={})
    app.trading.strategies = FakeStore()
    app.trading.tv = FakeTv()
    app.trading.mt5 = FakeMt5()
    app.voice = FakeVoice()
    app.config.set("voice.selftest", {"ok": True, "cer": 0.06, "script_ok": True, "ttfa_ms": 1180.0})
    seed(app)
    core = CoreThread()
    core.start()
    app.bus.bind_loop(core.loop)
    qapp = QApplication.instance() or QApplication(sys.argv[:1])
    controller = ui.build(app, core, show=False)
    island = controller.island
    written: list[Path] = []

    def settle(ms: int = 400) -> None:
        end = time.monotonic() + ms / 1000.0
        while time.monotonic() < end:
            qapp.processEvents()
            time.sleep(0.01)

    def shot(name: str) -> None:
        settle(420)
        if island.state in ("listening", "speaking"):
            # Freeze a mid-sentence voice level (the meter decays when no LevelMeter arrives).
            island._level = 0.72
            island._phase = 1.3
        pix = render_widget(island)
        for bg in ("dark", "light"):
            path = out_dir / f"island_{name}_{bg}.png"
            compose(pix, path, bg)
            written.append(path)

    island.show()
    scenes = [
        ("idle", lambda: island.set_state("idle")),
        ("listening", lambda: (island.set_state("listening", "cascade"), island.set_level("mic", 0.7),
                               island.handle_event(Caption(text="گۆڵد لەسەر پازدە خولەک پیشان", role="user")))),
        ("thinking", lambda: (island.set_state("thinking", "live"),
                              island.handle_event(Transcript(role="user", text="هێڵی پشتگیری و بەرگری بکێشە",
                                                             source="live")))),
        ("speaking", lambda: (island.set_state("speaking", "live"), island.set_level("speaker", 0.8),
                              island.handle_event(Caption(
                                  text="باشە، چوار هێڵم لەسەر چارتی زێڕ کێشا: دوو پشتگیری لە ٢٦٥١ و ٢٦٤٠، "
                                       "و دوو بەرگری لە ٢٦٨٧ و ٢٦٩٥. ئەگەر بتەوێت ئاگادارکەرەوەش دادەنێم.",
                                  role="assistant", final=True)))),
        ("working", lambda: (island.set_state("working"),
                             island.handle_event(WorkerProgress(task_id="t1", step=3, max_steps=7,
                                                                text_ckb="فۆڵدەری پڕۆژەکە دروست دەکەم…")))),
        ("tool", lambda: (island.set_state("thinking"),
                          island.handle_event(ToolStarted(call_id="c1", name="draw_on_chart", args={}))),),
        ("error", lambda: (island.set_state("error"),
                           island.handle_event(ToolFinished(call_id="c2", name="open_app", ok=False,
                                                            summary="TradingView is not installed.")))),
        ("muted", lambda: (island.clear_caption(), island.set_state("muted"))),
        ("confirm", lambda: (island.set_state("speaking", "cascade"),
                             island.handle_event(ConfirmRequest(
                                 confirm_id="k1", question_ckb="ئەم فەرمانە جێبەجێ بکەم؟ فایلەکانی ناو "
                                 "فۆڵدەری Downloads دەسڕێتەوە.", detail="Remove-Item ~/Downloads/*.tmp",
                                 tool_name="run_powershell", expires_at=time.time() + 18)))),
    ]
    for name, action in scenes:
        island.clear_progress()         # each shot shows only its own scene
        action()
        shot(name)
    island.card.clear()
    island.hide()

    panel = controller.ensure_panel()
    panel.resize(1180, 780)
    panel.show()
    settle(600)
    path = out_dir / "panel_chat_empty.png"       # what the user sees on first open
    panel.grab().save(str(path))
    written.append(path)
    events = [
        ComponentStatus(component="voice", state="ok"), ComponentStatus(component="omniroute", state="ok"),
        ComponentStatus(component="tradingview", state="degraded", detail="no CDP port"),
        ComponentStatus(component="mt5", state="ok"),
        Transcript(role="user", text="ترەیدینگ ڤیو بکەرەوە", source="live"),
        ToolStarted(call_id="a", name="open_app", args={"name": "TradingView"}, source="live"),
        ToolFinished(call_id="a", name="open_app", ok=True, summary="Opened TradingView", duration_ms=820,
                     source="live"),
        Transcript(role="assistant", text="ترەیدینگ ڤیو کرایەوە و بە چارتەکەوە پەیوەست بووم.", source="live"),
        Transcript(role="user", text="زێڕ شی بکەرەوە بە ستراتیژییەکەم", source="text"),
        Transcript(role="assistant", text="نرخی زێڕ ئێستا ٢٦٧٤ـە. ئاراستەی H4 سەرەوەیە و نزمیی ئاسیا ڕاماڵراوە، "
                   "بەڵام هێشتا شکاندنی پێکهاتە لە M15 نەبووە — بۆیە چاوەڕێ دەکەین. ناوچەی FVG لە نێوان "
                   "٢٦٥١ و ٢٦٥٦ م لەسەر چارت کێشا.", source="text"),
        Caption(text="ئەگەر نرخ گەیشتە ناوچەکە", role="user", final=False),
        VoiceState(state="listening", engine="cascade"),
    ]
    for ev in events:
        controller.bridge.deliver(ev)
    for key in ("chat", "strategies", "monitor", "activity", "settings"):
        panel.show_page(key)
        if key == "strategies":
            settle(500)
            page = panel.pages["strategies"]
            if page.list.count():
                page.list.setCurrentRow(0)
        settle(900)
        pix = panel.grab()
        path = out_dir / f"panel_{key}.png"
        pix.save(str(path))
        written.append(path)
        if key == "settings":
            from PySide6.QtWidgets import QScrollArea
            area = panel.pages["settings"].findChild(QScrollArea)
            bar = area.verticalScrollBar()
            for suffix, value in (("2", bar.maximum() // 2), ("3", bar.maximum())):
                bar.setValue(value)
                settle(300)
                path = out_dir / f"panel_settings_{suffix}.png"
                panel.grab().save(str(path))
                written.append(path)
    # tray icon + orb sheet
    from sam.ui.orb import orb_pixmap
    from PySide6.QtGui import QPainter, QPixmap
    from PySide6.QtCore import Qt
    sheet = QPixmap(8 * 90, 100)
    sheet.fill(Qt.GlobalColor.transparent)
    p = QPainter(sheet)
    for i, state in enumerate(("idle", "listening", "thinking", "speaking", "working", "error", "muted", "sleeping")):
        p.drawPixmap(i * 90 + 5, 10, orb_pixmap(80, state=state, glow=True))
    p.end()
    path = out_dir / "orbs.png"
    compose(sheet, path, "dark")
    written.append(path)
    tray_pix = controller.tray.icon.icon().pixmap(32, 32)
    path = out_dir / "tray_32.png"
    tray_pix.save(str(path))
    written.append(path)
    controller.shutdown()
    core.stop()
    app.close()
    shutil.rmtree(home, ignore_errors=True)      # the temporary SAM_HOME
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    for p in main(target):
        print(p)
