"""Integration smoke on this PC: the whole of SAM 2 started for real.

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe acceptance\\integration_smoke.py [--home C:\\Users\\samit\\Desktop\\SAM-Agent]

Phase A -- the real program, exactly as the sign-in shortcut starts it
(``pythonw SAM.pyw --background --home <home>``): time to a visible island,
``startup:*`` timings, idle RAM after 15 s, no console window, then a clean
``SAM.pyw --quit`` and no SAM 2 process left.

Phase B -- the same start-up path in THIS process (``sam.__main__.main``: UI
first, packages starting on the core thread, island on screen) driven through
the TEXT path the chat page uses (``app.submit_text`` -> the one
Conversation), with the real free models of the configured providers:

  a) "سڵاو"                           short Sorani reply, no self-introduction, no tool
  b) "ترەیدینگ ڤیو بکەرەوە"            open_app/tv_open ok, chart connected
  c) "گۆڵد لەسەر ١٥ خولەک پیشان بدە"    tv_set_chart ok, chart shows gold on M15
  d) "شیکاری گۆڵد بکە و ئاستەکان بکێشە"  analysis + SAM drawings on the chart
     "هێڵەکانت بسڕەوە"                 only SAM's drawings removed (user drawings unchanged)
  e) "نۆتپاد بکەرەوە"                  Notepad opens (closed again if it is a new window)

then the voice CASCADE inside the same running app on a recorded Sorani clip
(no Gemini key: Live waits for the user's key) through a file-fed microphone.

Safety: the speaker is a null output stream (nothing is ever played), the
real microphone is never opened (a silent file source replaces it), every
confirmation request is declined at once, the chart's symbol/timeframe are
restored and every SAM drawing made here is removed, the smoke conversation
and its turns/analyses are deleted afterwards, MT5 is read only. Keys are
read only by SAM 2's own Secrets code; nothing prints them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from _common import ROOT, Acceptance

import launcher_startup as proc  # process_tree, visible_windows, memory_mb, terminate

DEFAULT_HOME = r"C:\Users\samit\Desktop\SAM-Agent"
WORK = ROOT / "work" / "integ"
ISLAND_TARGET_MS, RAM_TARGET_MB, IDLE_S = 3000, 400, 15
PACE_S = 20.0   # --pace: seconds between typed commands
KEY_SHAPES = re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{30,}|AQ\.[A-Za-z0-9_\-]{20,}|"
                        r"sk-[A-Za-z0-9]{32,}")
COMMANDS = {
    "a": "سڵاو",
    "b": "ترەیدینگ ڤیو بکەرەوە",
    "c": "گۆڵد لەسەر ١٥ خولەک پیشان بدە",
    "d": "شیکاری گۆڵد بکە و ئاستەکان بکێشە",
    "d2": "هێڵەکانت بسڕەوە",
    "e": "نۆتپاد بکەرەوە",
}


# -- phase A: the real program -------------------------------------------------------------------

def phase_real_process(acc: Acceptance, home: Path, pythonw: Path) -> None:
    logdir = WORK / "logs-A"
    env = {**os.environ, "SAM_LOG_DIR": str(logdir), "PYTHONIOENCODING": "utf-8"}
    launcher = [str(pythonw), str(ROOT / "SAM.pyw"), "--home", str(home)]
    since = time.time()
    began = time.perf_counter()
    child = subprocess.Popen(launcher + ["--background"], env=env, cwd=str(ROOT))
    try:
        with acc.check(f"A: island visible < {ISLAND_TARGET_MS} ms (real SAM.pyw --background)") as c:
            windows: list[dict] = []
            while time.perf_counter() - began < 60 and child.poll() is None:
                windows = proc.visible_windows(set(proc.process_tree(child.pid)))
                if windows:
                    break
                time.sleep(0.03)
            c.data["ms"] = round((time.perf_counter() - began) * 1000)
            c.data["windows"] = windows
            assert windows, f"no visible SAM window (exit code {child.poll()})"
            c.detail = f"{c.data['ms']} ms"
            assert c.data["ms"] < ISLAND_TARGET_MS, c.detail

        with acc.check("A: core ready + start-up breakdown") as c:
            rows = wait_rows(home, "SELECT stage, ms FROM timings WHERE stage LIKE 'startup:%' AND at >= ? ORDER BY id",
                             (since,), until=lambda r: any(s == "startup:core_ready" for s, _ in r), timeout_s=60)
            c.data = {stage: round(ms) for stage, ms in rows}
            assert "startup:core_ready" in c.data, "no startup:core_ready timing"
            slow = sorted(((ms, s) for s, ms in rows if s.startswith("startup:start:")), reverse=True)[:3]
            c.detail = (f"island_visible {c.data.get('startup:island_visible')} ms, core_ready "
                        f"{c.data['startup:core_ready']} ms; slowest start: "
                        + ", ".join(f"{s.split(':')[-1]} {ms:.0f} ms" for ms, s in slow))

        with acc.check("A: no console window") as c:
            tree = proc.process_tree(child.pid)
            consoles = [w for w in proc.visible_windows(set(tree))
                        if w["class"] in ("ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS")]
            c.data = {"processes": sorted(set(tree.values())), "consoles": consoles}
            assert not consoles, consoles

        with acc.check(f"A: RAM < {RAM_TARGET_MB} MB after {IDLE_S} s idle") as c:
            time.sleep(IDLE_S)
            tree = proc.process_tree(child.pid)
            usage = {pid: proc.memory_mb(pid) for pid, exe in tree.items() if exe.lower().startswith("python")}
            main_pid = max(usage, key=lambda p: usage[p].get("working_set_mb", 0)) if usage else child.pid
            c.data = {"per_process": {str(k): v for k, v in usage.items()}, "sam_pid": main_pid}
            working = usage.get(main_pid, {}).get("working_set_mb", 0.0)
            c.detail = f"working set {working} MB, private {usage.get(main_pid, {}).get('private_mb')} MB"
            assert 0 < working < RAM_TARGET_MB, c.detail

        with acc.check("A: packages loaded, none failed (sam2.log)") as c:
            text = read_log(logdir)
            c.data["errors"] = [line[-240:] for line in text.splitlines() if " ERROR " in line][:8]
            assert "package" not in " ".join(c.data["errors"]), c.data["errors"]

        with acc.check("A: clean quit (SAM.pyw --quit), no SAM 2 process left") as c:
            t0 = time.perf_counter()
            quit_run = subprocess.run(launcher[:2] + ["--quit"], env=env, cwd=str(ROOT), timeout=60)
            try:
                child.wait(30)
            except subprocess.TimeoutExpired:
                pass
            c.data = {"quit_exit": quit_run.returncode, "ms": round((time.perf_counter() - t0) * 1000),
                      "left": proc.process_tree(child.pid) if child.poll() is None else {}}
            c.detail = f"stopped in {c.data['ms']} ms"
            assert quit_run.returncode == 0 and child.poll() is not None, c.data
            assert "SAM 2 stopped" in read_log(logdir), "no clean-stop line in sam2.log"
    finally:
        if child.poll() is None:  # only ever SAM's own python processes, never OmniRoute
            tree = proc.process_tree(child.pid)
            proc.terminate([pid for pid, exe in tree.items() if exe.lower().startswith("python")])


def wait_rows(home: Path, sql: str, params: tuple, *, until: Any, timeout_s: float) -> list[tuple]:
    import sqlite3

    db = home / "data" / "sam2.sqlite3"
    deadline = time.monotonic() + timeout_s
    rows: list[tuple] = []
    while time.monotonic() < deadline:
        try:
            conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=2)
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            rows = []
        if until(rows):
            return rows
        time.sleep(0.25)
    return rows


def read_log(logdir: Path) -> str:
    path = logdir / "sam2.log"
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


# -- phase B helpers: no sound, no room audio ----------------------------------------------------

class NullOutputStream:
    """sounddevice.RawOutputStream stand-in: pulls audio in real time, plays nothing."""

    def __init__(self, **kwargs: Any) -> None:
        self.callback = kwargs["callback"]
        self.block = int(kwargs["blocksize"])
        self.rate = int(kwargs["samplerate"])
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="null-speaker", daemon=True).start()

    def _run(self) -> None:
        buf = bytearray(self.block * 2)
        period, due = self.block / self.rate, time.perf_counter()
        while not self._stop.is_set():
            self.callback(memoryview(buf), self.block, None, None)
            due += period
            time.sleep(max(0.0, due - time.perf_counter()))

    def stop(self) -> None:
        self._stop.set()

    abort = close = stop


class FileMic:
    """MicStream stand-in: optional clip after 0.3 s of silence, then silence
    until stopped, paced in real time (30 ms frames, 16 kHz int16)."""

    FRAME = 16000 * 2 * 30 // 1000

    def __init__(self, pcm: bytes | None = None) -> None:
        self.pcm = pcm or b""
        self._queue: asyncio.Queue[bytes | None] | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def is_open(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        self._queue = asyncio.Queue()
        self._task = asyncio.ensure_future(self._feed())

    async def _feed(self) -> None:
        loop = asyncio.get_running_loop()
        audio = b"\x00" * (self.FRAME * 10) + self.pcm
        audio += b"\x00" * (-len(audio) % self.FRAME)
        due, offset = loop.time(), 0
        while True:
            chunk = audio[offset:offset + self.FRAME] if offset < len(audio) else b"\x00" * self.FRAME
            offset += self.FRAME
            assert self._queue is not None
            self._queue.put_nowait(chunk)
            due += 0.03
            await asyncio.sleep(max(0.0, due - loop.time()))

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
        if self._queue is not None:
            self._queue.put_nowait(None)

    async def frames(self):  # noqa: ANN201 - async iterator like MicStream.frames
        queue = self._queue
        if queue is None:
            return
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item


def silence_devices() -> None:
    """Before any App exists: null speaker stream, silent mic source."""
    import functools

    import sam.voice.engine as voice_engine

    voice_engine.Speaker = functools.partial(voice_engine.Speaker, stream_factory=NullOutputStream)  # type: ignore[misc]
    voice_engine.VoiceEngine._default_mic = lambda self: FileMic()  # type: ignore[method-assign]


# -- phase B: the text path in the running app -----------------------------------------------------

class Driver:
    def __init__(self, acc: Acceptance, app: Any, core: Any, since: float) -> None:
        self.acc, self.app, self.core, self.since = acc, app, core, since
        self.events: list[Any] = []
        self.report: dict[str, Any] = {}
        self.conversations: set[int] = set()

    def run(self, coro: Any, timeout: float = 60) -> Any:
        return self.core.submit(coro).result(timeout)

    def on_event(self, event: Any) -> None:
        from sam.events import ConfirmRequest

        self.events.append(event)
        if isinstance(event, ConfirmRequest):  # this smoke never approves anything
            self.app.confirm.resolve(event.confirm_id, False, via="click")

    def ask(self, key: str, timeout: float = 150) -> dict[str, Any]:
        from sam.events import ConfirmRequest, ToolFinished

        text = COMMANDS[key]
        mark, t0, began = len(self.events), time.time(), time.perf_counter()
        error = ""
        try:
            reply = self.run(self.app.submit_text(text), timeout)
        except Exception as exc:  # noqa: BLE001 - reported as the check's failure
            reply, error = "", f"{type(exc).__name__}: {exc}"
        total = round((time.perf_counter() - began) * 1000)
        new = self.events[mark:]
        tools = [{"name": e.name, "ok": e.ok, "ms": round(e.duration_ms or 0), "summary": e.summary[:160]}
                 for e in new if isinstance(e, ToolFinished)]
        rows = self.app.db.query("SELECT turn_id, kind, stage, ms, extra FROM timings WHERE at >= ? ORDER BY id",
                                 (t0 - 0.05,))
        stages = [f"{r['stage']}={r['ms']:.0f}" for r in rows if not r["stage"].startswith("startup")]
        models = sorted({json.loads(r["extra"]).get("model", "") for r in rows
                         if r["extra"] and r["stage"].startswith("llm")} - {""})
        if self.app.conversation is not None and self.app.conversation.conversation_id:
            self.conversations.add(int(self.app.conversation.conversation_id))
        result = {"text": text, "reply": reply, "total_ms": total, "tools": tools, "stages_ms": stages,
                  "models": models, "confirms_declined": sum(isinstance(e, ConfirmRequest) for e in new),
                  "error": error}
        self.report[key] = result
        time.sleep(PACE_S)  # a person's pace; Groq's free tier is ~8k tokens a minute (one request ~5k)
        return result

    def chart(self) -> dict[str, Any]:
        return self.run(self.app.trading.tv.chart_state(), 30)

    def notepads(self) -> set[int]:
        windows = self.run(self.app.hands.windows.list(), 30)
        return {w.hwnd for w in windows if w.process.lower() == "notepad.exe"}

    def scenario(self) -> None:
        acc, app = self.acc, self.app
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not app.db.scalar(
                "SELECT COUNT(*) FROM timings WHERE stage='startup:core_ready' AND at >= ?", (self.since,)):
            time.sleep(0.2)
        from sam.events import ConfirmRequest, ToolFinished, ToolStarted

        self.run(self._subscribe((ToolStarted, ToolFinished, ConfirmRequest)))
        with acc.check("B: island on screen, packages started") as c:
            rows = app.db.query("SELECT stage, ms FROM timings WHERE stage LIKE 'startup:%' AND at >= ?",
                                (self.since,))
            c.data = {r["stage"]: round(r["ms"]) for r in rows}
            c.data["island_windows"] = proc.visible_windows({os.getpid()})
            c.data["failed"] = list(app.failed)
            c.detail = f"island_visible {c.data.get('startup:island_visible')} ms, core_ready " \
                       f"{c.data.get('startup:core_ready')} ms"
            assert c.data["island_windows"] and not app.failed, c.data

        chart0 = self.chart() if app.trading.tv is not None and self.run(app.trading.tv.connect(), 30) else None
        owned0 = len(self.run(app.trading.tv.my_drawings(), 30)) if chart0 else 0
        pads0 = self.notepads()
        self.report["chart_before"] = {k: chart0.get(k) for k in ("symbol", "timeframe", "resolution", "user_drawings",
                                                                  "my_drawings")} if chart0 else None
        try:
            self._text_commands(chart0, pads0)
            self._cascade()
        finally:
            self._cleanup(chart0, owned0, pads0)

    async def _subscribe(self, types: tuple) -> None:
        self.app.bus.subscribe(types, self.on_event)

    def _text_commands(self, chart0: dict | None, pads0: set[int]) -> None:
        acc = self.acc
        with acc.check("B(a) 'سڵاو' -> short Sorani reply, no self-introduction, no tool") as c:
            r = self.ask("a")
            c.data = r
            from sam.textnorm import is_arabic_script
            c.detail = f"{r['total_ms']} ms: {r['reply']}"
            assert r["reply"] and is_arabic_script(r["reply"]), r
            assert not r["tools"], r["tools"]
            assert not re.search(r"سام|SAM|یاریدەدەر|ئەسیستەنت", r["reply"]), "self-introduction"
            assert len(r["reply"]) < 220, "not short"

        with acc.check("B(b) 'ترەیدینگ ڤیو بکەرەوە' -> TradingView chart connected") as c:
            r = self.ask("b")
            ok_tools = [t for t in r["tools"] if t["name"] in ("open_app", "tv_open") and t["ok"]]
            state = self.chart()
            c.data = {**r, "chart": {k: state.get(k) for k in ("symbol", "timeframe")},
                      "connected": bool(self.app.trading.tv.connected)}
            c.detail = f"{r['total_ms']} ms, tools {[t['name'] + ':' + str(t['ms']) for t in r['tools']]}: {r['reply']}"
            assert ok_tools and self.app.trading.tv.connected, r

        with acc.check("B(c) 'گۆڵد لەسەر ١٥ خولەک پیشان بدە' -> gold on M15") as c:
            r = self.ask("c")
            state = self.chart()
            c.data = {**r, "chart": {k: state.get(k) for k in ("symbol", "canonical", "timeframe")}}
            c.detail = f"{r['total_ms']} ms, chart {state.get('symbol')} {state.get('timeframe')}: {r['reply']}"
            assert any(t["name"] == "tv_set_chart" and t["ok"] for t in r["tools"]), r["tools"]
            assert state.get("canonical") == "XAUUSD" and state.get("timeframe") == "M15", c.data["chart"]

        with acc.check("B(d) 'شیکاری گۆڵد بکە و ئاستەکان بکێشە' -> analysis + SAM drawings") as c:
            r = self.ask("d", timeout=180)
            state = self.chart()
            mine = self.run(self.app.trading.tv.my_drawings(), 30)
            c.data = {**r, "my_drawings": len(mine), "kinds": sorted({d["kind"] for d in mine}),
                      "user_drawings": state.get("user_drawings")}
            c.detail = f"{r['total_ms']} ms, {len(mine)} SAM drawings {c.data['kinds']}: {r['reply']}"
            assert any(t["name"] in ("analyze_market", "draw_on_chart") and t["ok"] for t in r["tools"]), r["tools"]
            assert mine, "no SAM drawing on the chart"

        with acc.check("B(d2) 'هێڵەکانت بسڕەوە' -> only SAM drawings removed") as c:
            r = self.ask("d2")
            state = self.chart()
            before_user = (chart0 or {}).get("user_drawings")
            c.data = {**r, "my_drawings": state.get("my_drawings"), "user_drawings": state.get("user_drawings"),
                      "user_drawings_before": before_user}
            c.detail = f"{r['total_ms']} ms, SAM left {state.get('my_drawings')}, user {before_user} -> " \
                       f"{state.get('user_drawings')}: {r['reply']}"
            assert any(t["name"] == "clear_my_drawings" and t["ok"] for t in r["tools"]), r["tools"]
            assert state.get("my_drawings") == 0 and state.get("user_drawings") == before_user, c.data

        with acc.check("B(e) 'نۆتپاد بکەرەوە' -> Notepad opens") as c:
            r = self.ask("e")
            time.sleep(1.0)
            new = self.notepads() - pads0
            c.data = {**r, "new_notepad_windows": len(new), "notepad_was_open": bool(pads0)}
            c.detail = f"{r['total_ms']} ms, new windows {len(new)}: {r['reply']}"
            assert any(t["name"] == "open_app" and t["ok"] for t in r["tools"]), r["tools"]
            assert new or pads0, "no Notepad window"

    def _cascade(self) -> None:
        acc, app = self.acc, self.app
        with acc.check("B(voice) cascade on a recorded Sorani clip, inside the running app") as c:
            has_gemini = app.secrets.has("gemini_api_key")
            c.data["gemini_key"] = has_gemini
            c.data["live"] = "self-test needs a Gemini key" if not has_gemini else "key present"
            if not app.secrets.has("kurdishtts_stt_api_key"):
                c.skip("no KurdishTTS STT key")
            sys.path.insert(0, str(ROOT / "acceptance"))
            from voice_live import CLIP_NAME, load_clip
            pcm = load_clip(CLIP_NAME)
            if pcm is None:
                c.skip("clip not found")
            app.voice._mic_factory = lambda: FileMic(pcm)
            since = time.time()
            self.run(app.voice.start_listening(), 30)
            deadline = time.monotonic() + 60
            rows: list[dict] = []
            while time.monotonic() < deadline:
                rows = app.db.query("SELECT turn_id, stage, ms FROM timings WHERE kind='cascade' AND at >= ? "
                                    "ORDER BY id", (since,))
                if any(r["stage"] == "total" for r in rows):
                    break
                time.sleep(0.25)
            self.run(app.voice.stop_listening(), 30)
            turns = app.db.query("SELECT role, text, source FROM turns WHERE at >= ? AND source='cascade' ORDER BY id",
                                 (since,))
            c.data.update({"engine": app.voice.engine_name or app.voice.choose_engine(),
                           "stages_ms": {r["stage"]: round(r["ms"]) for r in rows}, "turns": turns,
                           "audio_bytes_played": getattr(app.voice.speaker, "bytes_played", None)})
            ttfa = c.data["stages_ms"].get("first_audio")
            answer = c.data["stages_ms"].get("first_answer_audio")
            c.detail = (f"engine {c.data['engine']}, TTFA {ttfa} ms (target 4500), first answer audio {answer} ms, "
                        "turns: " + " | ".join(f"{t['role']}: {t['text']}" for t in turns))
            assert ttfa is not None and turns, c.data
            # The acknowledgement alone is not a pass (review 2026-09-24: a 4987 ms TTFA
            # "passed" while the real answer was «ببورە، نەکرا.» after a failed tool).
            from sam.brain.responder import SORANI_NO_MODEL, SORANI_NOT_DONE, SORANI_NOT_UNDERSTOOD
            said = " ".join(t["text"] for t in turns if t["role"] == "assistant")
            failed_tools = [t["text"] for t in turns if t["role"] == "tool" and ": failed" in t["text"]]
            assert answer is not None, "no answer audio after the acknowledgement"
            assert not any(x in said for x in (SORANI_NO_MODEL, SORANI_NOT_DONE, SORANI_NOT_UNDERSTOOD)), said
            assert not failed_tools, failed_tools
        app.voice._mic_factory = lambda: FileMic()

    def _cleanup(self, chart0: dict | None, owned0: int, pads0: set[int]) -> None:
        app = self.app
        with self.acc.check("B: cleanup (chart restored, SAM drawings gone, new Notepad closed, test data removed)") as c:
            if chart0 is not None:
                if owned0 == 0:
                    c.data["cleared"] = self.run(app.trading.tv.clear(), 30)
                state = self.chart()
                if state.get("symbol") != chart0.get("symbol"):
                    self.run(app.trading.tv.set_symbol(chart0["symbol"]), 60)
                if state.get("resolution") != chart0.get("resolution"):
                    self.run(app.trading.tv.set_timeframe(chart0.get("timeframe") or chart0["resolution"]), 60)
                after = self.chart()
                c.data["chart_after"] = {k: after.get(k) for k in ("symbol", "resolution", "user_drawings",
                                                                   "my_drawings")}
                assert after.get("symbol") == chart0.get("symbol") and after.get("resolution") == chart0.get(
                    "resolution"), c.data["chart_after"]
                assert after.get("user_drawings") == chart0.get("user_drawings")
            for hwnd in self.notepads() - pads0:
                self.run(app.hands.windows.close(hwnd), 30)
            c.data["notepad_left"] = len(self.notepads() - pads0)
            ids = sorted(self.conversations | {r["id"] for r in app.db.query(
                "SELECT id FROM conversations WHERE started_at >= ?", (self.since,))})
            with app.db.transaction():
                for cid in ids:
                    app.db.execute("DELETE FROM turns WHERE conversation_id=?", (cid,))
                    app.db.execute("DELETE FROM brain_conversation_state WHERE conversation_id=?", (cid,))
                    app.db.execute("DELETE FROM conversations WHERE id=?", (cid,))
                app.db.execute("DELETE FROM facts WHERE created_at >= ? AND source='extracted'", (self.since,))
                app.db.execute("DELETE FROM analyses WHERE at >= ?", (self.since,))
            if app.conversation is not None:
                app.conversation.conversation_id = None
            c.data["conversations_removed"] = ids
            assert c.data["notepad_left"] == 0


def phase_text_path(acc: Acceptance, home: Path) -> dict[str, Any]:
    import sam.__main__ as sam_main
    import sam.ui
    from sam import winapp

    os.environ["SAM_LOG_DIR"] = str(WORK / "logs-B")
    silence_devices()
    since = time.time()
    real_run = sam.ui.run
    holder: dict[str, Any] = {}

    def run_with_driver(app: Any, core: Any, *, started: float | None = None) -> int:
        prior = app.config.get("memory.extract_on_sleep", True)
        app.config.set("memory.extract_on_sleep", False)   # no model call / facts from the smoke conversation
        driver = holder["driver"] = Driver(acc, app, core, since)

        def go() -> None:
            try:
                driver.scenario()
            except Exception as exc:  # noqa: BLE001
                with acc.check("B: scenario crashed"):
                    raise exc
            finally:
                if prior is True:     # the default: leave no settings row behind
                    app.config.reset("memory.extract_on_sleep")
                else:
                    app.config.set("memory.extract_on_sleep", prior)
                if not winapp.signal_quit():
                    from PySide6.QtCore import QMetaObject, Qt
                    from PySide6.QtWidgets import QApplication
                    QMetaObject.invokeMethod(QApplication.instance(), "quit", Qt.ConnectionType.QueuedConnection)
        threading.Thread(target=go, name="smoke-driver", daemon=True).start()
        return real_run(app, core, started=started)

    sam.ui.run = run_with_driver
    try:
        code = sam_main.main(["--home", str(home)])
    finally:
        sam.ui.run = real_run
    with acc.check("B: clean shutdown, no SAM 2 instance left") as c:
        c.data = {"exit_code": code, "instance_running": winapp.instance_running(),
                  "stopped_line": "SAM 2 stopped" in read_log(WORK / "logs-B")}
        assert code == 0 and not c.data["instance_running"] and c.data["stopped_line"], c.data
    return holder["driver"].report if "driver" in holder else {}


def other_sam2_processes() -> list[str]:
    """Python processes running SAM 2 other than this script (should be none).
    Only python*.exe: the shells that launched this script also mention it."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { "
          f"$_.ProcessId -ne {os.getpid()} -and $_.CommandLine -match 'SAM\\.pyw|-m sam( |$)' }} | "
          "ForEach-Object { \"$($_.ProcessId) $($_.Name)\" }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    return [line for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", default=os.environ.get("SAM_HOME") or DEFAULT_HOME)
    parser.add_argument("--skip-a", action="store_true")
    parser.add_argument("--skip-b", action="store_true")
    parser.add_argument("--pace", type=float, default=20.0, help="seconds between typed commands")
    args = parser.parse_args()
    global PACE_S
    PACE_S = args.pace
    home = Path(args.home)
    WORK.mkdir(parents=True, exist_ok=True)
    acc = Acceptance("integration_smoke")
    from sam import winapp

    if winapp.instance_running():
        acc.skip("SAM 2 is already running; quit it first (this check never touches it)")
        return acc.finish()
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    usage_sql = "SELECT provider, model, kind, requests, errors, rate_limited FROM usage_counters ORDER BY day"
    usage0 = wait_rows(home, usage_sql, (), until=lambda r: True, timeout_s=2)
    if not args.skip_a:
        phase_real_process(acc, home, pythonw)
    report: dict[str, Any] = {}
    if not args.skip_b:
        report = phase_text_path(acc, home)
    with acc.check("no SAM 2 process left; no key material in the logs") as c:
        left = other_sam2_processes()
        logs = read_log(WORK / "logs-A") + read_log(WORK / "logs-B")
        usage1 = wait_rows(home, usage_sql, (), until=lambda r: True, timeout_s=2)
        c.data = {"processes_left": left, "key_shapes_in_logs": len(KEY_SHAPES.findall(logs)),
                  "usage_delta": usage_delta(usage0, usage1)}
        assert not left and not c.data["key_shapes_in_logs"], c.data
    (WORK / "smoke-report.json").write_text(json.dumps({"commands": report, "result": acc.result()},
                                                       ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return acc.finish()


def usage_delta(before: list[tuple], after: list[tuple]) -> dict[str, list[int]]:
    """Requests/errors/429s spent by this run per provider:model:kind (today's rows)."""
    def total(rows: list[tuple]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for provider, model, kind, requests, errors, limited in rows:
            item = out.setdefault(f"{provider}:{model}:{kind}", [0, 0, 0])
            item[0] += requests or 0
            item[1] += errors or 0
            item[2] += limited or 0
        return out
    b, a = total(before), total(after)
    return {k: [v[i] - b.get(k, [0, 0, 0])[i] for i in range(3)] for k, v in a.items() if v != b.get(k)}


if __name__ == "__main__":
    sys.exit(main())
