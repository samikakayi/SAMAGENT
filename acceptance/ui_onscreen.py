"""Real on-screen check of the island (run by hand; ~5 s on the desktop).

Usage:  .venv\\Scripts\\python.exe acceptance\\ui_onscreen.py [out_dir]

Shows ONLY the island (no tray icon, no panel) at the top centre of the real
desktop with a temporary SAM_HOME (no keys, no network, no microphone), then:
- checks that showing it did not change the foreground window (no focus steal)
  and that the Win32 styles WS_EX_NOACTIVATE / WS_EX_TOOLWINDOW / WS_EX_TOPMOST
  are set;
- measures process CPU while idle (timer stopped) and while animating (fps);
- checks that a fully transparent pixel of the island window lets clicks through
  (WindowFromPoint returns another window);
- grabs only the island's own screen region in three states -> PNGs;
- closes everything. Nothing is clicked, typed or recorded.

Exit code / JSON per acceptance/_common.py (0 pass, 1 fail, 77 skip).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, Acceptance  # noqa: E402

if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
    del os.environ["QT_QPA_PLATFORM"]

DEFAULT_OUT = Path(os.environ.get("SAM_UI_SHOTS", ROOT / "work" / "ui-shots"))
IDLE_CPU_MAX_PCT = 2.0          # idle island: the animation timer is stopped
ANIMATING_CPU_MAX_PCT = 35.0    # one core, 60 fps listening with a live level meter
SLOW_CPU_MAX_PCT = 25.0         # one core, 30 fps thinking comet (sam/ui/island.py SLOW_FRAME_MS)
ISLAND_VISIBLE_MAX_MS = 3000.0  # design acceptance 2 (whole app); the UI alone must be well under it


def main(out_dir: Path, acc: Acceptance) -> None:
    import ctypes
    from ctypes import wintypes

    from PySide6.QtCore import QEventLoop, QPoint, QTimer
    from PySide6.QtWidgets import QApplication

    import sam.ui as ui
    from sam.app import App
    from sam.bridge import CoreThread
    from sam.events import Caption, VoiceState

    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.WindowFromPoint.argtypes = [wintypes.POINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
    get_style = user32.GetWindowLongPtrW
    get_style.restype = ctypes.c_ssize_t
    get_style.argtypes = [wintypes.HWND, ctypes.c_int]

    out_dir.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="sam-ui-onscreen-"))
    (home / "data").mkdir()
    app = App(home, environ={}, llm_backends={})
    core = CoreThread()
    core.start()
    app.bus.bind_loop(core.loop)
    qapp = QApplication.instance() or QApplication(sys.argv[:1])

    def settle(seconds: float) -> None:
        """Run the real Qt event loop (no busy polling, so CPU numbers are honest)."""
        loop = QEventLoop()
        QTimer.singleShot(int(seconds * 1000), loop.quit)
        loop.exec()

    controller = None
    try:
        with acc.check("platform is the real Windows desktop") as c:
            c.detail = qapp.platformName()
            assert qapp.platformName() == "windows", c.detail

        before = user32.GetForegroundWindow()
        started = time.perf_counter()
        controller = ui.build(app, core, show=False)
        island = controller.island
        island.show()
        qapp.processEvents()
        visible_ms = (time.perf_counter() - started) * 1000
        settle(0.4)
        with acc.check("island visible quickly (UI build + show)") as c:
            c.data["island_visible_ms"] = round(visible_ms, 1)
            c.detail = f"{visible_ms:.0f} ms"
            assert visible_ms < ISLAND_VISIBLE_MAX_MS

        hwnd = int(island.winId())
        ex = get_style(hwnd, -20)
        after = user32.GetForegroundWindow()
        with acc.check("showing the island does not steal focus") as c:
            c.data.update(foreground_before=int(before or 0), foreground_after=int(after or 0), island_hwnd=hwnd)
            assert int(after or 0) != hwnd, "the island became the foreground window"
            assert after == before, "the foreground window changed"
        with acc.check("Win32 styles: NOACTIVATE + TOOLWINDOW + TOPMOST") as c:
            c.data.update(noactivate=bool(ex & 0x08000000), toolwindow=bool(ex & 0x80), topmost=bool(ex & 0x8),
                          layered=bool(ex & 0x80000))
            assert ex & 0x08000000 and ex & 0x80 and ex & 0x8, c.data

        geo = island.geometry()
        with acc.check("top centre of the primary screen, DPI aware") as c:
            area = qapp.primaryScreen().availableGeometry()
            c.data.update(geometry_logical=[geo.x(), geo.y(), geo.width(), geo.height()],
                          device_pixel_ratio=island.devicePixelRatioF(),
                          screen=[area.x(), area.y(), area.width(), area.height()])
            assert abs(geo.center().x() - area.center().x()) <= 2 and geo.top() <= area.top() + 20, c.data

        with acc.check("transparent margin lets clicks through") as c:
            pill = island._pill_rect()
            probe = island.mapToGlobal(QPoint(2, int(pill.center().y())))   # alpha 0 (outside the shadow)
            dpr = island.devicePixelRatioF()
            sg = island.screen().geometry()
            px = int(sg.x() + (probe.x() - sg.x()) * dpr)
            py = int(sg.y() + (probe.y() - sg.y()) * dpr)
            hit = user32.WindowFromPoint(wintypes.POINT(px, py))
            root = user32.GetAncestor(hit, 2) if hit else None
            c.data["hit_other_window"] = bool(hit) and int(root or 0) != hwnd
            assert c.data["hit_other_window"]

        def grab(name: str) -> str:
            g = island.geometry()
            pix = island.screen().grabWindow(0, g.x(), g.y(), g.width(), g.height())
            path = out_dir / f"onscreen_{name}.png"
            pix.save(str(path))
            return str(path)

        shots = [grab("idle")]
        with acc.check("idle: animation timer stopped, ~0% CPU") as c:
            t0, c0 = time.perf_counter(), time.process_time()
            settle(1.5)
            cpu = (time.process_time() - c0) / (time.perf_counter() - t0) * 100
            c.data.update(idle_timer_active=island.animating, idle_cpu_pct=round(cpu, 2))
            c.detail = f"{cpu:.2f}% CPU"
            assert not island.animating and cpu < IDLE_CPU_MAX_PCT

        controller.bridge.deliver(VoiceState(state="listening", engine="cascade"))
        controller.bridge.deliver(Caption(text="گۆڵد لەسەر ١٥ خولەک پیشان بدە", role="user", final=False))
        with acc.check("listening: <= 60 fps, bounded CPU") as c:
            feeder = QTimer()
            feeder.setInterval(40)   # LevelMeter arrives at <= 25 Hz
            feeder.timeout.connect(lambda: island.set_level("mic", 0.25 + 0.6 * abs(((time.monotonic() * 3.1) % 2) - 1)))
            feeder.start()
            # Steady state: the caption's 260 ms grow animation and the first
            # build of this colour's cached orb layers are one-off costs.
            settle(0.5)
            t0, c0, frames0, paint0 = time.perf_counter(), time.process_time(), island.frames, island.paint_ms
            settle(2.0)
            feeder.stop()
            elapsed = time.perf_counter() - t0
            cpu = (time.process_time() - c0) / elapsed * 100
            fps = (island.frames - frames0) / elapsed
            frames = max(1, island.frames - frames0)
            paint = (island.paint_ms - paint0) / frames
            c.data.update(animating_cpu_pct=round(cpu, 2), animating_fps=round(fps, 1),
                          paint_ms_per_frame=round(paint, 2))
            c.detail = f"{fps:.0f} fps, {cpu:.1f}% CPU, paint {paint:.2f} ms/frame"
            assert fps <= 61 and cpu < ANIMATING_CPU_MAX_PCT
        shots.append(grab("listening"))

        controller.bridge.deliver(VoiceState(state="thinking", engine="cascade"))
        settle(0.6)                     # the mic level decays first (60 fps until it is quiet)
        with acc.check("thinking: slow motion at ~30 fps") as c:
            t0, c0, frames0 = time.perf_counter(), time.process_time(), island.frames
            settle(1.5)
            elapsed = time.perf_counter() - t0
            cpu = (time.process_time() - c0) / elapsed * 100
            fps = (island.frames - frames0) / elapsed
            c.data.update(thinking_cpu_pct=round(cpu, 2), thinking_fps=round(fps, 1))
            c.detail = f"{fps:.0f} fps, {cpu:.1f}% CPU"
            assert 15 <= fps <= 32 and cpu < SLOW_CPU_MAX_PCT
        shots.append(grab("thinking"))

        controller.bridge.deliver(VoiceState(state="speaking", engine="cascade"))
        controller.bridge.deliver(Caption(text="باشە، چارتی زێڕم لەسەر پازدە خولەک کردەوە.", role="assistant",
                                          final=True))
        feeder = QTimer()
        feeder.setInterval(40)
        feeder.timeout.connect(lambda: island.set_level("speaker", 0.7))
        feeder.start()
        settle(0.8)
        feeder.stop()
        shots.append(grab("speaking"))
        with acc.check("screenshots of the island region") as c:
            c.data["shots"] = shots
            assert all(Path(s).stat().st_size > 1000 for s in shots)

        controller.shutdown()
        controller = None
        island.deleteLater()
        settle(0.2)
        with acc.check("focus unchanged after closing") as c:
            assert user32.GetForegroundWindow() == before
    finally:
        if controller is not None:
            controller.shutdown()
        core.stop()
        app.close()
        shutil.rmtree(home, ignore_errors=True)      # the temporary SAM_HOME


if __name__ == "__main__":
    acceptance = Acceptance("ui_onscreen")
    if sys.platform != "win32":
        acceptance.skip("needs the Windows desktop")
    else:
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT, acceptance)
    sys.exit(acceptance.finish())
