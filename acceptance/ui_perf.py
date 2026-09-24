"""Island cost per state on the real desktop + orb paint cost (run by hand, ~15 s).

Usage:  .venv\\Scripts\\python.exe acceptance\\ui_perf.py

Shows ONLY the island (temporary SAM_HOME: no keys, no network, no microphone,
nothing spoken) and measures, with the real Qt event loop:
- per state: frames per second, process CPU (% of one core), paint ms/frame --
  listening with a moving voice level (60 fps expected), listening while the
  user is silent and thinking (30 fps expected), idle (timer stopped, 0 fps);
- the orb's own paint time per frame at the screen's device pixel ratio
  (300 frames into an image), to see what the cached pixmaps save.
These are the numbers quoted in sam/ui/island.py (FRAME_MS) and sam/ui/orb.py.
The machine is shared, so CPU figures move by +-10 points between runs; the
checks only guard against regressions (e.g. 60 fps in a slow state).

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
from _common import Acceptance  # noqa: E402

if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
    del os.environ["QT_QPA_PLATFORM"]


def main(acc: Acceptance) -> None:
    from PySide6.QtCore import QEventLoop, QPointF, Qt, QTimer
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication

    import sam.ui as ui
    from sam.app import App
    from sam.bridge import CoreThread
    from sam.events import Caption, VoiceState
    from sam.ui import theme
    from sam.ui.orb import paint_orb

    home = Path(tempfile.mkdtemp(prefix="sam-ui-perf-"))
    (home / "data").mkdir()
    app = App(home, environ={}, llm_backends={})
    core = CoreThread()
    core.start()
    app.bus.bind_loop(core.loop)
    qapp = QApplication.instance() or QApplication(sys.argv[:1])

    def settle(seconds: float) -> None:
        loop = QEventLoop()
        QTimer.singleShot(int(seconds * 1000), loop.quit)
        loop.exec()

    controller = None
    try:
        controller = ui.build(app, core, show=False)
        island = controller.island
        island.show()
        settle(0.5)

        def measure(check, seconds: float = 2.5) -> tuple[float, float]:
            t0, c0, f0, p0 = time.perf_counter(), time.process_time(), island.frames, island.paint_ms
            settle(seconds)
            elapsed = time.perf_counter() - t0
            frames = island.frames - f0
            fps, cpu = frames / elapsed, (time.process_time() - c0) / elapsed * 100
            paint = (island.paint_ms - p0) / max(1, frames)
            check.data.update(fps=round(fps, 1), cpu_pct_one_core=round(cpu, 1), paint_ms_per_frame=round(paint, 2),
                              timer_interval_ms=island._timer.interval() if island.animating else None)
            check.detail = f"{fps:.0f} fps, {cpu:.1f}% of one core, paint {paint:.2f} ms/frame"
            return fps, cpu

        controller.bridge.deliver(VoiceState(state="listening", engine="cascade"))
        controller.bridge.deliver(Caption(text="گۆڵد لەسەر ١٥ خولەک پیشان بدە", role="user", final=False))
        feeder = QTimer()
        feeder.setInterval(40)                       # LevelMeter arrives at <= 25 Hz
        feeder.timeout.connect(lambda: island.set_level("mic", 0.25 + 0.6 * abs(((time.monotonic() * 3.1) % 2) - 1)))
        feeder.start()
        settle(0.4)
        with acc.check("listening with a voice level: <= 60 fps") as c:
            fps, cpu = measure(c)
            assert 40 <= fps <= 61 and cpu < 45, c.detail
        feeder.stop()
        settle(1.2)                                  # the level decays, the clock drops to 30 fps
        with acc.check("listening, silent: ~30 fps") as c:
            fps, cpu = measure(c)
            assert 15 <= fps <= 32 and cpu < 25, c.detail
        controller.bridge.deliver(VoiceState(state="thinking", engine="cascade"))
        settle(0.3)
        with acc.check("thinking: ~30 fps") as c:
            fps, cpu = measure(c)
            assert 15 <= fps <= 32 and cpu < 25, c.detail
        controller.bridge.deliver(VoiceState(state="idle", engine="cascade"))
        island.clear_caption()
        settle(0.8)                                  # collapse animation ends
        with acc.check("idle: no frames, ~0% CPU") as c:
            fps, cpu = measure(c, 2.0)
            assert not island.animating and fps < 1 and cpu < 2, c.detail

        dpr = island.devicePixelRatioF()
        image = QImage(round(160 * dpr), round(160 * dpr), QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(dpr)
        for state in ("listening", "thinking", "idle"):
            with acc.check(f"orb paint, {state} (300 frames at dpr {dpr:g})") as c:
                color = theme.qcolor(theme.state_color(state))
                t0 = time.perf_counter()
                for i in range(300):
                    image.fill(Qt.GlobalColor.transparent)
                    p = QPainter(image)
                    paint_orb(p, QPointF(80, 80), 17.0, color, state=state, level=0.6 if state == "listening" else 0.0,
                              phase=i * 0.016)
                    p.end()
                ms = (time.perf_counter() - t0) / 300 * 1000
                c.data["ms_per_frame"] = round(ms, 3)
                c.detail = f"{ms:.2f} ms/frame"
                assert ms < 4.0, c.detail
    finally:
        if controller is not None:
            controller.shutdown()
        core.stop()
        app.close()
        shutil.rmtree(home, ignore_errors=True)      # the temporary SAM_HOME


if __name__ == "__main__":
    acceptance = Acceptance("ui_perf")
    if sys.platform != "win32":
        acceptance.skip("needs the Windows desktop")
    else:
        main(acceptance)
    sys.exit(acceptance.finish())
