"""Live: a NORMAL launch (Desktop / Start-menu shortcut) shows the island AND the panel.

    set SAM_HOME=C:\\Users\\samit\\Desktop\\SAM-Agent
    .venv\\Scripts\\python.exe acceptance\\launcher_normal_launch.py

Real use on 2026-09-24: after a normal launch the log said "panel requested"
but the panel stayed hidden until SAM.pyw was run a second time. This starts
``pythonw SAM.pyw --home <SAM_HOME>`` once, waits for both windows (the small
island and the large panel), checks they are still visible 3 s later, then
asks SAM to quit (``SAM.pyw --quit``) and makes sure no SAM 2 process is left.

Nothing is spoken or recorded: listening stays off after a launch (push-to-talk),
there are no alerts in a fresh DB, and the voice self-test does not run when a
verdict is stored. If SAM 2 already runs, the check is skipped (it never
touches the user's own SAM).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from _common import ROOT, Acceptance, sam_home

import launcher_startup as proc

PANEL_MIN_HEIGHT = 300      # physical pixels; the island pill is ~60-130 px tall (DPI-scaled)


def _split(windows: list[dict]) -> tuple[list[dict], list[dict]]:
    panels = [w for w in windows if w["size"][1] >= PANEL_MIN_HEIGHT]
    islands = [w for w in windows if w["size"][1] < PANEL_MIN_HEIGHT]
    return islands, panels


def main() -> int:
    acc = Acceptance("launcher_normal_launch")
    launcher = proc._load_launcher()  # noqa: SLF001
    if launcher.instance_running():
        acc.skip("SAM 2 is already running on this PC; quit it first (this check never touches it)")
        return acc.finish()
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    home = sam_home()
    command = [str(pythonw), str(ROOT / "SAM.pyw"), "--home", str(home)]
    started: subprocess.Popen | None = None
    try:
        began = time.perf_counter()
        started = subprocess.Popen(command, cwd=str(ROOT))

        with acc.check("island and panel visible after one normal launch") as c:
            islands: list[dict] = []
            panels: list[dict] = []
            while time.perf_counter() - began < 30 and started.poll() is None:
                islands, panels = _split(proc.visible_windows(set(proc.process_tree(started.pid))))
                if islands and panels:
                    break
                time.sleep(0.1)
            c.data = {"ms": round((time.perf_counter() - began) * 1000), "islands": islands, "panels": panels}
            assert started.poll() is None, f"SAM exited with code {started.returncode}"
            assert islands, "no island window"
            assert panels, "the panel did not appear (the evening's bug)"
            c.detail = f"island + panel in {c.data['ms']} ms"

        with acc.check("both still visible 3 s later") as c:
            time.sleep(3.0)
            islands, panels = _split(proc.visible_windows(set(proc.process_tree(started.pid))))
            c.data = {"islands": len(islands), "panels": len(panels)}
            assert islands and panels, c.data

        with acc.check("SAM.pyw --quit ends SAM 2") as c:
            t0 = time.perf_counter()
            subprocess.run([str(pythonw), str(ROOT / "SAM.pyw"), "--quit"], cwd=str(ROOT), timeout=30)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and started.poll() is None:
                time.sleep(0.2)
            c.data = {"exit_code": started.poll(), "ms": round((time.perf_counter() - t0) * 1000)}
            assert started.poll() is not None, "still running 20 s after --quit"
    finally:
        if started is not None and started.poll() is None:
            tree = proc.process_tree(started.pid)
            proc.terminate([pid for pid, exe in tree.items() if exe.lower().startswith("python")])
    return acc.finish()


if __name__ == "__main__":
    sys.exit(main())
