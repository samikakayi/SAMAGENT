"""Live check: SAM's Settings page through Windows UI Automation (run by hand).

Shows the real panel (temp home: no keys, no model calls, nothing saved in the
user's data) on the Settings page in a child process, then -- from this
process, as a screen reader or another program would -- lists the page's
controls through UIA and operates three of them with UIA patterns (Toggle on
"always listen" and on the "regular voice" engine chip, RangeValue on the
conversation timeout). The child reports the saved settings.

Before the acceptance fix (2026-09-24) the five key rows read «پاشەکەوت» /
«تاقیکردنەوە» alike, eight controls had no name, and toggling an engine chip
through UIA changed nothing (it listened to 'clicked' only).

    .venv\\Scripts\\python.exe acceptance\\ui_settings_uia.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHILD = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
work = sys.argv[2]
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from sam.app import App
from sam.bridge import CoreThread
app = App(os.path.join(work, "home"), environ={}, llm_backends={})
core = CoreThread(); core.start(); app.bus.bind_loop(core.loop)
qapp = QApplication(sys.argv[:1])
from sam.ui.qtbridge import QtBridge
from sam.ui.panel import Panel
bridge = QtBridge(app, core); bridge.attach()
panel = Panel(app, bridge)
panel.show_and_raise("settings")
QTimer.singleShot(800, lambda: open(os.path.join(work, "hwnd.txt"), "w").write(str(int(panel.winId()))))
def done():
    json.dump({"always": app.config.get("voice.always_listening"), "engine": app.config.get("voice.engine"),
               "timeout": app.config.get("voice.conversation_timeout_s")}, open(os.path.join(work, "state.json"), "w"))
    qapp.quit()
QTimer.singleShot(12000, done)
qapp.exec()
bridge.detach(); core.stop(); app.close()
'''
INTERACTIVE = {"ButtonControl", "EditControl", "ComboBoxControl", "CheckBoxControl", "SpinnerControl"}


def main() -> int:
    import uiautomation as auto

    work = Path(tempfile.mkdtemp(prefix="sam2-uia-"))
    (work / "home" / "data").mkdir(parents=True)
    script = work / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    child = subprocess.Popen([str(pythonw if pythonw.exists() else sys.executable), str(script), str(ROOT), str(work)])
    for _ in range(80):
        if (work / "hwnd.txt").exists():
            break
        time.sleep(0.25)
    else:
        child.kill()
        print("FAIL: the panel did not appear")
        return 1
    time.sleep(0.5)
    root = auto.ControlFromHandle(int((work / "hwnd.txt").read_text()))
    controls: list[tuple[str, str]] = []

    def walk(control: object, depth: int = 0) -> None:
        for item in control.GetChildren():       # type: ignore[attr-defined]
            if item.ControlTypeName in INTERACTIVE:
                controls.append((item.ControlTypeName, item.Name))
            if depth < 20:
                walk(item, depth + 1)
    walk(root)
    unnamed = [kind for kind, name in controls if not name and kind != "ButtonControl"]
    buttons = [name for kind, name in controls if kind == "ButtonControl" and name not in ("Minimize", "Maximize",
                                                                                            "Close")]
    print(f"{len(controls)} interactive controls; unnamed: {unnamed}; duplicate buttons: "
          f"{sorted({b for b in buttons if buttons.count(b) > 1})}")
    always = root.CheckBoxControl(searchDepth=20, Name="هەمیشە گوێ بگرە")
    always.GetPattern(auto.PatternId.TogglePattern).Toggle()
    cascade = root.CheckBoxControl(searchDepth=20, Name="بزوێنەری دەنگ: دەنگی ئاسایی")
    cascade.GetPattern(auto.PatternId.TogglePattern).Toggle()
    spin = root.SpinnerControl(searchDepth=20, Name="ماوەی گفتوگۆ دوای بێدەنگی بە چرکە")
    spin.GetPattern(auto.PatternId.RangeValuePattern).SetValue(60)
    child.wait(30)
    state = json.loads((work / "state.json").read_text(encoding="utf-8"))
    ok = not unnamed and state == {"always": True, "engine": "cascade", "timeout": 60}
    print("saved through UIA:", state, "->", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
