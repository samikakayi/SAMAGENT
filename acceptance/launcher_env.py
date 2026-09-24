"""Live: the install/launch environment of this PC (read-only).

- the venv runs Python 3.13 and every requirement imports;
- OmniRoute: installed? answering on 127.0.0.1:20128? (only a GET of its
  status endpoint -- never started or stopped here);
- the installer's dry run succeeds and names the three shortcuts;
- which SAM shortcuts exist today and whether they start SAM 2 or v1
  (reported only: the switch-over is the lead's step);
- is SAM 2 running (its single-instance mutex)?
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from _common import ROOT, Acceptance, sam_home

IMPORTS = ["google.genai", "websockets", "httpx", "numpy", "sounddevice", "webrtcvad", "PySide6.QtWidgets",
           "MetaTrader5", "win32api", "comtypes", "uiautomation", "mss", "PIL", "rapidfuzz",
           "winrt.windows.media.ocr", "tzdata"]


def main() -> int:
    acc = Acceptance("launcher_env")

    with acc.check("venv Python is 3.13") as c:
        c.data["python"] = sys.version.split()[0]
        c.data["executable"] = sys.executable
        assert sys.version_info[:2] == (3, 13), sys.version

    with acc.check("every requirement imports") as c:
        missing = []
        for name in IMPORTS:
            result = subprocess.run([sys.executable, "-c", f"import {name}"], capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                missing.append(name)
        c.data["missing"] = missing
        assert not missing, missing

    with acc.check("OmniRoute gateway") as c:
        from sam import omniroute

        c.data = {"installed": omniroute.is_installed(), "command": omniroute.find_command() or "",
                  "port_open": omniroute.port_open(), "answering": omniroute.is_running()}
        c.detail = ("answering" if c.data["answering"] else
                    "installed but not answering" if c.data["installed"] else "not installed")
        if c.data["installed"]:
            assert c.data["command"], "installed but the omniroute command is missing"

    powershell = shutil.which("powershell")
    with acc.check("installer dry run") as c:
        if not powershell:
            c.skip("Windows PowerShell not found")
        result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                                 str(ROOT / "scripts" / "install.ps1"), "-DryRun", "-SamHome", str(sam_home())],
                                capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace")
        c.data["tail"] = result.stdout.strip().splitlines()[-12:]
        assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
        assert "Dry run finished: nothing was changed." in result.stdout

    with acc.check("current SAM shortcuts (informational)") as c:
        if not powershell:
            c.skip("Windows PowerShell not found")
        links = {
            "desktop": Path(os.path.expandvars(r"%USERPROFILE%\Desktop\SAM.lnk")),
            "start_menu": Path(os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\SAM.lnk")),
            "startup": Path(os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SAM (background).lnk")),
        }
        for key, path in links.items():
            if not path.exists():
                c.data[key] = "missing"
                continue
            script = f"(New-Object -ComObject WScript.Shell).CreateShortcut('{path}').Arguments"
            args = subprocess.run([powershell, "-NoProfile", "-Command", script], capture_output=True, text=True,
                                  timeout=60).stdout.strip()
            c.data[key] = "SAM 2" if "SAM.pyw" in args else ("SAM v1" if "sam_desktop.pyw" in args else "other")
        c.detail = ", ".join(f"{k}: {v}" for k, v in c.data.items())

    with acc.check("SAM 2 running (informational)") as c:
        import importlib.machinery
        import importlib.util

        loader = importlib.machinery.SourceFileLoader("sam_launcher_probe", str(ROOT / "SAM.pyw"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        c.data["running"] = module.instance_running()
        c.detail = "yes" if c.data["running"] else "no"

    return acc.finish()


if __name__ == "__main__":
    sys.exit(main())
