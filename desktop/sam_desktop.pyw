"""SAM as a desktop program: start it quietly, open it in a window of its own.

Double-clicking the SAM shortcut starts the backend (through start.ps1, so
Ollama and the rest come up exactly as they do from a terminal) and the local
OmniRoute gateway when it is installed, with no console window, then opens SAM
in an app window of its own -- no tabs, no address bar. Closing that window
does not stop SAM: the hands-free listener lives in the backend, so "Hey SAM"
keeps working. The tray icon opens, restarts or quits it.

Run with --background at Windows sign-in: everything starts, no window opens.

This exists because the browser tab made SAM feel like a web page that had to
stay open. Nothing here changes how SAM itself runs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 8765 is SAM's documented default, but on this machine another project's
# `python -m http.server 8765` holds it, so the desktop program uses 8877 --
# the port start.ps1 itself suggests when 8765 is taken.
PORT = int(os.environ.get("SAM_DESKTOP_PORT", "8877"))
URL = f"http://127.0.0.1:{PORT}"
APP_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "SAM"
LOG_PATH = APP_DIR / "sam-desktop.log"
OMNIROUTE_HOME = Path.home() / ".omniroute"
OMNIROUTE_URL = "http://127.0.0.1:20128"

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
# Started services must outlive this launcher (and its tray) and never flash a
# console window.
BACKGROUND_FLAGS = CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def log(message: str) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def _get_json(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback only
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - "not answering" is the answer
        return None


def sam_running() -> bool:
    # Only a responder that says it is SAM counts, as in start.ps1: something
    # else on the port must not be mistaken for a running SAM.
    health = _get_json(f"{URL}/api/health")
    return bool(health and health.get("name") == "SAM")


def omniroute_running() -> bool:
    try:
        urllib.request.urlopen(f"{OMNIROUTE_URL}/api/auth/status", timeout=2)  # noqa: S310
        return True
    except urllib.error.HTTPError:
        return True  # it answered; a refusal still means it is up
    except Exception:  # noqa: BLE001
        return False


def wait_until(check, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.5)
    return check()


def _spawn(command: list[str], cwd: Path, env: dict | None = None) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    output = LOG_PATH.open("a", encoding="utf-8")
    subprocess.Popen(  # noqa: S603 - fixed local commands
        command, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
        creationflags=BACKGROUND_FLAGS, close_fds=True,
    )


def start_omniroute() -> None:
    """Start the local OmniRoute gateway if it is set up and not already up.

    SAM's chat goes through it first (sam-fast); without it SAM falls back to
    Groq directly, so a failure here is logged, never fatal. It runs from its
    own data folder because it also reads a .env from the working directory,
    and SAM's .env must not leak into it.
    """
    if not (OMNIROUTE_HOME / ".env").is_file() or omniroute_running():
        return
    command = shutil.which("omniroute") or str(Path(os.environ.get("APPDATA", "")) / "npm" / "omniroute.cmd")
    if not Path(command).is_file():
        log("OmniRoute is set up but its command was not found; SAM will use Groq directly.")
        return
    env = {**os.environ, "OMNIROUTE_CLI_SKIP_REPO_ENV": "1"}
    log("Starting OmniRoute.")
    _spawn([command, "serve", "--no-open", "--no-tray"], OMNIROUTE_HOME, env)


def start_sam() -> None:
    """Start SAM through start.ps1, hidden, unless it is already answering.

    start.ps1 is the one way SAM is started, so Ollama, the port checks and
    everything else behave exactly as they do from a terminal.
    """
    if sam_running():
        return
    log(f"Starting SAM on port {PORT}.")
    _spawn([
        "powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-WindowStyle", "Hidden", "-File", str(ROOT / "start.ps1"), "-Port", str(PORT), "-NoBrowser",
    ], ROOT)


def _edge() -> str | None:
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            candidate = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if candidate.is_file():
                return str(candidate)
    return None


def open_window() -> None:
    """SAM in an app window of its own, with its own browser profile.

    The separate profile keeps SAM's window, its microphone permission and its
    size apart from the user's normal browsing.
    """
    edge = _edge()
    if edge:
        subprocess.Popen([  # noqa: S603
            edge, f"--app={URL}", f"--user-data-dir={APP_DIR / 'window'}",
            "--no-first-run", "--no-default-browser-check", "--window-size=1400,900",
        ])
    else:
        webbrowser.open(URL)


def stop_sam() -> None:
    """Stop the backend listening on SAM's port, and the start.ps1 that launched it.

    Stopping start.ps1's PowerShell lets nothing restart it; its child Ollama is
    left to start.ps1's own cleanup rules. OmniRoute keeps running: it is shared
    and cheap to leave up.
    """
    try:
        import psutil
    except ImportError:
        log("psutil is missing; cannot stop SAM from the tray.")
        return
    for connection in psutil.net_connections(kind="inet"):
        if connection.status == psutil.CONN_LISTEN and connection.laddr and connection.laddr.port == PORT and connection.pid:
            try:
                process = psutil.Process(connection.pid)
                family = [process, *process.parents()]
                for member in family:
                    command = " ".join(member.cmdline()).lower()
                    if "sam_backend" in command or "start.ps1" in command:
                        member.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    log("SAM stopped from the tray.")


def ensure_running(show_window: bool) -> bool:
    start_omniroute()
    start_sam()
    ready = wait_until(sam_running, 90)
    if not ready:
        log(f"SAM did not answer on {URL} within 90 s; see {LOG_PATH}.")
    if show_window:
        open_window()
    return ready


# --- Icon ---------------------------------------------------------------------

def make_icon_image(size: int = 256):
    """SAM's mark: a white S on the blue of the app's own avatar."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=size // 5, fill=(79, 124, 255, 255))
    font = None
    for name in ("segoeuib.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, int(size * 0.68))
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    box = draw.textbbox((0, 0), "S", font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    draw.text(((size - width) / 2 - box[0], (size - height) / 2 - box[1]), "S", font=font, fill="white")
    return image


def write_icon(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    make_icon_image(256).save(path, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


# --- Tray ---------------------------------------------------------------------

def single_instance() -> bool:
    """True for the first tray in this Windows session; a second launch only opens the window."""
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\SAMDesktopTray")
    return bool(handle) and ctypes.windll.kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS


def tray_available() -> bool:
    try:
        import pystray  # noqa: F401
    except ImportError:
        return False
    return True


def run_tray() -> None:
    import pystray

    def restart(icon, _item):
        stop_sam()
        wait_until(lambda: not sam_running(), 15)
        threading.Thread(target=ensure_running, args=(False,), daemon=True).start()

    def quit_sam(icon, _item):
        stop_sam()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Open SAM", lambda icon, item: open_window(), default=True),
        pystray.MenuItem("Restart SAM", restart),
        pystray.MenuItem("Quit SAM", quit_sam),
    )
    pystray.Icon("SAM", make_icon_image(64), "SAM", menu).run()


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM desktop launcher")
    parser.add_argument("--background", action="store_true", help="start SAM without opening its window")
    parser.add_argument("--write-icon", type=Path, help="write SAM's .ico to this path and exit")
    args = parser.parse_args()

    if args.write_icon:
        write_icon(args.write_icon)
        return 0
    if not single_instance():
        # The tray is already up: this launch only needs the window.
        if not args.background:
            if sam_running():
                open_window()
            else:
                ensure_running(show_window=True)
        return 0
    if not tray_available():
        # No tray: start everything in the foreground, then leave SAM running.
        ensure_running(show_window=not args.background)
        return 0
    threading.Thread(target=ensure_running, args=(not args.background,), daemon=True).start()
    run_tray()
    return 0


if __name__ == "__main__":
    sys.exit(main())
