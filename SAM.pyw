"""SAM 2 desktop launcher -- run with ``pythonw.exe`` (no console window).

Shortcuts made by ``scripts/install.ps1`` start this file:

    .venv\\Scripts\\pythonw.exe SAM.pyw --home <SAM_HOME>               (Desktop / Start menu)
    .venv\\Scripts\\pythonw.exe SAM.pyw --home <SAM_HOME> --background  (Startup: sign-in)

What it does, in order:
1. pythonw has no stdout/stderr (``sys.stdout is None``): anything printed or
   any traceback would vanish, so both go (redacted) to
   ``%LOCALAPPDATA%\\SAM\\sam.log``; hard crashes (faulthandler) to
   ``sam-crash.log`` next to it. The app's own detailed log stays at
   ``%LOCALAPPDATA%\\SAM2\\logs\\sam2.log`` (``sam.config.Config.log_dir``).
2. Single instance: if SAM 2 already runs (its named mutex exists), a normal
   launch asks it to show its panel (``sam.winapp.signal_show``) and exits; a
   ``--background`` launch just exits.
3. Starts the local OmniRoute gateway early when it is installed and not
   listening (``sam.omniroute``: the v1 launcher's exact rules -- run from
   ``~/.omniroute``, OMNIROUTE_CLI_SKIP_REPO_ENV=1, CREATE_NO_WINDOW |
   CREATE_NEW_PROCESS_GROUP, never DETACHED_PROCESS).
4. Runs ``sam.__main__.main`` (core thread + Qt UI). ``--background`` sets
   ``SAM_BACKGROUND=1`` for the UI (island only, panel hidden); a normal
   launch opens the panel once the UI is listening for show requests.
5. If start-up fails, a Sorani/English message box names the log file (a
   pythonw program has no other way to tell the user).

Extra arguments (``--no-ui``, ``--console``, ``--check``, ``--after-pid N``)
are passed through to ``python -m sam``. ``--quit`` asks a running SAM 2 to
shut down cleanly (same path as tray Quit) and waits for it. ``--write-icon
PATH`` only writes SAM's .ico (used by the installer) and exits.
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
APP_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "SAM"
LOG_PATH = APP_DIR / "sam.log"
CRASH_LOG_PATH = APP_DIR / "sam-crash.log"
SHOW_WAIT_S = 90.0          # the first start after sign-in can be slow on this busy PC

MB_ICONERROR = 0x10
MB_SETFOREGROUND = 0x10000
SYNCHRONIZE = 0x00100000
ERROR_ACCESS_DENIED = 5

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# numpy's OpenBLAS reserves a buffer per hardware thread (24 on this Ryzen AI 9
# HX 370). Measured 2026-09-24 with acceptance/launcher_startup.py on an idle
# SAM 2: private bytes 852 MB by default vs 148 MB with 2 threads, same ~205 MB
# working set. SAM's numpy work (indicators on <= 500 bars, 20 ms audio frames)
# gains nothing from more BLAS threads. Must be set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

_log = logging.getLogger("sam.launcher")
_crash_file: Any = None


def _redact(text: str) -> str:
    try:
        from sam.secrets import redact
        return redact(text)
    except Exception:  # noqa: BLE001 - redaction must never stop the launcher
        return text


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            if record.exc_info and record.exc_info[1] is not None:
                message += "\n" + "".join(traceback.format_exception(*record.exc_info))
                record.exc_info, record.exc_text = None, None
            record.msg, record.args = _redact(message), None
        except Exception:  # noqa: BLE001
            pass
        return True


class _LogStream:
    """File-like stand-in for the missing stdout/stderr of pythonw: every
    complete line goes to the launcher log, redacted."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger, self._level, self._buffer = logger, level, ""
        self._lock = threading.Lock()
        self._inside = threading.local()

    def write(self, text: str) -> int:
        # A failing log handler reports to sys.stderr, i.e. back here: drop
        # re-entrant writes instead of recursing.
        if getattr(self._inside, "active", False):
            return len(text)
        with self._lock:
            self._buffer += str(text)
            *lines, self._buffer = self._buffer.split("\n")
        self._inside.active = True
        try:
            for line in lines:
                if line.strip():
                    self._logger.log(self._level, "%s", line.rstrip())
        finally:
            self._inside.active = False
        return len(text)

    def flush(self) -> None:
        with self._lock:
            rest, self._buffer = self._buffer, ""
        if rest.strip():
            self._logger.log(self._level, "%s", rest.rstrip())

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


def setup_logging(log_path: Path | None = None) -> logging.Logger:
    """Launcher log (rotating, redacted); stdout/stderr captured under pythonw."""
    global _crash_file
    path = log_path or LOG_PATH
    _log.setLevel(logging.INFO)
    _log.propagate = False
    if not any(getattr(h, "_sam_launcher", False) for h in _log.handlers):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler: logging.Handler = logging.handlers.RotatingFileHandler(
                path, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        except OSError:
            handler = logging.NullHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handler.addFilter(_RedactingFilter())
        handler._sam_launcher = True  # type: ignore[attr-defined]
        _log.addHandler(handler)
    if sys.stdout is None:
        sys.stdout = _LogStream(_log, logging.INFO)  # type: ignore[assignment]
    if sys.stderr is None:
        logging.raiseExceptions = False  # nobody can read handler errors under pythonw
        sys.stderr = _LogStream(_log, logging.ERROR)  # type: ignore[assignment]
        # Native crashes (e.g. inside Qt) print only frame names, never values.
        try:
            import faulthandler
            _crash_file = open(path.parent / CRASH_LOG_PATH.name, "a", encoding="utf-8")  # noqa: SIM115
            faulthandler.enable(_crash_file)
        except (OSError, RuntimeError):
            pass
    return _log


def instance_running(name: str | None = None) -> bool:
    """True when SAM 2's single-instance mutex exists (without creating it:
    ``sam.__main__`` must be the one to own it)."""
    if os.name != "nt":
        return False
    from sam.winapp import MUTEX_NAME

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenMutexW.restype = ctypes.c_void_p
    kernel32.OpenMutexW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
    handle = kernel32.OpenMutexW(SYNCHRONIZE, False, name or MUTEX_NAME)
    if handle:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        return True
    return ctypes.get_last_error() == ERROR_ACCESS_DENIED  # exists, other integrity level


def signal_show() -> bool:
    from sam.winapp import signal_show as _signal_show
    return _signal_show()


def request_panel_when_ready(timeout_s: float = SHOW_WAIT_S, *, poll_s: float = 0.5,
                             signal: Callable[[], bool] | None = None) -> threading.Thread:
    """Open the panel once the UI listens for show requests (normal launch).

    The UI creates its show event in ``sam.winapp.watch_show_requests``; until
    then ``signal_show`` returns False, so poll for it. Gives up quietly (e.g.
    a headless run has no UI)."""
    send = signal or signal_show

    def run() -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                if send():
                    _log.info("panel requested")
                    return
            except Exception:  # noqa: BLE001
                return
            time.sleep(poll_s)
        _log.info("the UI did not listen for show requests within %.0f s", timeout_s)

    thread = threading.Thread(target=run, name="sam-open-panel", daemon=True)
    thread.start()
    return thread


def start_omniroute() -> str:
    try:
        from sam import omniroute
        return omniroute.start_early()
    except Exception as exc:  # noqa: BLE001 - never block SAM over the gateway
        _log.warning("OmniRoute start check failed: %s", exc)
        return "failed"


def show_error(message: str) -> None:
    """A message box: the only way a console-less program can tell the user."""
    try:
        ctypes.windll.user32.MessageBoxW(None, message, "SAM", MB_ICONERROR | MB_SETFOREGROUND)
    except Exception:  # noqa: BLE001
        pass


def error_text(log_path: Path | None = None) -> str:
    path = log_path or LOG_PATH
    return ("SAM نەیتوانی دەست پێبکات.\n"
            f"وردەکارییەکان لەم فایلەدان:\n{path}\n\n"
            "SAM could not start. The details are in the file above.")


def run_app(argv: list[str]) -> int:
    from sam.__main__ import main
    return int(main(argv) or 0)


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog="SAM.pyw", add_help=False)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--write-icon", type=Path, default=None)
    parser.add_argument("--home", default=None)
    parser.add_argument("--no-omniroute", action="store_true")
    return parser.parse_known_args(argv)


def launch(argv: list[str] | None = None) -> int:
    args, rest = parse(list(sys.argv[1:] if argv is None else argv))
    log = setup_logging()
    if args.write_icon:
        from sam.icon import write_icon
        write_icon(args.write_icon)
        return 0
    if "--quit" in rest:
        # Ask the running SAM 2 to shut down cleanly (installer, smoke tests);
        # must come before the single-instance check, which would show it.
        code = run_app(["--quit"])
        log.info("quit request: %s", "stopped" if code == 0 else "still running")
        return code
    headless = any(flag in rest for flag in ("--check", "--no-ui"))
    log.info("launch: background=%s home=%s args=%s", args.background, args.home or "(default)", rest)
    if "--check" not in rest and "--after-pid" not in " ".join(rest) and instance_running():
        if not args.background:
            log.info("SAM is already running: asking it to show its panel")
            signal_show()
        else:
            log.info("SAM is already running: nothing to do for a background start")
        return 0
    os.environ["SAM_BACKGROUND"] = "1" if args.background else "0"
    if not args.no_omniroute and "--check" not in rest:
        log.info("OmniRoute: %s", start_omniroute())
    if not args.background and not headless:
        request_panel_when_ready()
    interactive = "--check" not in rest   # --check is scripted: never block on a message box
    started = time.monotonic()
    try:
        code = run_app(rest + (["--home", args.home] if args.home else []))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except BaseException:  # noqa: BLE001 - report every failure, then exit non-zero
        log.exception("SAM crashed")
        if interactive:
            show_error(error_text())
        return 1
    log.info("SAM exited with code %s after %.0f s", code, time.monotonic() - started)
    if code not in (0, None) and interactive:
        show_error(error_text())
    return int(code or 0)


if __name__ == "__main__":
    sys.exit(launch())
