"""``python -m sam`` -- start SAM 2 (UI unless --no-ui).

Options:
  --home PATH   SAM_HOME override (folder with .env and data/)
  --no-ui       run the core only (voice/tools/monitor), Ctrl+C to quit
  --console     also log to stderr
  --check       build the app, register packages, print a JSON status (no key
                values, nothing started, no mic/network) and exit
  --after-pid N wait for process N to exit first (used by tray "Restart")
  --quit        ask the running SAM 2 to shut down cleanly and wait for it

Start-up order (design acceptance 2: island visible < 3 s). The UI is shown
FIRST while ``app.start()`` runs on the core thread. Measured 2026-09-24 by the
launcher stage: waiting for ``app.start()`` before building the UI put the
island at 2.1-2.2 s on an idle PC and 3.3-5.3 s under load, because voice
start alone took 0.56-5.4 s. Packages are already registered (tools, slots)
before the UI is built, and the UI reads late component states back through
``sam.ui.status_seed``, so nothing depends on ``start()`` having finished.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from typing import Any

# numpy's OpenBLAS reserves a buffer per hardware thread (24 on this PC).
# Measured 2026-09-24 (launcher stage): private bytes 852 MB by default vs
# 148 MB with 2 threads, same ~205 MB working set; SAM's numpy work (500-bar
# indicators, 20 ms audio frames) gains nothing from more threads. SAM.pyw
# sets the same default; this covers plain ``python -m sam`` runs. Must be set
# before numpy is first imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

START_WAIT_ON_EXIT_S = 25.0   # quit during start-up: let start() settle before stop()
QUIT_WAIT_S = 30.0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sam", description="SAM 2 desktop assistant")
    parser.add_argument("--home", default=None)
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--console", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--quit", action="store_true", help="ask the running SAM 2 to shut down cleanly")
    parser.add_argument("--after-pid", type=int, default=0,
                        help="wait (<= 15 s) for this process to exit first (tray 'Restart')")
    return parser.parse_args(argv)


def _wait_for_exit(pid: int, timeout_s: float = 15.0) -> None:
    """Restart support: the old instance must release the mutex first."""
    import ctypes

    SYNCHRONIZE = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid) if pid else 0
    if handle:
        ctypes.windll.kernel32.WaitForSingleObject(handle, int(timeout_s * 1000))
        ctypes.windll.kernel32.CloseHandle(handle)


def request_quit(timeout_s: float = QUIT_WAIT_S) -> int:
    """``--quit``: 0 = SAM stopped (or was not running), 1 = it did not stop in time."""
    from .winapp import instance_running, signal_quit

    if not instance_running():
        return 0
    deadline = time.monotonic() + timeout_s
    sent = False
    while time.monotonic() < deadline:
        if not sent:
            sent = signal_quit()      # False while the instance is still starting
        if not instance_running():
            return 0
        time.sleep(0.2)
    return 1


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args = _parse(argv)
    if args.quit:
        return request_quit()
    from .app import App, setup_logging
    from .bridge import CoreThread
    from .winapp import acquire_single_instance, set_dpi_awareness, signal_show

    if args.check:
        app = App(args.home)
        setup_logging(app, console=args.console)
        status = {"load": app.load_packages(), **app.status()}
        print(json.dumps(app.redact_obj(status), ensure_ascii=False, indent=1))
        app.close()
        return 0 if not app.failed else 1

    if args.after_pid:
        _wait_for_exit(args.after_pid)
    if not acquire_single_instance():
        signal_show()
        return 0

    app = App(args.home)
    setup_logging(app, console=args.console)
    log = logging.getLogger("sam")
    app.load_packages()
    app.timing.record("startup:registered", (time.perf_counter() - started) * 1000.0, kind="startup")
    core = CoreThread()
    core.start()
    # Bind before anything publishes from another thread (UI, hotkey, audio):
    # app.start() binds it too, but the UI attaches while start() is running.
    app.bus.bind_loop(core.loop)
    start_future = core.submit(app.start())
    start_future.add_done_callback(lambda f: _core_ready(app, f, started, log))
    try:
        if args.no_ui:
            set_dpi_awareness()
            start_future.result(timeout=90)
            return _run_headless(log)
        try:
            from . import ui
        except ImportError as exc:
            log.error("UI unavailable (%s); running headless", exc)
            start_future.result(timeout=90)
            return _run_headless(log)
        return int(ui.run(app, core, started=started) or 0)
    finally:
        _shutdown(app, core, start_future, log)


def _core_ready(app: Any, future: Any, started: float, log: logging.Logger) -> None:
    """Runs on the core thread when ``app.start()`` finishes."""
    if future.cancelled():
        return
    error = future.exception()
    if error is not None:
        log.error("core start failed: %s", app.redact(f"{type(error).__name__}: {error}"))
        return
    app.timing.record("startup:core_ready", (time.perf_counter() - started) * 1000.0, kind="startup")


def _shutdown(app: Any, core: Any, start_future: Any, log: logging.Logger) -> None:
    """Clean stop: packages in reverse order (voice stops listening, CDP and
    MT5 close, the monitor stops), then the loop, then the DB (SQLite
    checkpoints the WAL when its last connection closes)."""
    try:
        start_future.result(timeout=START_WAIT_ON_EXIT_S)
    except Exception:  # noqa: BLE001 - start errors are already logged; stop anyway
        pass
    try:
        core.run_sync(app.stop(), timeout=20)
    except Exception:  # noqa: BLE001
        log.exception("shutdown error")
    core.stop()
    app.close()
    from .winapp import release_single_instance

    release_single_instance()
    log.info("SAM 2 stopped")


def _run_headless(log: logging.Logger) -> int:
    from .winapp import watch_quit_requests

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    watch_quit_requests(stop.set)
    log.info("SAM 2 running without UI; Ctrl+C to quit")
    while not stop.wait(0.5):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
