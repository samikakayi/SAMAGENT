"""Start the local OmniRoute gateway (http://127.0.0.1:20128) when SAM needs it.

OmniRoute is the user's OpenAI-compatible gateway (combos ``sam-fast``,
``sam-strong``, ``sam-vision``). SAM only ever STARTS it, never stops it: it is
shared with other tools and cheap to leave running.

Ported from v1 ``desktop/sam_desktop.pyw`` (tag v1-final) with its lessons:

- Only when it is installed (``~/.omniroute/.env`` exists) and not already up.
- Run from ``~/.omniroute`` with ``OMNIROUTE_CLI_SKIP_REPO_ENV=1``: its CLI
  also loads a ``.env`` from the working directory (read in
  ``omniroute/bin/omniroute.mjs::loadEnvFile``), and SAM's ``.env`` must not
  leak into it.
- ``CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`` and NEVER
  ``DETACHED_PROCESS``: with DETACHED_PROCESS, v1's hidden ``powershell.exe``
  exited at once with code 0 and no output, so the first v1 install opened a
  window onto a SAM that never started. CREATE_NO_WINDOW alone gives the
  child a hidden console of its own (the ``omniroute.cmd`` shim needs one).
  The process group keeps a Ctrl+C in a dev console from reaching it.
- Child output goes to ``<log_dir>/omniroute.log`` (a file, not a pipe: the
  gateway outlives SAM, and a pipe whose reader has gone would break its
  writes). Its start-up output only names the ``.env`` paths it loaded, never
  values (checked in ``omniroute.mjs``).

Two callers, one start: ``SAM.pyw`` calls :func:`start_early` before the app
is even built (the gateway takes seconds to boot), and ``App.start`` calls
:func:`ensure_running` in the background. A module-level record of the spawn
makes the second call wait for the port instead of starting a second copy.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("sam.omniroute")

DEFAULT_URL = "http://127.0.0.1:20128"
HOST, PORT = "127.0.0.1", 20128
STATUS_PATH = "/api/auth/status"
START_WAIT_S = 45.0          # a cold Next.js start of the gateway takes seconds
RESPAWN_GUARD_S = 60.0       # a spawn younger than this is waited for, not repeated

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008   # listed only so tests can assert it is never used
BACKGROUND_FLAGS = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP

_lock = threading.Lock()
_spawned_at: float | None = None


def omniroute_home() -> Path:
    """``~/.omniroute`` (OmniRoute's own data folder, holding its ``.env``)."""
    return Path.home() / ".omniroute"


def is_installed(home: Path | None = None) -> bool:
    return ((home or omniroute_home()) / ".env").is_file()


def default_log_dir() -> Path:
    """Same folder as ``Config.log_dir`` (used before the app exists)."""
    value = os.environ.get("SAM_LOG_DIR")
    if value:
        return Path(value)
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "SAM2" / "logs"


def port_open(host: str = HOST, port: int = PORT, timeout: float = 0.3) -> bool:
    """Fast TCP probe (a closed loopback port refuses at once)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# Loopback only: never route the probe through a system proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def is_running(url: str = DEFAULT_URL, timeout: float = 2.0) -> bool:
    """True when OmniRoute answers HTTP at ``url`` (any status counts: a 401
    from its auth endpoint still means the gateway is up -- v1 behaviour)."""
    try:
        with _OPENER.open(url.rstrip("/") + STATUS_PATH, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:  # noqa: BLE001 - "not answering" is the answer
        return False


def find_command() -> str | None:
    """The ``omniroute`` CLI: PATH first, then npm's default global folder."""
    found = shutil.which("omniroute")
    if found:
        return found
    candidate = Path(os.environ.get("APPDATA", "")) / "npm" / "omniroute.cmd"
    return str(candidate) if candidate.is_file() else None


def _spawn(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    """Start the gateway detached from SAM's lifetime; return its pid."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as output:
        output.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} SAM starts OmniRoute ---\n".encode())
        output.flush()
        process = subprocess.Popen(  # noqa: S603 - fixed local command
            command, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
            creationflags=BACKGROUND_FLAGS, close_fds=True)
    return process.pid


def start_if_needed(*, log_dir: Path | None = None, home: Path | None = None,
                    probe: Callable[[], bool] = port_open) -> str:
    """Synchronous start decision + spawn (no waiting for the port).

    Returns "not_installed" | "already_running" | "starting" (spawned now or
    by an earlier call less than ``RESPAWN_GUARD_S`` ago) | "failed".
    """
    global _spawned_at
    home = home or omniroute_home()
    if not is_installed(home):
        return "not_installed"
    with _lock:
        if _spawned_at is not None and time.monotonic() - _spawned_at < RESPAWN_GUARD_S:
            return "starting"
        if probe():
            return "already_running"
        command = find_command()
        if not command or not Path(command).is_file():
            log.warning("OmniRoute is set up but its command was not found; SAM uses Groq/Gemini directly")
            return "not_installed"
        env = {**os.environ, "OMNIROUTE_CLI_SKIP_REPO_ENV": "1"}
        try:
            pid = _spawn([command, "serve", "--no-open", "--no-tray"], home, env,
                         (log_dir or default_log_dir()) / "omniroute.log")
        except OSError as exc:
            log.error("could not start OmniRoute: %s", exc)
            return "failed"
        _spawned_at = time.monotonic()
        log.info("OmniRoute started (pid %s) from %s", pid, home)
        return "starting"


def start_early() -> str:
    """For ``SAM.pyw``: kick the gateway off before the app is built."""
    return start_if_needed()


def spawned_recently() -> bool:
    return _spawned_at is not None and time.monotonic() - _spawned_at < RESPAWN_GUARD_S


async def ensure_running(app: Any, *, wait_s: float = START_WAIT_S, poll_s: float = 0.5) -> str:
    """Make sure the gateway is up; returns "already_running" | "started" |
    "not_installed" | "failed". Publishes ``ComponentStatus("omniroute")``.
    Never stops OmniRoute."""
    if not _uses_gateway(app):
        # Without its client key SAM cannot use the gateway, so there is nothing
        # to start. This also keeps every test App (temp home, no keys) from
        # probing or launching the real gateway through App.start().
        app.publish_status("omniroute", "unconfigured", "no OmniRoute client key (LITELLM_API_KEY)")
        return "not_installed"
    url = _base_url(app)
    home = omniroute_home()
    if not is_installed(home):
        app.publish_status("omniroute", "unconfigured", "OmniRoute is not installed on this PC")
        return "not_installed"
    if not spawned_recently() and await asyncio.to_thread(is_running, url):
        app.publish_status("omniroute", "ok", "running")
        return "already_running"
    parts = urllib.parse.urlsplit(url)
    if parts.hostname not in ("127.0.0.1", "localhost"):
        # The gateway is configured on another machine: nothing to start here.
        app.publish_status("omniroute", "down", "remote gateway is not answering")
        return "failed"
    port = parts.port or PORT
    state = await asyncio.to_thread(start_if_needed, log_dir=app.config.log_dir, home=home,
                                    probe=lambda: port_open(HOST, port))
    if state == "already_running":
        # The port is open but HTTP did not answer a moment ago: give it time.
        state = "starting"
    if state != "starting":
        app.publish_status("omniroute", "down" if state == "failed" else "unconfigured", state)
        return state
    app.publish_status("omniroute", "degraded", "starting")
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if await asyncio.to_thread(is_running, url):
            app.publish_status("omniroute", "ok", "started")
            log.info("OmniRoute is answering at %s", url)
            return "started"
        await asyncio.sleep(poll_s)
    log.warning("OmniRoute did not answer within %.0f s; see %s and ~/.omniroute/logs",
                wait_s, app.config.log_dir / "omniroute.log")
    app.publish_status("omniroute", "down", "did not answer after start")
    return "failed"


def _uses_gateway(app: Any) -> bool:
    try:
        return bool(app.secrets.has("litellm_api_key"))
    except Exception:  # noqa: BLE001
        return False


def _base_url(app: Any) -> str:
    """Gateway root from the provider setting (``.../v1`` -> root)."""
    try:
        value = str(app.config.get("providers.omniroute.base_url") or DEFAULT_URL)
    except Exception:  # noqa: BLE001
        value = DEFAULT_URL
    value = value.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def _reset_for_tests() -> None:
    global _spawned_at
    _spawned_at = None


__all__ = ["is_running", "ensure_running", "start_if_needed", "start_early", "is_installed", "port_open",
           "find_command", "BACKGROUND_FLAGS", "DEFAULT_URL"]
