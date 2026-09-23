"""Starting and stopping the one n8n SAM installed, and nothing else.

This is deliberately not a process manager. It knows a single location, builds
a single command from constants, and will only ever stop a process it can
prove is that installation. Nothing about the command comes from a request, a
workflow, a model or a setting a model can reach: the frontend can say "start"
and "stop", and that is the whole vocabulary.

The reason for the care is that a generic "run this binary" endpoint on a
local-first desktop agent is a remote code execution feature with a friendly
name. So: fixed paths, fixed arguments, loopback-only binding, and an
ownership check before anything is killed.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx

# Where a SAM-managed n8n lives. An operator may relocate it with an
# environment variable before SAM starts; nothing reachable from the web
# surface, a workflow or a model can change it.
DEFAULT_ROOT = Path(os.getenv("LOCALAPPDATA") or Path.home() / ".local") / "SAM"
RUNTIME_ENV = "SAM_N8N_RUNTIME_DIR"
DATA_ENV = "SAM_N8N_DATA_DIR"

HOST = "127.0.0.1"
DEFAULT_PORT = 5678
# Measured, not guessed: a warm start of n8n 2.40.5 on Windows took just
# over three minutes to answer /healthz, because it brings up task runners
# and a workflow index before it listens. Returning STARTING is honest, but
# returning it for a start that was simply going to work is not useful.
START_TIMEOUT_SECONDS = 420.0
HEALTH_POLL_SECONDS = 2.0
STOP_TIMEOUT_SECONDS = 30.0

# The one setting key this module owns.
RECORD_SETTING_KEY = "n8n_runtime_record"


class RuntimeState(StrEnum):
    NOT_INSTALLED = "NOT_INSTALLED"
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    UNHEALTHY = "UNHEALTHY"
    PORT_CONFLICT = "PORT_CONFLICT"
    UNKNOWN_PROCESS = "UNKNOWN_PROCESS"


@dataclass(slots=True)
class RuntimeRecord:
    """Non-secret facts about the managed install. Never a key or a password."""

    managed: bool = True
    runtime_path: str = ""
    data_path: str = ""
    host: str = HOST
    port: int = DEFAULT_PORT
    version: str = ""
    pid: int | None = None
    last_started_at: str = ""
    last_stopped_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeStatus:
    state: RuntimeState
    detail: str = ""
    record: RuntimeRecord = field(default_factory=RuntimeRecord)
    url: str = ""
    healthy: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value, "detail": self.detail, "url": self.url,
            "healthy": self.healthy, **self.record.as_dict(),
        }


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


class ManagedN8nRuntime:
    """One installation, one command, one process SAM is willing to stop."""

    def __init__(self, database: Any, *, runtime_dir: Path | None = None,
                 data_dir: Path | None = None, port: int = DEFAULT_PORT) -> None:
        self.database = database
        self.runtime_dir = Path(runtime_dir or os.getenv(RUNTIME_ENV)
                                or DEFAULT_ROOT / "n8n-runtime")
        self.data_dir = Path(data_dir or os.getenv(DATA_ENV) or DEFAULT_ROOT / "n8n-data")
        self.port = int(port)

    # -- what is installed --------------------------------------------------
    @property
    def entrypoint(self) -> Path:
        return self.runtime_dir / "node_modules" / "n8n" / "bin" / "n8n"

    @property
    def package_json(self) -> Path:
        return self.runtime_dir / "node_modules" / "n8n" / "package.json"

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def installed(self) -> bool:
        return self.entrypoint.is_file()

    def version(self) -> str:
        try:
            import json

            return str(json.loads(self.package_json.read_text(encoding="utf-8")).get("version") or "")
        except Exception:  # noqa: BLE001 - a missing or broken package.json is "unknown"
            return ""

    # -- the persisted record ------------------------------------------------
    def _read_record(self) -> RuntimeRecord:
        try:
            stored = self.database.get_settings().get(RECORD_SETTING_KEY) or {}
        except Exception:  # noqa: BLE001 - status must not fail on storage
            stored = {}
        record = RuntimeRecord(
            runtime_path=str(self.runtime_dir), data_path=str(self.data_dir),
            host=HOST, port=self.port, version=self.version(),
        )
        if isinstance(stored, dict):
            for key in ("pid", "last_started_at", "last_stopped_at"):
                if stored.get(key) not in (None, ""):
                    setattr(record, key, stored[key])
        return record

    def _write_record(self, **changes: Any) -> RuntimeRecord:
        record = self._read_record()
        for key, value in changes.items():
            setattr(record, key, value)
        try:
            self.database.update_settings({RECORD_SETTING_KEY: {
                "pid": record.pid,
                "last_started_at": record.last_started_at,
                "last_stopped_at": record.last_stopped_at,
            }})
        except Exception:  # noqa: BLE001
            pass
        return record

    # -- process identity ----------------------------------------------------
    def _owned_process(self, pid: int | None) -> Any:
        """The process at `pid`, but only if it is demonstrably ours.

        Ownership is the command line naming this exact runtime directory.
        A PID alone proves nothing: the number gets reused, and killing by
        name would end somebody else's node.
        """
        if not pid:
            return None
        try:
            import psutil

            process = psutil.Process(int(pid))
            line = " ".join(process.cmdline() or [])
        except Exception:  # noqa: BLE001 - gone, denied, or psutil unavailable
            return None
        # The entrypoint, not the directory. A directory substring is only as
        # specific as whatever the operator put in SAM_N8N_RUNTIME_DIR -- set
        # to something broad, it would match unrelated processes and this
        # method decides what may be killed. The full path to n8n's own bin
        # script is the narrowest evidence available and is what the command
        # line actually contains.
        marker = str(self.entrypoint).replace("\\", "/").lower()
        if marker and marker in line.replace("\\", "/").lower():
            return process
        return None

    def _port_owner(self) -> Any:
        """Whatever is listening on the managed port, if anything."""
        try:
            import psutil

            for connection in psutil.net_connections(kind="inet"):
                if (connection.laddr and connection.laddr.port == self.port
                        and connection.status == psutil.CONN_LISTEN):
                    return psutil.Process(connection.pid) if connection.pid else None
        except Exception:  # noqa: BLE001 - psutil needs privileges for some sockets
            return None
        return None

    # -- health --------------------------------------------------------------
    def healthy(self, timeout: float = 4.0) -> bool:
        """Ready means *usable by SAM*, which is later than merely alive.

        n8n answers /healthz while it is still bringing the public API up, so
        a start that only waited for that reported RUNNING and then handed SAM
        a client that got 404s. The public API is what SAM actually needs, so
        that is what readiness means here. It is probed without a key on
        purpose: an unauthenticated 401 proves the route is mounted, and this
        check has no business handling a credential.
        """
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                if client.get(f"{self.url}/healthz").status_code != 200:
                    return False
                return client.get(f"{self.url}/api/v1/workflows").status_code in (200, 401, 403)
        except httpx.HTTPError:
            return False

    # -- status --------------------------------------------------------------
    def status(self) -> RuntimeStatus:
        record = self._read_record()
        if not self.installed():
            return RuntimeStatus(RuntimeState.NOT_INSTALLED, record=record, url=self.url,
                                 detail=f"No managed n8n is installed under {self.runtime_dir}.")
        healthy = self.healthy()
        owned = self._owned_process(record.pid)

        if healthy:
            if owned is not None:
                return RuntimeStatus(RuntimeState.RUNNING, record=record, url=self.url, healthy=True,
                                     detail=f"n8n {record.version} is answering on {self.url}.")
            listener = self._port_owner()
            if listener is not None and self._owned_process(listener.pid) is not None:
                record = self._write_record(pid=listener.pid)
                return RuntimeStatus(RuntimeState.RUNNING, record=record, url=self.url, healthy=True,
                                     detail="Reattached to the managed n8n already running.")
            # Something is serving this port and SAM cannot show it is the
            # managed install, so SAM will not claim it and will not stop it.
            return RuntimeStatus(RuntimeState.UNKNOWN_PROCESS, record=record, url=self.url, healthy=True,
                                 detail=(f"Something is answering on port {self.port} that SAM cannot "
                                         "identify as its managed n8n. SAM will not stop it."))
        if owned is not None:
            return RuntimeStatus(RuntimeState.UNHEALTHY, record=record, url=self.url,
                                 detail="The managed n8n process is alive but not answering yet.")
        listener = self._port_owner()
        if listener is not None:
            return RuntimeStatus(RuntimeState.PORT_CONFLICT, record=record, url=self.url,
                                 detail=(f"Port {self.port} is held by another process, so the managed "
                                         "n8n cannot start there."))
        return RuntimeStatus(RuntimeState.STOPPED, record=record, url=self.url,
                             detail=f"n8n {record.version} is installed and stopped.")

    # -- start ---------------------------------------------------------------
    def _environment(self) -> dict[str, str]:
        """The only configuration this process ever gets, built from constants."""
        return {
            **os.environ,
            "N8N_LISTEN_ADDRESS": HOST,   # loopback only: never 0.0.0.0
            "N8N_HOST": HOST,
            "N8N_PORT": str(self.port),
            "N8N_PROTOCOL": "http",
            "N8N_USER_FOLDER": str(self.data_dir),
            "N8N_DIAGNOSTICS_ENABLED": "false",
            "N8N_PERSONALIZATION_ENABLED": "false",
            "N8N_VERSION_NOTIFICATIONS_ENABLED": "false",
            "N8N_TEMPLATES_ENABLED": "false",
        }

    def start(self, *, timeout: float = START_TIMEOUT_SECONDS) -> RuntimeStatus:
        current = self.status()
        if current.state is RuntimeState.RUNNING:
            return current
        if current.state in (RuntimeState.NOT_INSTALLED, RuntimeState.PORT_CONFLICT,
                             RuntimeState.UNKNOWN_PROCESS):
            # Nothing to start, or something else owns the port. Either way
            # SAM does not get to force it.
            return current
        if current.state is RuntimeState.UNHEALTHY:
            # An owned process is already coming up. Launching a second one
            # only produces a process that dies on the taken port and a
            # STOPPED verdict for an instance that was about to work.
            return self._await_health(timeout, self._read_record(),
                                      "The managed n8n was already starting")
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

        # Fixed executable, fixed arguments. No shell, and nothing here comes
        # from a caller.
        command = ["node", str(self.entrypoint), "start"]
        try:
            process = subprocess.Popen(  # noqa: S603 - argument vector is constant
                command, cwd=str(self.runtime_dir), env=self._environment(),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            return RuntimeStatus(RuntimeState.STOPPED, record=self._read_record(), url=self.url,
                                 detail=f"Could not launch the managed n8n: {exc}")

        record = self._write_record(pid=process.pid, last_started_at=_now())
        return self._await_health(timeout, record, f"n8n {record.version} started", process=process)

    def _await_health(self, timeout: float, record: RuntimeRecord, started_note: str,
                      *, process: Any = None) -> RuntimeStatus:
        """Wait for the public API, not merely for a process that has not died."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.healthy():
                return RuntimeStatus(RuntimeState.RUNNING, record=record, url=self.url, healthy=True,
                                     detail=f"{started_note} on {self.url}.")
            if process is not None and process.poll() is not None:
                return RuntimeStatus(RuntimeState.STOPPED, record=self._write_record(pid=None),
                                     url=self.url,
                                     detail=f"The managed n8n exited during startup (code {process.returncode}).")
            if process is None and self._owned_process(record.pid) is None:
                return RuntimeStatus(RuntimeState.STOPPED, record=self._write_record(pid=None),
                                     url=self.url, detail="The managed n8n stopped while starting up.")
            time.sleep(HEALTH_POLL_SECONDS)
        return RuntimeStatus(RuntimeState.STARTING, record=record, url=self.url,
                             detail=(f"n8n was launched but had not answered within {int(timeout)}s. "
                                     "First runs migrate the database and can take longer."))

    # -- stop ----------------------------------------------------------------
    def stop(self, *, timeout: float = STOP_TIMEOUT_SECONDS) -> RuntimeStatus:
        record = self._read_record()
        process = self._owned_process(record.pid)
        if process is None:
            listener = self._port_owner()
            candidate = self._owned_process(listener.pid) if listener is not None else None
            if candidate is None:
                if listener is not None:
                    # Refusing is the feature. SAM stops what it started, and
                    # nothing it merely found.
                    return RuntimeStatus(
                        RuntimeState.UNKNOWN_PROCESS, record=record, url=self.url,
                        healthy=self.healthy(),
                        detail=(f"SAM cannot show that the process on port {self.port} is its managed "
                                "n8n, so it will not stop it."))
                return RuntimeStatus(RuntimeState.STOPPED, record=self._write_record(pid=None),
                                     url=self.url, detail="The managed n8n is not running.")
            process = candidate

        try:
            children = process.children(recursive=True)
            process.terminate()
            import psutil

            _gone, alive = psutil.wait_procs([process, *children], timeout=timeout)
            for survivor in alive:
                survivor.kill()
        except Exception as exc:  # noqa: BLE001 - already gone, or denied
            if self._owned_process(record.pid) is not None:
                return RuntimeStatus(RuntimeState.RUNNING, record=record, url=self.url, healthy=True,
                                     detail=f"The managed n8n could not be stopped: {type(exc).__name__}")
        record = self._write_record(pid=None, last_stopped_at=_now())
        return RuntimeStatus(RuntimeState.STOPPED, record=record, url=self.url,
                             detail="The managed n8n was stopped. Its data is preserved.")
