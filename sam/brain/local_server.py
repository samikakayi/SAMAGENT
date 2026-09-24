"""The local Ollama server behind SAM's last-resort brain: found, started on
demand, stopped on quit (only when SAM started it).

- SAM never starts Ollama at launch. The first request that needs the local
  brain (every cloud rung resting/failing/offline) or a prewarm starts
  ``ollama serve`` -- and only when nothing already listens on the host: SAM v1
  keeps its own ``ollama serve`` on 127.0.0.1:11434 and SAM 2 simply shares it.
- Hidden window: ``CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP``, never
  ``DETACHED_PROCESS`` (a detached console child is what made the v1 desktop
  launcher fail, commit 6148b85). Output goes to %LOCALAPPDATA%\\SAM2\\logs\\ollama.log.
- Paths are settings so the switch-over keeps working after v1's files move:
  ``llm.local.ollama_exe`` / ``llm.local.models_dir``; empty = found under
  SAM_HOME (``tools/ollama*/ollama.exe``, ``data/ollama-models``), then the
  normal Ollama install / PATH and Ollama's own model folder.
- GPU: this PC's Radeon 890M is not used. Ollama 0.33.1 drops integrated GPUs
  unless ``OLLAMA_IGPU_ENABLE=1``; with it, llama-server crashed while fitting
  either model to Vulkan memory (exit 0xe06d7363, "AMD driver is too old";
  measured 2026-09-24). The child therefore never gets that variable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("sam.local_brain")

DEFAULT_HOST = "127.0.0.1:11434"
START_WAIT_S = 20.0
# Variables of the parent that would change how the child uses the GPU or
# where it listens: SAM decides those itself.
_DROP_ENV = ("OLLAMA_IGPU_ENABLE", "OLLAMA_HOST", "OLLAMA_MODELS")


def split_host(host: str) -> tuple[str, int]:
    value = (host or DEFAULT_HOST).strip()
    for prefix in ("http://", "https://"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    value = value.rstrip("/")
    name, _, port = value.rpartition(":")
    if not name:
        return value or "127.0.0.1", 11434
    try:
        return name, int(port)
    except ValueError:
        return name, 11434


def is_listening(host: str, timeout_s: float = 0.3) -> bool:
    name, port = split_host(host)
    try:
        with socket.create_connection((name, port), timeout=timeout_s):
            return True
    except OSError:
        return False


class OllamaServer:
    """Find / start / stop ``ollama serve`` for SAM (thread-safe enough: one
    asyncio lock around starting)."""

    def __init__(self, config: Any, *, popen: Any = None, listening: Any = None) -> None:
        self.config = config
        self._popen = popen or subprocess.Popen
        self._listening = listening or is_listening
        self._proc: Any = None
        self._lock: asyncio.Lock | None = None
        self._log_file: Any = None
        self.started_by_sam = False
        self.last_error = ""

    # -- settings -------------------------------------------------------------------------------
    def _setting(self, key: str, default: Any) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:  # noqa: BLE001
            value = default
        return default if value in (None, "") else value

    @property
    def host(self) -> str:
        return str(self._setting("llm.local.host", DEFAULT_HOST))

    @property
    def base_url(self) -> str:
        name, port = split_host(self.host)
        return f"http://{name}:{port}"

    def _home(self) -> Path | None:
        home = getattr(self.config, "home", None)
        return Path(home) if home else None

    def exe_path(self) -> Path | None:
        """The ollama.exe to start: the setting, else SAM_HOME/tools/ollama*/
        (newest folder name first), else the per-user install, else PATH."""
        explicit = str(self._setting("llm.local.ollama_exe", "") or "")
        if explicit:
            path = Path(os.path.expandvars(explicit)).expanduser()
            return path if path.is_file() else None
        candidates: list[Path] = []
        home = self._home()
        if home is not None:
            tools = home / "tools"
            try:
                folders = sorted((p for p in tools.glob("ollama*") if p.is_dir()), key=lambda p: p.name, reverse=True)
            except OSError:
                folders = []
            candidates += [f / "ollama.exe" for f in folders]
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        found = shutil.which("ollama")
        return Path(found) if found else None

    def models_dir(self) -> Path | None:
        """OLLAMA_MODELS for a server SAM starts: the setting, else
        <data_dir>/ollama-models when it exists (v1 kept qwen3:8b and
        qwen3.5:4b there, 8.1 GB), else None = Ollama's own default."""
        explicit = str(self._setting("llm.local.models_dir", "") or "")
        if explicit:
            return Path(os.path.expandvars(explicit)).expanduser()
        data_dir = getattr(self.config, "data_dir", None)
        if data_dir:
            folder = Path(data_dir) / "ollama-models"
            if folder.is_dir():
                return folder
        return None

    def available(self) -> bool:
        """Something can answer: a server listens, or SAM can start one."""
        return self.listening() or self.exe_path() is not None

    def listening(self) -> bool:
        return bool(self._listening(self.host))

    def running_ours(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- start / stop -----------------------------------------------------------------------------
    def _env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _DROP_ENV}
        name, port = split_host(self.host)
        env["OLLAMA_HOST"] = f"{name}:{port}"
        models = self.models_dir()
        if models is not None:
            env["OLLAMA_MODELS"] = str(models)
        keep = str(self._setting("llm.local.keep_alive", "5m"))
        env.setdefault("OLLAMA_KEEP_ALIVE", keep)
        return env

    def _log_path(self) -> Path:
        log_dir = getattr(self.config, "log_dir", None)
        folder = Path(log_dir) if log_dir else Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "SAM2" / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "ollama.log"

    def _spawn(self, exe: Path) -> Any:
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        self._log_file = open(self._log_path(), "ab")  # noqa: SIM115 - kept open for the child's lifetime
        return self._popen([str(exe), "serve"], env=self._env(), cwd=str(exe.parent), stdin=subprocess.DEVNULL,
                           stdout=self._log_file, stderr=subprocess.STDOUT, creationflags=flags)

    async def ensure(self, *, wait_s: float = START_WAIT_S) -> bool:
        """True when a server answers on the host (starting one if needed)."""
        if await asyncio.to_thread(self.listening):
            return True
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if await asyncio.to_thread(self.listening):
                return True
            if not self.running_ours():
                exe = await asyncio.to_thread(self.exe_path)
                if exe is None:
                    self.last_error = "ollama.exe not found"
                    return False
                try:
                    self._proc = await asyncio.to_thread(self._spawn, exe)
                except OSError as exc:
                    self.last_error = f"start failed: {type(exc).__name__}"
                    log.warning("could not start ollama: %s", type(exc).__name__)
                    return False
                self.started_by_sam = True
                log.info("started ollama serve (pid %s) on %s", getattr(self._proc, "pid", "?"), self.host)
            ends = time.monotonic() + wait_s
            while time.monotonic() < ends:
                if await asyncio.to_thread(self.listening):
                    return True
                if not self.running_ours():
                    self.last_error = "ollama exited during start"
                    return False
                await asyncio.sleep(0.25)
            self.last_error = "ollama did not answer in time"
            return False

    async def stop(self) -> bool:
        """Stop the server only if SAM started it (never SAM v1's or the
        user's). Kills the process tree: ollama serve runs llama-server
        children that would otherwise keep the model in memory."""
        proc, self._proc = self._proc, None
        stopped = False
        if proc is not None and proc.poll() is None:
            stopped = await asyncio.to_thread(self._kill_tree, proc)
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None
        return stopped

    def _kill_tree(self, proc: Any) -> bool:
        pid = getattr(proc, "pid", None)
        if os.name == "nt" and pid:
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=10,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - already gone
            pass
        return True

    def status(self) -> dict[str, Any]:
        exe = self.exe_path()
        return {"host": self.host, "listening": self.listening(), "exe": str(exe) if exe else None,
                "models_dir": str(self.models_dir() or ""), "started_by_sam": self.started_by_sam and self.running_ours(),
                "last_error": self.last_error}


__all__ = ["OllamaServer", "is_listening", "split_host", "DEFAULT_HOST"]
