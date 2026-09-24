"""Paths, ``.env`` values and DB-backed settings with defaults.

- ``SAM_HOME`` (env var) is the folder holding ``.env`` and ``data/``; default is
  the repository root. Development runs use ``SAM_HOME=C:\\Users\\samit\\Desktop\\SAM-Agent``
  so the real ``.env``, key store and MT5/TradingView are used.
- ``.env`` is parsed into a private dict. Values are NEVER copied into
  ``os.environ`` (child processes such as PowerShell/VS Code must not inherit
  keys) and never logged. Names SAM 2 reads: LITELLM_BASE_URL, LITELLM_API_KEY
  (or LITELLM_MASTER_KEY), LITELLM_FAST_MODEL, LITELLM_STRONG_MODEL,
  LITELLM_VISION_MODEL, OPENROUTER_BASE_URL, GROQ_BASE_URL, SAM_DATA_DIR.
- Settings live in the ``settings`` table as JSON; ``DEFAULTS`` supplies every
  known key. Packages add their own keys with ``config.register_defaults``.
  Model IDs are settings (Google/Groq rename models often: Gemini 2.5 access
  was restricted on 2026-09-18, reports/reference-agents.json).
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_FILENAME = "sam2.sqlite3"
V1_DB_FILENAME = "sam.sqlite3"


def resolve_home(explicit: str | Path | None = None) -> Path:
    value = explicit or os.environ.get("SAM_HOME") or REPO_ROOT
    return Path(value).expanduser().resolve()


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal dotenv parser (KEY=VALUE, quotes, ``export``, # comments).

    Never logs or raises on content; a malformed line is skipped.
    """
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, _, value = line.partition("=")
        name = name.strip()
        if not name.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # Inline comment only when preceded by whitespace (URLs may hold '#').
            for marker in (" #", "\t#"):
                if marker in value:
                    value = value.split(marker, 1)[0].rstrip()
        values[name] = value
    return values


def _env_models(env: Callable[[str], str | None]) -> dict[str, str]:
    return {
        "fast": env("LITELLM_FAST_MODEL") or "sam-fast",
        "strong": env("LITELLM_STRONG_MODEL") or "sam-strong",
        "vision": env("LITELLM_VISION_MODEL") or "sam-vision",
    }


# The local brain (sam/brain/llm_ollama.py, llm_local.py, local_server.py):
# Ollama on this PC, the implicit last rung of every ladder. Measured
# 2026-09-24 on CPU (the Radeon 890M crashes Ollama's Vulkan loader): qwen3:8b
# answers a Sorani command in 2.6-9.2 s once warm (prompt cache), qwen3.5:4b in
# 13-19 s (no prefix reuse); a cold first answer ~1.5 min (busy PC).
LOCAL_BRAIN_DEFAULTS: dict[str, Any] = {
    "llm.local.enabled": True,
    "llm.local.model": "qwen3:8b",
    "llm.local.fallback_models": ["qwen3.5:4b"],   # used when the main model is not installed
    "llm.local.host": "127.0.0.1:11434",
    "llm.local.ollama_exe": "",        # empty = SAM_HOME/tools/ollama*/ollama.exe, then the Ollama install / PATH
    "llm.local.models_dir": "",        # empty = <data>/ollama-models when it exists, else Ollama's default
    "llm.local.keep_alive": "5m",      # the model stays loaded a few minutes after use
    "llm.local.num_ctx": 8192,         # SAM's prompt is ~4.1k tokens before history
    "llm.local.max_tokens": 1024,      # 8 tok/s on CPU: an unbounded reply could run for minutes
    "llm.local.timeout_s": 150,        # a cold first answer took 8 s load + 86 s prompt (busy PC)
    "llm.local.temperature": 0.3,
    "llm.local.think": False,          # Qwen3 would think before every tool call
    "llm.local.vision": False,         # qwen3:8b has no vision; images never go to the local rung
    "llm.local.stop_on_quit": True,    # only a server SAM started itself
    "llm.local.prewarm": True,         # load the model while the cloud rests, before the user waits for it
}


def build_defaults(env: Callable[[str], str | None]) -> dict[str, Any]:
    """Every known setting with its default. Model refs are "provider:model".

    Ladder order follows measurements: Groq gpt-oss-20b median 0.8 s for chat
    vs 8.0 s for OmniRoute sam-fast without reasoning limits
    (reports/audit-latency.json); gemini-3.5-flash-lite has ~500 RPD vs ~20 RPD
    for gemini-3.8-flash on the free tier (reports/computer-control.json).
    """
    om = _env_models(env)
    return {
        # --- app ---
        "app.timezone": "Asia/Baghdad",
        "app.user_name": "",
        "app.single_instance": True,
        # --- llm (brain/llm.py) ---
        "llm.ladder.chat": ["groq:openai/gpt-oss-20b", f"omniroute:{om['fast']}", "gemini:gemini-3.5-flash-lite"],
        "llm.ladder.strong": [f"omniroute:{om['strong']}", "gemini:gemini-3.5-flash-lite", "groq:openai/gpt-oss-120b"],
        "llm.ladder.vision": ["gemini:gemini-3.5-flash-lite", "groq:qwen/qwen3.8-27b", f"omniroute:{om['vision']}"],
        "llm.ladder.extract": ["gemini:gemini-3.5-flash-lite", f"omniroute:{om['fast']}", "groq:openai/gpt-oss-20b"],
        "llm.ladder.hard": ["gemini:gemini-3.8-flash", f"omniroute:{om['strong']}", "gemini:gemini-3.5-flash-lite"],
        # Natural Sorani wording. One live sample each (2026-09-24, same tool loop):
        # Groq gpt-oss-20b replied in 0.57 s but with broken Sorani; OmniRoute
        # sam-fast (Gemini Flash-Lite) in 5.0 s with natural Sorani.
        "llm.ladder.sorani": [f"omniroute:{om['fast']}", "gemini:gemini-3.5-flash-lite", "groq:openai/gpt-oss-120b"],
        # Stop using a model for the day before Google/Groq starts refusing it.
        "llm.daily_caps": {"gemini:gemini-3.8-flash": 18, "gemini:gemini-3.5-flash-lite": 450,
                           "gemini:gemini-3.1-flash-lite": 450, "groq:qwen/qwen3.8-27b": 900},
        "llm.cooldown_429_s": 60,
        "llm.cooldown_auth_s": 600,
        "llm.cooldown_down_s": 30,
        # A 5xx/timeout that took >= slow_failure_s is not retried on the same
        # rung and cools it for cooldown_slow_s (OmniRoute 503s took 34-39 s, a
        # gemini-3.1-flash-lite call timed out at 40 s; an overloaded rung stays slow).
        "llm.slow_failure_s": 8,
        "llm.cooldown_slow_s": 180,
        "llm.strike_decay_s": 1800,       # a rung with no failure for 30 min regains its place (brain/llm.py)
        "llm.timeout_s": 40,
        "llm.max_tokens": 4096,
        "llm.reasoning": "low",            # minimal|low|medium|high
        "llm.verify_on_start": True,
        "providers.groq.base_url": env("GROQ_BASE_URL") or "https://api.groq.com/openai/v1",
        "providers.openrouter.base_url": env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
        "providers.omniroute.base_url": env("LITELLM_BASE_URL") or "http://127.0.0.1:20128/v1",
        **LOCAL_BRAIN_DEFAULTS,
        # --- no-AI fast path (brain/fastpath.py): common commands without a model ---
        "brain.fastpath.enabled": True,
        # --- confirm (brain/confirm.py) ---
        "confirm.timeout_s": 20,
        # --- voice (sam/voice) ---
        "voice.engine": "auto",            # auto|live|cascade
        "voice.live_model": "gemini-3.8-live",
        "voice.live_fallback_model": "gemini-3.1-flash-live-preview",
        "voice.voice_name": "Kore",
        "voice.tts_provider": "gemini",    # gemini|kurdishtts
        "voice.tts_model": "gemini-3.8-flash-lite-tts",
        "voice.stt_provider": "kurdishtts",  # kurdishtts|gemini
        "voice.stt_fallback_model": "gemini-3.5-flash-lite",
        "voice.hotkey": "ctrl+alt+space",
        "voice.conversation_timeout_s": 45,
        "voice.always_listening": False,
        "voice.silence_ms": 600,
        "voice.watchdog_s": 5,
        "voice.input_device": None,
        "voice.output_device": None,
        "voice.selftest": None,            # last self-test result dict
        # --- conversation / worker / memory (sam/brain) ---
        "conversation.history_turns": 12,
        "conversation.max_tool_rounds": 6,
        "conversation.speak_typed_replies": False,
        "worker.max_steps": 25,
        "memory.extract_on_sleep": True,
        # --- hands (sam/hands) ---
        "hands.projects_dir": str(Path.home() / "SAM Projects"),
        "hands.vscode_path": str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) /
                                 "Programs" / "Microsoft VS Code" / "bin" / "code.cmd"),
        "hands.vision_daily_budget": 450,
        "hands.screen_act_max_steps": 12,
        "hands.powershell_timeout_s": 45,
        # --- trading (sam/trading) ---
        "trading.default_symbol": "XAUUSD",
        "trading.default_timeframes": ["H4", "H1", "M15"],
        "trading.tv_port": 9222,
        "trading.tv_aumid": "TradingView.Desktop_n534cwy3pjxzj!TradingView.Desktop",
        "trading.symbol_map": {},          # user overrides: {"XAUUSD": {"mt5": "XAUUSD", "tv": "OANDA:XAUUSD"}}
        "trading.monitor_interval_s": 2.0,
        "trading.min_rr": 1.5,
        # --- ui (sam/ui) ---
        "ui.island_pos": None,             # [x, y] after the user drags it
        "ui.font_family": "Vazirmatn",
        "ui.show_english": True,
        # --- migration ---
        "migrate.v1_done": False,
    }


class Config:
    """Paths + env + settings. Thread-safe. ``attach_db`` before get/set."""

    def __init__(self, home: str | Path | None = None, *, env_file: Path | None = None,
                 environ: dict[str, str] | None = None) -> None:
        self.home = resolve_home(home)
        self._environ = environ if environ is not None else os.environ
        self._env_file = parse_env_file(env_file or self.home / ".env")
        data_value = self.env_value("SAM_DATA_DIR")
        data_dir = Path(data_value).expanduser() if data_value else Path("data")
        self.data_dir = (data_dir if data_dir.is_absolute() else self.home / data_dir).resolve()
        self.db_path = self.data_dir / DB_FILENAME
        self.v1_db_path = self.data_dir / V1_DB_FILENAME
        self.workspace_dir = self.home / "workspace"
        # Logs live outside SAM_HOME: the dev SAM_HOME is v1's folder, where
        # SAM 2 may only create data/sam2.sqlite3.
        log_dir = self.env_value("SAM_LOG_DIR")
        self.log_dir = Path(log_dir) if log_dir else Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "SAM2" / "logs"
        self.defaults: dict[str, Any] = build_defaults(self.env_value)
        self._db: Any | None = None
        self._cache: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._on_change: Callable[[str, Any], None] | None = None

    # -- env -----------------------------------------------------------------
    def env_value(self, name: str) -> str | None:
        """Process environment first, then ``.env``. Never log the result."""
        value = self._environ.get(name)
        if value:
            return value
        value = self._env_file.get(name)
        return value or None

    def env_names(self) -> list[str]:
        """Names defined in .env (no values) -- for diagnostics."""
        return sorted(self._env_file)

    # -- settings ------------------------------------------------------------
    def attach_db(self, db: Any, on_change: Callable[[str, Any], None] | None = None) -> None:
        with self._lock:
            self._db = db
            self._cache.clear()
            self._on_change = on_change

    def register_defaults(self, defaults: dict[str, Any]) -> None:
        """Add a package's settings keys. Existing keys keep their default."""
        with self._lock:
            for key, value in defaults.items():
                self.defaults.setdefault(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._cache:
                return copy.deepcopy(self._cache[key])
            value: Any = None
            found = False
            if self._db is not None:
                row = self._db.query_one("SELECT value FROM settings WHERE key=?", (key,))
                if row is not None:
                    try:
                        value, found = json.loads(row["value"]), True
                    except (json.JSONDecodeError, TypeError):
                        found = False
            if not found:
                if key in self.defaults:
                    value = self.defaults[key]
                else:
                    return default
            self._cache[key] = value
            return copy.deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        """Persist a setting (JSON) and notify ``SettingsChanged``."""
        encoded = json.dumps(value, ensure_ascii=False)
        with self._lock:
            if self._db is None:
                raise RuntimeError("Config.set before attach_db")
            self._db.execute(
                "INSERT INTO settings(key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, encoded, time.time()))
            self._cache[key] = json.loads(encoded)
            callback = self._on_change
        if callback is not None:
            callback(key, value)

    def reset(self, key: str) -> None:
        with self._lock:
            if self._db is not None:
                self._db.execute("DELETE FROM settings WHERE key=?", (key,))
            self._cache.pop(key, None)
            callback = self._on_change
        if callback is not None:
            callback(key, self.get(key))

    def all(self) -> dict[str, Any]:
        """Every known setting with its effective value (no secrets live here)."""
        keys = set(self.defaults)
        if self._db is not None:
            keys.update(r["key"] for r in self._db.query("SELECT key FROM settings"))
        return {k: self.get(k) for k in sorted(keys)}


__all__ = ["Config", "resolve_home", "parse_env_file", "build_defaults", "REPO_ROOT", "DB_FILENAME",
           "LOCAL_BRAIN_DEFAULTS"]
