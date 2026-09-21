from __future__ import annotations

import json
import ctypes
import os
import threading

GLOBAL_ABORT_EVENT = threading.Event()
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def is_elevated_windows_process() -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


@dataclass(slots=True)
class Settings:
    """Runtime configuration. Secret values are deliberately excluded from public views."""

    project_root: Path = field(default_factory=_default_project_root)
    workspace_root: Path | None = None
    data_dir: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    default_provider: str = "auto"
    default_model: str = "qwen3.5:4b"
    model_mode: str = "AUTO"
    permission_mode: str = "guarded"
    ollama_base_url: str = "http://127.0.0.1:11434"
    litellm_base_url: str = "http://127.0.0.1:4000/v1"
    litellm_api_key: str | None = None
    litellm_fast_model: str = "sam-fast"
    litellm_strong_model: str = "sam-strong"
    litellm_vision_model: str = "sam-vision"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str | None = None
    openrouter_fast_model: str = "openrouter/auto"
    openrouter_strong_model: str = "openrouter/auto"
    openrouter_vision_model: str = "openrouter/auto"
    openrouter_http_referer: str | None = None
    openrouter_title: str = "SAM Local Agent"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"
    max_tool_iterations: int = 8
    command_timeout_seconds: int = 45
    max_file_bytes: int = 1_500_000
    max_output_chars: int = 24_000
    approval_ttl_minutes: int = 30
    daily_budget_usd: float = 2.0
    monthly_budget_usd: float = 30.0
    budget_warning_ratio: float = 0.8
    budget_hard_ratio: float = 1.0
    computer_control_enabled: bool = False
    screen_access_enabled: bool = False
    market_data_provider: str = "metatrader5"
    default_trading_theory: str = "default"
    minimum_rr: float = 1.5
    setup_monitor_interval_seconds: int = 10
    voice_mode: str = "PUSH_TO_TALK"
    voice_language: str = "ckb-IQ"
    voice_vad_threshold: float = 0.035
    voice_silence_ms: int = 800
    voice_wake_word: str = "SAM"
    # Which KurdishTTS voice speaks Sorani replies; blank picks the first Sorani
    # speaker the provider offers.
    sorani_speaker_id: str = ""
    local_stt_model: str = "small"
    local_tts_voice: str = ""
    allow_lan: bool = False
    allow_unsafe_system_actions: bool = False
    allow_cloud_secret_access: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["http://127.0.0.1:8765", "http://localhost:8765"])

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.workspace_root = Path(self.workspace_root or self.project_root / "workspace").expanduser().resolve()
        self.data_dir = Path(self.data_dir or self.project_root / "data").expanduser().resolve()
        self.permission_mode = self.permission_mode.strip().lower()
        if self.permission_mode not in {"guarded", "strict", "trusted"}:
            raise ValueError("SAM_PERMISSION_MODE must be guarded, strict, or trusted")
        self.model_mode = self.model_mode.strip().upper()
        if self.model_mode not in {"AUTO", "LOCAL_ONLY", "CLOUD_ONLY", "MANUAL"}:
            raise ValueError("SAM_MODEL_MODE must be AUTO, LOCAL_ONLY, CLOUD_ONLY, or MANUAL")
        self.voice_mode = self.voice_mode.strip().upper()
        if self.voice_mode not in {"PUSH_TO_TALK", "CONVERSATION", "ALWAYS_LISTENING", "WAKE_WORD"}:
            raise ValueError("SAM_VOICE_MODE is invalid")
        if self.minimum_rr <= 0:
            raise ValueError("SAM_MINIMUM_RR must be positive")

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or _default_project_root()).resolve()
        # Process/OS variables win over the optional local development file.
        load_dotenv(root / ".env", override=False)
        port = int(os.getenv("SAM_PORT", "8765"))
        workspace_value = Path(os.getenv("SAM_WORKSPACE", "workspace")).expanduser()
        data_value = Path(os.getenv("SAM_DATA_DIR", "data")).expanduser()
        workspace_path = workspace_value if workspace_value.is_absolute() else root / workspace_value
        data_path = data_value if data_value.is_absolute() else root / data_value
        origins_raw = os.getenv("SAM_CORS_ORIGINS", "")
        origins = [item.strip() for item in origins_raw.split(",") if item.strip()]
        return cls(
            project_root=root,
            workspace_root=workspace_path,
            data_dir=data_path,
            host=os.getenv("SAM_HOST", "127.0.0.1"),
            port=port,
            default_provider=os.getenv("SAM_PROVIDER", "auto").strip().lower(),
            default_model=os.getenv("SAM_MODEL", "qwen3.5:4b").strip(),
            model_mode=os.getenv("SAM_MODEL_MODE", "AUTO").strip().upper(),
            permission_mode=os.getenv("SAM_PERMISSION_MODE", "guarded").strip().lower(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            litellm_base_url=os.getenv("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1").rstrip("/"),
            litellm_api_key=os.getenv("LITELLM_MASTER_KEY") or os.getenv("LITELLM_API_KEY"),
            litellm_fast_model=os.getenv("LITELLM_FAST_MODEL", "sam-fast"),
            litellm_strong_model=os.getenv("LITELLM_STRONG_MODEL", "sam-strong"),
            litellm_vision_model=os.getenv("LITELLM_VISION_MODEL", "sam-vision"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_fast_model=os.getenv("OPENROUTER_FAST_MODEL", "openrouter/auto"),
            openrouter_strong_model=os.getenv("OPENROUTER_STRONG_MODEL", "openrouter/auto"),
            openrouter_vision_model=os.getenv("OPENROUTER_VISION_MODEL", "openrouter/auto"),
            openrouter_http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
            openrouter_title=os.getenv("OPENROUTER_TITLE", "SAM Local Agent"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            max_tool_iterations=int(os.getenv("SAM_MAX_TOOL_ITERATIONS", "8")),
            command_timeout_seconds=int(os.getenv("SAM_COMMAND_TIMEOUT", "45")),
            max_file_bytes=int(os.getenv("SAM_MAX_FILE_BYTES", "1500000")),
            max_output_chars=int(os.getenv("SAM_MAX_OUTPUT_CHARS", "24000")),
            approval_ttl_minutes=int(os.getenv("SAM_APPROVAL_TTL_MINUTES", "30")),
            daily_budget_usd=float(os.getenv("SAM_DAILY_BUDGET_USD", "2.0")),
            monthly_budget_usd=float(os.getenv("SAM_MONTHLY_BUDGET_USD", "30.0")),
            budget_warning_ratio=float(os.getenv("SAM_BUDGET_WARNING_RATIO", "0.8")),
            budget_hard_ratio=float(os.getenv("SAM_BUDGET_HARD_RATIO", "1.0")),
            computer_control_enabled=_env_bool("SAM_COMPUTER_CONTROL", False),
            screen_access_enabled=_env_bool("SAM_SCREEN_ACCESS", False),
            market_data_provider=os.getenv("SAM_MARKET_DATA_PROVIDER", "metatrader5").strip().lower(),
            default_trading_theory=os.getenv("SAM_DEFAULT_TRADING_THEORY", "default").strip().lower(),
            minimum_rr=float(os.getenv("SAM_MINIMUM_RR", "1.5")),
            setup_monitor_interval_seconds=int(os.getenv("SAM_MONITOR_INTERVAL", "10")),
            voice_mode=os.getenv("SAM_VOICE_MODE", "PUSH_TO_TALK").strip().upper(),
            voice_language=os.getenv("SAM_VOICE_LANGUAGE", "ckb-IQ"),
            voice_vad_threshold=float(os.getenv("SAM_VOICE_VAD_THRESHOLD", "0.035")),
            voice_silence_ms=int(os.getenv("SAM_VOICE_SILENCE_MS", "800")),
            voice_wake_word=os.getenv("SAM_VOICE_WAKE_WORD", "SAM"),
            sorani_speaker_id=os.getenv("SAM_SORANI_SPEAKER_ID", ""),
            local_stt_model=os.getenv("SAM_LOCAL_STT_MODEL", "small"),
            local_tts_voice=os.getenv("SAM_LOCAL_TTS_VOICE", ""),
            allow_lan=_env_bool("SAM_ALLOW_LAN", False),
            allow_unsafe_system_actions=_env_bool("SAM_ALLOW_UNSAFE_SYSTEM_ACTIONS", False),
            allow_cloud_secret_access=_env_bool("SAM_ALLOW_CLOUD_SECRET_ACCESS", False),
            cors_origins=origins or [f"http://127.0.0.1:{port}", f"http://localhost:{port}"],
        )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "sam.sqlite3"

    def prepare(self) -> None:
        if is_elevated_windows_process():
            raise PermissionError("SAM refuses to start as Administrator. Open a normal PowerShell window and start it again.")
        workspace = self.workspace_root.resolve(strict=False)
        data_dir = self.data_dir.resolve(strict=False)
        forbidden_roots = {Path(workspace.anchor), Path.home().resolve(strict=False)}
        for variable in ("SYSTEMROOT", "WINDIR", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA"):
            value = os.getenv(variable)
            if value:
                forbidden_roots.add(Path(value).resolve(strict=False))
        if workspace in forbidden_roots:
            raise ValueError("SAM_WORKSPACE must be a narrow project directory, not a drive, profile, or protected system root.")
        if data_dir == workspace or data_dir.is_relative_to(workspace) or workspace.is_relative_to(data_dir):
            raise ValueError("SAM_DATA_DIR and SAM_WORKSPACE must be separate, non-overlapping directories.")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for secret_name in ("openai_api_key", "openrouter_api_key", "litellm_api_key"):
            result.pop(secret_name, None)
        for key in ("project_root", "workspace_root", "data_dir"):
            result[key] = str(result[key])
        result["openai_configured"] = bool(self.openai_api_key)
        result["openrouter_configured"] = bool(self.openrouter_api_key)
        result["litellm_key_configured"] = bool(self.litellm_api_key)
        return result

    def save_public_overrides(self, values: dict[str, Any]) -> None:
        allowed = {
            "default_provider", "default_model", "model_mode", "openai_model", "permission_mode",
            "openrouter_fast_model", "openrouter_strong_model", "openrouter_vision_model",
            "max_tool_iterations", "command_timeout_seconds", "daily_budget_usd", "monthly_budget_usd",
            "computer_control_enabled", "screen_access_enabled", "default_trading_theory", "minimum_rr",
            "voice_mode", "voice_language", "voice_vad_threshold", "voice_silence_ms", "voice_wake_word",
            "sorani_speaker_id",
        }
        payload = {key: value for key, value in values.items() if key in allowed}
        (self.data_dir / "settings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
