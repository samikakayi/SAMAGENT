from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .routing_profiles import MAX_FREE_CANDIDATES

from .config import is_placeholder_model


class ConversationCreate(BaseModel):
    title: str = Field(default="New conversation", max_length=160)
    provider: Literal["auto", "ollama", "litellm", "openrouter", "openai"] = "auto"
    model: str | None = None


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=160)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("title cannot be blank")
        return normalized


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=100_000)
    conversation_id: str | None = None
    provider: Literal["auto", "ollama", "litellm", "openrouter", "openai"] | None = None
    model: str | None = None


class DirectToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None


class ApprovalDecision(BaseModel):
    decision: Literal["approved", "denied"]
    note: str = Field(default="", max_length=1000)


class MemoryCreate(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)
    tags: list[str] = Field(default_factory=list, max_length=30)
    importance: float = Field(default=0.5, ge=0, le=1)
    source_conversation_id: str | None = None
    domain: Literal["session", "user", "project", "trading", "task", "episodic"] = "user"


class SettingsUpdate(BaseModel):
    default_provider: Literal["auto", "ollama", "litellm", "openrouter", "openai"] | None = None
    default_model: str | None = None
    model_mode: Literal["AUTO", "LOCAL_ONLY", "CLOUD_ONLY", "MANUAL"] | None = None
    openai_model: str | None = None
    openrouter_fast_model: str | None = None
    openrouter_strong_model: str | None = None
    openrouter_vision_model: str | None = None
    # The real model a run switches to when the configured one is unavailable.
    # Blank clears it; the pair is checked together where the current values
    # are known, because a partial update sees only one half of it.
    fallback_model: str | None = Field(default=None, max_length=160)
    fallback_enabled: bool | None = None
    # Which models a run may reach. Absent leaves the stored value alone, so an
    # installation that never sets it keeps routing exactly as it did before.
    routing_profile: Literal["FREE", "BALANCED", "PREMIUM"] | None = None
    # Ordered "provider/model" references FREE and BALANCED try in this order.
    free_candidates: list[str] | None = Field(default=None, max_length=MAX_FREE_CANDIDATES)
    # Where SAM sends workflow imports. Only this value ever decides the
    # target; a workflow's own contents never do.
    n8n_base_url: str | None = Field(default=None, max_length=300)
    permission_mode: Literal["guarded", "strict", "trusted"] | None = None
    max_tool_iterations: int | None = Field(default=None, ge=1, le=20)
    command_timeout_seconds: int | None = Field(default=None, ge=3, le=300)
    daily_budget_usd: float | None = Field(default=None, ge=0, le=100000)
    monthly_budget_usd: float | None = Field(default=None, ge=0, le=1000000)
    computer_control_enabled: bool | None = None
    screen_access_enabled: bool | None = None
    default_trading_theory: str | None = Field(default=None, min_length=1, max_length=80)
    minimum_rr: float | None = Field(default=None, gt=0, le=20)
    voice_mode: Literal["PUSH_TO_TALK", "CONVERSATION", "ALWAYS_LISTENING", "WAKE_WORD"] | None = None
    voice_language: str | None = Field(default=None, min_length=2, max_length=32)
    voice_vad_threshold: float | None = Field(default=None, ge=0.001, le=1)
    voice_silence_ms: int | None = Field(default=None, ge=100, le=10000)
    voice_wake_word: str | None = Field(default=None, min_length=1, max_length=40)
    # Which KurdishTTS voice speaks Sorani replies; blank means the default.
    sorani_speaker_id: str | None = Field(default=None, max_length=80)

    @field_validator("free_candidates")
    @classmethod
    def check_free_candidates(cls, value: list[str] | None) -> list[str] | None:
        """Reject a malformed entry rather than dropping it silently.

        A candidate the operator cannot see was discarded is worse than an
        error: FREE would quietly have one fewer model than they configured.
        """
        if value is None:
            return value
        cleaned: list[str] = []
        for item in value:
            provider, _, model = str(item).partition("/")
            if not provider.strip() or not model.strip():
                raise ValueError(f"{item!r} is not a provider/model reference")
            reference = f"{provider.strip().lower()}/{model.strip()}"
            if reference not in cleaned:
                cleaned.append(reference)
        return cleaned

    @field_validator("fallback_model")
    @classmethod
    def check_fallback_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            return ""
        if not re.fullmatch(r"[A-Za-z0-9._:/-]+", normalized):
            raise ValueError("fallback_model must be a model identifier such as vendor/model:tag")
        if is_placeholder_model(normalized):
            raise ValueError("fallback_model must name a real model, not a test double")
        return normalized

    def provided(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class TradingAnalysisRequest(BaseModel):
    symbol: str = Field(default="XAUUSD", min_length=2, max_length=64, pattern=r"^[A-Za-z0-9._:/-]+$")
    timeframes: list[str] = Field(default_factory=lambda: ["H1", "M15", "M5", "M1"], min_length=1, max_length=8)
    theories: list[str] = Field(default_factory=lambda: ["default"], min_length=1, max_length=8)
    count: int = Field(default=600, ge=100, le=5000)
    minimum_rr: float | None = Field(default=None, gt=0, le=20)


class MarketSnapshotRequest(BaseModel):
    symbol: str = Field(default="XAUUSD", min_length=2, max_length=64, pattern=r"^[A-Za-z0-9._:/-]+$")
    timeframes: list[str] = Field(default_factory=lambda: ["M1", "M5", "M15", "H1"], min_length=1, max_length=8)


class TradingViewActionRequest(BaseModel):
    action: Literal[
        "launch", "focus", "set_symbol", "set_timeframe", "capture",
        "calibrate", "auto_calibrate", "verify_calibration",
    ]
    symbol: str | None = Field(default=None, min_length=2, max_length=64, pattern=r"^[A-Za-z0-9._:/-]+$")
    timeframe: str | None = Field(default=None, min_length=1, max_length=16)
    price_a: float | None = None
    y_a: float | None = None
    price_b: float | None = None
    y_b: float | None = None


class CustomTheoryCreate(BaseModel):
    definition: dict[str, Any]


class SetupMonitorRequest(BaseModel):
    enabled: bool


class SetupCreateRequest(BaseModel):
    theory: str = Field(default="default", min_length=1, max_length=120)


class JournalCreate(BaseModel):
    symbol: str = Field(min_length=2, max_length=64)
    theory: str = Field(min_length=1, max_length=120)
    setup_id: str | None = None
    payload: dict[str, Any]


class DrawLevelRequest(BaseModel):
    annotation: str = Field(min_length=2, max_length=32)
    price: float
    # When given, the chart must be showing this instrument before anything is drawn.
    symbol: str | None = Field(default=None, max_length=64)
    label: str = Field(default="", max_length=64)
    theory: str = Field(default="", max_length=64)
    setup_id: str | None = None
    layer: str | None = None


class DrawAnalysisRequest(BaseModel):
    theory: str = Field(default="", max_length=64)
    setup_id: str | None = None


class ClearDrawingsRequest(BaseModel):
    symbol: str | None = Field(default=None, max_length=64)
    layer: str | None = None
    theory: str | None = Field(default=None, max_length=64)
    setup_id: str | None = None
    all_owned: bool = False


class ChartLayerRequest(BaseModel):
    layer: str
    visible: bool
    symbol: str | None = Field(default=None, max_length=64)


class VoiceListenRequest(BaseModel):
    max_seconds: float = Field(default=12.0, ge=1.0, le=60.0)
    device: int | None = Field(default=None, ge=0)
    language: str | None = Field(default=None, max_length=16)


class VoiceSpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: str | None = Field(default=None, max_length=16)


class BacktestRequest(BaseModel):
    symbol: str = Field(default="XAUUSD", min_length=2, max_length=64, pattern=r"^[A-Za-z0-9._:/-]+$")
    timeframe: str = Field(default="M15", min_length=1, max_length=16)
    trigger: str | None = Field(default=None, max_length=64)
    count: int = Field(default=3000, ge=300, le=20000)
    stop_atr_multiple: float = Field(default=1.5, gt=0, le=10)
    reward_multiple: float = Field(default=2.0, gt=0, le=20)
    max_bars: int = Field(default=60, ge=5, le=500)


class TwoAnchorDrawRequest(BaseModel):
    annotation: str = Field(min_length=2, max_length=32)
    price_a: float
    minutes_a: float = Field(ge=0, le=1440)
    price_b: float
    minutes_b: float = Field(ge=0, le=1440)
    label: str = Field(default="", max_length=64)
    theory: str = Field(default="", max_length=64)
    setup_id: str | None = None
    layer: str | None = None


class CredentialMetadata(BaseModel):
    """The non-secret facts about a key, which only its issuer knows.

    n8n's public API has no endpoint for reading an API key's own scopes or
    expiry, so the moment the operator pastes the key is the only honest
    opportunity to learn them. Recorded separately from the value, and only
    these three fields -- anything else offered is ignored.
    """

    scopes: list[str] | None = Field(default=None, max_length=200)
    created_at: str | None = Field(default=None, max_length=40)
    expires_at: str | None = Field(default=None, max_length=40)


class CredentialRequest(BaseModel):
    name: Literal["openrouter_api_key", "openai_api_key", "litellm_api_key",
                  "groq_api_key", "gemini_api_key", "n8n_api_key"]
    # The value is accepted, never echoed. Responses carry only a fingerprint.
    value: str = Field(min_length=8, max_length=400)
    metadata: CredentialMetadata | None = None


class GannDrawRequest(BaseModel):
    symbol: str = Field(default="XAUUSD", max_length=64, pattern=r"^[A-Za-z0-9._:/-]+$")
    timeframe: str = Field(default="M15", max_length=16)
    max_rays: int = Field(default=5, ge=1, le=9)
    setup_id: str | None = None


class PitchforkDrawRequest(BaseModel):
    symbol: str = Field(default="XAUUSD", max_length=64, pattern=r"^[A-Za-z0-9._:/-]+$")
    timeframe: str = Field(default="M15", max_length=16)
    variant: Literal["andrews", "schiff", "modified_schiff"] = "andrews"
    setup_id: str | None = None


class ReplayStartRequest(BaseModel):
    symbol: str = Field(default="XAUUSD", max_length=64, pattern=r"^[A-Za-z0-9._:/-]+$")
    timeframe: str = Field(default="M15", max_length=16)
    count: int = Field(default=1500, ge=100, le=20000)
    start_offset: int = Field(default=300, ge=50, le=19000)
    session_id: str = Field(default="default", max_length=64)


class ReplayControlRequest(BaseModel):
    action: Literal["advance", "pause", "resume", "stop", "status", "report"]
    session_id: str = Field(default="default", max_length=64)
    bars: int = Field(default=1, ge=1, le=1000)


class ReplayScanRequest(BaseModel):
    trigger: str = Field(min_length=2, max_length=64)
    session_id: str = Field(default="default", max_length=64)
    bars: int = Field(default=200, ge=1, le=5000)
    stop_atr_multiple: float = Field(default=1.5, gt=0, le=10)
    reward_multiple: float = Field(default=2.0, gt=0, le=20)


class ComposedStrategyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    context: str = Field(default="", max_length=200)
    setup: str = Field(default="", max_length=200)
    confirmation: str = Field(default="", max_length=200)
    entry_trigger: str = Field(default="", max_length=64)
    invalidation: str = Field(default="", max_length=200)
    stop: str = Field(default="", max_length=200)
    targets: list[str] = Field(default_factory=list, max_length=8)
    timeframes: list[str] = Field(default_factory=list, max_length=8)
    minimum_rr: float = Field(default=1.5, gt=0, le=50)
    direction: Literal["LONG", "SHORT", "BOTH"] = "BOTH"


class AgentTaskCreate(BaseModel):
    """Start an autonomous run."""

    goal: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = None
    constraints: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("goal")
    @classmethod
    def normalize_goal(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("goal cannot be blank")
        return normalized


class AgentTaskApproval(BaseModel):
    approval_id: str = Field(min_length=1, max_length=120)
    decision: Literal["approved", "denied"]
    note: str = Field(default="", max_length=2_000)
