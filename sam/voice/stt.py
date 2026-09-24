"""Sorani speech-to-text for the cascade: KurdishTTS first, Gemini fallback.

KurdishTTS (port of v1 sam_backend/sorani.py, contract verified live by v1):
``POST {base}/stt-proxy`` multipart ``file`` (WAV) + ``dialect=sorani``, header
``x-api-key``. Free plan: 2 h of STT a month, transcripts cut at 500
characters, 403 past quota (kurdishtts.com/pricing, fetched 2026-09-24).
v1 measured: ~0.5 s utterances come back empty about 3 times in 4, ~1 s
utterances are recognised reliably; so short utterances are padded with
silence to 1 s (cheap; whether padding helps is NOT measured).

Gemini (fallback, only with a Gemini key): the utterance goes inline as WAV
to ``voice.stt_fallback_model`` (gemini-3.5-flash-lite, ~500 RPD free) with a
strict verbatim-Sorani prompt. Google's dedicated STT models list no Kurdish
(reports/realtime-voice.json), so this general model is the only Google path;
its Sorani accuracy is undocumented.

Gemini STT never retries inside the SDK (genai_client.py), must answer within
``voice.stt_gemini_timeout_s`` (8 s), and a 429 rests it (quota.py) instead of
being asked again for the next utterance.

KurdishTTS STT must answer within ``voice.kurdishtts_stt_timeout_s`` (6 s)
plus the utterance's length (adversarial review 2026-09-24, stall_probe.py: a
request that was accepted but never answered cost the client's 30 s read
timeout before the Gemini fallback). A timeout, network error or 5xx rests it
60 s: the router then asks the other provider FIRST (a resting provider is
still tried last, so SAM never stops listening because of a rest).

One persistent ``httpx.AsyncClient`` per provider: a new TLS connection costs
0.25-0.35 s on this network (measured 2026-09-24, reports/realtime-voice.json).
Keys are read at call time and sent only to their own provider; errors never
carry them.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from ..textnorm import normalize_ckb
from . import kurdish_http, strings
from .audio import MIC_RATE, pcm16_to_wav, pcm_seconds, silence
from .genai_client import make_client
from .notices import VoiceNotice
from .quota import rests

if TYPE_CHECKING:  # httpx costs ~41 ms to import (measured 2026-09-24): loaded on first request
    import httpx

log = logging.getLogger("sam.voice.stt")

KURDISHTTS_BASE = "https://www.kurdishtts.com/api"
MIN_RELIABLE_S = 1.0

GEMINI_STT_SYSTEM = (
    "You are a speech-to-text engine for Central Kurdish (Sorani) spoken in Iraqi Kurdistan. "
    "Output ONLY the verbatim transcript of the audio: exactly the words spoken, in the order spoken. "
    "Write Sorani in Kurdish Arabic script with the correct Kurdish letters (ە ێ ۆ ڕ ڵ ی ک), never Arabic ي or ك, "
    "never Latin letters for Kurdish words. Keep English words, app names and market symbols as spoken "
    "(for example TradingView, XAUUSD) in Latin letters. Never translate, answer, explain, summarise or correct "
    "the speaker. If the speaker speaks English, transcribe the English. If there is no intelligible speech, "
    "output nothing."
)


class SttError(Exception):
    """kind: unconfigured | auth | quota | rate_limit | network | server | bad_request."""

    def __init__(self, kind: str, message: str = "", *, provider: str = "", status: int | None = None,
                 retry_after: float | None = None) -> None:
        super().__init__(f"{provider or 'stt'} {kind}: {message}".strip())
        self.kind = kind
        self.provider = provider
        self.status = status
        self.retry_after = retry_after


@dataclass
class SttResult:
    text: str
    provider: str
    model: str = ""
    ms: float = 0.0
    audio_s: float = 0.0
    detected_dialect: str | None = None
    truncated: bool = False
    attempts: list[str] = field(default_factory=list)


def clean_transcript(text: str) -> str:
    """Unify Arabic ي/ك to Kurdish ی/ک and collapse spaces; keep case and punctuation."""
    return normalize_ckb(text or "", lower=False).strip().strip("\"'«»").strip()


def month_prefix(provider: str = "kurdishtts") -> str:
    """Iraq-time month key matching ``usage_counters.day`` (YYYY-MM)."""
    from zoneinfo import ZoneInfo
    zone = "America/Los_Angeles" if provider == "gemini" else "Asia/Baghdad"
    return _dt.datetime.now(ZoneInfo(zone)).strftime("%Y-%m")


def month_units(db: Any, provider: str, kind: str) -> float:
    """Units (seconds / characters) used this month for provider+kind."""
    if db is None:
        return 0.0
    try:
        value = db.scalar("SELECT COALESCE(SUM(units),0) FROM usage_counters WHERE provider=? AND kind=? "
                          "AND day LIKE ?", (provider, kind, month_prefix(provider) + "-%"))
        return float(value or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _status_kind(status: int) -> str:
    if status in (401,):
        return "auth"
    if status == 403:
        return "quota"      # KurdishTTS: plan inactive / credit exceeded (openapi 1.2.0)
    if status == 429:
        return "rate_limit"
    if status >= 500:
        return "server"
    return "bad_request"


class KurdishTtsStt:
    provider = "kurdishtts"
    model = "stt-proxy"

    def __init__(self, app: Any, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.app = app
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._exhausted_month: str | None = None

    def _setting(self, key: str, default: Any) -> Any:
        return self.app.config.get(key, default)

    REST = "kurdishtts_stt"

    def configured(self) -> bool:
        if not self.app.secrets.has("kurdishtts_stt_api_key"):
            return False
        return self._exhausted_month != month_prefix()

    def resting(self) -> bool:
        return rests(self.app).resting(self.REST)

    def _timeout_s(self, spoken_s: float) -> float:
        try:
            base = float(self._setting("voice.kurdishtts_stt_timeout_s", 6.0))
        except (TypeError, ValueError):
            base = 6.0
        return max(1.0, base) + max(0.0, spoken_s)

    def budget_left_s(self) -> float:
        budget = float(self._setting("voice.kurdishtts_monthly_stt_s", 7200))
        return budget - month_units(self.app.db, self.provider, "stt")

    def _http(self) -> httpx.AsyncClient:
        """The pool shared with KurdishTTS text-to-speech, kept alive 120 s (kurdish_http.py:
        the default 5 s cost a new TLS handshake on every spoken turn); a test
        transport gets its own client."""
        if self._transport is None:
            return kurdish_http.shared(self.app).client()
        if self._client is None:
            self._client = kurdish_http.new_client(self._transport)
        return self._client

    async def transcribe(self, pcm: bytes, rate: int = MIC_RATE) -> SttResult:
        import httpx

        key = self.app.secrets.get("kurdishtts_stt_api_key")
        if not key:
            raise SttError("unconfigured", provider=self.provider)
        if self.budget_left_s() <= 0:
            raise SttError("quota", "monthly STT budget used", provider=self.provider)
        spoken = pcm_seconds(pcm, rate)
        if spoken < MIN_RELIABLE_S:
            pcm = pcm + silence((MIN_RELIABLE_S - spoken) * 1000, rate)
        seconds = max(spoken, MIN_RELIABLE_S)  # count what is uploaded (the provider bills the file)
        wav = pcm16_to_wav(pcm, rate)
        url = str(self._setting("voice.kurdishtts_base_url", KURDISHTTS_BASE)).rstrip("/") + "/stt-proxy"
        started = time.perf_counter()
        wait_s = self._timeout_s(seconds)
        try:
            response = await self._http().post(url, headers={"x-api-key": key},
                                               files={"file": ("audio.wav", wav, "audio/wav")},
                                               data={"dialect": "sorani"},
                                               timeout=httpx.Timeout(wait_s, connect=min(5.0, wait_s)))
        except httpx.TimeoutException as exc:
            self._count(seconds, error=True)
            rests(self.app).on_transient(self.REST, "timeout")
            raise SttError("network", f"{type(exc).__name__} after {wait_s:.1f} s", provider=self.provider) from None
        except httpx.HTTPError as exc:
            self._count(seconds, error=True)
            rests(self.app).on_transient(self.REST, "network")
            raise SttError("network", type(exc).__name__, provider=self.provider) from None
        ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            kind = _status_kind(response.status_code)
            if kind == "server":
                rests(self.app).on_transient(self.REST, "server")
            if kind == "quota":
                self._exhausted_month = month_prefix()
                try:
                    self.app.bus.publish_threadsafe(VoiceNotice(kind="quota", text_ckb=strings.KURDISH_STT_MONTH,
                                                                detail="kurdishtts_stt"))
                except Exception:  # noqa: BLE001
                    pass
            self._count(seconds, error=True, rate_limited=kind == "rate_limit")
            detail = self.app.redact(response.text[:160])
            raise SttError(kind, f"HTTP {response.status_code} {detail}", provider=self.provider,
                           status=response.status_code)
        try:
            payload = response.json()
        except ValueError:
            self._count(seconds, error=True)
            raise SttError("server", "response was not JSON", provider=self.provider) from None
        if not isinstance(payload, dict):
            raise SttError("server", "unexpected response shape", provider=self.provider)
        self._count(seconds)
        return SttResult(text=clean_transcript(str(payload.get("text") or "")), provider=self.provider,
                         model=self.model, ms=ms, audio_s=seconds,
                         detected_dialect=payload.get("detected_dialect"),
                         truncated=bool(payload.get("truncated")))

    def _count(self, seconds: float, *, error: bool = False, rate_limited: bool = False) -> None:
        try:
            self.app.db.bump_usage(self.provider, self.model, kind="stt", errors=int(error),
                                   rate_limited=int(rate_limited), units=0.0 if error else round(seconds, 2))
        except Exception:  # noqa: BLE001
            log.debug("usage count failed", exc_info=True)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class GeminiStt:
    """Inline-audio transcription with a Gemini text model (fallback)."""

    provider = "gemini"

    def __init__(self, app: Any, *, client_factory: Callable[[str], Any] | None = None) -> None:
        self.app = app
        self._client_factory = client_factory
        self._client: Any = None
        self._client_fp: str | None = None
        self._cool_until = 0.0

    @property
    def model(self) -> str:
        return str(self.app.config.get("voice.stt_fallback_model", "gemini-3.5-flash-lite"))

    REST = "gemini_stt"

    def configured(self) -> bool:
        return (self.app.secrets.has("gemini_api_key") and time.monotonic() >= self._cool_until
                and not rests(self.app).resting(self.REST))

    def _over_cap(self) -> bool:
        caps = self.app.config.get("llm.daily_caps", {}) or {}
        cap = caps.get(f"gemini:{self.model}")
        if not cap:
            return False
        used = (self.app.db.usage_for("gemini", self.model) or {}).get("requests", 0)
        return used >= int(cap)

    def _genai(self, key: str) -> Any:
        import hashlib
        fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]
        if self._client is None or fingerprint != self._client_fp:
            self._client = make_client(key, factory=self._client_factory)  # SDK retries OFF
            self._client_fp = fingerprint
        return self._client

    async def transcribe(self, pcm: bytes, rate: int = MIC_RATE) -> SttResult:
        key = self.app.secrets.get("gemini_api_key")
        if not key:
            raise SttError("unconfigured", provider=self.provider)
        if self._over_cap():
            raise SttError("quota", "daily cap reached", provider=self.provider)
        from google.genai import types

        seconds = pcm_seconds(pcm, rate)
        client = self._genai(key)
        contents = [types.Content(role="user", parts=[
            types.Part.from_bytes(data=pcm16_to_wav(pcm, rate), mime_type="audio/wav"),
            types.Part(text="Transcribe this audio verbatim."),
        ])]
        config_kwargs: dict[str, Any] = {"system_instruction": GEMINI_STT_SYSTEM, "temperature": 0.0,
                                         "max_output_tokens": 4096,
                                         "thinking_config": types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)}
        started = time.perf_counter()
        response = None
        timeout_s = float(self.app.config.get("voice.stt_gemini_timeout_s", 8.0) or 8.0)
        for attempt in range(2):
            try:
                response = await asyncio.wait_for(client.aio.models.generate_content(
                    model=self.model, contents=contents, config=types.GenerateContentConfig(**config_kwargs)),
                    max(0.5, timeout_s - (time.perf_counter() - started)))
                break
            except Exception as exc:  # noqa: BLE001 - mapped below
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                text = str(exc)
                if attempt == 0 and code == 400 and "think" in text.lower():
                    config_kwargs.pop("thinking_config", None)  # model refuses MINIMAL: retry with its default
                    continue
                self._bump(seconds, error=True, rate_limited=code == 429)
                if code == 429:
                    rests(self.app).on_rate_limit(self.REST, exc)
                    raise SttError("rate_limit", "429", provider=self.provider, status=429) from None
                if code in (401, 403) or (code == 400 and "api key" in text.lower()):
                    self._cool_until = time.monotonic() + 600.0
                    raise SttError("auth", str(code), provider=self.provider, status=code) from None
                import httpx

                if isinstance(exc, (httpx.HTTPError, OSError, asyncio.TimeoutError)):
                    rests(self.app).on_transient(self.REST, "timeout" if isinstance(exc, asyncio.TimeoutError)
                                                 else "network")
                    raise SttError("network", type(exc).__name__, provider=self.provider) from None
                if isinstance(code, int) and code >= 500:
                    rests(self.app).on_transient(self.REST, "server")
                raise SttError("server", self.app.redact(f"{type(exc).__name__}: {text}")[:200],
                               provider=self.provider, status=code if isinstance(code, int) else None) from None
        ms = (time.perf_counter() - started) * 1000.0
        text = ""
        try:
            text = response.text or ""  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - blocked/empty candidates
            text = ""
        usage = getattr(response, "usage_metadata", None)
        self._bump(seconds, tokens_in=int(getattr(usage, "prompt_token_count", 0) or 0),
                   tokens_out=int(getattr(usage, "candidates_token_count", 0) or 0))
        return SttResult(text=clean_transcript(text), provider=self.provider, model=self.model, ms=ms,
                         audio_s=seconds)

    def _bump(self, seconds: float, *, error: bool = False, rate_limited: bool = False, tokens_in: int = 0,
              tokens_out: int = 0) -> None:
        try:
            self.app.db.bump_usage("gemini", self.model, kind="stt", errors=int(error),
                                   rate_limited=int(rate_limited), tokens_in=tokens_in, tokens_out=tokens_out,
                                   units=0.0 if error else round(seconds, 2))
        except Exception:  # noqa: BLE001
            log.debug("usage count failed", exc_info=True)

    async def aclose(self) -> None:
        self._client = None


class SttRouter:
    """Provider order from ``voice.stt_provider``; fall back only when a
    provider FAILS (a surprising or empty transcript is not a failure -- v1
    rule: never re-run audio through another recogniser because of content)."""

    def __init__(self, app: Any, providers: dict[str, Any]) -> None:
        self.app = app
        self.providers = providers
        self.last_provider: str | None = None

    @classmethod
    def default(cls, app: Any) -> "SttRouter":
        return cls(app, {"kurdishtts": KurdishTtsStt(app), "gemini": GeminiStt(app)})

    def order(self) -> list[Any]:
        """The setting's provider first -- but a resting one (after a timeout /
        5xx) goes to the end: still tried when nothing else answers."""
        first = str(self.app.config.get("voice.stt_provider", "kurdishtts"))
        names = [first] + [n for n in self.providers if n != first]
        providers = [self.providers[n] for n in names if n in self.providers]
        awake = [p for p in providers if not bool(getattr(p, "resting", lambda: False)())]
        return awake + [p for p in providers if p not in awake]

    def configured(self) -> bool:
        return any(p.configured() for p in self.order())

    def status(self) -> dict[str, Any]:
        return {name: {"configured": p.configured()} for name, p in self.providers.items()}

    async def transcribe(self, pcm: bytes, rate: int = MIC_RATE) -> SttResult:
        attempts: list[str] = []
        last: SttError | None = None
        for provider in self.order():
            if not provider.configured():
                attempts.append(f"{provider.provider}:unconfigured")
                continue
            try:
                result = await provider.transcribe(pcm, rate)
            except SttError as exc:
                attempts.append(f"{provider.provider}:{exc.kind}")
                last = exc
                log.warning("stt %s failed: %s", provider.provider, exc.kind)
                continue
            result.attempts = attempts + [f"{provider.provider}:ok"]
            self.last_provider = provider.provider
            return result
        if last is None:
            raise SttError("unconfigured", "no STT provider configured", provider="stt")
        raise last

    async def aclose(self) -> None:
        for provider in self.providers.values():
            try:
                await provider.aclose()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["SttRouter", "KurdishTtsStt", "GeminiStt", "SttResult", "SttError", "clean_transcript",
           "month_units", "month_prefix", "GEMINI_STT_SYSTEM", "KURDISHTTS_BASE"]
