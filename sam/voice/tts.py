"""Sorani text-to-speech for the cascade: Gemini 3.8 Flash-Lite TTS or KurdishTTS.

Gemini (``voice.tts_model`` = gemini-3.8-flash-lite-tts, GA 2026-09-22, lists
"Central Kurdish"; docs saved in the lead's scratchpad voice/speech-generation.txt):
``client.aio.interactions.create(model, input=[user_input text], response_format=
{"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
generation_config={"speech_config": [{"voice": NAME}]}, stream=True)`` yields
``step.delta`` events whose ``delta.type == "audio"`` carries base64 headerless
16-bit 24 kHz mono PCM. 3.8 TTS reads the text VERBATIM (stage directions would
be spoken), so delivery style only goes in ``speech_metadata.style`` and the
docs advise leaving it empty for voice agents ("Leave the per-turn style field
empty, or send one short constant string").

KurdishTTS (alternative): ``POST {base}/tts-stream`` JSON ``{text, speaker_id,
model_version, stream_format: "pcm"}`` -> raw 16-bit mono PCM at 24 kHz
(openapi 1.2.0). Free plan: 20,000 characters a month, 500 per request, free
Sorani speakers v4 ``sorani_1``/``sorani_986`` (v3 ``sorani_85``/``sorani_214``).
v1's reply of 552 characters was refused and went silent, so every request here
is <= ``voice.tts_max_chars`` (480) and the month's characters are counted.

Never stall (real use 2026-09-24 20:51: a 429 plus the SDK's own ~27 s retry
kept the island on «بیردەکەمەوە» for ~40 s): the Gemini client never retries
(genai_client.py); the first audio must arrive within ``voice.tts_first_audio_s``
(2.5 s) or the request is dropped; a 429 / timeout / 5xx rests Gemini TTS
(quota.py: until the Pacific-midnight reset for a daily quota, 60-180 s for a
per-minute one) and the router moves to KurdishTTS for this sentence and the
following ones at once. After the first audio, a gap of more than
``voice.tts_chunk_gap_s`` (3 s) ends the piece and rests Gemini too.

KurdishTTS gets the same treatment (adversarial review 2026-09-24,
stall_probe.py: a KurdishTTS request that was accepted but never answered held
every piece for the client's 30 s read timeout, and a network error did not
rest it, so the next piece waited again): no audio within
``voice.kurdishtts_tts_first_audio_s`` (5 s) or a gap that long between
chunks -> the piece ends and KurdishTTS rests 60 s; while every provider
rests, SAM's answer is shown as text (cascade.py says so once a minute).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from . import kurdish_http
from .genai_client import make_client
from .numbers_ckb import verbalize_numbers
from .quota import rests
from .speech_text import split_for_tts
from .stt import KURDISHTTS_BASE, month_prefix, month_units
from .tts_cache import PhraseCache

if TYPE_CHECKING:  # ~41 ms import (measured 2026-09-24): loaded on first request, not in register()
    import httpx

log = logging.getLogger("sam.voice.tts")


class TtsError(Exception):
    """kind: unconfigured | auth | quota | rate_limit | network | server | bad_request | empty."""

    def __init__(self, kind: str, message: str = "", *, provider: str = "", status: int | None = None) -> None:
        super().__init__(f"{provider or 'tts'} {kind}: {message}".strip())
        self.kind = kind
        self.provider = provider
        self.status = status


def _even(pcm: bytes, carry: bytes) -> tuple[bytes, bytes]:
    """Join a leftover odd byte with the next chunk; return (even bytes, new carry)."""
    data = carry + pcm
    cut = len(data) - (len(data) % 2)
    return data[:cut], data[cut:]


class GeminiTts:
    provider = "gemini"

    def __init__(self, app: Any, *, client_factory: Callable[[str], Any] | None = None) -> None:
        self.app = app
        self._client_factory = client_factory
        self._client: Any = None
        self._client_fp: str | None = None
        self._cool_until = 0.0
        self._send_store = True

    @property
    def model(self) -> str:
        return str(self.app.config.get("voice.tts_model", "gemini-3.8-flash-lite-tts"))

    REST = "gemini_tts"

    def configured(self) -> bool:
        return (self.app.secrets.has("gemini_api_key") and time.monotonic() >= self._cool_until
                and not rests(self.app).resting(self.REST))

    def resting(self) -> bool:
        """Has a key but rests after a quota / timeout / server error."""
        return self.app.secrets.has("gemini_api_key") and rests(self.app).resting(self.REST)

    def voice_id(self) -> str:
        """What the audio sounds like (phrase-cache key): model, voice, style."""
        return f"{self.model}|{self.app.config.get('voice.voice_name', 'Kore')}|" \
               f"{str(self.app.config.get('voice.tts_style', '') or '').strip()}"

    def _genai(self, key: str) -> Any:
        import hashlib
        fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]
        if self._client is None or fingerprint != self._client_fp:
            self._client = make_client(key, factory=self._client_factory)  # SDK retries OFF
            self._client_fp = fingerprint
        return self._client

    def request(self, text: str) -> dict[str, Any]:
        content: dict[str, Any] = {"type": "text", "text": text}
        style = str(self.app.config.get("voice.tts_style", "") or "").strip()
        if style:
            content["annotations"] = [{"type": "speech_metadata", "style": style}]
        body: dict[str, Any] = {
            "model": self.model,
            "input": [{"type": "user_input", "content": [content]}],
            "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
            "generation_config": {"speech_config": [{"voice": str(self.app.config.get("voice.voice_name", "Kore"))}]},
            "stream": True,
        }
        if self._send_store:
            body["store"] = False  # privacy: nothing to keep server-side (retried without it if refused)
        return body

    def _map_error(self, exc: Exception) -> TtsError:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        text = str(exc)
        if code == 429:
            rests(self.app).on_rate_limit(self.REST, exc)
            return TtsError("rate_limit", "429", provider=self.provider, status=429)
        if code in (401, 403) or (code == 400 and "api key" in text.lower()):
            self._cool_until = time.monotonic() + 600.0
            return TtsError("auth", str(code), provider=self.provider, status=code)
        import httpx

        if isinstance(exc, (httpx.HTTPError, OSError, asyncio.TimeoutError)):
            rests(self.app).on_transient(self.REST, "network")
            return TtsError("network", type(exc).__name__, provider=self.provider)
        kind = "bad_request" if isinstance(code, int) and 400 <= code < 500 else "server"
        if kind == "server":
            rests(self.app).on_transient(self.REST, "server")
        return TtsError(kind, self.app.redact(f"{type(exc).__name__}: {text}")[:200], provider=self.provider,
                        status=code if isinstance(code, int) else None)

    def _first_audio_s(self) -> float:
        try:
            return max(0.5, float(self.app.config.get("voice.tts_first_audio_s", 2.5)))
        except (TypeError, ValueError):
            return 2.5

    def _chunk_gap_s(self) -> float:
        try:
            return max(0.5, float(self.app.config.get("voice.tts_chunk_gap_s", 3.0)))
        except (TypeError, ValueError):
            return 3.0

    def _too_slow(self, text: str) -> TtsError:
        rests(self.app).on_transient(self.REST, "timeout")
        self._bump(text, error=True)
        return TtsError("timeout", f"no audio within {self._first_audio_s():g} s", provider=self.provider)

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        key = self.app.secrets.get("gemini_api_key")
        if not key:
            raise TtsError("unconfigured", provider=self.provider)
        client = self._genai(key)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._first_audio_s()   # hard deadline for the FIRST audio
        events = None
        for attempt in range(2):
            try:
                events = await asyncio.wait_for(client.aio.interactions.create(**self.request(text)),
                                                max(0.05, deadline - loop.time()))
                break
            except asyncio.TimeoutError:
                raise self._too_slow(text) from None
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if attempt == 0 and code == 400 and "store" in str(exc).lower():
                    self._send_store = False
                    continue
                self._bump(text, error=True)
                raise self._map_error(exc) from None
        carry = b""
        produced = 0
        iterator = events.__aiter__()  # type: ignore[union-attr]
        try:
            while True:
                try:
                    if produced:
                        # A stream that stalls mid-sentence must not hold the voice line
                        # (every later piece, alert and confirmation question waits on it).
                        event = await asyncio.wait_for(iterator.__anext__(), self._chunk_gap_s())
                    else:
                        event = await asyncio.wait_for(iterator.__anext__(), max(0.05, deadline - loop.time()))
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    raise self._too_slow(text) from None
                kind = getattr(event, "event_type", None)
                if kind == "error":
                    raise TtsError("server", str(getattr(event, "error", "") or "stream error")[:200],
                                   provider=self.provider)
                if kind != "step.delta":
                    continue
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) != "audio" or not getattr(delta, "data", None):
                    continue
                pcm, carry = _even(base64.b64decode(delta.data), carry)
                if pcm:
                    produced += len(pcm)
                    yield pcm
        except TtsError as exc:
            if exc.kind != "timeout":   # _too_slow already counted it
                self._bump(text, error=True)
            raise
        except Exception as exc:  # noqa: BLE001
            self._bump(text, error=True)
            raise self._map_error(exc) from None
        finally:
            closer = getattr(events, "aclose", None) or getattr(events, "close", None)
            if closer is not None:  # barge-in: stop the HTTP stream now, not at garbage collection
                try:
                    result = closer()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:  # noqa: BLE001
                    pass
        if not produced:
            self._bump(text, error=True)
            raise TtsError("empty", "no audio", provider=self.provider)
        self._bump(text)

    async def synthesize(self, text: str) -> bytes:
        """Whole 24 kHz PCM for ``text`` (self-test)."""
        parts = [chunk async for chunk in self.stream(text)]
        return b"".join(parts)

    def _bump(self, text: str, *, error: bool = False) -> None:
        try:
            self.app.db.bump_usage("gemini", self.model, kind="tts", errors=int(error),
                                   units=0.0 if error else float(len(text)))
        except Exception:  # noqa: BLE001
            log.debug("usage count failed", exc_info=True)

    async def aclose(self) -> None:
        self._client = None


class KurdishTts:
    provider = "kurdishtts"
    model = "tts-stream"

    def __init__(self, app: Any, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.app = app
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._exhausted_month: str | None = None
        self._warned_month: str | None = None

    def _setting(self, key: str, default: Any) -> Any:
        return self.app.config.get(key, default)

    def month_used(self) -> float:
        return month_units(self.app.db, self.provider, "tts")

    def budget(self) -> float:
        return float(self._setting("voice.kurdishtts_monthly_tts_chars", 20000))

    REST = "kurdishtts_tts"

    def configured(self) -> bool:
        if not self.app.secrets.has("kurdishtts_tts_api_key"):
            return False
        if self._exhausted_month == month_prefix() or rests(self.app).resting(self.REST):
            return False
        return self.month_used() < self.budget()

    def resting(self) -> bool:
        return self.app.secrets.has("kurdishtts_tts_api_key") and rests(self.app).resting(self.REST)

    def _first_audio_s(self) -> float:
        try:
            return max(0.5, float(self._setting("voice.kurdishtts_tts_first_audio_s", 5.0)))
        except (TypeError, ValueError):
            return 5.0

    def voice_id(self) -> str:
        return f"{self._setting('voice.kurdishtts_speaker', 'sorani_1')}|" \
               f"{self._setting('voice.kurdishtts_model_version', 'v4')}"

    def _http(self) -> httpx.AsyncClient:
        """The pool shared with KurdishTTS speech-to-text, kept alive 120 s (kurdish_http.py:
        the default 5 s cost a new TLS handshake on every spoken turn); a test
        transport gets its own client."""
        if self._transport is None:
            return kurdish_http.shared(self.app).client()
        if self._client is None:
            self._client = kurdish_http.new_client(self._transport)
        return self._client

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        import httpx

        key = self.app.secrets.get("kurdishtts_tts_api_key")
        if not key:
            raise TtsError("unconfigured", provider=self.provider)
        url = str(self._setting("voice.kurdishtts_base_url", KURDISHTTS_BASE)).rstrip("/") + "/tts-stream"
        body = {"text": text[: int(self._setting("voice.tts_max_chars", 480))],
                "speaker_id": str(self._setting("voice.kurdishtts_speaker", "sorani_1")),
                "model_version": str(self._setting("voice.kurdishtts_model_version", "v4")),
                "stream_format": "pcm"}
        carry = b""
        produced = 0
        wait_s = self._first_audio_s()
        # read = the longest wait for the first bytes AND between chunks (httpx per-request timeout).
        timeout = httpx.Timeout(wait_s, connect=min(5.0, wait_s))
        try:
            async with self._http().stream("POST", url, headers={"x-api-key": key}, json=body,
                                           timeout=timeout) as response:
                if response.status_code != 200:
                    raw = await response.aread()
                    status = response.status_code
                    kind = {401: "auth", 403: "quota", 429: "rate_limit"}.get(status,
                                                                              "server" if status >= 500 else "bad_request")
                    if kind == "quota":
                        self._exhausted_month = month_prefix()
                    if kind in ("server", "rate_limit"):
                        rests(self.app).on_transient(self.REST, kind)
                    self._bump(body["text"], error=True)
                    raise TtsError(kind, f"HTTP {status} {self.app.redact(raw[:160].decode('utf-8', 'replace'))}",
                                   provider=self.provider, status=status)
                ctype = response.headers.get("content-type", "")
                if "json" in ctype or "html" in ctype:
                    raw = await response.aread()
                    self._bump(body["text"], error=True)
                    raise TtsError("server", f"unexpected {ctype}", provider=self.provider)
                async for chunk in response.aiter_bytes():
                    pcm, carry = _even(chunk, carry)
                    if pcm:
                        produced += len(pcm)
                        yield pcm
        except TtsError:
            raise
        except httpx.TimeoutException as exc:
            self._bump(body["text"], error=True)
            rests(self.app).on_transient(self.REST, "timeout")
            kind = "timeout" if not produced else "stalled"
            raise TtsError(kind, f"{type(exc).__name__} after {wait_s:g} s", provider=self.provider) from None
        except httpx.HTTPError as exc:
            self._bump(body["text"], error=True)
            rests(self.app).on_transient(self.REST, "network")
            raise TtsError("network", type(exc).__name__, provider=self.provider) from None
        if not produced:
            raise TtsError("empty", "no audio", provider=self.provider)
        self._bump(body["text"])
        self._check_budget()

    def _bump(self, text: str, *, error: bool = False) -> None:
        try:
            self.app.db.bump_usage(self.provider, self.model, kind="tts", errors=int(error),
                                   units=0.0 if error else float(len(text)))
        except Exception:  # noqa: BLE001
            log.debug("usage count failed", exc_info=True)

    def _check_budget(self) -> None:
        used, budget = self.month_used(), self.budget()
        month = month_prefix()
        if budget > 0 and used >= 0.8 * budget and self._warned_month != month:
            self._warned_month = month
            detail = f"KurdishTTS: {int(used)}/{int(budget)} characters used this month"
            self.app.publish_status("kurdishtts", "degraded", detail)
            try:
                self.app.db.log_activity("voice", "kurdishtts_budget", ok=False, summary=detail, source="tts")
            except Exception:  # noqa: BLE001
                pass

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


CACHE_CHUNK = 4800  # 100 ms of 24 kHz PCM per write when replaying a cached phrase


class TtsRouter:
    """``voice.tts_provider`` first when configured, then the other one. A
    provider that fails BEFORE its first chunk is skipped for this piece; after
    audio started, a failure just ends the piece (never repeat words).

    Short fixed phrases (the brain's acknowledgements) replay from the
    persistent ``PhraseCache`` (sam/voice/tts_cache.py: measured 2.0 s saved
    on the first audio of tool-using turns with KurdishTTS)."""

    def __init__(self, app: Any, providers: dict[str, Any], *, cache: PhraseCache | None = None) -> None:
        self.app = app
        self.providers = providers
        self.cache = cache
        self.last_provider: str | None = None
        self.last_cached = False

    @classmethod
    def default(cls, app: Any) -> "TtsRouter":
        return cls(app, {"gemini": GeminiTts(app), "kurdishtts": KurdishTts(app)},
                   cache=PhraseCache(getattr(app, "db", None)))

    def register_phrases(self, phrases: Any) -> int:
        """Fixed phrases that may be served from / stored in the phrase cache."""
        return self.cache.register(phrases) if self.cache is not None else 0

    def order(self) -> list[Any]:
        first = str(self.app.config.get("voice.tts_provider", "kurdishtts"))
        names = [first] + [n for n in self.providers if n != first]
        return [self.providers[n] for n in names if n in self.providers]

    def configured(self) -> bool:
        return any(p.configured() for p in self.order())

    def resting_only(self) -> bool:
        """No provider can speak now, but one has a key and only rests (a
        quota / timeout rest), so the answer is shown as text for a while."""
        return not self.configured() and any(bool(getattr(p, "resting", lambda: False)())
                                             for p in self.providers.values())

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {name: {"configured": p.configured()} for name, p in self.providers.items()}
        if self.cache is not None:
            out["cache"] = self.cache.status()
        out["rests"] = rests(self.app).status()
        return out

    def max_chars(self) -> int:
        return int(self.app.config.get("voice.tts_max_chars", 480))

    def low_budget(self, share: float = 0.2) -> bool:
        """True when only KurdishTTS can speak and less than ``share`` of its
        monthly characters is left (Gemini TTS, when configured, has no such cap)."""
        gemini = self.providers.get("gemini")
        if gemini is not None and gemini.configured():
            return False
        kurdish = self.providers.get("kurdishtts")
        if kurdish is None or not hasattr(kurdish, "budget"):
            return False
        budget = float(kurdish.budget())
        return budget > 0 and float(kurdish.month_used()) >= budget * (1.0 - float(share))

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """PCM 24 kHz for ``text`` (split into <= max_chars requests). Numbers
        with decimals are spoken as Sorani words (numbers_ckb.py; setting voice.tts_verbalize_numbers
        decimals|all|off)."""
        mode = str(self.app.config.get("voice.tts_verbalize_numbers", "decimals") or "off")
        if mode in ("decimals", "all", "True"):
            text = verbalize_numbers(text, integers=mode == "all")
        for piece in split_for_tts(text, self.max_chars()) or []:
            async for chunk in self._stream_piece(piece):
                yield chunk

    def _voice_id(self, provider: Any) -> str:
        getter = getattr(provider, "voice_id", None)
        return str(getter()) if callable(getter) else ""

    async def _stream_piece(self, text: str, *, only: Any = None) -> AsyncIterator[bytes]:
        last: TtsError | None = None
        cache = self.cache if self.cache is not None and self.cache.cacheable(text) else None
        for provider in ([only] if only is not None else self.order()):
            if not provider.configured():
                continue
            voice = self._voice_id(provider)
            cached = cache.get(provider.provider, voice, text) if cache is not None else None
            if cached:
                self.last_provider, self.last_cached = provider.provider, True
                for offset in range(0, len(cached), CACHE_CHUNK):
                    yield cached[offset:offset + CACHE_CHUNK]
                return
            started = False
            collected: list[bytes] | None = [] if cache is not None else None
            try:
                async for chunk in provider.stream(text):
                    started = True
                    self.last_provider, self.last_cached = provider.provider, False
                    if collected is not None:
                        collected.append(chunk)
                    yield chunk
                # Reached only when the provider finished normally: a piece cut
                # short by a barge-in (generator closed) is never cached.
                if collected:
                    cache.put(provider.provider, voice, text, b"".join(collected))  # type: ignore[union-attr]
                return
            except TtsError as exc:
                log.warning("tts %s failed: %s", provider.provider, exc.kind)
                if started:
                    return
                last = exc
        raise last or TtsError("unconfigured", "no TTS provider configured", provider="tts")

    async def prewarm(self, phrases: Any, *, idle: Any = None, pause_s: float = 0.5) -> int:
        """Synthesize missing short phrases into the cache (background, one at
        a time). ``idle()`` -> False makes it wait, so it never competes with a
        real reply for the provider. Returns how many were added.

        Never with Gemini TTS: its free quota is a handful of requests a day,
        and on 2026-09-24 the self-test plus this prewarm used it up before
        the user's first sentence. Off by default (``voice.tts_prewarm``):
        phrases are cached the first time they are really spoken."""
        if self.cache is None:
            return 0
        phrases = list(phrases)
        self.cache.register(phrases)
        added = 0
        for text in phrases:
            if not self.cache.cacheable(text):
                continue
            provider = next((p for p in self.order() if p.configured() and p.provider != "gemini"), None)
            if provider is None:
                return added
            if self.cache.has(provider.provider, self._voice_id(provider), text):
                continue
            while idle is not None and not idle():
                await asyncio.sleep(pause_s)
            try:
                async for _chunk in self._stream_piece(text, only=provider):
                    pass
                added += 1
            except TtsError as exc:
                log.info("tts prewarm stopped: %s", exc.kind)
                return added
        return added

    async def aclose(self) -> None:
        for provider in self.providers.values():
            try:
                await provider.aclose()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["TtsRouter", "GeminiTts", "KurdishTts", "TtsError"]
