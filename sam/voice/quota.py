"""Rests for voice providers after a quota/rate error, and when quotas reset.

Tonight's failure (sam2.log 2026-09-24 20:51-20:59): Gemini TTS answered 429
to every request for 8 minutes; the old code rested it only 60 s, so almost
every sentence first waited for another 429 (plus the SDK's own ~27 s retry,
fixed in genai_client.py) before KurdishTTS spoke. Now:

- a DAILY quota 429 (the error names a per-day quota: ``PerDay`` /
  ``per_day`` / "per day") rests the provider until the next reset: Gemini's
  free-tier daily quotas reset at midnight Pacific time (= 10:00 in Iraq
  while California is on daylight time, 11:00 in winter);
- a per-minute 429 rests it for the server's ``retryDelay`` clamped to
  60-180 s; an unlabelled 429 rests 120 s, and a second one within 10 minutes
  of the first counts as daily (the free TTS quota is only a handful of
  requests a day);
- a timeout / 5xx rests it 60 s (the next sentence goes to KurdishTTS at once).

Rests are stored in setting ``voice.rests`` so a restart does not spend the
first request of the day on a known 429. A daily rest publishes a
``VoiceNotice`` for the island («سنووری ئەمڕۆی دەنگی Gemini پڕە — دوای
کاتژمێر ١٠ی بەیانی»).
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
import time
from typing import Any

from . import strings
from .notices import VoiceNotice

log = logging.getLogger("sam.voice.quota")

PACIFIC = "America/Los_Angeles"
LOCAL = "Asia/Baghdad"
SETTING = "voice.rests"
MINUTE_MIN_S, MINUTE_MAX_S, UNLABELLED_S, TRANSIENT_S = 60.0, 180.0, 120.0, 60.0
ESCALATE_WITHIN_S = 600.0

_DAILY = re.compile(r"per\s*day|perday|per_day|requestsperday|daily", re.IGNORECASE)
_MINUTE = re.compile(r"per\s*minute|perminute|per_minute", re.IGNORECASE)
_DELAY = re.compile(r"retry(?:[_ ]?delay)?[\"']?\s*[:=]?\s*[\"']?(\d+(?:\.\d+)?)\s*s\b|retry in (\d+(?:\.\d+)?)\s*s",
                    re.IGNORECASE)
_EASTERN = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def _zone(name: str) -> Any:
    from zoneinfo import ZoneInfo
    return ZoneInfo(name)


def next_pacific_midnight(now: float | None = None) -> float:
    """Unix time of the next 00:00 in California (Gemini's daily quota reset)."""
    now = time.time() if now is None else now
    local = _dt.datetime.fromtimestamp(now, _zone(PACIFIC))
    midnight = (local + _dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


def reset_time_ckb(when: float) -> str:
    """«کاتژمێر ١٠ی بەیانی» for a unix time, in Iraq time, Sorani digits."""
    local = _dt.datetime.fromtimestamp(when, _zone(LOCAL))
    hour = local.hour
    part = ("بەیانی" if 5 <= hour < 12 else "نیوەڕۆ" if hour == 12 else "دوانیوەڕۆ" if 13 <= hour < 17
            else "ئێوارە" if 17 <= hour < 20 else "شەو")
    h12 = hour % 12 or 12
    clock = f"{h12}" if local.minute == 0 else f"{h12}:{local.minute:02d}"
    return f"کاتژمێر {clock.translate(_EASTERN)}ی {part}"


def error_text(exc: BaseException) -> str:
    parts = [str(exc)]
    for attr in ("details", "message", "body", "response_json"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def classify_429(exc_or_text: Any) -> tuple[str, float | None]:
    """("daily" | "minute" | "unlabelled", retry delay in s or None)."""
    text = exc_or_text if isinstance(exc_or_text, str) else error_text(exc_or_text)
    match = _DELAY.search(text)
    delay = float(match.group(1) or match.group(2)) if match else None
    if _DAILY.search(text):
        return "daily", delay
    if _MINUTE.search(text):
        return "minute", delay
    return "unlabelled", delay


class ProviderRests:
    """Named rests ("gemini_tts", "gemini_stt", ...) persisted in ``voice.rests``."""

    def __init__(self, app: Any, *, clock: Any = time.time) -> None:
        self.app = app
        self._clock = clock
        self._last_429: dict[str, float] = {}

    def _all(self) -> dict[str, Any]:
        try:
            value = self.app.config.get(SETTING, {}) or {}
        except Exception:  # noqa: BLE001
            value = {}
        return dict(value) if isinstance(value, dict) else {}

    def until(self, name: str) -> float:
        entry = self._all().get(name)
        until = float(entry.get("until", 0.0)) if isinstance(entry, dict) else 0.0
        return until if until > self._clock() else 0.0

    def resting(self, name: str) -> bool:
        return self.until(name) > 0.0

    def reason(self, name: str) -> str:
        entry = self._all().get(name)
        return str(entry.get("reason", "")) if isinstance(entry, dict) and self.until(name) else ""

    def rest(self, name: str, *, seconds: float | None = None, until: float | None = None,
             reason: str = "", notify: bool = True) -> float:
        now = self._clock()
        until = float(until if until is not None else now + float(seconds or 0.0))
        rests = {k: v for k, v in self._all().items() if isinstance(v, dict) and float(v.get("until", 0)) > now}
        rests[name] = {"until": round(until, 1), "reason": reason, "at": round(now, 1)}
        try:
            self.app.config.set(SETTING, rests)
        except Exception:  # noqa: BLE001 - a rest that is not persisted still works in memory? no: log it
            log.warning("could not store the %s rest", name)
        log.info("voice provider %s rests %.0f s (%s)", name, until - now, reason)
        if notify and reason == "daily" and name.startswith("gemini"):
            self._notice("quota", strings.GEMINI_VOICE_DAILY.format(time=reset_time_ckb(until)), name, until)
        return until

    def on_rate_limit(self, name: str, exc: Any) -> float:
        """Rest ``name`` after a 429 (daily -> next reset; see module docstring)."""
        kind, delay = classify_429(exc)
        now = self._clock()
        previous = self._last_429.get(name)
        self._last_429[name] = now
        if kind == "unlabelled" and previous is not None and now - previous <= ESCALATE_WITHIN_S + UNLABELLED_S:
            kind = "daily"  # a second 429 right after a rest: the day's quota is gone
        if kind == "daily":
            return self.rest(name, until=next_pacific_midnight(now), reason="daily")
        if kind == "minute":
            seconds = min(MINUTE_MAX_S, max(MINUTE_MIN_S, delay or MINUTE_MIN_S))
        else:
            seconds = UNLABELLED_S
        until = self.rest(name, seconds=seconds, reason=kind)
        self._notice("quota", strings.GEMINI_VOICE_REST, name, until)
        return until

    def on_transient(self, name: str, reason: str = "timeout") -> float:
        return self.rest(name, seconds=TRANSIENT_S, reason=reason, notify=False)

    def _notice(self, kind: str, text: str, detail: str, until: float) -> None:
        try:
            self.app.bus.publish_threadsafe(VoiceNotice(kind=kind, text_ckb=text, detail=detail, until=until))
        except Exception:  # noqa: BLE001
            pass

    def status(self) -> dict[str, Any]:
        now = self._clock()
        return {k: {"until": v.get("until"), "reason": v.get("reason"), "left_s": round(float(v["until"]) - now)}
                for k, v in self._all().items() if isinstance(v, dict) and float(v.get("until", 0)) > now}


def rests(app: Any) -> ProviderRests:
    """One ``ProviderRests`` per app (TTS, STT and the engine share it)."""
    holder = getattr(app, "_voice_rests", None)
    if holder is None:
        holder = ProviderRests(app)
        try:
            app._voice_rests = holder  # noqa: SLF001 - a per-app cache, like kurdish_http.shared
        except Exception:  # noqa: BLE001 - an app object without attributes (tests)
            pass
    return holder


__all__ = ["ProviderRests", "rests", "classify_429", "next_pacific_midnight", "reset_time_ckb", "error_text",
           "PACIFIC", "SETTING"]
