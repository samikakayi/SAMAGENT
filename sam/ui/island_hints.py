"""What the island says besides the state word: quota and listening notices.

Real use on 2026-09-24: when every free model quota was used up the island
only showed «بیردەکەمەوە» and then SAM said «ببورە، ئێستا ناتوانم پەیوەندی بە
مۆدێلەکانەوە بکەم...»; nothing told the user it would work again after 10:00.
The voice package now publishes ``VoiceNotice`` events (sam/voice/notices.py):

- ``closed``  -- listening closed by itself (push-to-talk window over);
- ``ignored`` -- speech that was not the user's voice / had no «سام»;
- ``quota``   -- Gemini voice resting (daily: until the Pacific-midnight reset);
- ``models``  -- no model could answer (with the reset time when Gemini's daily
  quota is the reason);
- ``enroll``  -- voice enrollment progress / result;
- ``voiceprint`` -- the voiceprint exists but its model cannot run, so every
  nearby voice is accepted until the enrollment is repeated.

Every notice is shown once as the caption line. A ``models`` notice also
replaces the idle status word with «سنووری ئەمڕۆ پڕە» until its reset time
(or 10 minutes when no reset time is known), or until SAM answers normally
again. Pure Python (no Qt): island.py asks ``status_override`` and ``tone``.
"""

from __future__ import annotations

import time
from typing import Any

STICKY_STATES = frozenset({"idle", "sleeping", "error"})
MODELS_WORD = "سنووری ئەمڕۆ پڕە"
DEFAULT_STICKY_S = 600.0
TONES = {"closed": "system", "ignored": "system", "enroll": "system", "quota": "alert", "models": "danger",
         "voiceprint": "alert"}


class IslandHints:
    def __init__(self, clock: Any = time.time) -> None:
        self._clock = clock
        self._sticky_text = ""
        self._sticky_until = 0.0
        self.last_kind = ""

    def on_notice(self, event: Any) -> tuple[str, str] | None:
        """(caption text, tone) to show for a ``VoiceNotice``-like event."""
        kind = str(getattr(event, "kind", "") or "")
        text = str(getattr(event, "text_ckb", "") or "")
        if not text:
            return None
        self.last_kind = kind
        if kind == "models":
            until = float(getattr(event, "until", 0.0) or 0.0)
            self._sticky_text = MODELS_WORD
            self._sticky_until = until if until > self._clock() else self._clock() + DEFAULT_STICKY_S
        return text, TONES.get(kind, "system")

    def on_answer(self, role: str, text: str) -> None:
        """A normal assistant reply: the models work again."""
        if role == "assistant" and text and "مۆدێلەکان" not in text:
            self.clear_sticky()

    def clear_sticky(self) -> None:
        self._sticky_text, self._sticky_until = "", 0.0

    def status_override(self, state: str) -> str | None:
        if self._sticky_text and state in STICKY_STATES:
            if self._clock() < self._sticky_until:
                return self._sticky_text
            self.clear_sticky()
        return None


def is_notice(event: Any) -> bool:
    return type(event).__name__ == "VoiceNotice" and hasattr(event, "text_ckb")


__all__ = ["IslandHints", "is_notice", "MODELS_WORD"]
