"""What the island says besides the state word: quota and listening notices.

Real use on 2026-09-24: when every free model quota was used up the island
only showed «بیردەکەمەوە» and then SAM said «ببورە، ئێستا ناتوانم پەیوەندی بە
مۆدێلەکانەوە بکەم...»; nothing told the user it would work again after 10:00.
The voice package now publishes ``VoiceNotice`` events (sam/voice/notices.py):

- ``closed``  -- listening closed by itself (push-to-talk window over);
- ``ignored`` -- speech that had no «سام» (always-listening);
- ``not_recognized`` -- the voiceprint did not accept a follow-up / barge-in
  (real use 2026-09-25: the user's own voice was rejected four times and the
  island said nothing useful). The engine sends it once per episode; the
  caption shows «دەنگەکەت نەناسرایەوە — کلیک بکە» and the status word stays
  «کلیک بکە» while listening/idle for ``HINT_S`` -- or until the user clicks
  (the click re-opens the owner's turn) or SAM takes the user's words (a user
  caption) -- so repeated tries are never silently ignored;
- ``quota``   -- Gemini voice resting (daily: until the Pacific-midnight reset);
- ``models``  -- no model could answer (with the reset time when Gemini's daily
  quota is the reason);
- ``enroll``  -- voice enrollment progress / result;
- ``voiceprint`` -- the voiceprint exists but its model cannot run, so every
  nearby voice is accepted until the enrollment is repeated;
- ``local`` / ``cloud`` -- the brain switched to the local Ollama model (every
  cloud rung resting/offline; sam/brain/llm_local.py) or back. While on the
  local brain the idle/thinking status word is «مێشکی ناوخۆیی».

Every notice is shown once as the caption line. A ``models`` notice also
replaces the idle status word with «سنووری ئەمڕۆ پڕە» until its reset time
(or 10 minutes when no reset time is known), or until SAM answers normally
again. Pure Python (no Qt): island.py asks ``status_override`` and ``tone``.
"""

from __future__ import annotations

import time
from typing import Any

STICKY_STATES = frozenset({"idle", "sleeping", "error"})
LOCAL_STATES = frozenset({"idle", "sleeping", "thinking"})
HINT_STATES = frozenset({"listening", "idle"})
MODELS_WORD = "سنووری ئەمڕۆ پڕە"
LOCAL_WORD = "مێشکی ناوخۆیی"
CLICK_WORD = "کلیک بکە"
DEFAULT_STICKY_S = 600.0
HINT_S = 20.0
TONES = {"closed": "system", "ignored": "system", "enroll": "system", "quota": "alert", "models": "danger",
         "voiceprint": "alert", "local": "alert", "cloud": "system", "not_recognized": "alert"}


class IslandHints:
    def __init__(self, clock: Any = time.time) -> None:
        self._clock = clock
        self._sticky_text = ""
        self._sticky_until = 0.0
        self._hint_until = 0.0
        self.last_kind = ""
        self.local_brain = False

    def on_notice(self, event: Any) -> tuple[str, str] | None:
        """(caption text, tone) to show for a ``VoiceNotice``-like event."""
        kind = str(getattr(event, "kind", "") or "")
        text = str(getattr(event, "text_ckb", "") or "")
        if not text:
            return None
        self.last_kind = kind
        if kind == "local":
            self.local_brain = True
        elif kind == "cloud":
            self.local_brain = False
        if kind == "models":
            until = float(getattr(event, "until", 0.0) or 0.0)
            self._sticky_text = MODELS_WORD
            self._sticky_until = until if until > self._clock() else self._clock() + DEFAULT_STICKY_S
        if kind == "not_recognized":
            self._hint_until = self._clock() + HINT_S
        return text, TONES.get(kind, "system")

    def on_answer(self, role: str, text: str) -> None:
        """A normal assistant reply: the models work again."""
        if role == "assistant" and text and "مۆدێلەکان" not in text:
            self.clear_sticky()

    def on_user_words(self) -> None:
        """SAM took the user's words (a user caption): the «کلیک بکە» hint is over."""
        self.clear_hint()

    def clear_sticky(self) -> None:
        self._sticky_text, self._sticky_until = "", 0.0

    def clear_hint(self) -> None:
        """The user clicked (the owner's turn is open again) or SAM took a request."""
        self._hint_until = 0.0

    @property
    def hint_active(self) -> bool:
        return self._clock() < self._hint_until

    def status_override(self, state: str) -> str | None:
        if state in HINT_STATES and self.hint_active:
            return CLICK_WORD
        if self._sticky_text and state in STICKY_STATES:
            if self._clock() < self._sticky_until:
                return self._sticky_text
            self.clear_sticky()
        if self.local_brain and state in LOCAL_STATES:
            return LOCAL_WORD
        return None


def is_notice(event: Any) -> bool:
    return type(event).__name__ == "VoiceNotice" and hasattr(event, "text_ckb")


__all__ = ["IslandHints", "is_notice", "MODELS_WORD", "LOCAL_WORD", "CLICK_WORD", "HINT_S"]
