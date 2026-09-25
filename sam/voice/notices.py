"""Voice-package events for the UI (compatible additions to sam/events.py:
defined here, forwarded to the UI like every bus event by ``UiAdapter``).

- ``VoiceNotice``: a short Sorani line for the island -- listening closed,
  speech ignored (not the user / too far), a quota used up until a time.
- ``VoiceEnrollRequest``: the user asked (by voice) to record their
  voiceprint; the Settings voice card opens the enrollment dialog.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..events import Event


@dataclass(frozen=True, slots=True)
class VoiceNotice(Event):
    kind: str                 # closed | ignored | not_recognized | quota | models | enroll | voiceprint | local | cloud
    text_ckb: str
    detail: str = ""
    until: float = 0.0        # unix time the notice stays true (quota rests); 0 = transient


@dataclass(frozen=True, slots=True)
class VoiceEnrollRequest(Event):
    source: str = "voice"


__all__ = ["VoiceNotice", "VoiceEnrollRequest"]
