"""The small interface LiveVoice and CascadeVoice use to talk back to the
VoiceEngine (state words for the island, activity for the conversation
window, fallbacks). ``RecordingHooks`` is a no-op implementation for tests
and standalone use."""

from __future__ import annotations

from typing import Protocol


class VoiceHooks(Protocol):
    def set_state(self, state: str, detail: str = "") -> None: ...
    def activity(self) -> None: ...
    def live_stalled(self, pcm: bytes, user_text: str, eos_at: float, published: bool) -> None: ...
    def live_failed(self, reason: str) -> None: ...
    async def speak_fallback(self, text: str, source: str) -> None: ...


class RecordingHooks:
    """Records every call (tests); does nothing else."""

    def __init__(self) -> None:
        self.states: list[tuple[str, str]] = []
        self.activities = 0
        self.stalls: list[tuple[bytes, str, float, bool]] = []
        self.failures: list[str] = []
        self.fallback_spoken: list[tuple[str, str]] = []

    def set_state(self, state: str, detail: str = "") -> None:
        self.states.append((state, detail))

    def activity(self) -> None:
        self.activities += 1

    def live_stalled(self, pcm: bytes, user_text: str, eos_at: float, published: bool) -> None:
        self.stalls.append((pcm, user_text, eos_at, published))

    def live_failed(self, reason: str) -> None:
        self.failures.append(reason)

    async def speak_fallback(self, text: str, source: str) -> None:
        self.fallback_spoken.append((text, source))

    @property
    def last_state(self) -> str | None:
        return self.states[-1][0] if self.states else None


__all__ = ["VoiceHooks", "RecordingHooks"]
