"""A closed-loop living room for the listening tests (no audio I/O, no network).

The adversarial review of 2026-09-24 found that fixed-window simulations hid
the real failure: an accepted TV utterance produced an answer, the answer
opened a follow-up window, and the next TV utterance was accepted -- 61 STT and
61 model calls in 300 s from 3 clicks. This harness closes that loop: the REAL
VoiceEngine, FramePipeline, ListeningPolicy, NearFieldGate, Endpointer and
CascadeVoice run on a VIRTUAL clock (the voice modules' ``time`` is replaced),
fed frame by frame; STT (1 s), the brain (1 s) and TTS (3 s of speech on a
virtual speaker) are fakes that take virtual time, so every answer really
reopens (or closes) listening.

Sound: square waves (the VAD is the energy stand-in of the listening tests).
The user speaks at -20 dBFS; the "TV" says 2 s sentences at -38..-42 dBFS with
quiet syllable gaps (-60 dBFS) and 0.9 s pauses, so its frames pass the gate
while the user's level is unknown -- the evening's situation.
"""

from __future__ import annotations

import array
import asyncio
import time as _real_time
from dataclasses import dataclass, field
from typing import Any

from sam.voice.stt import SttResult

FRAME_S = 0.03
USER_TEXT = "نرخی زێڕ چەندە؟"
TV_TEXT = "کچێکی پێکەنین وەرگیراوە باوکە گیان عوسمان"
_TV_PATTERN = (-38.0, -40.0, -42.0, -39.0, -60.0, -41.0, -40.0, -60.0)


class VirtualClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t

    def perf_counter(self) -> float:
        return self.t

    def time(self) -> float:
        return 1.79e9 + self.t

    @staticmethod
    def sleep(seconds: float) -> None:
        _real_time.sleep(seconds)


CLOCK = VirtualClock()


async def vwait(seconds: float) -> None:
    """Wait ``seconds`` of VIRTUAL time (the scene driver advances it)."""
    end = CLOCK.t + seconds
    while CLOCK.t < end:
        await asyncio.sleep(0)


def frame(db: float, pitch: int) -> bytes:
    if db <= -90:
        return bytes(960)
    amp = int(32767 * 10 ** (db / 20))
    return array.array("h", [amp if (i // pitch) % 2 else -amp for i in range(480)]).tobytes()


class VSpeaker:
    """``playing`` lasts the virtual duration of what was written."""

    def __init__(self) -> None:
        self.until = 0.0
        self.epoch = 0
        self.gain = 1.0
        self.is_open = False
        self.device = None
        self.underflows = 0
        self.last_error = None

    @property
    def playing(self) -> bool:
        return CLOCK.t < self.until

    def write(self, pcm: bytes, *, epoch: int | None = None, rate: int = 24000) -> bool:
        if not pcm or (epoch is not None and epoch != self.epoch):
            return False
        self.until = max(self.until, CLOCK.t) + len(pcm) / 2 / rate
        return True

    def flush(self) -> int:
        self.until = CLOCK.t
        self.epoch += 1
        return 0

    def buffered_ms(self) -> float:
        return max(0.0, self.until - CLOCK.t) * 1000.0

    async def wait_idle(self, timeout: float | None = None, poll_s: float = 0.01) -> bool:
        while CLOCK.t < self.until:
            await asyncio.sleep(0)
        return True

    async def open(self) -> bool:
        self.is_open = True
        return True

    def close(self) -> None:
        self.is_open = False

    async def close_if_idle(self) -> bool:
        return False


class VTts:
    last_provider = "fake-tts"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def configured(self) -> bool:
        return True

    def status(self) -> dict[str, Any]:
        return {}

    def max_chars(self) -> int:
        return 480

    async def stream(self, text: str):
        self.texts.append(text)
        await vwait(0.3)
        for _ in range(3):                      # 3 s of speech
            yield b"\x10\x00" * 24000

    async def aclose(self) -> None:
        pass


class VStt:
    """Loud audio (the user at -20 dBFS) -> the user's words, else the TV's."""

    last_provider = "fake-stt"

    def __init__(self, user_text: str = USER_TEXT) -> None:
        self.user_text = user_text
        self.calls: list[str] = []

    def configured(self) -> bool:
        return True

    def status(self) -> dict[str, Any]:
        return {}

    async def transcribe(self, pcm: bytes, rate: int = 16000) -> SttResult:
        samples = array.array("h")
        samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
        who = "user" if samples and max(abs(x) for x in samples) > 1500 else "tv"
        self.calls.append(who)
        await vwait(1.0)
        return SttResult(text=self.user_text if who == "user" else TV_TEXT, provider="fake", ms=1000.0)

    async def aclose(self) -> None:
        pass


@dataclass
class Scene:
    seconds: float
    clicks: list[float]
    tv: bool = True
    user_db: float = -20.0
    user_delay_s: float = 1.0
    user_len_s: float = 1.5
    llm_calls: list[str] = field(default_factory=list)

    def frame_at(self, t: float) -> bytes:
        for click in self.clicks:
            if click + self.user_delay_s <= t < click + self.user_delay_s + self.user_len_s:
                return frame(self.user_db, 8)
        if self.tv:
            cycle = t % 2.9                        # 2.0 s sentence + 0.9 s pause
            if cycle < 2.0:
                return frame(_TV_PATTERN[int(t / FRAME_S) % len(_TV_PATTERN)], 3)
        return frame(-96.0, 8)

    def llm(self):
        async def stream(text: str, turn: Any):
            self.llm_calls.append(text)
            await vwait(1.0)
            yield "باشە، ئەوە دەکەم."
        return stream


async def run_scene(eng: Any, scene: Scene) -> dict[str, Any]:
    """Drive ``eng`` through ``scene`` on the virtual clock."""
    if eng._watch_task is not None:  # noqa: SLF001 - the window watcher is driven by hand
        eng._watch_task.cancel()
    eng.gate._clock = CLOCK.monotonic  # noqa: SLF001
    start = CLOCK.t
    frames = int(scene.seconds / FRAME_S)
    clicks = {int(c / FRAME_S) for c in scene.clicks}
    mic_open_s = 0.0
    for index in range(frames):
        CLOCK.t += FRAME_S
        if index in clicks and not eng.listening:
            await eng.start_listening()
        if eng.listening:
            mic_open_s += FRAME_S
            await eng._on_frame(scene.frame_at(CLOCK.t - start))  # noqa: SLF001
        if index % 8 == 0:
            await eng._window_tick()  # noqa: SLF001
        for _ in range(4):
            await asyncio.sleep(0)
    return {"stt": list(eng.stt.calls), "llm": list(scene.llm_calls), "mic_open_s": round(mic_open_s, 1)}


__all__ = ["CLOCK", "Scene", "VSpeaker", "VStt", "VTts", "run_scene", "USER_TEXT", "TV_TEXT", "vwait"]
