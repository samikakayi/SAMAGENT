"""Local voice activity detection: frame classifier + utterance endpointer.

Used by both engines:
- Cascade: decides when the user finished speaking (then STT runs) and
  detects barge-in while SAM speaks.
- Live: the server does its own VAD; the local endpointer only times the end
  of speech (TTFA and the 5 s watchdog) and keeps the last utterance's audio
  so the cascade can answer it if Live stalls.

webrtcvad (``webrtcvad-wheels``, imported lazily: 0.1 s) classifies 30 ms
frames; an RMS floor stops it from triggering on hum. The ring-buffer trigger
follows the py-webrtcvad example. v1's end-of-speech wait was 20 x 30 ms =
0.6 s (sam_backend/wake.py:79) and the Live docs recommend 500-800 ms of
silence, so the default is ``voice.silence_ms`` = 600.
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass, field
from typing import Literal

from .audio import MIC_RATE, SAMPLE_BYTES


class FrameClassifier:
    """speech = webrtcvad says voiced AND the frame is louder than a floor."""

    def __init__(self, *, rate: int = MIC_RATE, aggressiveness: int = 2, energy_floor: float = 0.004) -> None:
        self.rate = rate
        self.aggressiveness = max(0, min(3, int(aggressiveness)))
        self.energy_floor = energy_floor
        self._vad = None
        self._failed = False

    def _get(self):  # noqa: ANN202
        if self._vad is None and not self._failed:
            try:
                import webrtcvad

                self._vad = webrtcvad.Vad(self.aggressiveness)
            except Exception:  # noqa: BLE001 - fall back to energy only
                self._failed = True
        return self._vad

    def is_speech(self, frame: bytes, rms: float) -> bool:
        if rms < self.energy_floor:
            return False
        vad = self._get()
        if vad is None:
            return rms >= self.energy_floor * 4
        try:
            return bool(vad.is_speech(frame, self.rate))
        except Exception:  # noqa: BLE001 - odd frame size (last partial block)
            return rms >= self.energy_floor * 4


@dataclass
class VadEvent:
    kind: Literal["start", "end"]
    pcm: bytes = b""               # "end": pre-roll + speech (+ a little trailing silence)
    speech_ms: float = 0.0         # voiced duration
    eos_at: float = 0.0            # perf_counter() of the last voiced frame
    too_short: bool = False        # "end" of a blip (cough, click): ignore
    forced: bool = False           # max utterance length reached


@dataclass
class Endpointer:
    """Turns classified frames into start/end-of-utterance events."""

    frame_ms: int = 30
    start_window_ms: int = 240     # look-back window for the start trigger
    start_ratio: float = 0.6       # voiced share of that window that means "speech started"
    silence_ms: int = 600          # unvoiced time that ends an utterance
    preroll_ms: int = 300          # audio kept from before the trigger (first syllable)
    min_speech_ms: int = 250       # shorter voiced time = blip
    max_utterance_s: float = 30.0
    trailing_ms: int = 200         # silence kept after the last voiced frame
    in_speech: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._window: collections.deque[tuple[bytes, bool]] = collections.deque(
            maxlen=max(1, self.start_window_ms // self.frame_ms))
        self._preroll: collections.deque[bytes] = collections.deque(maxlen=max(1, self.preroll_ms // self.frame_ms))
        self._frames: list[bytes] = []
        self._voiced = 0
        self._silent_run = 0
        self._last_voiced_at = 0.0
        self._last_voiced_index = 0
        # e.g. 5 voiced frames among the last 8 (>= 150 ms of speech within 240 ms)
        self._start_needed = max(2, math.ceil(self.start_ratio * (self._window.maxlen or 1)))

    def reset(self) -> None:
        self.in_speech = False
        self._window.clear()
        self._preroll.clear()
        self._frames = []
        self._voiced = 0
        self._silent_run = 0

    @property
    def current_speech_ms(self) -> float:
        return self._voiced * self.frame_ms if self.in_speech else 0.0

    def frames_so_far(self) -> list[bytes]:
        """The current utterance's frames (pre-roll included) while in speech:
        what Live receives once the utterance is accepted (engine frames.py)."""
        return list(self._frames) if self.in_speech else []

    def process(self, frame: bytes, speech: bool, now: float | None = None) -> VadEvent | None:
        now = time.perf_counter() if now is None else now
        if not self.in_speech:
            self._window.append((frame, speech))
            voiced = sum(1 for _, s in self._window if s)
            if speech and voiced >= self._start_needed:
                self.in_speech = True
                self._frames = list(self._preroll) + [f for f, _ in self._window]
                self._voiced = voiced
                self._silent_run = 0
                self._last_voiced_at = now
                self._last_voiced_index = len(self._frames)
                self._window.clear()
                self._preroll.clear()
                return VadEvent("start")
            self._preroll.append(frame)
            return None
        self._frames.append(frame)
        if speech:
            self._voiced += 1
            self._silent_run = 0
            self._last_voiced_at = now
            self._last_voiced_index = len(self._frames)
        else:
            self._silent_run += 1
        too_long = len(self._frames) * self.frame_ms >= self.max_utterance_s * 1000
        if self._silent_run * self.frame_ms >= self.silence_ms or too_long:
            keep = min(len(self._frames), self._last_voiced_index + self.trailing_ms // self.frame_ms)
            pcm = b"".join(self._frames[:keep])
            speech_ms = self._voiced * self.frame_ms
            ended_by_silence = self._silent_run * self.frame_ms >= self.silence_ms
            event = VadEvent("end", pcm=pcm, speech_ms=speech_ms, eos_at=self._last_voiced_at,
                             too_short=speech_ms < self.min_speech_ms, forced=too_long and not ended_by_silence)
            self.reset()
            return event
        return None


def pcm_ms(pcm: bytes, rate: int = MIC_RATE) -> float:
    return len(pcm) / (SAMPLE_BYTES * rate) * 1000.0


__all__ = ["FrameClassifier", "Endpointer", "VadEvent", "pcm_ms"]
