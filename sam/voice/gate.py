"""Near-field gate: only speech close to the microphone reaches STT or Live.

Why (real use, 2026-09-24 evening, sam2.log + DB): in the old always-open
conversation window SAM transcribed the TV and a family conversation
(«کچێکی پێکەنین وەرگیراوە باوکە گیان عوسمان...») as commands -- 149 KurdishTTS
STT calls in a day (the free plan is 2 h a month) and one LLM request per
sound, which used up Groq, Gemini and OmniRoute. Speech into a headset (or at
arm's length) is louder at the mic than a TV or people across the room, so
each 30 ms frame is classified:

    near-field speech = webrtcvad says voiced AND level_db >= threshold_db
    threshold_db = clamp(max(floor + margin, background + 4, user_level - 8),
                         abs_min_db, ceiling)

- ``floor``: p30 of ALL frame levels of the last 8 s (continuous noise: fan,
  a TV that never pauses). ``margin_db`` = ``voice.gate_margin_db`` (14).
- ``user_level``: the median level of the user's own speech, measured by the
  voice enrollment or learned from push-to-talk turns (``learn_user_level``).
  Then ``ceiling`` = user_level - 4 dB, so the user always passes. Without it
  the ceiling is ``voice.gate_ceiling_db`` (-30 dBFS).

  Learning is guarded (adversarial review 2026-09-24): with a TV that never
  pauses, the first utterance after a click is often the TV or the user
  merged with it, and the old code stored its median at once (-35 to -39
  dBFS in the closed-loop simulation, user at -26) -- the gate then let the
  TV through in that session and every later one. Now an utterance is a
  candidate only when it looks like near-field speech (median >= -38 dBFS,
  >= 18 dB above the room floor, not within 6 dB of a rejected talker); the
  FIRST level is stored only after 3 candidates agree within 6 dB, and a
  stored level moves at most 2 dB per turn (outliers > 10 dB away ignored).
- ``background``: the loudest recent utterance that turned out NOT to be the
  user (rejected by "only my voice" or the name check), kept 120 s.

Measured offline (work/voiceeval/sim_gate.py: real webrtcvad + this gate +
the endpointer on 30 s scenes of reverberant TV speech from other voices and
four dry user utterances; nothing played or recorded):
- user level known: TV/family 12-16 dB below the user -> 0 false utterances
  in 30 s and 4/4 user utterances (user -26 dBFS vs TV -40/-45, user -30 vs
  family -42); TV within ~9 dB of the user (laptop mic, loud TV) still leaks
  (5-7 utterances / 30 s) -- that is what "only my voice" is for.
- user level unknown: an intermittent TV sets no floor (its pauses are
  silence), so only ``abs_min_db`` applies and TV at -40 passes (6 utts /
  30 s). Push-to-talk windows (listening.py) bound that exposure to ~8 s per
  click and ~6 s after each answer.

Frames that fail the gate count as SILENCE for the endpointer (vad.py): a TV
talking under the user no longer keeps an utterance open for 30 s, and a
rejected talker never starts one -- no STT, no model call.

Pure Python on floats; ~3 us per frame plus a sort of <= 267 values every
``update_every`` frames.
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass, field
from typing import Any

SILENT_DB = -96.0
# User-level learning (see the module docstring).
LEARN_MIN_FRAMES = 10          # 0.3 s of near-field speech
START_FRAMES = 10              # frames that trigger an utterance start (endpointer window + margin)
LEARN_MIN_DB = -38.0           # far TV / family were -40..-46 dBFS in the simulations
LEARN_ABOVE_FLOOR_DB = 18.0
LEARN_BACKGROUND_DB = 6.0
LEARN_AGREE = 3
LEARN_SPREAD_DB = 6.0
LEARN_STEP_DB = 2.0
LEARN_OUTLIER_DB = 10.0


def level_db(rms: float) -> float:
    """dBFS of an RMS value in 0..1 (``pcm_rms``)."""
    return 20.0 * math.log10(rms) if rms > 1.6e-5 else SILENT_DB


@dataclass
class GateSettings:
    margin_db: float = 14.0
    abs_min_db: float = -50.0
    ceiling_db: float = -30.0
    min_voiced_ms: int = 300
    window_s: float = 8.0
    percentile: float = 30.0
    user_level_db: float | None = None   # the user's speech level (enrollment / learned from turns)
    user_drop_db: float = 8.0            # with a known user level: threshold >= user - 8 dB ...
    user_headroom_db: float = 4.0        # ... and never above user - 4 dB (the user always passes)
    bg_margin_db: float = 4.0            # a rejected talker's level + 4 dB
    bg_memory_s: float = 120.0

    @classmethod
    def from_config(cls, config: Any) -> "GateSettings":
        def num(key: str, default: float) -> float:
            try:
                return float(config.get(key, default))
            except (TypeError, ValueError):
                return default
        user = config.get("voice.gate_user_level_db", None)
        return cls(margin_db=num("voice.gate_margin_db", cls.margin_db),
                   abs_min_db=num("voice.gate_abs_min_db", cls.abs_min_db),
                   ceiling_db=num("voice.gate_ceiling_db", cls.ceiling_db),
                   min_voiced_ms=int(num("voice.gate_min_voiced_ms", cls.min_voiced_ms)),
                   user_level_db=float(user) if isinstance(user, (int, float)) else None)


@dataclass
class UtteranceLevels:
    """Levels of the near-field frames of the current utterance (diagnostics)."""

    near_db: list[float] = field(default_factory=list)
    floor_at_start: float = SILENT_DB
    threshold_at_start: float = 0.0

    def summary(self) -> dict[str, float]:
        if not self.near_db:
            return {"frames": 0}
        ordered = sorted(self.near_db)
        return {"frames": len(ordered), "p50_db": round(ordered[len(ordered) // 2], 1),
                "p90_db": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 1),
                "floor_db": round(self.floor_at_start, 1), "threshold_db": round(self.threshold_at_start, 1)}


class NearFieldGate:
    """Per-frame near-field classification with an adaptive floor."""

    def __init__(self, settings: GateSettings | None = None, *, frame_ms: int = 30, update_every: int = 5,
                 clock: Any = time.monotonic) -> None:
        self.settings = settings or GateSettings()
        self.frame_ms = frame_ms
        self.update_every = max(1, update_every)
        self._clock = clock
        self._history: collections.deque[float] = collections.deque(
            maxlen=max(10, int(self.settings.window_s * 1000 / frame_ms)))
        self._floor = SILENT_DB
        self._since_update = 0
        self._last_frame_at = 0.0
        self._background: collections.deque[tuple[float, float]] = collections.deque(maxlen=16)
        self._candidates: collections.deque[float] = collections.deque(maxlen=5)
        self.frames = 0
        self.near_frames = 0
        self.current = UtteranceLevels()

    # -- configuration ------------------------------------------------------------------------
    def configure(self, settings: GateSettings) -> None:
        self.settings = settings
        maxlen = max(10, int(settings.window_s * 1000 / self.frame_ms))
        if maxlen != self._history.maxlen:
            self._history = collections.deque(self._history, maxlen=maxlen)

    @property
    def floor_db(self) -> float:
        return self._floor

    @property
    def ceiling_db(self) -> float:
        s = self.settings
        if s.user_level_db is not None:
            # How loud the user really is at this mic is known (enrollment or
            # earlier turns): never demand more than that minus a headroom.
            return s.user_level_db - s.user_headroom_db
        return s.ceiling_db

    @property
    def threshold_db(self) -> float:
        s = self.settings
        wanted = self._floor + s.margin_db
        background = self.background_db
        if background is not None:
            wanted = max(wanted, background + s.bg_margin_db)
        if s.user_level_db is not None:
            wanted = max(wanted, s.user_level_db - s.user_drop_db)
        return max(s.abs_min_db, min(wanted, self.ceiling_db))

    @property
    def background_db(self) -> float | None:
        """Loudest recent speech that turned out NOT to be the user (an
        utterance rejected by "only my voice", the name check or a blip), for
        ``bg_memory_s``: that talker then stays below the threshold."""
        now = self._clock()
        while self._background and now - self._background[0][0] > self.settings.bg_memory_s:
            self._background.popleft()
        return max((db for _, db in self._background), default=None)

    def note_background(self, p50_db: float | None = None) -> None:
        """The current utterance was someone else (or noise): remember its level."""
        if p50_db is None:
            summary = self.current.summary()
            p50_db = summary.get("p50_db") if summary.get("frames") else None
        if p50_db is not None and p50_db > SILENT_DB:
            self._background.append((self._clock(), float(p50_db)))

    def learn_candidate(self, levels: dict[str, Any] | float | None = None) -> float | None:
        """The utterance's median level if it looks like the user's own
        near-field speech (see the module docstring), else None."""
        if isinstance(levels, (int, float)):
            p50, floor, frames = float(levels), self._floor, LEARN_MIN_FRAMES
        else:
            summary = levels if isinstance(levels, dict) else self.current.summary()
            frames = int(summary.get("frames", 0) or 0)
            p50 = summary.get("p50_db")
            floor = float(summary.get("floor_db", self._floor) if summary.get("floor_db") is not None else self._floor)
        if p50 is None or frames < LEARN_MIN_FRAMES:
            return None
        p50 = float(p50)
        if p50 < LEARN_MIN_DB or p50 < floor + LEARN_ABOVE_FLOOR_DB:
            return None
        background = self.background_db
        if background is not None and abs(p50 - background) <= LEARN_BACKGROUND_DB:
            return None
        return max(-60.0, min(-6.0, p50))

    def learn_user_level(self, levels: dict[str, Any] | float | None = None, *, weight: float = 0.3) -> float | None:
        """The utterance was probably the user: returns the user level to
        STORE now, or None (not a candidate, or the first level still needs
        more agreeing turns)."""
        candidate = self.learn_candidate(levels)
        if candidate is None:
            return None
        old = self.settings.user_level_db
        if old is None:
            self._candidates.append(candidate)
            recent = list(self._candidates)[-LEARN_AGREE:]
            if len(recent) < LEARN_AGREE or max(recent) - min(recent) > LEARN_SPREAD_DB:
                return None
            self._candidates.clear()
            self.settings.user_level_db = round(sorted(recent)[len(recent) // 2], 1)
            return self.settings.user_level_db
        if abs(candidate - old) > LEARN_OUTLIER_DB:
            return None
        step = max(-LEARN_STEP_DB, min(LEARN_STEP_DB, weight * (candidate - old)))
        self.settings.user_level_db = round(old + step, 1)
        return self.settings.user_level_db

    def forget_user_level(self) -> None:
        """«ئاستی دەنگم لەبیر بکە» (Settings) / a deleted voiceprint."""
        self.settings.user_level_db = None
        self._candidates.clear()

    # -- per frame ----------------------------------------------------------------------------------
    def classify(self, rms: float, voiced: bool) -> bool:
        """True when this frame is near-field speech (feeds the endpointer)."""
        now = self._clock()
        if self._last_frame_at and now - self._last_frame_at > 30.0:
            # The mic was closed for a while (push-to-talk): old levels say
            # little about the room now; keep the last floor as the seed.
            self._history.clear()
        self._last_frame_at = now
        db = level_db(rms)
        self._history.append(db)
        self._since_update += 1
        if self._since_update >= self.update_every or len(self._history) <= 12:
            self._since_update = 0
            self._floor = self._percentile()
        self.frames += 1
        near = bool(voiced) and db >= self.threshold_db
        if near:
            self.near_frames += 1
            self.current.near_db.append(db)
        return near

    def _percentile(self, values: Any = None) -> float:
        ordered = sorted(self._history if values is None else values)
        if not ordered:
            return SILENT_DB
        index = min(len(ordered) - 1, int(len(ordered) * self.settings.percentile / 100.0))
        return max(SILENT_DB, ordered[index])

    # -- utterance bookkeeping (engine calls on endpointer start/end) ----------------------------------------
    def utterance_started(self, frames: list[bytes] | None = None) -> None:
        # The room floor BEFORE this utterance: the frames that triggered the
        # start (~8) are left out; with too little history it is unknown.
        earlier = list(self._history)[:-START_FRAMES]
        floor = self._percentile(earlier) if len(earlier) >= START_FRAMES else SILENT_DB
        self.current = UtteranceLevels(floor_at_start=floor, threshold_at_start=self.threshold_db)

    def utterance_levels(self) -> dict[str, float]:
        return self.current.summary()

    def status(self) -> dict[str, Any]:
        return {"floor_db": round(self._floor, 1), "threshold_db": round(self.threshold_db, 1),
                "margin_db": self.settings.margin_db, "abs_min_db": self.settings.abs_min_db,
                "ceiling_db": round(self.ceiling_db, 1), "user_level_db": self.settings.user_level_db,
                "background_db": self.background_db, "level_candidates": len(self._candidates),
                "frames": self.frames, "near_frames": self.near_frames}


__all__ = ["NearFieldGate", "GateSettings", "UtteranceLevels", "level_db", "SILENT_DB"]
