"""Per-stage timings, persisted to the ``timings`` table.

v1 recorded no stage timings at all, so its 13-20 s time-to-first-sound could
only be estimated (reports/audit-latency.json). Every SAM 2 turn records its
stages here so latency work is driven by measurements.

Stage names (use these so the Activity tab can compare turns):
  end_of_speech, stt, llm_first_token, llm_total, tts_first_audio,
  first_audio (end of user speech -> first audio out = TTFA; in tool turns that
  is the cached acknowledgement), first_answer (first reply text that is not the
  acknowledgement), first_answer_audio (its first sound), live_connect,
  tool:<name>, confirm_wait, worker_step, analysis_engine, analysis_total,
  tv_cdp, mt5_fetch, startup:<phase>
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .events import new_id

log = logging.getLogger("sam.timing")


class TurnTimer:
    """Collects stages of one turn; ``finish()`` persists them in one go."""

    def __init__(self, timing: "Timing", kind: str, turn_id: str | None = None) -> None:
        self.timing = timing
        self.kind = kind
        self.turn_id = turn_id or new_id()
        self.t0 = time.perf_counter()
        self.stages: list[tuple[str, float, dict[str, Any] | None]] = []
        self._finished = False

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def mark(self, stage: str, **extra: Any) -> float:
        """Record ``stage`` at 'ms since the turn started' (e.g. first_audio)."""
        ms = self.elapsed_ms()
        self.stages.append((stage, ms, extra or None))
        return ms

    def mark_once(self, stage: str, **extra: Any) -> float | None:
        """``mark`` unless ``stage`` was already recorded in this turn (e.g.
        first_answer: the first words that are not the acknowledgement)."""
        if any(name == stage for name, _, _ in self.stages):
            return None
        return self.mark(stage, **extra)

    def add(self, stage: str, ms: float, **extra: Any) -> None:
        """Record an already-measured duration."""
        self.stages.append((stage, float(ms), extra or None))

    @contextmanager
    def stage(self, name: str, **extra: Any) -> Iterator[None]:
        """Measure the duration of a block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000.0, **extra)

    def finish(self, **extra: Any) -> dict[str, float]:
        """Persist (idempotent) and return {stage: ms}."""
        if not self._finished:
            self._finished = True
            total = self.elapsed_ms()
            self.stages.append(("total", total, extra or None))
            self.timing._persist(self.turn_id, self.kind, self.stages)
        return {name: ms for name, ms, _ in self.stages}

    def __enter__(self) -> "TurnTimer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()


class Timing:
    """Factory for turn timers plus one-off records. DB may be None (tests)."""

    def __init__(self, db: Any | None = None) -> None:
        self.db = db

    def turn(self, kind: str, turn_id: str | None = None) -> TurnTimer:
        return TurnTimer(self, kind, turn_id)

    def record(self, stage: str, ms: float, *, kind: str = "", turn_id: str | None = None, **extra: Any) -> None:
        self._persist(turn_id, kind, [(stage, float(ms), extra or None)])

    @contextmanager
    def measure(self, stage: str, *, kind: str = "", turn_id: str | None = None, **extra: Any) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - start) * 1000.0, kind=kind, turn_id=turn_id, **extra)

    def _persist(self, turn_id: str | None, kind: str, stages: list[tuple[str, float, dict[str, Any] | None]]) -> None:
        if self.db is None or not stages:
            return
        now = time.time()
        try:
            self.db.executemany(
                "INSERT INTO timings(at, turn_id, kind, stage, ms, extra) VALUES (?,?,?,?,?,?)",
                [(now, turn_id, kind, name, round(ms, 1),
                  json.dumps(extra, ensure_ascii=False, default=str) if extra else None)
                 for name, ms, extra in stages])
        except Exception:  # noqa: BLE001 - timing must never break a turn
            log.exception("could not persist timings")

    def recent(self, limit: int = 200, kind: str | None = None) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        if kind:
            return self.db.query("SELECT * FROM timings WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit))
        return self.db.query("SELECT * FROM timings ORDER BY id DESC LIMIT ?", (limit,))


__all__ = ["Timing", "TurnTimer"]
