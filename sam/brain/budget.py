"""Quota guard for the brain's optional model calls.

Three kinds of model request are not needed to answer the user: folding old
turns into a conversation summary, extracting durable facts when a
conversation sleeps, and rewording a fast picker's small talk. On 2026-09-24
(the user's first real evening test, sam2.log after 20:50) they competed with
the live turns for the same free quotas: four summary calls ran while the
user was talking (20:55:18-20:57:58), each walking the whole 'extract' ladder
(OmniRoute 6 s timeouts, 429s, a Gemini 503), and a fact extraction at
20:58:46 did the same -- while SAM answered «ببورە، ئێستا ناتوانم پەیوەندی بە
مۆدێلەکانەوە بکەم» to the user because every rung was resting.

Rules (``BackgroundBudget.reason_to_skip``):

- never while a live exchange runs (a turn in progress, the voice engine
  thinking/speaking/working, or speech/text within ``quiet_s``) -- except the
  reword, which is part of a turn by design;
- never while ANY rung of the live conversation ladders (or of the call's own
  ladder) rests after a failure: a resting rung means the free quotas are
  under pressure, and the next live turn needs what is left;
- never when the day's budget is low: ``daily_max`` summary + extraction
  requests a day (rewordings have their own ``reword_daily_max``), and a model
  with a daily cap (``llm.daily_caps``) is left to live turns once
  ``cap_share`` of its cap is used;
- one request per job: the job asks exactly one rung (the first usable one)
  and never walks a ladder.

Every decision is counted in ``usage_counters`` (provider ``background``,
model = purpose, kind = ran|skipped|deferred|failed) and logged.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import ladders

log = logging.getLogger("sam.brain.budget")

PURPOSES = ("summary", "extract", "reword")
DEFAULTS: dict[str, Any] = {
    "brain.background.daily_max": 30,     # summary + extraction requests a day
    "brain.background.reword_daily_max": 60,   # rewordings a day (their own bucket: one per small-talk turn)
    "brain.background.cap_share": 0.6,    # of a capped model's daily cap; the rest is kept for live turns
    "brain.background.quiet_s": 40,       # no summary/extraction within this long of the last speech/text
}
LIVE_VOICE_STATES = frozenset({"thinking", "speaking", "working"})
PROVIDER = "background"


class BackgroundBudget:
    """Decides whether an optional background model call may spend a request now."""

    def __init__(self, app: Any, *, clock: Callable[[], float] = time.time) -> None:
        self.app = app
        self.clock = clock

    def _setting(self, key: str) -> Any:
        try:
            value = self.app.config.get(key, DEFAULTS[key])
        except Exception:  # noqa: BLE001
            value = DEFAULTS[key]
        return DEFAULTS[key] if value is None else value

    # -- state of the conversation --------------------------------------------------------------------------------
    def live(self, *, recent: bool = True) -> bool:
        """A live exchange is running (or, with ``recent``, ended less than
        ``quiet_s`` ago; a conversation that just went to sleep has ended)."""
        conversation = getattr(self.app, "conversation", None)
        if int(getattr(conversation, "active_turns", 0) or 0) > 0:
            return True
        last = float(getattr(conversation, "last_activity", 0.0) or 0.0)
        if recent and last and self.clock() - last < float(self._setting("brain.background.quiet_s")):
            return True
        state = str(getattr(getattr(self.app, "voice", None), "state", "") or "")
        return state in LIVE_VOICE_STATES

    def watched_refs(self, refs: list[str] | None = None) -> list[str]:
        """The rungs whose rest means "quota pressure": the live picker and
        wording ladders plus the call's own ladder."""
        watched: list[str] = []
        try:
            live = ladders.auto_picker(self.app, "voice") + ladders.auto_wording(self.app)
        except Exception:  # noqa: BLE001
            live = []
        for ref in live + list(refs or []):
            if ref not in watched:
                watched.append(ref)
        return watched

    def _configured(self, ref: str) -> bool:
        try:
            backend = self.app.llm.backends.get(ref.split(":", 1)[0])
            return backend is not None and bool(backend.configured())
        except Exception:  # noqa: BLE001
            return False

    def _cap_low(self, ref: str) -> bool:
        try:
            caps = self.app.config.get("llm.daily_caps", {}) or {}
            cap = caps.get(ref)
            if not cap:
                return False
            provider, model = ref.split(":", 1)
            used = int(self.app.db.usage_for(provider, model).get("requests", 0) or 0)
        except Exception:  # noqa: BLE001
            return False
        return used >= float(cap) * float(self._setting("brain.background.cap_share"))

    def used_today(self, purposes: tuple[str, ...] = PURPOSES) -> int:
        """Requests the given background purposes spent today."""
        total = 0
        for purpose in purposes:
            try:
                total += int((self.app.db.usage_for(PROVIDER, purpose, "ran") or {}).get("requests", 0) or 0)
            except Exception:  # noqa: BLE001
                continue
        return total

    # -- decisions --------------------------------------------------------------------------------------------------
    def reason_to_skip(self, purpose: str, refs: list[str], *, allow_live: bool = False,
                       ended: bool = False) -> str | None:
        """Why ``purpose`` must not spend a request now, or None. ``ended``:
        the conversation has just ended (voice window asleep), so only a turn
        still running counts as live."""
        if not allow_live and self.live(recent=not ended):
            return "live"
        llm = getattr(self.app, "llm", None)
        if llm is None:
            return "no model client"
        for ref in self.watched_refs(refs):
            if self._configured(ref) and llm.cooling(ref):
                return f"cooling:{ref}"
        if purpose == "reword":
            if self.used_today(("reword",)) >= int(self._setting("brain.background.reword_daily_max")):
                return "daily budget used"
        elif self.used_today(("summary", "extract")) >= int(self._setting("brain.background.daily_max")):
            return "daily budget used"
        for ref in refs:
            if self._configured(ref) and self._cap_low(ref):
                return f"cap:{ref}"
        if not any(self._configured(ref) for ref in refs):
            return "no model configured"
        return None

    def pick(self, purpose: str, refs: list[str], *, allow_live: bool = False,
             ended: bool = False) -> tuple[list[str], str | None]:
        """(one rung to ask, None) or ([], reason to skip)."""
        reason = self.reason_to_skip(purpose, refs, allow_live=allow_live, ended=ended)
        if reason is not None:
            return [], reason
        for ref in refs:
            if self._configured(ref):
                return [ref], None
        return [], "no model configured"

    def record(self, purpose: str, outcome: str, reason: str = "") -> None:
        """Count one decision (ran | skipped | deferred | failed)."""
        if outcome != "ran":
            log.info("background %s %s: %s", purpose, outcome, reason or "-")
        try:
            self.app.db.bump_usage(PROVIDER, purpose, outcome)
        except Exception:  # noqa: BLE001 - counting must never break a turn
            log.debug("background counter failed", exc_info=True)

    def counts_today(self) -> dict[str, dict[str, int]]:
        """{purpose: {outcome: n}} for today (panel / tests)."""
        out: dict[str, dict[str, int]] = {}
        try:
            day = self.app.db.quota_day(PROVIDER)
            for row in self.app.db.query("SELECT model, kind, requests FROM usage_counters WHERE day=? AND provider=?",
                                         (day, PROVIDER)):
                out.setdefault(str(row["model"]), {})[str(row["kind"])] = int(row["requests"])
        except Exception:  # noqa: BLE001
            pass
        return out


def quota_reason(reason: str | None) -> bool:
    """True for reasons that mean "quota pressure" (fall back without a model
    now), False for "live" (just wait for a pause)."""
    return bool(reason) and reason != "live"


__all__ = ["BackgroundBudget", "DEFAULTS", "PURPOSES", "quota_reason"]
