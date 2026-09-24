"""Bar Replay research (ported from v1 ``sam_backend/trading/replay.py``).

Two separate things live here. The replay *state machine* and its anti-lookahead
guarantee are pure logic and fully testable. Driving TradingView's own Bar Replay
control is not: its toolbar state cannot be read back reliably from a screenshot,
so that half is reported PARTIAL and the deterministic historical engine in
`research.py` stays the primary research path.

The guarantee that matters is enforced here regardless of which driver is used:
at decision bar N the session hands out `candles[:N+1]` and nothing else, and an
outcome can only be resolved after the cursor has advanced past N.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .types import CapabilityState, Candle


class ReplayState(StrEnum):
    IDLE = "IDLE"
    LOADED = "LOADED"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"


class LookaheadError(RuntimeError):
    """Raised when something asks for data past the replay cursor."""


@dataclass(slots=True)
class ReplayDecision:
    """One recorded decision, plus the outcome resolved strictly afterwards."""

    cursor: int
    timestamp: str
    theory: str
    setup_state: str
    direction: str | None = None
    entry: float | None = None
    invalidation: float | None = None
    stop: float | None = None
    targets: list[float] = field(default_factory=list)
    notes: str = ""
    outcome: str | None = None
    outcome_price: float | None = None
    outcome_cursor: int | None = None
    r_multiple: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor, "timestamp": self.timestamp, "theory": self.theory,
            "setup_state": self.setup_state, "direction": self.direction, "entry": self.entry,
            "invalidation": self.invalidation, "stop": self.stop, "targets": list(self.targets),
            "notes": self.notes, "outcome": self.outcome, "outcome_price": self.outcome_price,
            "outcome_cursor": self.outcome_cursor,
            "r_multiple": round(self.r_multiple, 4) if self.r_multiple is not None else None,
        }


class ReplaySession:
    """A cursor over historical candles that never reveals the future.

    `visible()` is the only way to read candles, and it is bounded by the cursor.
    Outcome resolution is deliberately a separate call that refuses to run until
    the cursor has moved past the decision it is scoring.
    """

    def __init__(self, candles: list[Candle], *, start_index: int, symbol: str = "", timeframe: str = "") -> None:
        if not candles:
            raise ValueError("A replay session needs candles.")
        if not 0 <= start_index < len(candles):
            raise ValueError("start_index is outside the series.")
        self._candles = candles
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_index = start_index
        self.cursor = start_index
        self.state = ReplayState.LOADED
        self.decisions: list[ReplayDecision] = []

    # --- Cursor -----------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.cursor >= len(self._candles) - 1

    @property
    def timestamp(self) -> str:
        return self._candles[self.cursor].time.isoformat()

    def visible(self) -> list[Candle]:
        """Candles up to and including the cursor. The only legal read."""
        return self._candles[: self.cursor + 1]

    def candle_at_cursor(self) -> Candle:
        return self._candles[self.cursor]

    def peek(self, index: int) -> Candle:
        """Read a specific bar, refusing anything the cursor has not reached."""
        if index > self.cursor:
            raise LookaheadError(
                f"Bar {index} is ahead of the replay cursor at {self.cursor}; it is not knowable yet."
            )
        return self._candles[index]

    def advance(self, bars: int = 1) -> dict[str, Any]:
        if bars < 1:
            raise ValueError("advance() moves forward by at least one bar.")
        before = self.cursor
        self.cursor = min(self.cursor + bars, len(self._candles) - 1)
        self.state = ReplayState.FINISHED if self.finished else ReplayState.PLAYING
        return {
            "from": before, "to": self.cursor, "advanced": self.cursor - before,
            "timestamp": self.timestamp, "state": self.state.value, "finished": self.finished,
        }

    def pause(self) -> dict[str, Any]:
        if self.state is ReplayState.PLAYING:
            self.state = ReplayState.PAUSED
        return self.status()

    def resume(self) -> dict[str, Any]:
        if self.state is ReplayState.PAUSED:
            self.state = ReplayState.PLAYING
        return self.status()

    def stop(self) -> dict[str, Any]:
        self.state = ReplayState.IDLE
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state.value, "cursor": self.cursor, "start_index": self.start_index,
            "timestamp": self.timestamp, "bars_remaining": len(self._candles) - 1 - self.cursor,
            "finished": self.finished, "symbol": self.symbol, "timeframe": self.timeframe,
            "decisions": len(self.decisions),
        }

    # --- Decisions --------------------------------------------------------

    def record_decision(
        self,
        *,
        theory: str,
        setup_state: str,
        direction: str | None = None,
        entry: float | None = None,
        invalidation: float | None = None,
        stop: float | None = None,
        targets: list[float] | None = None,
        notes: str = "",
    ) -> ReplayDecision:
        decision = ReplayDecision(
            cursor=self.cursor, timestamp=self.timestamp, theory=theory, setup_state=setup_state,
            direction=direction, entry=entry, invalidation=invalidation, stop=stop,
            targets=list(targets or []), notes=notes,
        )
        self.decisions.append(decision)
        return decision

    def resolve_decisions(self) -> list[ReplayDecision]:
        """Score decisions using only bars the cursor has already passed."""
        for decision in self.decisions:
            if decision.outcome is not None or decision.entry is None or decision.stop is None:
                continue
            if decision.cursor >= self.cursor:
                # The cursor has not moved past it: its outcome is not knowable.
                continue
            target = decision.targets[0] if decision.targets else None
            long = (decision.direction or "").upper() == "BULLISH"
            risk = abs(decision.entry - decision.stop)
            for index in range(decision.cursor + 1, self.cursor + 1):
                candle = self._candles[index]
                hit_stop = candle.low <= decision.stop if long else candle.high >= decision.stop
                hit_target = target is not None and (candle.high >= target if long else candle.low <= target)
                if hit_stop:
                    decision.outcome, decision.outcome_price = "STOPPED", decision.stop
                elif hit_target:
                    decision.outcome, decision.outcome_price = "TARGET", target
                else:
                    continue
                decision.outcome_cursor = index
                decision.r_multiple = (
                    (decision.outcome_price - decision.entry) / risk if long
                    else (decision.entry - decision.outcome_price) / risk
                ) if risk else 0.0
                break
        return self.decisions

    def report(self) -> dict[str, Any]:
        resolved = [item for item in self.decisions if item.outcome]
        return {
            **self.status(),
            "recorded": [item.as_dict() for item in self.decisions],
            "resolved": len(resolved),
            "unresolved": len(self.decisions) - len(resolved),
            "total_r": round(sum(item.r_multiple or 0.0 for item in resolved), 4),
        }


class BarReplayResearch:
    """Replay research over provider history.

    v1 wrapped results in its ``StandardResult``; SAM 2 returns plain dicts
    ``{"ok", "error_code"?, "error"?, ...payload}``. ``fetch`` is any callable
    ``(symbol, timeframe, count) -> list[Candle]`` (tests pass a list maker;
    the app passes a thread-safe MT5 reader).
    """

    def __init__(self, fetch: Any = None, tradingview: Any = None) -> None:
        self.fetch = fetch
        self.tradingview = tradingview
        self._sessions: dict[str, ReplaySession] = {}

    def capability(self) -> dict[str, Any]:
        """Native TradingView replay is reported honestly as unverifiable."""
        return {
            "name": "bar_replay",
            "data_replay": {
                "state": CapabilityState.AVAILABLE.value,
                "engine": "provider-history cursor",
                "detail": "Replays real provider candles with an enforced anti-lookahead cursor.",
            },
            "tradingview_native_replay": {
                "state": CapabilityState.PARTIALLY_AVAILABLE.value,
                "reason": (
                    "TradingView's Bar Replay toolbar exposes no readable state, so activation, the "
                    "replay clock, and per-bar advancement cannot be independently confirmed from a "
                    "screenshot. Driving it blind would produce unverifiable research, so the "
                    "provider-history cursor is used instead."
                ),
            },
            "anti_lookahead": "Enforced: a decision at bar N is handed candles[:N+1] and nothing later.",
        }

    @staticmethod
    def _failure(message: str, code: str) -> dict[str, Any]:
        return {"ok": False, "error": message, "error_code": code}

    def start(self, symbol: str = "XAUUSD", timeframe: str = "M15", *, count: int = 1500,
              start_offset: int = 300, session_id: str = "default") -> dict[str, Any]:
        if self.fetch is None:
            return self._failure("No market data source is configured.", "MARKET_DATA_UNAVAILABLE")
        try:
            candles = list(self.fetch(symbol, timeframe, count))
        except Exception as exc:  # noqa: BLE001 - feed errors become results
            return self._failure(str(exc), "MARKET_DATA_UNAVAILABLE")
        if len(candles) <= start_offset + 10:
            return self._failure(f"Need more than {start_offset + 10} candles to replay; got {len(candles)}.",
                                 "INSUFFICIENT_HISTORY")
        session = ReplaySession(candles, start_index=start_offset, symbol=symbol, timeframe=timeframe)
        self._sessions[session_id] = session
        return {"ok": True, "session_id": session_id, **session.status()}

    def session(self, session_id: str = "default") -> ReplaySession | None:
        return self._sessions.get(session_id)

    def control(self, action: str, *, session_id: str = "default", bars: int = 1) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            return self._failure("No replay session is running. Start one first.", "NO_REPLAY_SESSION")
        try:
            if action == "advance":
                payload = session.advance(bars)
            elif action == "pause":
                payload = session.pause()
            elif action == "resume":
                payload = session.resume()
            elif action == "stop":
                payload = session.stop()
                self._sessions.pop(session_id, None)
            elif action == "status":
                payload = session.status()
            elif action == "report":
                session.resolve_decisions()
                payload = session.report()
            else:
                return self._failure(f"Unknown replay action: {action}", "UNKNOWN_ACTION")
        except (ValueError, LookaheadError) as exc:
            return self._failure(str(exc), "REPLAY_REFUSED")
        return {"ok": True, "session_id": session_id, **payload}

    def run_scan(self, trigger_id: str, *, session_id: str = "default", bars: int = 200,
                 stop_atr_multiple: float = 1.5, reward_multiple: float = 2.0) -> dict[str, Any]:
        """Walk the cursor forward, recording decisions and scoring them after the fact."""
        from .indicators import atr, latest
        from .research import TRIGGER_REGISTRY

        session = self._sessions.get(session_id)
        if session is None:
            return self._failure("No replay session is running.", "NO_REPLAY_SESSION")
        trigger = TRIGGER_REGISTRY.get(trigger_id)
        if trigger is None:
            return self._failure(f"Unknown trigger: {trigger_id}", "UNKNOWN_TRIGGER")
        started = time.perf_counter()
        long = trigger.direction.value == "BULLISH"
        for _ in range(bars):
            if session.finished:
                break
            visible = session.visible()
            if trigger.evaluate(visible):
                atr_value = latest(atr(visible, 14))
                if atr_value and atr_value > 0:
                    entry = visible[-1].close
                    distance = atr_value * stop_atr_multiple
                    session.record_decision(
                        theory=trigger_id, setup_state="ENTRY_READY", direction=trigger.direction.value,
                        entry=entry, invalidation=entry - distance if long else entry + distance,
                        stop=entry - distance if long else entry + distance,
                        targets=[entry + distance * reward_multiple if long else entry - distance * reward_multiple],
                        notes=trigger.confirmation)
            session.advance(1)
        session.resolve_decisions()
        return {"ok": True, "session_id": session_id, "trigger": trigger.as_dict(), **session.report(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "observations": ["Every decision was taken on candles closed at or before its own bar."]}
