"""Executable strategies: what can fire, whether it worked, and saving it.

A strategy is only saved if it maps onto predicates SAM can actually
evaluate, which is what keeps this side of the API honest."""

from __future__ import annotations

import asyncio

from ...schemas import BacktestRequest, ComposedStrategyRequest
from ..services import AppServices
from fastapi import FastAPI
from fastapi import HTTPException
from typing import Any




def _composed_conditions(payload: Any) -> list[dict[str, Any]]:
    """Map a composed strategy onto the executable predicates SAM supports.

    Only phrases that correspond to a real predicate become conditions; free text
    is preserved as documentation rather than pretending to be executable.
    """
    lowered = " ".join(
        str(part).lower() for part in (payload.context, payload.setup, payload.confirmation, payload.entry_trigger)
    )
    conditions: list[dict[str, Any]] = []
    timeframes = payload.timeframes or ["H1", "M15", "M5"]
    higher = timeframes[0]
    lower = timeframes[-1]
    if payload.direction in {"LONG", "BOTH"} and "demand" in lowered or "bullish" in lowered:
        conditions.append({"predicate": "trend_is", "value": "BULLISH", "timeframe": higher})
    elif payload.direction == "SHORT" or "supply" in lowered or "bearish" in lowered:
        conditions.append({"predicate": "trend_is", "value": "BEARISH", "timeframe": higher})
    if "sweep" in lowered or "liquidity" in lowered:
        conditions.append({"predicate": "has_liquidity_sweep", "timeframe": timeframes[min(1, len(timeframes) - 1)]})
    if "mss" in lowered or "choch" in lowered or "shift" in lowered:
        conditions.append({"predicate": "has_mss", "timeframe": lower})
    if "bos" in lowered or "break" in lowered:
        conditions.append({"predicate": "has_bos", "timeframe": lower})
    if "fvg" in lowered or "imbalance" in lowered or "gap" in lowered:
        conditions.append({"predicate": "has_active_fvg", "timeframe": lower})
    if not conditions:
        # A strategy must be executable to be saved at all.
        conditions.append({"predicate": "trend_is", "value": "BULLISH" if payload.direction != "SHORT" else "BEARISH",
                           "timeframe": higher})
    return conditions


def register_strategy_routes(application: FastAPI, sv: AppServices) -> None:
    database = sv.database
    trading = sv.trading
    @application.get("/api/trading/triggers")
    async def list_triggers() -> dict[str, Any]:
        return (await asyncio.to_thread(trading.list_entry_triggers)).as_dict()
    @application.post("/api/trading/backtest")
    async def run_backtest(payload: BacktestRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(
            trading.backtest, payload.symbol, payload.timeframe,
            trigger=payload.trigger, count=payload.count,
            stop_atr_multiple=payload.stop_atr_multiple,
            reward_multiple=payload.reward_multiple, max_bars=payload.max_bars,
        )
        database.add_audit(
            "research", result.status.value.lower(),
            f"Backtest {payload.trigger or 'all triggers'} on {payload.symbol} {payload.timeframe}",
            actor="user", details={"verified": result.verified, "error_code": result.error_code},
        )
        return result.as_dict()
    @application.post("/api/trading/strategies", status_code=201)
    async def save_strategy(payload: ComposedStrategyRequest) -> dict[str, Any]:
        """Persist a composed strategy as a versioned custom theory."""
        from ...trading.research import TRIGGER_REGISTRY

        if payload.entry_trigger and payload.entry_trigger not in TRIGGER_REGISTRY:
            raise HTTPException(400, f"Unknown entry trigger: {payload.entry_trigger}")
        definition = {
            "name": payload.name,
            "description": " + ".join(
                part for part in (payload.context, payload.setup, payload.confirmation, payload.entry_trigger) if part
            ) or payload.name,
            "kind": "composed_strategy",
            "context": payload.context, "setup": payload.setup, "confirmation": payload.confirmation,
            "entry_trigger": payload.entry_trigger, "entry": payload.entry_trigger or payload.confirmation,
            "invalidation": payload.invalidation or "A close through the setup level.",
            "stop": payload.stop, "targets": payload.targets or ["Nearest external liquidity"],
            "timeframes": payload.timeframes or ["H1", "M15", "M5"],
            "minimum_rr": payload.minimum_rr, "direction": payload.direction,
            "conditions": _composed_conditions(payload),
        }
        try:
            saved = await asyncio.to_thread(trading.save_custom_theory, definition)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        database.add_audit("trading", "success", f"Saved composed strategy {payload.name}",
                           actor="user", details={"version": saved.get("version")})
        return saved
    @application.get("/api/trading/strategies")
    async def list_strategies() -> dict[str, Any]:
        theories = await asyncio.to_thread(database.list_custom_theories)
        return {"strategies": theories, "count": len(theories)}
