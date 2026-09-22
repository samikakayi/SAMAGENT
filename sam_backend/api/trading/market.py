"""What the market is doing, and what SAM makes of it.

Reads and analysis: the snapshot, the theories applied to it, and the
geometry studies. Nothing here draws or persists a decision."""

from __future__ import annotations

import asyncio

from ...schemas import CustomTheoryCreate, MarketSnapshotRequest, TradingAnalysisRequest
from ..services import AppServices
from fastapi import FastAPI
from fastapi import HTTPException
from typing import Any


def register_market_routes(application: FastAPI, sv: AppServices) -> None:
    database = sv.database
    policy = sv.policy
    cancellation = sv.cancellation
    trading = sv.trading
    @application.get("/api/trading/status")
    async def trading_status(symbol: str = "XAUUSD") -> dict[str, Any]:
        return await asyncio.to_thread(trading.status, symbol)
    @application.get("/api/trading/capabilities")
    async def trading_capabilities(symbol: str = "XAUUSD") -> dict[str, Any]:
        return await asyncio.to_thread(trading.market_data.providers["metatrader5"].capabilities, symbol)
    @application.post("/api/trading/snapshot")
    async def trading_snapshot(payload: MarketSnapshotRequest) -> dict[str, Any]:
        return (await asyncio.to_thread(trading.market_snapshot, payload.symbol, payload.timeframes)).as_dict()
    @application.post("/api/trading/analyze")
    async def trading_analyze(payload: TradingAnalysisRequest) -> dict[str, Any]:
        token = cancellation.create()
        try:
            result = await asyncio.to_thread(
                trading.analyze,
                payload.symbol,
                payload.timeframes,
                payload.theories,
                count=payload.count,
                minimum_rr=payload.minimum_rr,
                task_id=token.task_id,
            )
            response = result.as_dict()
            response["task_id"] = token.task_id
            return response
        except RuntimeError as exc:
            if token.cancelled:
                return {"status": "CANCELLED", "executed": True, "verified": False, "data": None, "error": str(exc), "task_id": token.task_id}
            raise HTTPException(500, str(exc)) from exc
        finally:
            cancellation.complete(token.task_id)
    @application.get("/api/trading/theories")
    async def trading_theories() -> dict[str, Any]:
        return {"built_in": trading.knowledge.list(), "custom": database.list_custom_theories()}
    @application.post("/api/trading/theories/custom", status_code=201)
    async def save_theory(payload: CustomTheoryCreate) -> dict[str, Any]:
        if policy.contains_embedded_secret(payload.definition):
            raise HTTPException(400, "Custom theories cannot contain credentials")
        try:
            theory = trading.save_custom_theory(payload.definition)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        database.add_audit("custom_theory", "created", "Versioned custom theory saved", actor="user", details={"theory_id": theory["id"], "name": theory["name"], "version": theory["version"]})
        return {"theory": theory}
    @application.get("/api/trading/skills")
    async def trading_skills() -> dict[str, Any]:
        return {"skills": trading.skills.list()}
    @application.get("/api/trading/context")
    async def trading_context() -> dict[str, Any]:
        return {"context": database.get_trading_context()}
    @application.get("/api/trading/gann")
    async def gann_analysis(symbol: str = "XAUUSD", timeframe: str = "M15") -> dict[str, Any]:
        return (await asyncio.to_thread(trading.gann_analysis, symbol, timeframe)).as_dict()
    @application.get("/api/trading/pitchfork")
    async def pitchfork_analysis(symbol: str = "XAUUSD", timeframe: str = "M15", variant: str = "andrews") -> dict[str, Any]:
        return (await asyncio.to_thread(trading.pitchfork_analysis, symbol, timeframe, variant=variant)).as_dict()
