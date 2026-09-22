"""The chart itself: what TradingView is showing and what is drawn on it.

Every route here reaches the real application through the observer, so
calibration and containment rules apply to all of them alike."""

from __future__ import annotations

import asyncio

from ...schemas import ChartLayerRequest, ClearDrawingsRequest, DrawAnalysisRequest, DrawLevelRequest, GannDrawRequest, PitchforkDrawRequest, TradingViewActionRequest, TwoAnchorDrawRequest
from ..services import AppServices
from fastapi import FastAPI
from fastapi import HTTPException
from typing import Any


def register_chart_routes(application: FastAPI, sv: AppServices) -> None:
    database = sv.database
    trading = sv.trading
    @application.get("/api/tradingview/state")
    async def tradingview_state() -> dict[str, Any]:
        return trading.tradingview.observe().as_dict()
    @application.post("/api/tradingview/action")
    async def tradingview_action(payload: TradingViewActionRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        if payload.action in {"launch", "focus"}:
            result = await asyncio.to_thread(
                trading.tradingview.launch if payload.action == "launch" else trading.tradingview.focus
            )
        elif payload.action == "set_symbol":
            if not payload.symbol:
                raise HTTPException(400, "symbol is required")
            result = await asyncio.to_thread(trading.tradingview.set_symbol, payload.symbol)
        elif payload.action == "set_timeframe":
            if not payload.timeframe:
                raise HTTPException(400, "timeframe is required")
            result = await asyncio.to_thread(trading.tradingview.set_timeframe, payload.timeframe)
        elif payload.action == "capture":
            result = await asyncio.to_thread(trading.tradingview.capture)
        elif payload.action == "auto_calibrate":
            # Screen capture and OCR are blocking; they must never run on the loop.
            result = await asyncio.to_thread(trading.calibrate_chart)
        elif payload.action == "verify_calibration":
            result = await asyncio.to_thread(trading.verify_calibration)
        else:
            anchors = (payload.price_a, payload.y_a, payload.price_b, payload.y_b)
            if any(value is None for value in anchors):
                raise HTTPException(400, "price_a, y_a, price_b, and y_b are required")
            result = trading.drawing.calibrate_from_anchors(*anchors)  # type: ignore[arg-type]
        database.add_audit("tradingview", result.status.value.lower(), f"TradingView {payload.action}", actor="user", details={"verified": result.verified, "error_code": result.error_code})
        return result.as_dict()
    @application.get("/api/tradingview/drawings")
    async def list_drawings(
        symbol: str | None = None,
        layer: str | None = None,
        theory: str | None = None,
        setup_id: str | None = None,
        visible_only: bool = False,
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(
            trading.list_drawings, symbol=symbol, layer=layer, theory=theory,
            setup_id=setup_id, visible_only=visible_only,
        )
        return result.as_dict()
    @application.post("/api/tradingview/drawings")
    async def draw_level(payload: DrawLevelRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(
            trading.draw_annotation, payload.annotation, payload.price,
            label=payload.label, theory=payload.theory, setup_id=payload.setup_id,
            layer=payload.layer, symbol=payload.symbol,
        )
        database.add_audit(
            "tradingview", result.status.value.lower(), f"Draw {payload.annotation} at {payload.price}",
            actor="user", details={"verified": result.verified, "error_code": result.error_code},
        )
        return result.as_dict()
    @application.post("/api/tradingview/drawings/analysis")
    async def draw_analysis(payload: DrawAnalysisRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(trading.draw_analysis, None, theory=payload.theory, setup_id=payload.setup_id)
        database.add_audit(
            "tradingview", result.status.value.lower(), "Draw analysis plan",
            actor="user", details={"verified": result.verified, "error_code": result.error_code},
        )
        return result.as_dict()
    @application.post("/api/tradingview/drawings/clear")
    async def clear_drawings(payload: ClearDrawingsRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(
            trading.clear_drawings, symbol=payload.symbol, layer=payload.layer,
            theory=payload.theory, setup_id=payload.setup_id, all_owned=payload.all_owned,
        )
        database.add_audit(
            "tradingview", result.status.value.lower(), "Clear SAM drawings",
            actor="user", details={"verified": result.verified, "error_code": result.error_code},
        )
        return result.as_dict()
    @application.post("/api/tradingview/layers")
    async def set_chart_layer(payload: ChartLayerRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(trading.set_layer_visibility, payload.layer, payload.visible, payload.symbol)
        return result.as_dict()
    @application.post("/api/tradingview/drawings/object")
    async def draw_object(payload: TwoAnchorDrawRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(
            trading.draw_two_anchor, payload.annotation,
            payload.price_a, payload.minutes_a, payload.price_b, payload.minutes_b,
            label=payload.label, theory=payload.theory, setup_id=payload.setup_id, layer=payload.layer,
        )
        database.add_audit(
            "tradingview", result.status.value.lower(), f"Draw {payload.annotation} (two anchors)",
            actor="user", details={"verified": result.verified, "error_code": result.error_code},
        )
        return result.as_dict()
    @application.post("/api/tradingview/drawings/gann")
    async def draw_gann(payload: GannDrawRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(
            trading.draw_gann_fan, payload.symbol, payload.timeframe,
            max_rays=payload.max_rays, setup_id=payload.setup_id,
        )
        database.add_audit("tradingview", result.status.value.lower(), "Draw Gann fan",
                           actor="user", details={"verified": result.verified, "error_code": result.error_code})
        return result.as_dict()
    @application.post("/api/tradingview/drawings/pitchfork")
    async def draw_pitchfork(payload: PitchforkDrawRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(
            trading.draw_pitchfork, payload.symbol, payload.timeframe,
            variant=payload.variant, setup_id=payload.setup_id,
        )
        database.add_audit("tradingview", result.status.value.lower(), f"Draw {payload.variant} pitchfork",
                           actor="user", details={"verified": result.verified, "error_code": result.error_code})
        return result.as_dict()
