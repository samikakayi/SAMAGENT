"""Market analysis, the chart, replay research and strategies."""

from __future__ import annotations

import asyncio

from ..schemas import BacktestRequest, ComposedStrategyRequest, ChartLayerRequest, ClearDrawingsRequest, CustomTheoryCreate, DrawAnalysisRequest, DrawLevelRequest, GannDrawRequest, JournalCreate, MarketSnapshotRequest, PitchforkDrawRequest, ReplayControlRequest, ReplayScanRequest, ReplayStartRequest, SetupCreateRequest, SetupMonitorRequest, TradingAnalysisRequest, TradingViewActionRequest, TwoAnchorDrawRequest
from .services import AppServices
from fastapi import HTTPException
from fastapi import Query
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


def register_trading_routes(application: FastAPI, sv: AppServices) -> None:
    database = sv.database
    policy = sv.policy
    cancellation = sv.cancellation
    trading = sv.trading
    replay = sv.replay
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

    @application.get("/api/trading/setups")
    async def trading_setups(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        return {"setups": database.list_trading_setups(limit)}

    @application.post("/api/trading/setups", status_code=201)
    async def create_trading_setup(payload: SetupCreateRequest) -> dict[str, Any]:
        try:
            setup = trading.create_setup_from_last_analysis(payload.theory)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        database.add_audit("trading_setup", "created", "Setup saved from the latest verified analysis", actor="user", details={"setup_id": setup["id"], "theory": payload.theory})
        return {"setup": setup}

    @application.get("/api/trading/setups/{setup_id}")
    async def trading_setup(setup_id: str) -> dict[str, Any]:
        setup = database.get_trading_setup(setup_id)
        if setup is None:
            raise HTTPException(404, "Setup not found")
        return {"setup": setup, "events": database.list_setup_events(setup_id)}

    @application.post("/api/trading/setups/{setup_id}/monitor")
    async def monitor_setup(setup_id: str, payload: SetupMonitorRequest) -> dict[str, Any]:
        setup = database.set_setup_monitoring(setup_id, payload.enabled)
        if setup is None:
            raise HTTPException(404, "Setup not found")
        database.add_audit("setup_monitor", "started" if payload.enabled else "stopped", "Setup monitoring changed", actor="user", details={"setup_id": setup_id, "enabled": payload.enabled})
        return {"setup": setup}

    @application.get("/api/trading/journal")
    async def journal(query: str = "", limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        return {"entries": database.list_trading_journal(query, limit)}

    @application.post("/api/trading/journal", status_code=201)
    async def create_journal_entry(payload: JournalCreate) -> dict[str, Any]:
        return {"entry": database.add_trading_journal(payload.symbol, payload.theory, payload.payload, payload.setup_id)}

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
            result = trading.tradingview.calibrate(*anchors)  # type: ignore[arg-type]
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
    @application.get("/api/trading/gann")
    async def gann_analysis(symbol: str = "XAUUSD", timeframe: str = "M15") -> dict[str, Any]:
        return (await asyncio.to_thread(trading.gann_analysis, symbol, timeframe)).as_dict()

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

    @application.get("/api/trading/pitchfork")
    async def pitchfork_analysis(symbol: str = "XAUUSD", timeframe: str = "M15", variant: str = "andrews") -> dict[str, Any]:
        return (await asyncio.to_thread(trading.pitchfork_analysis, symbol, timeframe, variant=variant)).as_dict()

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

    @application.get("/api/research/replay/capability")
    async def replay_capability() -> dict[str, Any]:
        return replay.capability()

    @application.post("/api/research/replay/start")
    async def replay_start(payload: ReplayStartRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(
            replay.start, payload.symbol, payload.timeframe,
            count=payload.count, start_offset=payload.start_offset, session_id=payload.session_id,
        )
        return result.as_dict()

    @application.post("/api/research/replay/control")
    async def replay_control(payload: ReplayControlRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(replay.control, payload.action,
                                         session_id=payload.session_id, bars=payload.bars)
        return result.as_dict()

    @application.post("/api/research/replay/scan")
    async def replay_scan(payload: ReplayScanRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(
            replay.run_scan, payload.trigger, session_id=payload.session_id, bars=payload.bars,
            stop_atr_multiple=payload.stop_atr_multiple, reward_multiple=payload.reward_multiple,
        )
        return result.as_dict()

    @application.post("/api/trading/strategies", status_code=201)
    async def save_strategy(payload: ComposedStrategyRequest) -> dict[str, Any]:
        """Persist a composed strategy as a versioned custom theory."""
        from ..trading.research import TRIGGER_REGISTRY

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
