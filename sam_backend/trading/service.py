from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from typing import Any

from ..cancellation import CancellationManager
from ..contracts import StandardResult
from ..db import Database
from .analyst import DEFAULT_TIMEFRAMES, MarketAnalyst
from .calibration import ChartCalibrator
from .drawing import DrawingEngine, DrawRequest, Layer, TwoAnchorRequest
from .geometry import calculate_pitchfork_anchors, gann_box, gann_fan, square_of_nine_levels
from .market_data import MarketDataError, MarketDataService
from .registry import TradingKnowledgeRegistry, build_skill_registry
from .research import TRIGGER_REGISTRY, backtest_trigger, compare_triggers
from .tradingview import TradingViewController
from .types import Direction, SetupState, normalize_timeframe



class TradingService:
    def __init__(self, settings: Any, database: Database, cancellation: CancellationManager) -> None:
        self.settings = settings
        self.database = database
        self.cancellation = cancellation
        self.market_data = MarketDataService()
        self.knowledge = TradingKnowledgeRegistry()
        self.skills = build_skill_registry()
        self.tradingview = TradingViewController(
            settings.data_dir,
            computer_control=bool(getattr(settings, "computer_control_enabled", False)),
            screen_access=bool(getattr(settings, "screen_access_enabled", False)),
        )
        self.calibrator = ChartCalibrator()
        self.drawing = DrawingEngine(
            database=database,
            calibrator=self.calibrator,
            observe=self.tradingview.observe,
            focus=self.tradingview.focus,
            computer_control=bool(getattr(settings, "computer_control_enabled", False)),
            screen_access=bool(getattr(settings, "screen_access_enabled", False)),
        )
        self.analyst = MarketAnalyst(settings, database, self.market_data, self.knowledge, cancellation)

    def refresh_permissions(self) -> None:
        computer_control = bool(getattr(self.settings, "computer_control_enabled", False))
        screen_access = bool(getattr(self.settings, "screen_access_enabled", False))
        self.tradingview.computer_control = computer_control
        self.tradingview.screen_access = screen_access
        self.drawing.refresh_permissions(computer_control=computer_control, screen_access=screen_access)

    def status(self, symbol: str = "XAUUSD") -> dict[str, Any]:
        # One window scan, so both halves of the answer describe the same moment.
        observation = self.tradingview.observe()
        return {
            "market_data": self.market_data.health(),
            "capabilities": self.market_data.providers["metatrader5"].capabilities(symbol),
            "tradingview": observation.as_dict(),
            "drawing": self.drawing.capability(observation),
            "context": self.database.get_trading_context(),
            "monitors": [item for item in self.database.list_trading_setups(100) if item["monitor_enabled"]],
        }

    # --- Drawing -----------------------------------------------------------

    def calibrate_chart(self) -> StandardResult:
        return self.drawing.calibrate()

    def verify_calibration(self) -> StandardResult:
        return self.drawing.verify_calibration()

    def _chart_symbol_matches(self, expected: str | None) -> StandardResult | None:
        """Refuse to annotate a chart showing a different instrument.

        Price alone is not a safe guard: two instruments can trade in the same
        numeric range, and levels derived from one would then be drawn onto the
        other without anything looking wrong.
        """
        if not expected:
            return None
        state = self.tradingview.observe()
        shown = (state.symbol or "").upper()
        wanted = expected.upper()
        if not shown:
            return None
        # Providers and TradingView spell the same instrument differently, so a
        # containment match either way is treated as agreement.
        if shown == wanted or shown.endswith(wanted) or wanted.endswith(shown) or wanted in shown or shown in wanted:
            return None
        return StandardResult.failure(
            f"The chart is showing {shown} but this analysis is for {wanted}. Nothing was drawn: levels from one "
            f"instrument must never be placed on another. Switch the chart to {wanted}, or analyse {shown} instead.",
            error_code="SYMBOL_MISMATCH",
        )

    def draw_annotation(
        self,
        annotation: str,
        price: float,
        *,
        label: str = "",
        theory: str = "",
        setup_id: str | None = None,
        layer: str | None = None,
        symbol: str | None = None,
    ) -> StandardResult:
        mismatch = self._chart_symbol_matches(symbol)
        if mismatch is not None:
            return mismatch
        request = DrawRequest(
            annotation=annotation,
            price=float(price),
            label=label,
            theory=theory,
            setup_id=setup_id,
            layer=Layer[layer.upper()] if layer and layer.upper() in Layer.__members__ else None,
        )
        return self.drawing.draw(request)

    def draw_two_anchor(
        self,
        annotation: str,
        price_a: float,
        minutes_a: float,
        price_b: float,
        minutes_b: float,
        *,
        label: str = "",
        theory: str = "",
        setup_id: str | None = None,
        layer: str | None = None,
    ) -> StandardResult:
        return self.drawing.draw_two_anchor(TwoAnchorRequest(
            annotation=annotation,
            price_a=float(price_a), minutes_a=float(minutes_a),
            price_b=float(price_b), minutes_b=float(minutes_b),
            label=label, theory=theory, setup_id=setup_id,
            layer=Layer[layer.upper()] if layer and layer.upper() in Layer.__members__ else None,
        ))

    def draw_analysis(self, report: dict[str, Any] | None = None, *, theory: str = "", setup_id: str | None = None) -> StandardResult:
        """Turn a completed analysis report into a verified annotation plan."""
        source = report or self.analyst.latest
        if not source:
            return StandardResult.failure(
                "There is no completed analysis to draw. Run analyze_market first.",
                error_code="NO_ANALYSIS",
            )
        mismatch = self._chart_symbol_matches(source.get("symbol"))
        if mismatch is not None:
            return mismatch
        requests = [
            DrawRequest(annotation=annotation, price=float(price), label=label, theory=theory or source.get("theory", ""), setup_id=setup_id)
            for annotation, price, label in self._drawing_plan(source)
        ]
        if not requests:
            return StandardResult.failure(
                "The analysis produced no price levels worth annotating.", error_code="EMPTY_DRAWING_PLAN"
            )
        return self.drawing.draw_plan(requests)

    @staticmethod
    def _drawing_plan(report: dict[str, Any]) -> list[tuple[str, float, str]]:
        """Select the annotations worth placing, nearest levels and setup prices only."""
        plan: list[tuple[str, float, str]] = []
        setup = report.get("setup") or {}
        for key, annotation in (("entry", "entry"), ("stop", "stop"), ("invalidation", "invalidation")):
            value = setup.get(key)
            if isinstance(value, (int, float)):
                plan.append((annotation, float(value), annotation.upper()))
        for index, target in enumerate((setup.get("targets") or [])[:3], start=1):
            price = target.get("price") if isinstance(target, dict) else target
            if isinstance(price, (int, float)):
                plan.append((f"tp{index}", float(price), f"TP{index}"))
        for side, annotation in (("support", "support"), ("resistance", "resistance")):
            for level in (report.get(side) or [])[:3]:
                price = level.get("price") if isinstance(level, dict) else level
                if isinstance(price, (int, float)):
                    plan.append((annotation, float(price), f"{annotation.upper()} {price}"))
        return plan

    def list_drawings(self, **filters: Any) -> StandardResult:
        return self.drawing.list_owned(**filters)

    def clear_drawings(self, **filters: Any) -> StandardResult:
        return self.drawing.clear_owned(**filters)

    def set_layer_visibility(self, layer: str, visible: bool, symbol: str | None = None) -> StandardResult:
        return self.drawing.set_layer_visibility(layer, visible, symbol)

    # --- Gann and pitchfork ------------------------------------------------

    def _candles_for_symbol(self, symbol: str, timeframe: str, count: int = 600) -> list[Any]:
        return self.market_data.providers["metatrader5"].fetch(symbol, normalize_timeframe(timeframe), count).candles

    def gann_analysis(self, symbol: str = "XAUUSD", timeframe: str = "M15", *, bars_forward: int = 60) -> StandardResult:
        started = time.perf_counter()
        try:
            candles = self._candles_for_symbol(symbol, timeframe)
        except MarketDataError as exc:
            return StandardResult.failure(str(exc), error_code="MARKET_DATA_UNAVAILABLE", started_at=started)
        fan = gann_fan(candles, bars_forward=bars_forward)
        box = gann_box(candles)
        square = square_of_nine_levels(candles[-1].close) if candles else {"available": False}
        return StandardResult.success(
            {"symbol": symbol, "timeframe": normalize_timeframe(timeframe),
             "fan": fan, "box": box, "square_of_nine": square},
            verified=True, started_at=started,
            observations=["Gann anchors are analyst-chosen; every pivot used is reported with the output."],
        )

    def draw_gann_fan(self, symbol: str = "XAUUSD", timeframe: str = "M15", *, max_rays: int = 5,
                      setup_id: str | None = None) -> StandardResult:
        analysis = self.gann_analysis(symbol, timeframe)
        fan = (analysis.data or {}).get("fan") or {}
        if not fan.get("available"):
            return StandardResult.failure(fan.get("reason", "A Gann fan could not be built."), error_code="GANN_UNAVAILABLE")
        # Draw the shallow rays first: they stay on screen longest and carry the 1x1.
        rays = sorted(fan["rays"], key=lambda ray: abs(ray["slope_price_per_minute"]))[:max_rays]
        result = self.drawing.draw_line_plan(
            rays, annotation="gann_fan", theory="gann", layer=Layer.THEORY, setup_id=setup_id, max_lines=max_rays
        )
        if isinstance(result.data, dict):
            result.data["limitation"] = fan["limitation"]
            result.data["pivot"] = fan["pivot"]
        return result

    def pitchfork_analysis(self, symbol: str = "XAUUSD", timeframe: str = "M15", *, variant: str = "andrews") -> StandardResult:
        started = time.perf_counter()
        try:
            candles = self._candles_for_symbol(symbol, timeframe)
        except MarketDataError as exc:
            return StandardResult.failure(str(exc), error_code="MARKET_DATA_UNAVAILABLE", started_at=started)
        result = calculate_pitchfork_anchors(candles, variant=variant)
        if not result.get("available"):
            return StandardResult.failure(result.get("reason", "Pitchfork anchors unavailable."),
                                          error_code="PITCHFORK_UNAVAILABLE", started_at=started)
        return StandardResult.success({"symbol": symbol, "timeframe": normalize_timeframe(timeframe), **result},
                                      verified=True, started_at=started)

    def draw_pitchfork(self, symbol: str = "XAUUSD", timeframe: str = "M15", *, variant: str = "andrews",
                       setup_id: str | None = None) -> StandardResult:
        analysis = self.pitchfork_analysis(symbol, timeframe, variant=variant)
        if not analysis.verified:
            return analysis
        data = analysis.data
        lines = [
            {"label": "median", "start": data["median_line"]["start"], "end": data["median_line"]["end"]},
            {"label": "upper", "start": data["upper_parallel"]["start"], "end": data["upper_parallel"]["end"]},
            {"label": "lower", "start": data["lower_parallel"]["start"], "end": data["lower_parallel"]["end"]},
        ]
        result = self.drawing.draw_line_plan(
            lines, annotation=f"pitchfork_{variant}", theory="pitchfork",
            layer=Layer.THEORY, setup_id=setup_id, max_lines=3,
        )
        if isinstance(result.data, dict):
            result.data["limitation"] = data["limitation"]
            result.data["anchors"] = data["anchors"]
            result.data["variant"] = variant
        return result

    # --- Research ----------------------------------------------------------

    def list_entry_triggers(self) -> StandardResult:
        return StandardResult.success(
            {"triggers": [item.as_dict() for item in TRIGGER_REGISTRY.values()], "count": len(TRIGGER_REGISTRY)},
            verified=True,
        )

    def backtest(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "M15",
        *,
        trigger: str | None = None,
        count: int = 3000,
        stop_atr_multiple: float = 1.5,
        reward_multiple: float = 2.0,
        max_bars: int = 60,
    ) -> StandardResult:
        """Backtest one trigger, or compare them all, on real provider history."""
        started = time.perf_counter()
        normalized = normalize_timeframe(timeframe)
        try:
            batch = self.market_data.providers["metatrader5"].fetch(symbol, normalized, count)
        except MarketDataError as exc:
            return StandardResult.failure(str(exc), error_code="MARKET_DATA_UNAVAILABLE", started_at=started)
        options = {
            "timeframe": normalized, "symbol": batch.resolved_symbol,
            "stop_atr_multiple": stop_atr_multiple, "reward_multiple": reward_multiple, "max_bars": max_bars,
        }
        if trigger:
            if trigger not in TRIGGER_REGISTRY:
                return StandardResult.failure(f"Unknown entry trigger: {trigger}", error_code="UNKNOWN_TRIGGER", started_at=started)
            result = backtest_trigger(batch.candles, trigger, **options)
        else:
            result = compare_triggers(batch.candles, **options)
        payload = {
            **result,
            "provider": batch.metadata(),
            "note": (
                "Each decision saw only candles closed at or before its bar; outcomes were resolved on later "
                "bars, and a candle spanning both stop and target is counted as a loss."
            ),
        }
        if isinstance(result, dict) and result.get("available") is False:
            return StandardResult.failure(result.get("reason", "Backtest unavailable"),
                                          error_code="INSUFFICIENT_HISTORY", started_at=started)
        return StandardResult.success(payload, verified=True, started_at=started)


    # --- Analysis ------------------------------------------------------------
    # The pipeline lives in MarketAnalyst; these keep the public surface.

    def market_snapshot(self, symbol: str = "XAUUSD", timeframes: list[str] | None = None) -> StandardResult:
        return self.analyst.market_snapshot(symbol, timeframes)

    def analyze(
        self, symbol: str = "XAUUSD", timeframes: list[str] | None = None, theories: list[str] | None = None,
        *, count: int = 600, minimum_rr: float | None = None, task_id: str | None = None,
    ) -> StandardResult:
        return self.analyst.analyze(symbol, timeframes, theories, count=count, minimum_rr=minimum_rr, task_id=task_id)

    def route_natural_intent(self, message: str, *, task_id: str | None = None) -> StandardResult | None:
        """Handle deterministic trading/desktop intents without spending an LLM call."""
        lowered = message.lower().strip()
        if lowered in {"stop", "cancel", "sam stop", "وەستە", "سام وەستە", "هەڵوەشێنەوە"}:
            outcome = self.cancellation.emergency_stop("voice_or_text_stop")
            return StandardResult.success({**outcome, "spoken_summary_ckb": "وەستام. هەموو کارە cancellable ـەکان هەڵوەشێنرانەوە."})
        mentions_tradingview = "tradingview" in lowered or "ترەیدینگڤیو" in lowered or "تریدینگڤیو" in lowered
        open_terms = ("open", "focus", "بکەرەوە", "بخەرە پێشەوە")
        if mentions_tradingview and any(term in lowered for term in open_terms):
            self.refresh_permissions()
            result = self.tradingview.launch()
            if isinstance(result.data, dict):
                result.data["spoken_summary_ckb"] = "TradingView خرا پێشەوە." if result.verified else "TradingView دۆزرایەوە، بەڵام focus ـەکە تەواو پشتڕاست نەکرایەوە."
            return result
        timeframe_match = re.search(r"(?i)\b(1m|3m|5m|15m|30m|45m|1h|2h|4h|1d|1w)\b", message)
        switch_terms = ("بچۆ", "switch", "set timeframe", "timeframe")
        if mentions_tradingview and timeframe_match and any(term in lowered for term in switch_terms):
            self.refresh_permissions()
            return self.tradingview.set_timeframe(timeframe_match.group(1))
        analysis_terms = ("analyze", "analysis", "شیکار", "هەڵسەنگاندن", "entry", "ئینتری")
        if any(term in lowered for term in analysis_terms):
            symbol_match = re.search(r"\b(XAUUSD|XAGUSD|EURUSD|GBPUSD|BTCUSD|ETHUSD|US30|NAS100|SPX500)\b", message, re.I)
            symbol = symbol_match.group(1).upper() if symbol_match else str(self.database.get_trading_context().get("symbol") or "XAUUSD")
            timeframes = [normalize_timeframe(value) for value in re.findall(r"(?i)\b(?:1m|3m|5m|15m|30m|45m|1h|2h|4h|1d|1w)\b", message)]
            if not timeframes:
                timeframes = DEFAULT_TIMEFRAMES
            theory_terms = {
                "snr": ("snr", "support", "resistance", "پاڵپشتی", "بەرگری"),
                "wyckoff": ("wyckoff", "وایکۆف"),
                "ict": ("ict",),
                "smc": ("smc", "smart money"),
                "liquidity": ("liquidity", "لیکویدیتی"),
                "price_action": ("price action",),
            }
            theories = [theory for theory, markers in theory_terms.items() if any(marker in lowered for marker in markers)] or ["default"]
            return self.analyze(symbol, timeframes, theories, task_id=task_id)
        monitor_terms = ("monitor", "چاودێری")
        if any(term in lowered for term in monitor_terms) and ("setup" in lowered or "سێتەپ" in lowered):
            if self.analyst.latest is None:
                return StandardResult.failure("No completed analysis is available to monitor.", error_code="NO_SETUP_CONTEXT")
            setup = self.create_setup_from_analysis(self.analyst.latest, next(iter(self.analyst.latest.get("theories") or {"default": None})))
            enabled = self.database.set_setup_monitoring(setup["id"], True)
            return StandardResult.success({"setup": enabled, "spoken_summary_ckb": "چاودێریکردنی setup ـەکە دەستی پێکرد."})
        return None


    def create_setup_from_analysis(self, report: dict[str, Any], theory: str = "default") -> dict[str, Any]:
        setup = self.database.create_trading_setup(
            report["symbol"],
            report.get("feed"),
            theory,
            report.get("setup_state") or SetupState.NO_SETUP.value,
            {
                "direction": (report.get("long_scenario") or report.get("short_scenario") or {}).get("direction"),
                "entry": report.get("entry"),
                "stop": report.get("stop"),
                "targets": [price for price in (report.get("tp1"), report.get("tp2"), report.get("tp3")) if price is not None],
                "rr": report.get("rr"),
                "invalidation": report.get("invalidation"),
                "confidence": report.get("confidence"),
            },
        )
        self.database.add_trading_journal(report["symbol"], theory, report, setup["id"])
        return setup

    def create_setup_from_last_analysis(self, theory: str = "default") -> dict[str, Any]:
        if self.analyst.latest is None:
            raise ValueError("Run a market analysis before creating a monitored setup.")
        return self.create_setup_from_analysis(self.analyst.latest, theory)

    def poll_monitors(self) -> list[dict[str, Any]]:
        transitions: list[dict[str, Any]] = []
        for setup in self.database.list_trading_setups(500):
            if not setup["monitor_enabled"]:
                continue
            payload = setup["payload"]
            try:
                batch = self.market_data.providers["metatrader5"].fetch(setup["symbol"], "M1", 3)
            except Exception:
                continue
            price = batch.current_price
            if price is None:
                continue
            direction = payload.get("direction")
            entry = payload.get("entry")
            stop = payload.get("stop")
            targets = payload.get("targets") or []
            next_state = setup["state"]
            reason = None
            if direction == Direction.BULLISH.value:
                if stop is not None and price <= stop:
                    next_state, reason = SetupState.STOPPED.value, "Price reached the technical stop."
                elif len(targets) > 2 and price >= targets[2]:
                    next_state, reason = SetupState.TP3.value, "Price reached TP3."
                elif len(targets) > 1 and price >= targets[1]:
                    next_state, reason = SetupState.TP2.value, "Price reached TP2."
                elif targets and price >= targets[0]:
                    next_state, reason = SetupState.TP1.value, "Price reached TP1."
                elif entry is not None and setup["state"] == SetupState.ENTRY_READY.value and price >= entry:
                    next_state, reason = SetupState.ENTRY_TRIGGERED.value, "Price crossed the confirmed entry."
            elif direction == Direction.BEARISH.value:
                if stop is not None and price >= stop:
                    next_state, reason = SetupState.STOPPED.value, "Price reached the technical stop."
                elif len(targets) > 2 and price <= targets[2]:
                    next_state, reason = SetupState.TP3.value, "Price reached TP3."
                elif len(targets) > 1 and price <= targets[1]:
                    next_state, reason = SetupState.TP2.value, "Price reached TP2."
                elif targets and price <= targets[0]:
                    next_state, reason = SetupState.TP1.value, "Price reached TP1."
                elif entry is not None and setup["state"] == SetupState.ENTRY_READY.value and price <= entry:
                    next_state, reason = SetupState.ENTRY_TRIGGERED.value, "Price crossed the confirmed entry."
            if reason and next_state != setup["state"]:
                updated = self.database.transition_trading_setup(setup["id"], next_state, reason, {"last_price": price, "last_checked_at": datetime.now(UTC).isoformat()})
                transitions.append({"setup": updated, "reason": reason})
        return transitions

    def validate_custom_theory(self, definition: dict[str, Any]) -> dict[str, Any]:
        required = ["name", "description", "conditions", "invalidation", "targets"]
        missing = [key for key in required if not definition.get(key)]
        conditions = definition.get("conditions") or []
        errors = []
        if not isinstance(conditions, list):
            errors.append("conditions must be a list")
        else:
            allowed = {"trend_is", "has_liquidity_sweep", "has_bos", "has_mss", "has_active_fvg", "rsi_above", "rsi_below"}
            for index, condition in enumerate(conditions):
                if not isinstance(condition, dict) or condition.get("predicate") not in allowed:
                    errors.append(f"condition {index + 1} uses an unsupported predicate")
        if missing:
            errors.append("missing required fields: " + ", ".join(missing))
        return {"valid": not errors, "errors": errors, "supported_predicates": sorted({"trend_is", "has_liquidity_sweep", "has_bos", "has_mss", "has_active_fvg", "rsi_above", "rsi_below"})}

    def save_custom_theory(self, definition: dict[str, Any]) -> dict[str, Any]:
        validation = self.validate_custom_theory(definition)
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))
        return self.database.save_custom_theory(str(definition["name"]), {**definition, "schema_version": 1, "validated_at": datetime.now(UTC).isoformat()})
