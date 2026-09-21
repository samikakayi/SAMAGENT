from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from typing import Any

from ..cancellation import CancellationManager, CancellationToken
from ..contracts import CapabilityState, ExecutionStatus, StandardResult
from ..db import Database
from .analysis import entry_hunter, self_check, session_context, timeframe_analysis
from .calibration import ChartCalibrator
from .drawing import DrawingEngine, DrawRequest, Layer, TwoAnchorRequest
from .geometry import calculate_pitchfork_anchors, gann_box, gann_fan, square_of_nine_levels
from .market_data import MarketDataError, MarketDataService, MetaTrader5Provider
from .registry import TradingKnowledgeRegistry, build_skill_registry
from .research import TRIGGER_REGISTRY, backtest_trigger, compare_triggers
from .tradingview import TradingViewController
from .types import Direction, SetupDecision, SetupState, normalize_timeframe


DEFAULT_TIMEFRAMES = ["H1", "M15", "M5", "M1"]


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
        self._last_report: dict[str, Any] | None = None

    def refresh_permissions(self) -> None:
        computer_control = bool(getattr(self.settings, "computer_control_enabled", False))
        screen_access = bool(getattr(self.settings, "screen_access_enabled", False))
        self.tradingview.computer_control = computer_control
        self.tradingview.screen_access = screen_access
        self.drawing.refresh_permissions(computer_control=computer_control, screen_access=screen_access)

    def status(self, symbol: str = "XAUUSD") -> dict[str, Any]:
        return {
            "market_data": self.market_data.health(),
            "capabilities": self.market_data.providers["metatrader5"].capabilities(symbol),
            "tradingview": self.tradingview.observe().as_dict(),
            "drawing": self.drawing.capability(),
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
        source = report or self._last_report
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

    @staticmethod
    def _canonical_timeframes(timeframes: list[str] | None) -> list[str]:
        values = timeframes or DEFAULT_TIMEFRAMES
        result: list[str] = []
        for value in values:
            normalized = normalize_timeframe(value)
            if normalized not in result:
                result.append(normalized)
        if not result or len(result) > 8:
            raise ValueError("Choose between 1 and 8 distinct timeframes")
        return result

    def market_snapshot(self, symbol: str = "XAUUSD", timeframes: list[str] | None = None) -> StandardResult:
        started = time.perf_counter()
        requested = self._canonical_timeframes(timeframes)
        provider = self.market_data.providers["metatrader5"]
        snapshots: dict[str, Any] = {}
        batches: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for timeframe in requested:
            try:
                batch = provider.fetch(symbol, timeframe, 3)
                batches[timeframe] = batch
                candle = batch.candles[-1]
                snapshots[timeframe] = {
                    "metadata": batch.metadata(),
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "time": candle.time.isoformat(),
                }
            except Exception as exc:
                errors[timeframe] = str(exc)
        if not snapshots:
            return StandardResult.failure(
                "No live market snapshot was available.",
                error_code="MARKET_DATA_UNAVAILABLE",
                started_at=started,
                observations=list(errors.values()),
            )
        quality_verified = not errors and all(batch.data_quality_verified for batch in batches.values())
        status = ExecutionStatus.SUCCESS if quality_verified else ExecutionStatus.PARTIAL
        first = next(iter(snapshots.values()))["metadata"]
        if errors:
            error = "Some requested timeframes were unavailable."
        elif not quality_verified:
            error = "Market snapshot completed with provider timestamp or freshness verification gaps."
        else:
            error = None
        return StandardResult(
            status,
            True,
            quality_verified,
            data={
                "symbol": first["resolved_symbol"],
                "feed": first["feed"],
                "bid": first["bid"],
                "ask": first["ask"],
                "spread": (first["ask"] - first["bid"]) if first["ask"] is not None and first["bid"] is not None else None,
                "session": session_context(),
                "timeframes": snapshots,
                "errors": errors,
            },
            error=error,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            observations=[message for batch in batches.values() for message in batch.observations] + list(errors.values()),
        )

    def analyze(
        self,
        symbol: str = "XAUUSD",
        timeframes: list[str] | None = None,
        theories: list[str] | None = None,
        *,
        count: int = 600,
        minimum_rr: float | None = None,
        task_id: str | None = None,
    ) -> StandardResult:
        started = time.perf_counter()
        token = self.cancellation.get(task_id) if task_id else None
        requested_timeframes = self._canonical_timeframes(timeframes)
        requested_theories = theories or [str(getattr(self.settings, "default_trading_theory", "default"))]
        minimum_rr = float(minimum_rr if minimum_rr is not None else getattr(self.settings, "minimum_rr", 1.5))
        analyses: dict[str, dict[str, Any]] = {}
        batches: dict[str, Any] = {}
        errors: dict[str, str] = {}
        provider: MetaTrader5Provider = self.market_data.providers["metatrader5"]
        for timeframe in requested_timeframes:
            self._raise_if_cancelled(token)
            try:
                batch = provider.fetch(symbol, timeframe, count)
                batches[timeframe] = batch
                analysis = timeframe_analysis(batch)
                # Pattern engines rerun on the same candles the analysis used, so
                # they are kept on the analysis and stripped before serialization.
                analysis["_candles"] = batch.candles
                analyses[timeframe] = analysis
            except Exception as exc:
                errors[timeframe] = str(exc)
        if not analyses:
            return StandardResult.failure(
                "Analysis could not start because no requested timeframe returned verified OHLC data.",
                error_code="MARKET_DATA_UNAVAILABLE",
                started_at=started,
                observations=list(errors.values()),
            )
        self._raise_if_cancelled(token)
        theory_outputs: dict[str, Any] = {}
        for theory_name in requested_theories:
            try:
                theory = self.knowledge.get(theory_name)
                theory_outputs[theory.id] = self._execute_theory(theory.id, analyses)
            except KeyError as exc:
                custom = self.database.get_custom_theory(name=theory_name)
                if custom:
                    theory_outputs[f"custom:{custom['name']}"] = self._execute_custom_theory(custom, analyses)
                else:
                    theory_outputs[theory_name] = {"status": CapabilityState.UNAVAILABLE.value, "error": str(exc)}
        setup = entry_hunter(analyses, minimum_rr=minimum_rr)
        checks = self_check(analyses, setup, symbol, requested_timeframes)
        critical_failures = set(checks["critical_failures"])
        timestamp_failures = {
            "data_fresh", "timestamps_not_future", "quote_timestamps_verified",
        }
        if critical_failures & timestamp_failures:
            setup = {
                **setup,
                "decision": SetupDecision.WAIT.value,
                "state": SetupState.WATCH.value,
                "entry": None,
                "entry_zone": None,
                "technical_invalidation": None,
                "stop": None,
                "stop_distance": None,
                "targets": [],
                "rr": None,
                "reasons": [
                    "Provider timestamps or quote freshness could not be verified; analysis-only setup output is blocked until the feed clock is trustworthy."
                ],
                "missing_confirmation": checks["critical_failures"],
            }
        elif setup.get("decision") == SetupDecision.ENTRY_READY.value and not checks["passed"]:
            setup = {
                **setup,
                "decision": SetupDecision.WAIT.value,
                "state": SetupState.WAITING_FOR_TRIGGER.value,
                "entry": None,
                "entry_zone": None,
                "technical_invalidation": None,
                "stop": None,
                "stop_distance": None,
                "targets": [],
                "rr": None,
                "reasons": ["Self-check blocked ENTRY_READY because a critical verification failed."],
                "missing_confirmation": checks["critical_failures"],
            }
        first_batch = next(iter(batches.values()))
        report = self._report(
            requested_symbol=symbol,
            resolved_symbol=first_batch.resolved_symbol,
            feed=first_batch.feed,
            analyses=analyses,
            theories=theory_outputs,
            setup=setup,
            checks=checks,
            errors=errors,
        )
        report["spoken_summary_ckb"] = self._sorani_summary(report)
        self._last_report = report
        self.database.set_trading_context({
            "symbol": first_batch.resolved_symbol,
            "feed": first_batch.feed,
            "timeframes": requested_timeframes,
            "theories": requested_theories,
            "analysis_timestamp": datetime.now(UTC).isoformat(),
            "setup_state": setup.get("state"),
        })
        status = ExecutionStatus.PARTIAL if errors or not checks["passed"] else ExecutionStatus.SUCCESS
        self.database.add_audit(
            "trading_analysis",
            status.value.lower(),
            f"Deterministic analysis completed for {first_batch.resolved_symbol}",
            details={
                "symbol": first_batch.resolved_symbol,
                "feed": first_batch.feed,
                "timeframes": requested_timeframes,
                "theories": list(theory_outputs),
                "decision": setup.get("decision"),
                "self_check": checks["passed"],
                "errors": errors,
            },
        )
        return StandardResult(
            status,
            True,
            checks["passed"] and not errors,
            data=report,
            error="Analysis completed with verification gaps." if status == ExecutionStatus.PARTIAL else None,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            observations=[message for batch in batches.values() for message in batch.observations] + list(errors.values()),
        )

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
            if self._last_report is None:
                return StandardResult.failure("No completed analysis is available to monitor.", error_code="NO_SETUP_CONTEXT")
            setup = self.create_setup_from_analysis(self._last_report, next(iter(self._last_report.get("theories") or {"default": None})))
            enabled = self.database.set_setup_monitoring(setup["id"], True)
            return StandardResult.success({"setup": enabled, "spoken_summary_ckb": "چاودێریکردنی setup ـەکە دەستی پێکرد."})
        return None

    @staticmethod
    def _raise_if_cancelled(token: CancellationToken | None) -> None:
        if token and token.cancelled:
            raise RuntimeError(f"Task cancelled: {token.reason or 'user'}")

    def _execute_theory(self, theory_id: str, analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
        theory = self.knowledge.get(theory_id)
        if theory.health in {CapabilityState.UNAVAILABLE, CapabilityState.UNCONFIGURED}:
            return {
                "theory": theory.as_dict(),
                "status": theory.health.value,
                "facts": [],
                "observations": [],
                "interpretation": None,
                "reason": theory.limitations[0] if theory.limitations else "Required data is unavailable.",
            }
        facts: list[dict[str, Any]] = []
        observations: list[str] = []
        directions: list[Direction] = []
        for timeframe, analysis in analyses.items():
            direction = Direction(analysis["structure"]["trend"])
            directions.append(direction)
            facts.append({"timeframe": timeframe, "structure": direction.value, "current_price": analysis["current_price"]})
            observations.append(f"{timeframe}: {direction.value.lower()} structure, {len(analysis['snr'])} scored S/R levels, {len([gap for gap in analysis['fvg'] if gap['active']])} active FVG(s).")
        bullish = sum(direction == Direction.BULLISH for direction in directions)
        bearish = sum(direction == Direction.BEARISH for direction in directions)
        aggregate = Direction.BULLISH if bullish > bearish else Direction.BEARISH if bearish > bullish else Direction.NEUTRAL
        result: dict[str, Any] = {
            "theory": theory.as_dict(),
            "status": theory.health.value,
            "facts": facts,
            "observations": observations,
            "interpretation": {"direction": aggregate.value, "bullish_timeframes": bullish, "bearish_timeframes": bearish},
            "limitations": theory.limitations,
        }
        if theory_id == "snr":
            result["levels"] = {timeframe: analysis["snr"] for timeframe, analysis in analyses.items()}
        elif theory_id in {"smc", "ict", "liquidity"}:
            result["structure"] = {timeframe: analysis["structure"] for timeframe, analysis in analyses.items()}
            result["liquidity"] = {timeframe: analysis["liquidity"] for timeframe, analysis in analyses.items()}
            result["imbalances"] = {timeframe: analysis["fvg"] for timeframe, analysis in analyses.items()}
            if theory_id == "ict":
                result["session"] = session_context()
                result["unsupported_without_extra_data"] = ["SMT Divergence", "True Order Flow", "Economic Event Context"]
        elif theory_id == "wyckoff":
            result.update(self._wyckoff(analyses))
        elif theory_id == "volume_profile":
            result["profiles"] = {timeframe: analysis["volume_profile"] for timeframe, analysis in analyses.items()}
        elif theory_id == "vwap":
            result["vwap"] = {timeframe: analysis["indicators"]["vwap"] for timeframe, analysis in analyses.items()}
        elif theory_id in {"price_action", "candlesticks"}:
            result["patterns"] = {timeframe: analysis["candlestick_patterns"] for timeframe, analysis in analyses.items()}
        elif theory_id == "order_blocks":
            result["order_blocks"] = {timeframe: analysis.get("order_blocks", []) for timeframe, analysis in analyses.items()}
            fresh = sum(1 for blocks in result["order_blocks"].values() for block in blocks if block.get("fresh"))
            observations.append(f"{fresh} fresh order block(s) across the requested timeframes.")
        elif theory_id == "fibonacci":
            result["fibonacci"] = {timeframe: analysis.get("fibonacci") for timeframe, analysis in analyses.items()}
        elif theory_id == "harmonic":
            result["patterns"] = self._harmonics(analyses)
            if not any(result["patterns"].values()):
                observations.append("No XABCD structure passed ratio validation on the requested timeframes.")
        elif theory_id == "elliott":
            result["counts"] = self._elliott(analyses)
            observations.append("Alternate counts are ranked, never reduced to a single forced count.")
        elif theory_id == "sessions":
            result["opening_ranges"] = self._opening_ranges(analyses)
        return result

    def _harmonics(self, analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
        from .patterns import detect_harmonics

        return {
            timeframe: detect_harmonics(self._candles_for(analysis), timeframe)
            for timeframe, analysis in analyses.items()
        }

    def _elliott(self, analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
        from .patterns import rank_wave_counts

        return {
            timeframe: rank_wave_counts(self._candles_for(analysis), timeframe)
            for timeframe, analysis in analyses.items()
        }

    def _opening_ranges(self, analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
        from .patterns import opening_range

        ranges: dict[str, Any] = {}
        for timeframe, analysis in analyses.items():
            candles = self._candles_for(analysis)
            if not candles:
                continue
            ranges[timeframe] = {
                session: opening_range(candles, minutes=15, session=session)
                for session in ("london", "new_york")
            }
        return ranges

    def _candles_for(self, analysis: dict[str, Any]) -> list[Any]:
        """Candles kept alongside an analysis so pattern engines can rerun on them."""
        return analysis.get("_candles") or []

    @staticmethod
    def _wyckoff(analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
        primary_key = next((key for key in ("H1", "M30", "M15") if key in analyses), next(iter(analyses)))
        analysis = analyses[primary_key]
        statistics = analysis["statistics"]
        structure = analysis["structure"]["trend"]
        zscore = float(statistics.get("zscore") or 0)
        if abs(zscore) < 0.8 and structure == Direction.NEUTRAL.value:
            candidates = ["ACCUMULATION", "DISTRIBUTION"]
            phase = "B_OR_C_UNCONFIRMED"
        elif structure == Direction.BULLISH.value:
            candidates = ["MARKUP", "REACCUMULATION"]
            phase = "D_OR_E_UNCONFIRMED"
        elif structure == Direction.BEARISH.value:
            candidates = ["MARKDOWN", "REDISTRIBUTION"]
            phase = "D_OR_E_UNCONFIRMED"
        else:
            candidates = ["UNCLASSIFIED_RANGE"]
            phase = "UNCONFIRMED"
        return {
            "wyckoff": {
                "timeframe": primary_key,
                "alternate_interpretations": candidates,
                "phase_candidate": phase,
                "events": [],
                "confidence": 0.35,
                "warning": "A phase/event label is not confirmed without a validated range sequence and effort/result evidence.",
            }
        }

    @staticmethod
    def _execute_custom_theory(theory: dict[str, Any], analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
        definition = theory["definition"]
        conditions = definition.get("conditions") or []
        evaluated: list[dict[str, Any]] = []
        supported_predicates = {"trend_is", "has_liquidity_sweep", "has_bos", "has_mss", "has_active_fvg", "rsi_above", "rsi_below"}
        for condition in conditions:
            predicate = str(condition.get("predicate", "")) if isinstance(condition, dict) else ""
            if predicate not in supported_predicates:
                evaluated.append({"condition": condition, "passed": None, "status": "UNSUPPORTED_PREDICATE"})
                continue
            timeframe = normalize_timeframe(str(condition.get("timeframe", "M5")))
            analysis = analyses.get(timeframe)
            if analysis is None:
                evaluated.append({"condition": condition, "passed": None, "status": "MISSING_TIMEFRAME"})
                continue
            value = condition.get("value")
            if predicate == "trend_is":
                passed = analysis["structure"]["trend"] == str(value).upper()
            elif predicate == "has_liquidity_sweep":
                passed = any(item["type"] == str(value).upper() for item in analysis["liquidity"]["sweeps"])
            elif predicate == "has_bos":
                passed = (analysis["structure"].get("bos") or {}).get("direction") == str(value).upper()
            elif predicate == "has_mss":
                passed = (analysis["structure"].get("mss") or {}).get("direction") == str(value).upper()
            elif predicate == "has_active_fvg":
                passed = any(item["active"] and item["direction"] == str(value).upper() for item in analysis["fvg"])
            elif predicate == "rsi_above":
                passed = float(analysis["indicators"].get("rsi14") or -1) > float(value)
            else:
                passed = float(analysis["indicators"].get("rsi14") or 101) < float(value)
            evaluated.append({"condition": condition, "passed": passed, "status": "EVALUATED"})
        all_supported = all(item["status"] == "EVALUATED" for item in evaluated)
        all_passed = bool(evaluated) and all(item["passed"] is True for item in evaluated)
        return {
            "theory": {"id": theory["id"], "name": theory["name"], "version": theory["version"], "custom": True},
            "status": CapabilityState.AVAILABLE.value if all_supported else CapabilityState.PARTIALLY_AVAILABLE.value,
            "evaluated_conditions": evaluated,
            "setup_match": all_passed,
            "warning": None if all_supported else "Unsupported predicates were not guessed or executed.",
        }

    @staticmethod
    def _report(
        *,
        requested_symbol: str,
        resolved_symbol: str,
        feed: str | None,
        analyses: dict[str, dict[str, Any]],
        theories: dict[str, Any],
        setup: dict[str, Any],
        checks: dict[str, Any],
        errors: dict[str, str],
    ) -> dict[str, Any]:
        current = next((analyses[key] for key in ("M1", "M3", "M5", "M15", "H1") if key in analyses), next(iter(analyses.values())))
        all_levels = [level for analysis in analyses.values() for level in analysis["snr"]]
        price = current["current_price"]
        support = sorted((level for level in all_levels if level["price"] < price), key=lambda item: price - item["price"])[:5]
        resistance = sorted((level for level in all_levels if level["price"] > price), key=lambda item: item["price"] - price)[:5]
        htf = next((analyses[key] for key in ("H4", "H1", "M30", "M15") if key in analyses), current)
        return {
            "symbol": resolved_symbol,
            "requested_symbol": requested_symbol,
            "feed": feed,
            "current_price": price,
            "session": session_context(),
            "theories": theories,
            "htf_bias": htf["structure"]["trend"],
            "structure": {timeframe: analysis["structure"] for timeframe, analysis in analyses.items()},
            "support": support,
            "resistance": resistance,
            "supply": [zone for analysis in analyses.values() for zone in analysis["supply_demand"] if zone["type"] == "SUPPLY" and not zone["invalidated"]],
            "demand": [zone for analysis in analyses.values() for zone in analysis["supply_demand"] if zone["type"] == "DEMAND" and not zone["invalidated"]],
            "liquidity": {timeframe: analysis["liquidity"] for timeframe, analysis in analyses.items()},
            "order_block": {"status": CapabilityState.PARTIALLY_AVAILABLE.value, "candidates": [], "warning": "Strict order-block validation is not yet satisfied by any candidate."},
            "fvg": {timeframe: analysis["fvg"] for timeframe, analysis in analyses.items()},
            "volume": {timeframe: {"kind": analysis["metadata"]["volume_kind"], "profile": analysis["volume_profile"]} for timeframe, analysis in analyses.items()},
            # Raw candles are working state for the pattern engines, never output.
            "timeframes": {
                timeframe: {key: value for key, value in analysis.items() if not key.startswith("_")}
                for timeframe, analysis in analyses.items()
            },
            "long_scenario": setup if setup.get("direction") == Direction.BULLISH.value else None,
            "short_scenario": setup if setup.get("direction") == Direction.BEARISH.value else None,
            "setup_state": setup.get("state"),
            "decision": setup.get("decision"),
            "entry": setup.get("entry"),
            "stop": setup.get("stop"),
            "tp1": (setup.get("targets") or [{}])[0].get("price") if setup.get("targets") else None,
            "tp2": (setup.get("targets") or [{}, {}])[1].get("price") if len(setup.get("targets") or []) > 1 else None,
            "tp3": (setup.get("targets") or [{}, {}, {}])[2].get("price") if len(setup.get("targets") or []) > 2 else None,
            "rr": setup.get("rr"),
            "invalidation": setup.get("technical_invalidation"),
            "confidence": TradingService._confidence(setup, checks),
            "missing_confirmation": setup.get("missing_confirmation", []),
            "no_trade_reason": setup.get("reasons") if setup.get("decision") != SetupDecision.ENTRY_READY.value else None,
            "self_check": checks,
            "data_errors": errors,
            "uncertainty_model": {"fact": "Provider OHLC/ticks", "observation": "Deterministic structures/levels", "interpretation": "Theory output", "setup": setup.get("decision"), "confirmation": setup.get("confirmations"), "invalidation": setup.get("technical_invalidation")},
        }

    @staticmethod
    def _confidence(setup: dict[str, Any], checks: dict[str, Any]) -> float:
        confirmations = setup.get("confirmations") or {}
        present = sum(bool(value) for value in confirmations.values())
        total = max(1, len(confirmations))
        score = present / total * 0.75 + (0.25 if checks.get("passed") else 0)
        return round(min(1.0, max(0.0, score)), 3)

    @staticmethod
    def _sorani_summary(report: dict[str, Any]) -> str:
        bias = report.get("htf_bias")
        bias_text = {"BULLISH": "HTF bullish ـە", "BEARISH": "HTF bearish ـە", "NEUTRAL": "HTF neutral ـە"}.get(str(bias), "HTF دیار نییە")
        decision = report.get("decision")
        if decision == SetupDecision.ENTRY_READY.value:
            return f"{bias_text}. Entry پشتڕاست کراوەتەوە. Entry {report.get('entry')}, Stop {report.get('stop')}, RR {report.get('rr')}."
        missing = report.get("missing_confirmation") or []
        missing_text = ", ".join(str(item) for item in missing[:3])
        if decision == SetupDecision.NO_TRADE.value:
            return f"{bias_text}. ئێستا NO TRADE ـە. {missing_text or 'شوێنی entry و RR گونجاو نییە.'}"
        return f"{bias_text}. ئێستا {decision or 'WAIT'} ـە؛ entry هێشتا پشتڕاست نەکراوەتەوە. {missing_text}"

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
        if self._last_report is None:
            raise ValueError("Run a market analysis before creating a monitored setup.")
        return self.create_setup_from_analysis(self._last_report, theory)

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
