"""Runs the theories over live market data and assembles the report.

The primitives -- structure, levels, liquidity, patterns -- live in
analysis.py as pure functions over candles. This is the thing that decides
which of them to run, over which timeframes, for which theories, and folds
the answers into one report the rest of SAM can act on.

It also owns the one piece of state the trading layer has: the latest
completed report. The chart draws from it and a monitored setup is created
from it, so it has exactly one writer, here, and readers ask for it by name
rather than each keeping a copy.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from ..cancellation import CancellationManager, CancellationToken
from ..contracts import CapabilityState, ExecutionStatus, StandardResult
from ..db import Database
from .analysis import entry_hunter, self_check, session_context, timeframe_analysis
from .market_data import MarketDataService, MetaTrader5Provider
from .registry import TradingKnowledgeRegistry
from .types import Direction, SetupDecision, SetupState, normalize_timeframe

DEFAULT_TIMEFRAMES = ["H1", "M15", "M5", "M1"]


class MarketAnalyst:
    def __init__(
        self, settings: Any, database: Database, market_data: MarketDataService,
        knowledge: TradingKnowledgeRegistry, cancellation: CancellationManager,
    ) -> None:
        self.settings = settings
        self.database = database
        self.market_data = market_data
        self.knowledge = knowledge
        self.cancellation = cancellation
        # The latest completed report, PARTIAL included: a report with
        # verification gaps is still the most recent view of the market and
        # says so in its status. A fetch that fails outright leaves it alone.
        self.latest: dict[str, Any] | None = None

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
        self.latest = report
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
            "confidence": MarketAnalyst._confidence(setup, checks),
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
