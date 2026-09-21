from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..contracts import CapabilityState, SkillDefinition, SkillRegistry


@dataclass(slots=True)
class TheoryDefinition:
    id: str
    name: str
    aliases: list[str]
    category: str
    version: str
    description: str
    assumptions: list[str]
    required_data: list[str]
    preferred_timeframes: list[str]
    supported_markets: list[str]
    rules: list[str]
    entry_rules: list[str]
    invalidation_rules: list[str]
    target_rules: list[str]
    limitations: list[str]
    health: CapabilityState
    speculative: bool = False
    executable_components: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["health"] = self.health.value
        return payload


def _theory(
    theory_id: str,
    name: str,
    *,
    aliases: list[str] | None = None,
    category: str = "technical",
    required_data: list[str] | None = None,
    health: CapabilityState = CapabilityState.PARTIALLY_AVAILABLE,
    components: list[str] | None = None,
    limitations: list[str] | None = None,
    speculative: bool = False,
) -> TheoryDefinition:
    return TheoryDefinition(
        id=theory_id,
        name=name,
        aliases=aliases or [],
        category=category,
        version="1.0.0",
        description=f"Independent {name} analysis module. It never borrows a conclusion from another theory merely to create agreement.",
        assumptions=["Input candles are chronological and originate from a named, verified feed."],
        required_data=required_data or ["OHLC"],
        preferred_timeframes=["H1", "M15", "M5", "M1"],
        supported_markets=["forex", "metals", "crypto", "indices"],
        rules=["Collect theory-specific evidence.", "Report conflicts and missing prerequisites.", "Do not force a directional result."],
        entry_rules=["An entry needs an independent lower-timeframe trigger and technical invalidation."],
        invalidation_rules=["Invalidate only at a theory-defined structural price."],
        target_rules=["Use detected structure, liquidity, profile, or measured objectives; never invent targets."],
        limitations=limitations or [],
        health=health,
        speculative=speculative,
        executable_components=components or [],
    )


def builtin_theories() -> dict[str, TheoryDefinition]:
    ready = CapabilityState.AVAILABLE
    partial = CapabilityState.PARTIALLY_AVAILABLE
    unavailable = CapabilityState.UNAVAILABLE
    definitions = [
        _theory("default", "SAM Default Market Model", aliases=["default", "standard"], health=ready, components=["market_structure", "snr", "supply_demand", "liquidity", "fvg", "session", "entry_hunter", "risk_reward"]),
        _theory("price_action", "Price Action", health=ready, components=["candles", "swings", "market_structure", "candlestick_patterns"]),
        _theory("market_structure", "Market Structure", aliases=["structure", "bos", "choch", "mss"], health=ready, components=["swings", "hh_hl_lh_ll", "bos", "choch", "mss"]),
        _theory("snr", "Support and Resistance", aliases=["support resistance", "s/r", "rbs", "sbr"], health=ready, components=["snr_scoring", "role_reversal", "break_retest"]),
        _theory("supply_demand", "Supply and Demand", aliases=["supply", "demand"], health=ready, components=["rbr", "dbd", "rbd", "dbr", "zone_freshness"]),
        _theory("liquidity", "Liquidity", aliases=["bsl", "ssl", "sweep", "raid"], health=ready, components=["equal_highs", "equal_lows", "sweeps"]),
        _theory("smc", "Smart Money Concepts", aliases=["smart money"], health=ready, components=["internal_external_structure", "bos", "choch", "mss", "liquidity", "fvg", "premium_discount"]),
        _theory("ict", "ICT", aliases=["inner circle trader"], health=partial, components=["liquidity", "dealing_range", "premium_discount", "fvg", "mss", "bos", "sessions"], limitations=["SMT, Silver Bullet, and event context require synchronized companion markets and richer session data."]),
        _theory("order_blocks", "Order Blocks", aliases=["ob", "breaker", "mitigation block"], health=ready, components=["displacement_origin", "structure_break", "mitigation", "breaker", "invalidation"], limitations=["A candidate is promoted only when the departure exceeds an ATR-scaled displacement and breaks structure, so most opposite candles are rejected."]),
        _theory("imbalance", "FVG and Imbalance", aliases=["fvg", "ifvg", "liquidity void"], health=ready, components=["three_candle_fvg", "partial_fill", "full_fill"]),
        _theory("wyckoff", "Wyckoff", health=partial, components=["range_detection", "effort_result", "spring_upthrust_candidates", "alternate_interpretations"], limitations=["Event labels and phases remain probabilistic and expose alternate interpretations."]),
        _theory("amd", "Accumulation Manipulation Distribution", aliases=["power of three"], health=partial, components=["range", "liquidity_sweep", "displacement"]),
        _theory("vsa", "Volume Spread Analysis", required_data=["OHLC", "Volume"], health=partial, components=["spread", "relative_tick_volume", "effort_result"], limitations=["Tick volume is marked partial when centralized traded volume is unavailable."]),
        _theory("volume_profile", "Volume Profile", required_data=["OHLC", "Volume"], health=ready, components=["poc", "vah", "val", "hvn", "lvn"], limitations=["A candle is assigned to its typical-price bin; this is not a true footprint profile."]),
        _theory("vwap", "VWAP", required_data=["OHLC", "Volume"], health=ready, components=["session_vwap", "reclaim", "rejection"]),
        _theory("candlesticks", "Candlestick Patterns", health=ready, components=["doji", "hammer", "shooting_star", "engulfing", "marubozu"], limitations=["Patterns are observations and require market context."]),
        _theory("classical_patterns", "Classical Chart Patterns", health=partial, components=["range", "breakout", "measured_move"], limitations=["Complex formations require additional geometric validation before a setup can be ENTRY_READY."]),
        _theory("fibonacci", "Fibonacci", aliases=["fib", "ote"], health=ready, components=["retracement", "extension", "projection", "ote", "confluence"], limitations=["Anchor selection remains theory-dependent; the anchoring leg is always reported alongside the levels."]),
        _theory("pivots", "Pivot Points", health=ready, components=["standard", "fibonacci", "woodie", "camarilla"]),
        _theory("indicators", "Indicator Strategies", health=ready, components=["sma", "ema", "wma", "rsi", "macd", "stochastic", "atr", "bollinger", "keltner", "adx", "parabolic_sar", "ichimoku"]),
        _theory("sessions", "Session and Time Strategies", health=ready, components=["tokyo", "london", "new_york", "nyse", "dst", "opening_range_breakout"]),
        _theory("statistics", "Statistical Market Analysis", health=ready, components=["returns", "volatility", "zscore", "percentiles", "rolling_range"]),
        _theory("gann", "Gann", health=partial, components=["angles", "price_time_relationships"], limitations=["Subjective anchors are always labeled."]),
        _theory("elliott", "Elliott Wave", health=ready, components=["rule_validation", "alternate_counts", "wave_fib_relationships"], limitations=["The three inviolable impulse rules are enforced; counts are ranked and alternates retained rather than reduced to one answer."]),
        _theory("harmonic", "Harmonic Patterns", health=ready, components=["xabcd_ratios", "prz", "targets", "invalidation"], limitations=["Ten definitions are validated by ratio envelope; a shape failing any constrained leg is not reported at all."]),
        _theory("pitchfork", "Pitchfork", health=partial, components=["median_line", "parallels"], limitations=["Reliable screen drawing requires verified chart calibration."]),
        _theory("fractal", "Fractal Market Geometry", health=ready, components=["swing_fractals", "nested_structure", "multi_scale_structure"], limitations=["Fractality is descriptive, not a guarantee of prediction."]),
        _theory("experimental_cycles", "Experimental Cycles", category="experimental", health=partial, components=["custom_time_cycles", "price_time_cycles"], limitations=["Planetary and lunar interpretations are speculative and never enter the standard confluence score."], speculative=True),
        _theory("polar_price_time", "Polar Price-Time Models", category="experimental", health=partial, components=["cartesian_to_polar", "normalized_angle"], limitations=["User formulas must be explicitly supplied and versioned."], speculative=True),
        _theory("grid", "Grid Research", category="risk_research", health=partial, components=["arithmetic_grid", "geometric_grid"], limitations=["Not a default execution or risk model."]),
        _theory("martingale", "Martingale and Anti-Martingale Research", category="risk_research", health=partial, components=["exposure_projection", "drawdown_projection"], limitations=["Never activated automatically and never submits orders."]),
        _theory("order_flow", "Order Flow", required_data=["Bid", "Ask", "Aggressor Side"], health=unavailable, limitations=["A compatible verified bid/ask and aggressor-side feed is not configured."]),
        _theory("footprint", "Footprint", required_data=["Footprint", "Aggressor Side"], health=unavailable, limitations=["Never synthesized from OHLC candles."]),
        _theory("cvd", "CVD", required_data=["True Delta"], health=unavailable, limitations=["A valid delta source is required."]),
        _theory("dom", "Depth of Market", required_data=["Level II", "Order Book"], health=unavailable, limitations=["A verified Level II source is required."]),
        _theory("tape", "Tape Reading", required_data=["Trade Prints", "Aggressor Side"], health=unavailable, limitations=["A verified time-and-sales source is required."]),
        _theory("macro", "Macro Event", category="fundamental", required_data=["Economic Calendar"], health=unavailable, limitations=["Upcoming events are never fabricated."]),
        _theory("carry", "Carry Trade", category="fundamental", required_data=["Interest Rates"], health=unavailable, limitations=["A verified rates source is required."]),
        _theory("intermarket", "Intermarket Analysis", category="fundamental", required_data=["Synchronized Multi-Market OHLC"], health=unavailable, limitations=["Correlations are measured, never assumed permanent."]),
        _theory("sentiment", "Sentiment", category="fundamental", required_data=["Sentiment"], health=unavailable, limitations=["COT or retail positioning provider is not configured."]),
    ]
    return {definition.id: definition for definition in definitions}


def build_skill_registry(theories: dict[str, TheoryDefinition] | None = None) -> SkillRegistry:
    registry = SkillRegistry()

    def register(
        skill_id: str,
        name: str,
        category: str,
        *,
        health: CapabilityState = CapabilityState.AVAILABLE,
        tools: list[str] | None = None,
        data: list[str] | None = None,
        verification: str = "Recompute output invariants and retain provider metadata.",
        notes: list[str] | None = None,
    ) -> None:
        registry.register(SkillDefinition(
            id=skill_id,
            name=name,
            category=category,
            version="1.0.0",
            required_tools=tools or [],
            optional_tools=[],
            inputs={"type": "object"},
            outputs={"type": "object"},
            prerequisites=[],
            data_requirements=data or [],
            model_requirements=[],
            permission_requirements=[],
            cancellable=True,
            timeout_seconds=120,
            max_retries=2,
            verification_policy=verification,
            failure_states=["UNAVAILABLE", "MISSING_DATA", "CANCELLED", "TIMEOUT", "VERIFICATION_FAILED"],
            health=health,
            notes=notes or [],
        ))

    register("market_data", "Market Data", "trading", tools=["get_ohlcv"], data=["OHLC"])
    register("market_data_capabilities", "Market Data Capabilities", "trading")
    register("multi_timeframe", "Multi-Timeframe Analysis", "trading", data=["OHLC"])
    register("market_structure", "Market Structure", "trading", data=["OHLC"])
    register("snr_expert", "SnR Expert", "trading", data=["OHLC"])
    register("order_block", "Order Block", "trading", data=["OHLC"], verification="A block requires ATR-scaled displacement plus a confirmed structure break.")
    register("imbalance", "FVG and Imbalance", "trading", data=["OHLC"])
    register("fibonacci", "Fibonacci", "trading", data=["OHLC"])
    register("harmonic", "Harmonic Patterns", "trading", data=["OHLC"], verification="Every constrained XABCD leg ratio must fall inside its envelope.")
    register("elliott_wave", "Elliott Wave", "trading", data=["OHLC"], verification="The three impulse rules are checked before a count is called valid.")
    register("opening_range", "Opening Range Breakout", "trading", data=["OHLC"], verification="The range is built only from candles closing inside the session window.")
    register("liquidity", "Liquidity", "trading", data=["OHLC"])
    register("entry_hunter", "Entry Hunter", "trading", data=["OHLC", "Multiple Timeframes"])
    register("no_trade", "No-Trade Decision", "trading")
    register("risk_reward", "Risk and Reward", "trading")
    register("theory_execution", "Theory Execution", "trading")
    register("theory_comparison", "Theory Comparison", "trading")
    register("learn_custom_theory", "Learn Custom Theory", "memory")
    register("theory_memory", "Theory Memory", "memory")
    register("setup_monitor", "Setup Monitor", "monitoring")
    register("trading_journal", "Trading Journal", "memory")
    register(
        "strategy_research", "Strategy Research", "research",
        tools=["backtest_strategy", "list_entry_triggers"], data=["OHLC"],
        verification="Each decision sees only closed candles up to its bar; outcomes resolve on later bars only.",
        notes=["A candle touching both stop and target is scored as a loss, never as a win."],
    )
    register("entry_triggers", "Entry Trigger Registry", "trading", tools=["list_entry_triggers"], data=["OHLC"])
    register("self_check", "Self Check", "verification")
    register("hallucination_guard", "Trading Hallucination Guard", "verification")
    register("tradingview_observer", "TradingView Observer", "desktop", tools=["get_tradingview_state"], verification="Verify process, exact window handle, title, and geometry at observation time.")
    register("tradingview_symbol", "TradingView Symbol", "desktop", health=CapabilityState.PARTIALLY_AVAILABLE, tools=["set_tradingview_symbol"], verification="Window title must confirm the selected symbol.")
    register("tradingview_timeframe", "TradingView Timeframe", "desktop", health=CapabilityState.PARTIALLY_AVAILABLE, tools=["set_tradingview_timeframe"], verification="Toolbar OCR must read the selected interval after the command; otherwise return PARTIAL.")
    register(
        "chart_calibration", "Chart Calibration", "desktop",
        tools=["calibrate_chart", "verify_chart_calibration"],
        verification="Several price-axis labels must agree on one linear fit within the residual tolerance.",
        notes=["Automatic calibration reads the price axis with the offline Windows OCR engine.", "Logarithmic price scales are not supported yet."],
    )
    register(
        "tradingview_drawing", "TradingView Drawing", "desktop",
        tools=["draw_tradingview_level", "draw_analysis_on_chart", "clear_sam_drawings", "set_chart_layer"],
        verification="Compare the chart before and after the command and require a pixel change at the calibrated price row.",
        notes=["Price-level annotations and verified two-anchor trend/Fibonacci draws are supported after chart calibration.", "SAM only ever selects or deletes annotations it recorded as its own."],
    )
    register(
        "drawing_ownership", "Drawing Ownership", "desktop",
        tools=["list_sam_drawings", "clear_sam_drawings"],
        verification="Every annotation is persisted with symbol, timeframe, layer, theory, price, and viewport before it is verified.",
    )
    register(
        "gann_drawing", "Gann Drawing", "desktop",
        tools=["draw_tradingview_object"], data=["OHLC"],
        verification="Each ray is drawn as a trendline and confirmed by pixel coverage along its path.",
        notes=["TradingView's native Gann Fan cannot be placed through automation; a verified trendline fan is drawn instead."],
    )
    register(
        "pitchfork_drawing", "Pitchfork Drawing", "desktop",
        tools=["draw_tradingview_object"], data=["OHLC"],
        verification="The median line and both parallels are drawn and each confirmed independently.",
        notes=["The native pitchfork tool needs a three-click sequence whose intermediate state cannot be confirmed."],
    )
    register(
        "bar_replay", "Bar Replay Research", "research",
        health=CapabilityState.PARTIALLY_AVAILABLE, data=["OHLC"],
        verification="A decision at bar N sees candles up to N only; outcomes resolve on later bars.",
        notes=["Provider-history replay is available; driving TradingView's own Bar Replay control is not verifiable."],
    )
    register(
        "strategy_composer", "Strategy Composer", "memory",
        verification="A composed strategy is refused unless it maps onto executable predicates.",
    )
    register(
        "chart_verification", "Chart Verification", "verification",
        verification="A drawing is reported present only after an independent capture shows the expected change.",
    )
    register("voice_realtime", "Realtime Voice", "voice", health=CapabilityState.PARTIALLY_AVAILABLE, notes=["Browser VAD/interim STT and streaming speech chunks are available; local faster-whisper/Piper depend on optional models."])
    register("agent_recovery", "Agent Recovery", "verification")
    register("open_application", "Open Application", "windows", tools=["launch_app"])
    register("file_management", "File Management", "windows", tools=["list_files", "read_file", "write_file", "replace_text", "delete_path"])
    register("project_debug", "Project Debug", "windows", tools=["list_files", "read_file", "write_file", "run_terminal", "run_python"])
    register("browser_research", "Browser Research", "windows", tools=["browser_automate"])
    for theory in (theories or builtin_theories()).values():
        register(
            f"theory.{theory.id}",
            theory.name,
            "theory",
            health=theory.health,
            data=theory.required_data,
            notes=theory.limitations,
        )
    return registry


class TradingKnowledgeRegistry:
    def __init__(self) -> None:
        self._theories = builtin_theories()

    def get(self, theory: str) -> TheoryDefinition:
        normalized = theory.strip().lower()
        if normalized in self._theories:
            return self._theories[normalized]
        for definition in self._theories.values():
            if normalized == definition.name.lower() or normalized in {alias.lower() for alias in definition.aliases}:
                return definition
        raise KeyError(f"Unknown trading theory: {theory}")

    def list(self) -> list[dict[str, Any]]:
        return [definition.as_dict() for definition in sorted(self._theories.values(), key=lambda item: item.name)]

