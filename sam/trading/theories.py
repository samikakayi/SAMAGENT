"""The 40-theory trading catalogue, kept as knowledge (the user's "theories
that were written").

Source: v1 ``sam_backend/trading/registry.py::builtin_theories`` (ids, aliases,
categories, components, limitations and honest health labels are unchanged).
v1's descriptions were one generic sentence for all 40; SAM 2 adds a real
English description and a Sorani name/description per theory so SAM can
explain a theory in the user's language, and the predicates each theory's
rules can be checked with (``strategies.py`` uses them when ingesting a
card). ``engine.analyst.run_theory`` executes the ones with engine support.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..textnorm import normalize_ckb
from .engine.types import CapabilityState

READY = CapabilityState.AVAILABLE
PARTIAL = CapabilityState.PARTIALLY_AVAILABLE
UNAVAILABLE = CapabilityState.UNAVAILABLE


@dataclass(slots=True)
class Theory:
    id: str
    name: str
    name_ckb: str
    description: str
    description_ckb: str
    aliases: list[str] = field(default_factory=list)
    aliases_ckb: list[str] = field(default_factory=list)
    category: str = "technical"
    version: str = "1.0.0"
    required_data: list[str] = field(default_factory=lambda: ["OHLC"])
    preferred_timeframes: list[str] = field(default_factory=lambda: ["H1", "M15", "M5", "M1"])
    supported_markets: list[str] = field(default_factory=lambda: ["forex", "metals", "crypto", "indices"])
    components: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    health: CapabilityState = PARTIAL
    speculative: bool = False
    predicates: list[str] = field(default_factory=list)

    # v1 field name, kept for code that reads it.
    @property
    def executable_components(self) -> list[str]:
        return self.components

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["health"] = self.health.value
        return payload

    def summary(self) -> dict[str, Any]:
        """Compact form for tool results (fits the model context)."""
        return {"id": self.id, "name": self.name, "name_ckb": self.name_ckb, "category": self.category,
                "health": self.health.value, "description": self.description,
                "description_ckb": self.description_ckb, "components": self.components,
                "limitations": self.limitations, "speculative": self.speculative,
                "checkable_with": self.predicates, "required_data": self.required_data}


def _t(theory_id: str, name: str, name_ckb: str, description: str, description_ckb: str, **kw: Any) -> Theory:
    return Theory(id=theory_id, name=name, name_ckb=name_ckb, description=description,
                  description_ckb=description_ckb, **kw)


def builtin_theories() -> dict[str, Theory]:
    items = [
        _t("default", "SAM Default Market Model", "مۆدێلی بنەڕەتی سام",
           "Multi-timeframe model: higher-timeframe structure sets the bias; an entry needs a liquidity sweep, a "
           "lower-timeframe MSS/BOS and a confirming candle, with a structural stop and level-based targets.",
           "ئاراستە لە کاتە گەورەکانەوە دیاری دەکرێت؛ چوونەژوورەوە پێویستی بە ڕاماڵینی لیکویدیتی، شکانی پێکهاتە "
           "لە کاتی بچووک و مۆمێکی پشتڕاستکەرەوە هەیە، ستۆپ لەسەر پێکهاتە و ئامانج لەسەر ئاستەکانە.",
           aliases=["default", "standard"], aliases_ckb=["بنەڕەتی", "ستاندارد"], health=READY,
           components=["market_structure", "snr", "supply_demand", "liquidity", "fvg", "session", "entry_hunter",
                       "risk_reward"], predicates=["trend_is", "swept", "mss_or_bos", "candle_pattern", "rr_at_least"]),
        _t("price_action", "Price Action", "پرایس ئاکشن (جووڵەی نرخ)",
           "Reads raw candles and swings (highs, lows, structure, candle formations) without indicators.",
           "خوێندنەوەی مۆم و لووتکە و نزمایی نرخ بەبێ ئیندیکەیتەر.",
           aliases_ckb=["پرایس ئاکشن", "جووڵەی نرخ"], health=READY,
           components=["candles", "swings", "market_structure", "candlestick_patterns"],
           predicates=["candle_pattern", "trend_is", "near_level", "displacement"]),
        _t("market_structure", "Market Structure", "پێکهاتەی بازاڕ",
           "Swing highs/lows labelled HH/HL/LH/LL; a BOS continues the trend, a CHoCH/MSS signals a possible reversal.",
           "دیاریکردنی لووتکە و نزمایی (HH، HL، LH، LL)؛ BOS درێژەی ترێندە و CHoCH یان MSS نیشانەی گۆڕانی ئاراستەیە.",
           aliases=["structure", "bos", "choch", "mss"], aliases_ckb=["پێکهاتە", "ستراکچەر"], health=READY,
           components=["swings", "hh_hl_lh_ll", "bos", "choch", "mss"], predicates=["trend_is", "mss_or_bos", "choch", "bos"]),
        _t("snr", "Support and Resistance", "پشتگیری و بەرگری",
           "Horizontal levels scored by reactions, recency and timeframe; a broken level flips role (RBS/SBR).",
           "ئاستە ئاسۆییەکان بە ژمارەی کاردانەوە، نوێیی و کات هەڵدەسەنگێنرێن؛ ئاستی شکاو ڕۆڵی دەگۆڕێت (RBS/SBR).",
           aliases=["support resistance", "s/r", "rbs", "sbr"], aliases_ckb=["سەپۆرت", "ڕێزیستەنس", "ئاست"],
           health=READY, components=["snr_scoring", "role_reversal", "break_retest"], predicates=["near_level"]),
        _t("supply_demand", "Supply and Demand", "خستنەڕوو و داواکاری (سەپڵای و دیماند)",
           "Base candles before a strong departure form demand (DBR/RBR) or supply (RBD/DBD) zones; fresh zones are strongest.",
           "ناوچەی داواکاری (DBR/RBR) و خستنەڕوو (RBD/DBD) لە مۆمی بنچینە پێش جووڵەیەکی بەهێز پێکدێن؛ ناوچەی تازە بەهێزترە.",
           aliases=["supply", "demand"], aliases_ckb=["سەپڵای", "دیماند"], health=READY,
           components=["rbr", "dbd", "rbd", "dbr", "zone_freshness"], predicates=["in_supply_demand"]),
        _t("liquidity", "Liquidity", "لیکویدیتی",
           "Equal highs/lows hold resting orders (buy-side/sell-side liquidity); a sweep takes them and closes back.",
           "لووتکە و نزمایی یەکسان فەرمانی چاوەڕوانیان لە پشتە؛ ڕاماڵین ئەوانە دەبات و نرخ دەگەڕێتەوە.",
           aliases=["bsl", "ssl", "sweep", "raid"], aliases_ckb=["لیکویدیتی", "ڕاماڵین", "سویپ"], health=READY,
           components=["equal_highs", "equal_lows", "sweeps"], predicates=["swept", "liquidity_pool"]),
        _t("smc", "Smart Money Concepts", "چەمکەکانی پارەی زیرەک (SMC)",
           "Internal/external structure, BOS/CHoCH, liquidity, fair value gaps and premium/discount pricing.",
           "پێکهاتەی ناوەوە و دەرەوە، BOS و CHoCH، لیکویدیتی، FVG و پرێمیەم/دیسکاونت.",
           aliases=["smart money"], aliases_ckb=["سمارت مەنی", "ئێس ئێم سی"], health=READY,
           components=["internal_external_structure", "bos", "choch", "mss", "liquidity", "fvg", "premium_discount"],
           predicates=["mss_or_bos", "swept", "in_fvg", "in_order_block", "premium_discount"]),
        _t("ict", "ICT", "ICT (ئای سی تی)",
           "Dealing range, premium/discount, liquidity raids, fair value gaps, MSS and killzone timing.",
           "مەودای مامەڵە، پرێمیەم و دیسکاونت، ڕاماڵینی لیکویدیتی، FVG، MSS و کاتی کیلزۆنەکان.",
           aliases=["inner circle trader"], aliases_ckb=["ئای سی تی"], health=PARTIAL,
           components=["liquidity", "dealing_range", "premium_discount", "fvg", "mss", "bos", "sessions"],
           limitations=["SMT, Silver Bullet, and event context require synchronized companion markets and richer "
                        "session data."],
           predicates=["swept", "mss_or_bos", "in_fvg", "in_ote", "killzone", "premium_discount"]),
        _t("order_blocks", "Order Blocks", "ئۆردەر بلۆک",
           "The last opposite candle before a displacing move that breaks structure; mitigation and breakers are tracked.",
           "دوایین مۆمی پێچەوانە پێش جووڵەیەکی بەهێز کە پێکهاتە دەشکێنێت؛ گەڕانەوە بۆی و بوونی بە بریکەر چاودێری دەکرێن.",
           aliases=["ob", "breaker", "mitigation block"], aliases_ckb=["ئۆردەر بلۆک", "بریکەر"], health=READY,
           components=["displacement_origin", "structure_break", "mitigation", "breaker", "invalidation"],
           limitations=["A candidate is promoted only when the departure exceeds an ATR-scaled displacement and "
                        "breaks structure, so most opposite candles are rejected."], predicates=["in_order_block"]),
        _t("imbalance", "FVG and Imbalance", "بۆشایی نرخ (FVG) و ناهاوسەنگی",
           "Three-candle fair value gaps; partial and full fills are tracked.",
           "بۆشایی نرخی سێ مۆمی (FVG)؛ پڕبوونەوەی بەشێک یان تەواوی چاودێری دەکرێت.",
           aliases=["fvg", "ifvg", "liquidity void"], aliases_ckb=["بۆشایی نرخ", "ئێف ڤی جی"], health=READY,
           components=["three_candle_fvg", "partial_fill", "full_fill"], predicates=["in_fvg", "fvg_present"]),
        _t("wyckoff", "Wyckoff", "وایکۆف",
           "Accumulation/distribution phases and events, always reported with alternate interpretations.",
           "قۆناغەکانی کەڵەکەکردن و دابەشکردن، هەمیشە لەگەڵ لێکدانەوەی جێگرەوە.",
           aliases_ckb=["وایکۆف"], health=PARTIAL,
           components=["range_detection", "effort_result", "spring_upthrust_candidates", "alternate_interpretations"],
           limitations=["Event labels and phases remain probabilistic and expose alternate interpretations."],
           predicates=["swept", "volume_spike", "trend_is"]),
        _t("amd", "Accumulation Manipulation Distribution", "کەڵەکەکردن، دەستکاری، دابەشکردن (AMD)",
           "Power of three: a range (accumulation), a false move (manipulation), then the real move (distribution).",
           "هێزی سێ: مەودا (کەڵەکەکردن)، جووڵەی درۆ (دەستکاری)، پاشان جووڵەی ڕاستەقینە (دابەشکردن).",
           aliases=["power of three"], aliases_ckb=["پاوەر ئۆف سری", "ئەی ئێم دی"], health=PARTIAL,
           components=["range", "liquidity_sweep", "displacement"], predicates=["swept", "displacement", "session_is"]),
        _t("vsa", "Volume Spread Analysis", "شیکاری ڤۆلیۆم و مەودا (VSA)",
           "Effort (volume) versus result (spread); on FX and gold only tick volume exists.",
           "هەوڵ (ڤۆلیۆم) بەرامبەر ئەنجام (مەودای مۆم)؛ لە فۆرێکس و زێڕ تەنها تیک ڤۆلیۆم هەیە.",
           required_data=["OHLC", "Volume"], health=PARTIAL, components=["spread", "relative_tick_volume", "effort_result"],
           limitations=["Tick volume is marked partial when centralized traded volume is unavailable."],
           predicates=["volume_spike"]),
        _t("volume_profile", "Volume Profile", "پرۆفایلی ڤۆلیۆم",
           "Volume by price: point of control, value area high/low, high- and low-volume nodes.",
           "ڤۆلیۆم بەپێی نرخ: POC، سنووری ناوچەی بەها و گرێکانی ڤۆلیۆمی زۆر و کەم.",
           required_data=["OHLC", "Volume"], health=READY, components=["poc", "vah", "val", "hvn", "lvn"],
           limitations=["A candle is assigned to its typical-price bin; this is not a true footprint profile."]),
        _t("vwap", "VWAP", "VWAP (تێکڕای نرخ بە کێشی ڤۆلیۆم)",
           "Volume-weighted average price; reclaims and rejections of VWAP.",
           "تێکڕای نرخ بە کێشی ڤۆلیۆم؛ گەڕانەوە بۆ سەر VWAP یان ڕەتکردنەوەی.",
           required_data=["OHLC", "Volume"], health=READY, components=["session_vwap", "reclaim", "rejection"],
           predicates=["price_vs_vwap"]),
        _t("candlesticks", "Candlestick Patterns", "شێوەکانی مۆم",
           "Doji, hammer, shooting star, engulfing and marubozu: observations that need context.",
           "دۆجی، هامەر، شوتینگ ستار، ئینگەڵفینگ و مارووبۆزو؛ تەنها تێبینین و پاشخانیان پێویستە.",
           aliases_ckb=["مۆم", "کاندڵ"], health=READY,
           components=["doji", "hammer", "shooting_star", "engulfing", "marubozu"],
           limitations=["Patterns are observations and require market context."], predicates=["candle_pattern"]),
        _t("classical_patterns", "Classical Chart Patterns", "شێوە کلاسیکییەکانی چارت",
           "Ranges, breakouts and measured moves; triangles, flags and head-and-shoulders need extra validation.",
           "مەودا، شکان و جووڵەی پێوراو؛ سێگۆشە و ئاڵا و سەر و شان پشتڕاستکردنەوەی زیاتریان دەوێت.",
           health=PARTIAL, components=["range", "breakout", "measured_move"],
           limitations=["Complex formations require additional geometric validation before a setup can be ENTRY_READY."]),
        _t("fibonacci", "Fibonacci", "فیبۆناچی",
           "Retracements, extensions and the 0.618-0.79 OTE band on the latest swing leg.",
           "ڕیتڕەیسمێنت، ئێکستێنشن و ناوچەی OTE (0.618 تا 0.79) لەسەر دوایین شەپۆلی نرخ.",
           aliases=["fib", "ote"], aliases_ckb=["فیبۆ", "فیبۆناچی"], health=READY,
           components=["retracement", "extension", "projection", "ote", "confluence"],
           limitations=["Anchor selection remains theory-dependent; the anchoring leg is always reported alongside "
                        "the levels."], predicates=["in_ote", "premium_discount"]),
        _t("pivots", "Pivot Points", "خاڵەکانی پیڤۆت",
           "Standard, Fibonacci, Woodie and Camarilla pivot levels from the prior period.",
           "ئاستەکانی پیڤۆتی ستاندارد، فیبۆناچی، وودی و کامارێلا لە ماوەی پێشوو.",
           aliases_ckb=["پیڤۆت"], health=READY, components=["standard", "fibonacci", "woodie", "camarilla"]),
        _t("indicators", "Indicator Strategies", "ستراتیژی ئیندیکەیتەرەکان",
           "Moving averages, RSI, MACD, stochastic, ATR, Bollinger, Keltner, ADX, Parabolic SAR and Ichimoku.",
           "مووڤینگ ئەڤرێج، RSI، MACD، ستۆکاستیک، ATR، بۆلینگەر، کێڵتنەر، ADX، SAR و ئیچیمۆکو.",
           aliases_ckb=["ئیندیکەیتەر"], health=READY,
           components=["sma", "ema", "wma", "rsi", "macd", "stochastic", "atr", "bollinger", "keltner", "adx",
                       "parabolic_sar", "ichimoku"],
           predicates=["ema_cross", "price_vs_ema", "rsi_above", "rsi_below", "rsi_divergence", "adx_above"]),
        _t("sessions", "Session and Time Strategies", "سیشنەکان و کات",
           "Tokyo, London and New York sessions with daylight saving, and the opening-range breakout.",
           "سیشنەکانی تۆکیۆ، لەندەن و نیویۆرک لەگەڵ کاتی هاوین، و شکانی مەودای کردنەوە.",
           aliases_ckb=["سیشن", "کات"], health=READY,
           components=["tokyo", "london", "new_york", "nyse", "dst", "opening_range_breakout"],
           predicates=["session_is", "killzone", "opening_range_break", "time_window"]),
        _t("statistics", "Statistical Market Analysis", "شیکاری ئاماری بازاڕ",
           "Returns, volatility, z-score, percentiles and rolling range.",
           "گەڕانەوەی نرخ، ناجێگیری (ڤۆلەتیلیتی)، z-score و مەودای گەڕۆک.",
           aliases_ckb=["ئامار"], health=READY,
           components=["returns", "volatility", "zscore", "percentiles", "rolling_range"], predicates=["atr_at_least"]),
        _t("gann", "Gann", "گان",
           "Gann angles (1x1 = one ATR per bar), boxes and the Square of Nine; the anchor is always reported.",
           "گۆشەکانی گان، بۆکسی گان و چوارگۆشەی نۆ؛ خاڵی دەستپێک هەمیشە ڕوون دەکرێتەوە.",
           aliases_ckb=["گان"], health=PARTIAL, components=["angles", "price_time_relationships"],
           limitations=["Subjective anchors are always labeled."]),
        _t("elliott", "Elliott Wave", "شەپۆلی ئێلیۆت",
           "Impulse counts validated by the three hard rules; alternate counts are ranked, never forced.",
           "ژماردنی شەپۆلەکان بە سێ یاسا سەرەکییەکە پشتڕاست دەکرێتەوە؛ ژماردنی جێگرەوە ڕیز دەکرێن و هیچیان بەزۆر ناسەپێنرێت.",
           aliases_ckb=["ئێلیۆت", "ئیلیۆت"], health=READY,
           components=["rule_validation", "alternate_counts", "wave_fib_relationships"],
           limitations=["The three inviolable impulse rules are enforced; counts are ranked and alternates retained "
                        "rather than reduced to one answer."]),
        _t("harmonic", "Harmonic Patterns", "شێوە هارمۆنیکەکان",
           "XABCD patterns (Gartley, Bat, Butterfly, Crab, Shark, Cypher, AB=CD, 5-0) validated by ratio envelopes.",
           "شێوەکانی XABCD (گارتلی، بات، بەتەرفلای، کراب، شارک، سایفەر، AB=CD و 5-0) بە ڕێژەکانیان پشتڕاست دەکرێنەوە.",
           aliases_ckb=["هارمۆنیک"], health=READY, components=["xabcd_ratios", "prz", "targets", "invalidation"],
           limitations=["Ten definitions are validated by ratio envelope; a shape failing any constrained leg is not "
                        "reported at all."]),
        _t("pitchfork", "Pitchfork", "پیچفۆرک",
           "Andrews, Schiff and modified Schiff pitchforks from three swing pivots.",
           "پیچفۆرکی ئەندریوز، شیف و شیفی گۆڕاو لە سێ خاڵی وەرچەرخانی نرخ.",
           aliases_ckb=["پیچفۆرک"], health=PARTIAL, components=["median_line", "parallels"],
           limitations=["Anchors are subjective; the three pivots used are always reported."]),
        _t("fractal", "Fractal Market Geometry", "جیۆمەتری فراکتاڵی بازاڕ",
           "Swing fractals and nested structure across timeframes.",
           "فراکتاڵی لووتکە و نزمایی و پێکهاتەی هاوشێوە لە کاتە جیاوازەکاندا.",
           aliases_ckb=["فراکتاڵ"], health=READY,
           components=["swing_fractals", "nested_structure", "multi_scale_structure"],
           limitations=["Fractality is descriptive, not a guarantee of prediction."], predicates=["trend_is"]),
        _t("experimental_cycles", "Experimental Cycles", "خولە تاقیکارییەکان",
           "User-defined time cycles; speculative and never part of the confluence score.",
           "خولی کاتی کە بەکارهێنەر دیاری دەکات؛ تاقیکارییە و ناچێتە ناو هەڵسەنگاندنی گشتییەوە.",
           category="experimental", health=PARTIAL, components=["custom_time_cycles", "price_time_cycles"],
           limitations=["Planetary and lunar interpretations are speculative and never enter the standard confluence "
                        "score."], speculative=True),
        _t("polar_price_time", "Polar Price-Time Models", "مۆدێلی جەمسەری نرخ و کات",
           "Price and time in polar coordinates from formulas the user supplies; speculative.",
           "نرخ و کات بە هاوکێشەی جەمسەری کە بەکارهێنەر دەینووسێت؛ تاقیکارییە.",
           category="experimental", health=PARTIAL, components=["cartesian_to_polar", "normalized_angle"],
           limitations=["User formulas must be explicitly supplied and versioned."], speculative=True),
        _t("grid", "Grid Research", "توێژینەوەی گرید",
           "Arithmetic and geometric grid research; not an execution model.",
           "توێژینەوەی گریدی ئاسایی و ئەندازەیی؛ مۆدێلی مامەڵەکردن نییە.",
           category="risk_research", health=PARTIAL, components=["arithmetic_grid", "geometric_grid"],
           limitations=["Not a default execution or risk model."]),
        _t("martingale", "Martingale and Anti-Martingale Research", "توێژینەوەی مارتینگەیل و دژە-مارتینگەیل",
           "Exposure and drawdown projections of (anti-)martingale sizing; research only.",
           "پێشبینی مەترسی و دابەزینی سەرمایە لە مارتینگەیل و دژەکەی؛ تەنها بۆ توێژینەوە.",
           category="risk_research", health=PARTIAL, components=["exposure_projection", "drawdown_projection"],
           limitations=["Never activated automatically and never submits orders."]),
        _t("order_flow", "Order Flow", "ئۆردەر فلۆ (ڕەوتی فەرمانەکان)",
           "Aggressive buying and selling read from bid/ask and the aggressor side.",
           "کڕین و فرۆشتنی بەهێز لە bid و ask و لایەنی هێرشبەرەوە.",
           required_data=["Bid", "Ask", "Aggressor Side"], health=UNAVAILABLE,
           limitations=["A compatible verified bid/ask and aggressor-side feed is not configured."]),
        _t("footprint", "Footprint", "فووتپرینت",
           "Volume traded at bid and at ask inside each candle.",
           "ڤۆلیۆمی مامەڵەکراو لەسەر bid و ask لە ناو هەر مۆمێکدا.",
           required_data=["Footprint", "Aggressor Side"], health=UNAVAILABLE,
           limitations=["Never synthesized from OHLC candles."]),
        _t("cvd", "CVD", "CVD (کۆی دەلتای ڤۆلیۆم)",
           "Cumulative volume delta from true aggressor data.",
           "کۆی جیاوازی کڕین و فرۆشتنی ڕاستەقینە بە درێژایی کات.",
           required_data=["True Delta"], health=UNAVAILABLE, limitations=["A valid delta source is required."]),
        _t("dom", "Depth of Market", "قووڵایی بازاڕ (DOM)",
           "Level II order book depth.", "قووڵایی کتێبی فەرمانەکان (Level II).",
           required_data=["Level II", "Order Book"], health=UNAVAILABLE,
           limitations=["A verified Level II source is required."]),
        _t("tape", "Tape Reading", "خوێندنەوەی تەیپ",
           "Reading time-and-sales prints one by one.", "خوێندنەوەی مامەڵە جێبەجێکراوەکان یەک بە یەک.",
           required_data=["Trade Prints", "Aggressor Side"], health=UNAVAILABLE,
           limitations=["A verified time-and-sales source is required."]),
        _t("macro", "Macro Event", "ڕووداوە ئابوورییەکان (ماکرۆ)",
           "Scheduled economic events and their impact.", "ڕووداوە ئابوورییە خشتەکراوەکان و کاریگەرییان.",
           category="fundamental", required_data=["Economic Calendar"], health=UNAVAILABLE,
           limitations=["Upcoming events are never fabricated."], predicates=["no_news_blackout"]),
        _t("carry", "Carry Trade", "کاری ترەید (Carry Trade)",
           "Interest-rate differentials between currencies.", "جیاوازی ڕێژەی سوود لە نێوان دراوەکاندا.",
           category="fundamental", required_data=["Interest Rates"], health=UNAVAILABLE,
           limitations=["A verified rates source is required."]),
        _t("intermarket", "Intermarket Analysis", "شیکاری نێوان بازاڕەکان",
           "Measured correlations between markets, e.g. gold versus the dollar index.",
           "پەیوەندی پێوراو لە نێوان بازاڕەکاندا، بۆ نموونە زێڕ و ئیندێکسی دۆلار.",
           category="fundamental", required_data=["Synchronized Multi-Market OHLC"], health=UNAVAILABLE,
           limitations=["Correlations are measured, never assumed permanent."], predicates=["usdx_trend"]),
        _t("sentiment", "Sentiment", "هەستی بازاڕ (سێنتیمێنت)",
           "Positioning data such as COT and retail positioning.", "داتای هەڵوێستی ترەیدەران وەک COT.",
           category="fundamental", required_data=["Sentiment"], health=UNAVAILABLE,
           limitations=["COT or retail positioning provider is not configured."]),
    ]
    return {item.id: item for item in items}


THEORIES: dict[str, Theory] = builtin_theories()


def _keys(theory: Theory) -> list[str]:
    return [normalize_ckb(v, strip_punct=True) for v in
            (theory.id, theory.id.replace("_", " "), theory.name, theory.name_ckb, *theory.aliases, *theory.aliases_ckb)]


def find_theory(query: str, catalogue: dict[str, Theory] | None = None) -> Theory | None:
    """Exact id/name/alias (English or Sorani, normalised), then a fuzzy
    match (rapidfuzz WRatio >= 85) for spoken variants."""
    catalogue = catalogue or THEORIES
    wanted = normalize_ckb(query or "", strip_punct=True)
    if not wanted:
        return None
    for theory in catalogue.values():
        if wanted in _keys(theory):
            return theory
    try:
        from rapidfuzz import fuzz, process
    except ImportError:  # pragma: no cover - rapidfuzz is a pinned dependency
        return None
    choices = {f"{tid}\x00{i}": key for tid, t in catalogue.items() for i, key in enumerate(_keys(t)) if key}
    best = process.extractOne(wanted, choices, scorer=fuzz.WRatio, score_cutoff=85)
    return catalogue[best[2].split("\x00", 1)[0]] if best else None


def list_theories(catalogue: dict[str, Theory] | None = None) -> list[dict[str, Any]]:
    """One line per theory for the model/UI."""
    return [{"id": t.id, "name": t.name, "name_ckb": t.name_ckb, "health": t.health.value, "category": t.category}
            for t in (catalogue or THEORIES).values()]


class TradingKnowledgeRegistry:
    """v1-compatible lookup (``get`` raises KeyError on unknown names)."""

    def __init__(self) -> None:
        self._theories = builtin_theories()

    def get(self, theory: str) -> Theory:
        found = find_theory(theory, self._theories)
        if found is None:
            raise KeyError(f"Unknown trading theory: {theory}")
        return found

    def list(self) -> list[dict[str, Any]]:
        return [t.as_dict() for t in sorted(self._theories.values(), key=lambda item: item.name)]


__all__ = ["Theory", "THEORIES", "builtin_theories", "find_theory", "list_theories", "TradingKnowledgeRegistry"]
