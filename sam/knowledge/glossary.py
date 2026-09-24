"""Trading vocabulary shared by Sorani questions and English books (and the
other way round).

The user's trading books are mostly English while the questions are spoken
in Sorani, and an index cannot match «پشتگیری» against "support" by itself.
Each group below is one concept; when a question contains any spelling of a
group (Sorani, the loanwords people actually say, or English), the search
accepts every spelling of that group as the same concept. Sorani spellings
are written as SAM's users say them (letters ە ێ ۆ ڕ ڵ ی ک); matching uses the
folded search form, so Arabic-keyboard and OCR spellings match too.
"""

from __future__ import annotations

GROUPS: tuple[tuple[str, ...], ...] = (
    ("پشتگیری", "ساپۆرت", "support", "support level", "demand zone"),
    ("بەرگری", "ڕێزستەنس", "رێزیستانس", "resistance", "resistance level", "supply zone"),
    ("ترێند", "ترەند", "ڕەوت", "ئاراستە", "trend", "trending", "direction"),
    ("هێڵی ترێند", "trendline", "trend line"),
    ("مۆم", "کاندڵ", "candle", "candlestick"),
    ("قەبارە", "ڤۆلیوم", "volume"),
    ("ستۆپ لۆس", "ستۆپلۆس", "stop loss", "stop-loss", "stoploss", "protective stop"),
    ("تەیک پرۆفیت", "ئامانجی قازانج", "take profit", "profit target", "target"),
    ("مەترسی", "ڕیسک", "risk"),
    ("ڕیسک بۆ ڕیوارد", "ڕیسک و پاداشت", "risk reward", "risk to reward", "reward to risk"),
    ("شلەمەنی", "لیکویدیتی", "liquidity", "liquidity pool", "stop hunt"),
    ("ڕاماڵینی شلەمەنی", "سویپ", "sweep", "liquidity sweep", "liquidity grab", "stop run"),
    ("ئۆردەر بلۆک", "ئۆردەربلۆک", "order block", "orderblock"),
    ("فێر ڤاڵیو گاپ", "بۆشایی نرخ", "fair value gap", "fvg", "imbalance"),
    ("شکاندنی پێکهاتە", "break of structure"),
    ("گۆڕانی کاراکتەر", "چۆچ", "change of character", "choch", "market structure shift", "mss"),
    ("پێکهاتەی بازاڕ", "مارکێت ستراکچەر", "market structure", "structure"),
    ("شکاندن", "برێک ئاوت", "breakout", "break out"),
    ("گەڕانەوە", "پولباک", "پوڵباک", "pullback", "pull back", "retracement", "retest"),
    ("وەرگەڕان", "ڕیڤێرساڵ", "reversal", "reverse"),
    ("ڤیبۆناچی", "فیبۆناچی", "فیبۆ", "fibonacci", "fib", "golden ratio"),
    ("مامناوەندی جووڵاو", "مووڤینگ ئەڤرێج", "moving average", "exponential moving average"),
    ("ئار ئێس ئای", "rsi", "relative strength index"),
    ("دایڤێرجێنس", "divergence"),
    ("کڕین", "لۆنگ", "buy", "long position", "go long", "bullish entry"),
    ("فرۆشتن", "سێڵ", "شۆرت", "sell", "short position", "go short", "bearish entry"),
    ("بەرزبوونەوە", "بوڵیش", "bullish", "uptrend", "rally"),
    ("دابەزین", "بێریش", "bearish", "downtrend", "decline"),
    ("لووتکە", "high", "swing high", "peak", "higher high"),
    ("نزمترین", "بنکە", "swing low", "bottom", "lower low"),
    ("زێڕ", "گۆڵد", "گۆڵت", "gold", "xauusd"),
    ("دۆلار", "dollar", "usd", "dxy"),
    ("نرخ", "price"),
    ("بازاڕ", "مارکێت", "market"),
    ("ستراتیژی", "ستراتیژ", "strategy", "setup"),
    ("تیۆری", "theory"),
    ("شیکاری", "ئانالیز", "analysis", "analyze", "analyse"),
    ("چوارچێوەی کات", "تایم فرەیم", "تایمفرەیم", "timeframe", "time frame"),
    ("دانیشتن", "سێشن", "سەشن", "session", "killzone", "kill zone"),
    ("ئاسیا", "asia", "asian session"),
    ("لەندەن", "london", "london session"),
    ("نیویۆرک", "new york", "ny session"),
    ("هەواڵ", "نیوز", "news", "economic calendar"),
    ("دەروونناسی", "سایکۆلۆجی", "psychology", "discipline", "emotions"),
    ("بەڕێوەبردنی سەرمایە", "مانی مەنەجمێنت", "money management", "position sizing", "position size",
     "risk management"),
    ("لۆت", "lot size", "lots"),
    ("لیڤەرێج", "leverage", "margin"),
    ("سپرێد", "spread"),
    ("پاتێرن", "pattern", "chart pattern"),
    ("سەر و شان", "head and shoulders"),
    ("دوو لووتکە", "دەبڵ تۆپ", "double top"),
    ("دوو بنکە", "دەبڵ بۆتۆم", "double bottom"),
    ("سێگۆشە", "triangle"),
    ("ئاڵا", "flag", "pennant"),
    ("ڕەینج", "مەودا", "range", "consolidation", "sideways"),
    ("چوونەژوورەوە", "ئینتری", "entry", "entry point"),
    ("دەرچوون", "ئێگزیت", "exit"),
    ("ئیمپاڵس", "impulse", "impulsive move"),
    ("ڕاستکردنەوە", "کۆرێکشن", "correction", "corrective"),
    ("پرێمیۆم", "premium"),
    ("دیسکاونت", "discount"),
    ("باشترین شوێنی چوونەژوورەوە", "optimal trade entry"),
    ("ژۆرناڵ", "ڕۆژنامەی بازرگانی", "journal", "trading journal"),
    ("باکتێست", "back test", "backtest", "backtesting"),
    ("سکاڵپ", "سکالپینگ", "scalp", "scalping"),
    ("سوینگ", "swing", "swing trading"),
)

__all__ = ["GROUPS"]
