"""Sorani wording for analysis results and alerts (templates, no LLM).

v1's ``_sorani_summary`` was mostly English with Sorani glue ("HTF bearish ـە.
ئێستا WAIT ـە") -- one of the audit findings (reports/trading-intelligence.json).
Everything the user hears about the market is built here from numbers the
engine computed, so wording is consistent, fast (no model call) and never
invents a price.

Conventions: prices use Western digits (they read well in TTS and in RTL
text); a copula is never glued to a number ("نرخ 2651.3ە" reads badly), so
sentences are shaped to end in a word instead. SPOKEN sentences round prices
the way a trader says them (``spoken_price``: gold 4269.81 -> 4270, EURUSD
1.085431 -> 1.0854; persona rule "a price to a whole number or one decimal");
the panel text keeps the exact digits.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Iterable

SYMBOL_CKB: dict[str, str] = {
    "XAUUSD": "زێڕ", "XAGUSD": "زیو", "BTCUSD": "بیتکۆین", "ETHUSD": "ئیسریوم", "EURUSD": "یۆرۆ دۆلار",
    "GBPUSD": "پاوەند دۆلار", "USDJPY": "دۆلار ین", "USDX": "ئیندێکسی دۆلار", "NAS100": "ناسداک",
    "US30": "داو جۆنز", "USOIL": "نەوت",
}
TF_CKB: dict[str, str] = {
    "M1": "یەک خولەک", "M5": "پێنج خولەک", "M15": "پازدە خولەک", "M30": "نیو کاتژمێر",
    "H1": "یەک کاتژمێر", "H4": "چوار کاتژمێر", "D1": "ڕۆژانە", "W1": "هەفتانە", "MN1": "مانگانە",
}
TREND_CKB = {"up": "بەرەو سەرەوە", "down": "بەرەو خوارەوە", "range": "بێ ئاراستە"}
VERDICT_CKB = {"SETUP": "سێتەپ ئامادەیە", "WAIT": "چاوەڕێ بکە", "NO_TRADE": "ئێستا کاتی مامەڵە نییە"}
ZONE_CKB = {"fvg": "بۆشایی نرخ (FVG)", "order_block": "ئۆردەر بلۆک", "supply": "ناوچەی خستنەڕوو (سەپڵای)",
            "demand": "ناوچەی داواکاری (دیماند)", "ote": "ناوچەی OTE"}
SIDE_CKB = {"bullish": "کڕین", "bearish": "فرۆشتن"}
# Engine confirmation names and self-check names -> what the user is told is missing.
MISSING_CKB: dict[str, str] = {
    "liquidity_sweep": "ڕاماڵینی لیکویدیتی",
    "mss_or_bos": "شکانی پێکهاتەی بازاڕ",
    "candle_trigger": "مۆمێکی پشتڕاستکەرەوە",
    "higher_timeframe_bias": "هاوئاراستەبوونی کاتە گەورەکان",
    "Aligned higher-timeframe structure": "هاوئاراستەبوونی کاتە گەورەکان",
    "Directional higher-timeframe structure": "ئاراستەیەکی ڕوون لە کاتە گەورەکان",
    "Technical invalidation level": "ئاستێکی ڕوون بۆ ستۆپ",
    "Technical target": "ئامانجێکی تەکنیکی",
    "data_fresh": "داتای نوێ",
    "timestamps_not_future": "کاتی دروستی داتا",
    "quote_timestamps_verified": "نرخی نوێی بازاڕ",
    "requested_timeframes": "هەموو کاتە داواکراوەکان",
    "symbol_resolved": "ناسینەوەی هێما",
    "no_duplicates": "داتای پاک",
    "feed_identified": "سەرچاوەی داتا",
    "entry_confirmed": "پشتڕاستکردنەوەی چوونەژوورەوە",
    "technical_stop": "ستۆپی تەکنیکی",
    "technical_targets": "ئامانجی تەکنیکی",
}


def symbol_ckb(symbol: str) -> str:
    return SYMBOL_CKB.get((symbol or "").upper(), symbol or "")


def tf_ckb(tf: str) -> str:
    return TF_CKB.get(tf, tf)


def fmt_price(value: Any, digits: int | None = None) -> str:
    """Readable price: gold 2651.34 -> '2651.34', EURUSD 1.085431 -> '1.08543'."""
    if value is None:
        return "?"
    number = float(value)
    if digits is None:
        magnitude = abs(number)
        digits = 2 if magnitude >= 100 else 3 if magnitude >= 10 else 5
    text = f"{number:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def spoken_price(value: Any) -> str:
    """A price as it is said: >= 1000 whole, >= 100 one decimal, >= 1 four
    decimals at most (FX), else five (review 2026-09-24: "4269.81" was read)."""
    if value is None:
        return "?"
    number = float(value)
    magnitude = abs(number)
    digits = 0 if magnitude >= 1000 else 1 if magnitude >= 100 else 4 if magnitude >= 1 else 5
    return fmt_price(round(number, digits), digits)


def fmt_rr(value: Any) -> str:
    return "?" if value is None else f"{float(value):.1f}"


def join_ckb(items: Iterable[str]) -> str:
    """'a', 'b', 'c' -> 'a، b و c' (Sorani list join)."""
    values = [item for item in items if item]
    if len(values) <= 1:
        return "".join(values)
    return "، ".join(values[:-1]) + " و " + values[-1]


def missing_ckb(names: Iterable[Any], limit: int = 3) -> list[str]:
    out: list[str] = []
    for name in names:
        text = MISSING_CKB.get(str(name))
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def trend_sentence(trend: dict[str, str]) -> str:
    """{'H1': 'down', 'M15': 'down', 'M5': 'range'} -> Sorani clause."""
    if not trend:
        return "ترێند دیار نییە"
    groups: dict[str, list[str]] = {"up": [], "down": [], "range": []}
    for tf, word in trend.items():
        groups.setdefault(word, []).append(tf_ckb(tf))
    present = [key for key in ("down", "up", "range") if groups.get(key)]
    if len(present) == 1:
        only = present[0]
        # «لە هیچ کاتێکدا» reads as "never" (review 2026-09-24): name the timeframes.
        if only == "range":
            return "لە هیچ تایمفرەیمێکدا ترێندی ڕوون نییە"
        return f"ترێند لە هەموو تایمفرەیمەکاندا {TREND_CKB[only]}یە"
    parts = []
    for key in present:
        tfs = join_ckb(groups[key])
        parts.append(f"لە {tfs} بێ ئاراستەیە" if key == "range" else f"لە {tfs} {TREND_CKB[key]}یە")
    return "ترێند " + "، ".join(parts)


def _direction_word(direction: str | None) -> str:
    return "بەرەو سەرەوە" if direction == "long" else "بەرەو خوارەوە" if direction == "short" else ""


def _local_time(at: float | None = None, tz: str = "Asia/Baghdad") -> str:
    try:
        from zoneinfo import ZoneInfo
        moment = _dt.datetime.fromtimestamp(at or _dt.datetime.now().timestamp(), ZoneInfo(tz))
    except Exception:  # noqa: BLE001 - tzdata missing: fall back to UTC wording
        moment = _dt.datetime.now(_dt.UTC)
    return moment.strftime("%H:%M")


def _failed_rule_texts(report: dict[str, Any], kinds: tuple[str, ...] | None = None) -> list[str]:
    strategy = report.get("strategy") or {}
    out = []
    for rule in strategy.get("rules") or []:
        if rule.get("passed") is False and (kinds is None or rule.get("kind") in kinds):
            out.append(str(rule.get("text_ckb") or rule.get("text_en") or ""))
    return [text for text in out if text]


def _levels_sentence(report: dict[str, Any]) -> str:
    support = (report.get("support") or [None])[0]
    resistance = (report.get("resistance") or [None])[0]
    if support and resistance:
        return (f"نزیکترین بەرگری {spoken_price(resistance['price'])} و نزیکترین پشتگیری "
                f"{spoken_price(support['price'])}.")
    if resistance:  # price at its lows: nothing below to name
        return f"لە خوارەوە پشتگیری دیار نییە و نزیکترین بەرگری {spoken_price(resistance['price'])}."
    if support:
        return f"لە سەرەوە بەرگری دیار نییە و نزیکترین پشتگیری {spoken_price(support['price'])}."
    return ""


def spoken_summary(report: dict[str, Any]) -> str:
    """Two or three short spoken sentences (no Markdown, numbers from the engine)."""
    sym = symbol_ckb(report.get("symbol", ""))
    price = spoken_price(report.get("price"))
    if report.get("market_closed") or report.get("stale"):
        # Levels from history are exactly what a closed-market review needs
        # (review 2026-09-24: a Saturday "draw S/R" got nothing and "analysis after
        # the open"); only a setup needs a live market.
        head = (f"بازاڕی {sym} ئێستا داخراوە و دوایین نرخ {price} بوو؛ ئەمانە ئاستەکانن بۆ کرانەوەی بازاڕ."
                if report.get("market_closed") else f"داتای {sym} نوێ نییە، بۆیە سێتەپ پێشنیار ناکەم؛ ئاستەکان ئەمانەن.")
        return " ".join(part for part in (head, _levels_sentence(report)) if part)
    sentences = [f"{sym} ئێستا لەسەر {price} مامەڵە دەکرێت و {trend_sentence(report.get('trend') or {})}."]
    verdict = report.get("verdict")
    if verdict == "SETUP":
        sentences.append(
            f"سێتەپێکی {_direction_word(report.get('direction'))} ئامادەیە: چوونەژوورەوە {spoken_price(report.get('entry'))}، "
            f"ستۆپ {spoken_price(report.get('stop'))}، ئامانجی یەکەم {spoken_price(report.get('tp1'))}، "
            f"ڕێژەی قازانج بە مەترسی {fmt_rr(report.get('rr'))}.")
        sentences.append("ئەمە تەنها شیکارییە و بڕیاری کۆتایی هی خۆتە.")
        return " ".join(sentences)
    if verdict == "NO_TRADE":
        filters = _failed_rule_texts(report, ("filter", "risk"))
        if filters:
            sentences.append(f"ئێستا کاتی مامەڵە نییە، چونکە ئەم مەرجە جێبەجێ نەبووە: {filters[0]}.")
        elif report.get("rr") is not None:
            sentences.append(f"ئێستا کاتی مامەڵە نییە، چونکە ئامانجی نزیک تەنها {fmt_rr(report.get('rr'))} "
                             "هێندەی مەترسییەکە قازانج دەدات.")
        else:
            sentences.append("ئێستا کاتی مامەڵە نییە.")
    else:
        waiting = _failed_rule_texts(report)[:2] or missing_ckb(report.get("missing_confirmation") or [], 2)
        if waiting:
            sentences.append(f"هێشتا سێتەپ ئامادە نییە؛ چاوەڕێی {join_ckb(waiting)} دەکەم.")
        else:
            sentences.append("هێشتا سێتەپ ئامادە نییە.")
    levels = _levels_sentence(report)
    if levels:
        sentences.append(levels)
    return " ".join(sentences)


# Izafe forms for zones that have a side ("ئۆردەر بلۆکی کڕین").
_SIDED_ZONE_CKB = {"order_block": "ئۆردەر بلۆکی {side}", "fvg": "بۆشایی نرخی {side} (FVG)"}


def zone_label(zone: dict[str, Any]) -> str:
    kind = zone.get("kind", "")
    side = SIDE_CKB.get(str(zone.get("direction") or "").lower(), "")
    if kind in _SIDED_ZONE_CKB and side:
        return _SIDED_ZONE_CKB[kind].format(side=side)
    return ZONE_CKB.get(kind, kind)


def _zone_line(zone: dict[str, Any], digits: int | None) -> str:
    return (f"{zone_label(zone)} ({zone.get('tf')}): {fmt_price(zone.get('low'), digits)}–"
            f"{fmt_price(zone.get('high'), digits)}")


def full_text(report: dict[str, Any]) -> str:
    """The panel version: every number with its timeframe, one topic per line."""
    digits = report.get("digits")
    sym = report.get("symbol", "")
    source = {"tradingview": "چارتی TradingView", "mt5": "MetaTrader 5"}.get(report.get("data_source", ""),
                                                                           report.get("data_source", ""))
    lines = [f"شیکاری {symbol_ckb(sym)} ({sym}) · سەرچاوە: {source} · کاتژمێر {_local_time(report.get('at'))}",
             f"نرخ: {fmt_price(report.get('price'), digits)}"]
    trend = report.get("trend") or {}
    if trend:
        lines.append("ترێند: " + " · ".join(f"{tf_ckb(tf)}: {TREND_CKB.get(word, word)}" for tf, word in trend.items()))
    lines.append(f"بڕیار: {VERDICT_CKB.get(report.get('verdict', 'WAIT'), report.get('verdict', ''))}")
    if report.get("verdict") == "SETUP":
        targets = " · ".join(fmt_price(report.get(key), digits) for key in ("tp1", "tp2", "tp3") if report.get(key))
        lines.append(f"ئاراستە: {_direction_word(report.get('direction'))} · چوونەژوورەوە: "
                     f"{fmt_price(report.get('entry'), digits)} · ستۆپ: {fmt_price(report.get('stop'), digits)} · "
                     f"ئامانجەکان: {targets} · RR: {fmt_rr(report.get('rr'))}")
    else:
        reasons = missing_ckb(report.get("missing_confirmation") or [], 4)
        if reasons:
            lines.append("چاوەڕێی: " + join_ckb(reasons))
    potential = report.get("potential_plan")
    # A plan whose first target pays less than the risk is noise, not guidance.
    if report.get("verdict") != "SETUP" and potential and potential.get("stop") is not None and (potential.get("rr") or 0) >= 1:
        lines.append(f"ئەگەر پشتڕاست بووەوە ({_direction_word(potential.get('direction'))}): ستۆپ "
                     f"{fmt_price(potential.get('stop'), digits)} · ئامانجی یەکەم "
                     f"{fmt_price((potential.get('targets') or [{}])[0].get('price'), digits)} · RR "
                     f"{fmt_rr(potential.get('rr'))}")
    for key, title in (("resistance", "بەرگرییەکان"), ("support", "پشتگیرییەکان")):
        items = report.get(key) or []
        if items:
            lines.append(f"{title}: " + " · ".join(f"{fmt_price(i['price'], digits)} ({i.get('tf')})" for i in items[:4]))
    zones = report.get("zones") or []
    if zones:
        lines.append("ناوچەکان: " + " · ".join(_zone_line(zone, digits) for zone in zones[:5]))
    strategy = report.get("strategy")
    if strategy:
        marks = {True: "✓", False: "✗", None: "؟"}
        lines.append(f"ستراتیژی «{strategy.get('title_ckb') or strategy.get('id')}»:")
        for rule in strategy.get("rules") or []:
            how = {"vision": " (بە سەیرکردنی چارت)", "llm": " (نەپشکنراوە)"}.get(rule.get("how", ""), "")
            lines.append(f"  {marks.get(rule.get('passed'), '؟')} {rule.get('text_ckb') or rule.get('text_en')}{how}")
    failed = [c["name"] for c in report.get("checks") or [] if not c.get("passed")]
    if failed:
        lines.append("ئاگاداری داتا: " + join_ckb(missing_ckb(failed, 4) or failed[:4]))
    if report.get("feed_offset"):
        lines.append(f"جیاوازی نرخی TradingView و MT5: {fmt_price(report['feed_offset'], digits)}")
    lines.append("ئەمە تەنها شیکارییە، نەک ئامۆژگاری دارایی؛ سام هیچ مامەڵەیەک ناکات.")
    return "\n".join(lines)


__all__ = ["SYMBOL_CKB", "TF_CKB", "symbol_ckb", "tf_ckb", "fmt_price", "spoken_price", "fmt_rr", "join_ckb",
           "missing_ckb",
           "trend_sentence", "spoken_summary", "full_text", "TREND_CKB", "VERDICT_CKB", "ZONE_CKB"]
