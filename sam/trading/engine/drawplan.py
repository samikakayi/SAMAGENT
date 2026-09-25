"""Turn a report into chart drawings (items for ``TradingViewBridge.draw_many``).

v1's ``service._drawing_plan`` read ``report["setup"]``, a key the analyst
never produced, so entry/stop/TP were never drawn -- only S/R lines
(reports/trading-intelligence.json, defect 2). This plan reads the report's
top-level ``entry, stop, tp1..tp3`` keys, which ``analyst.build_report`` and
``core.apply_strategy`` always set together with ``verdict``.

Modes (docs/CONTRACTS.md 3.5): ``levels`` = support/resistance only (the fast
path for "هێڵی پشتگیری و بەرگری بکێشە"); ``full`` = levels + zones +
entry/stop/targets when the verdict is SETUP. ``price_offset`` shifts every
price into the chart's feed when the numbers came from MT5 but the chart
shows another feed (gold-api vs MT5 differed by 1.4 USD, measured).

Repair review 2026-09-24 (live drawing on PEPPERSTONE:XAUUSD, chart_after_analyze.png):
- a closed market or gold's daily break drew NOTHING (the plan returned [] on
  ``stale``): historic levels and zones are drawn now; only entry/stop/targets
  need a live market;
- clutter: three overlapping green order-block boxes plus an FVG inside them in
  a 12 USD band, labels printed over each other, resistance lines 1.6 apart.
  Now same-side zones overlapping by more than half are merged, one zone per
  side (the nearest) is drawn, extended to the right with alternating label
  corners, and lines closer than 0.3 ATR of the chart's timeframe are one line.
"""

from __future__ import annotations

from typing import Any

from .sorani import zone_label

ZONE_COLORS = {"bullish": "support", "bearish": "resistance"}
FIRST_BARS_AGO = 60  # left edge for zones whose origin time is unknown (OTE)
LINE_GAP_ATR = 0.3


def _p(value: float, offset: float) -> float:
    return round(float(value) + offset, 6)


def _atr(report: dict[str, Any]) -> float:
    indicators = report.get("indicators") or {}
    chart_tf = ((report.get("chart") or {}).get("timeframe")) or ""
    for tf in (chart_tf, "M15", "M5", "H1", "M30"):
        value = (indicators.get(tf) or {}).get("atr14") if tf else None
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    values = [v.get("atr14") for v in indicators.values() if isinstance(v, dict)]
    values = [float(v) for v in values if isinstance(v, (int, float)) and v > 0]
    return min(values) if values else 0.0


def _side(zone: dict[str, Any], price: float) -> str:
    direction = str(zone.get("direction") or "").lower()
    if direction in ("bullish", "bearish"):
        return direction
    return "bullish" if float(zone["high"]) <= price else "bearish"


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Overlap as a share of the smaller zone."""
    low, high = max(float(a["low"]), float(b["low"])), min(float(a["high"]), float(b["high"]))
    smaller = min(float(a["high"]) - float(a["low"]), float(b["high"]) - float(b["low"]))
    if high <= low:
        return 0.0
    return 1.0 if smaller <= 0 else (high - low) / smaller


def plan_zones(zones: list[dict[str, Any]], price: float, *, per_side: int = 1) -> list[dict[str, Any]]:
    """Merge same-side zones that overlap by more than half (the first -- the
    higher timeframe after ``collect_zones`` -- keeps its label), then keep the
    nearest ``per_side`` zones on each side of price."""
    merged: list[dict[str, Any]] = []
    for zone in zones:
        side = _side(zone, price)
        into = next((m for m in merged if m["_side"] == side and _overlap(m, zone) > 0.5), None)
        if into is None:
            merged.append({**zone, "_side": side})
        else:
            into["low"], into["high"] = min(into["low"], zone["low"]), max(into["high"], zone["high"])

    def distance(z: dict[str, Any]) -> float:
        return 0.0 if z["low"] <= price <= z["high"] else min(abs(price - z["low"]), abs(price - z["high"]))

    out: list[dict[str, Any]] = []
    for side in ("bearish", "bullish"):
        out += sorted((z for z in merged if z["_side"] == side), key=distance)[:per_side]
    return [{k: v for k, v in z.items() if k != "_side"} for z in out]


def build_draw_plan(report: dict[str, Any], mode: str = "full", *, price_offset: float = 0.0,
                    max_levels: int = 3, max_zones: int = 1) -> list[dict[str, Any]]:
    """Drawing items: ``{kind, points [{price, time|bars_ago}], text, color[, style]}``.
    ``max_zones`` is per side of price."""
    if mode not in ("levels", "full"):
        return []
    live = not (report.get("stale") or report.get("market_closed"))
    gap = _atr(report) * LINE_GAP_ATR
    items: list[dict[str, Any]] = []
    placed: list[float] = []
    for key, color, word in (("resistance", "resistance", "بەرگری"), ("support", "support", "پشتگیری")):
        count = 0
        for level in report.get(key) or []:
            if count >= max_levels:
                break
            price = float(level["price"])
            if any(abs(price - other) < gap for other in placed):
                continue
            placed.append(price)
            count += 1
            items.append({"kind": "horizontal_line", "points": [{"price": _p(price, price_offset)}],
                          "text": f"{word} {level.get('tf', '')}".strip(), "color": color})
    if mode == "levels":
        return items
    price_now = float(report.get("price") or 0.0)
    for index, zone in enumerate(plan_zones(list(report.get("zones") or []), price_now, per_side=max_zones)):
        left: dict[str, Any] = {"time": int(zone["time"])} if zone.get("time") else {"bars_ago": FIRST_BARS_AGO}
        items.append({
            "kind": "rectangle",
            "points": [{**left, "price": _p(zone["high"], price_offset)},
                       {"bars_ago": 0, "price": _p(zone["low"], price_offset)}],
            "text": f"{zone_label(zone)} {zone.get('tf', '')}".strip(),
            "color": "zone" if zone.get("kind") in ("fvg", "ote") else ZONE_COLORS.get(zone.get("direction"), "zone"),
            "style": {"extend_right": True, "label_valign": "top" if index % 2 == 0 else "bottom"},
        })
    if live and report.get("verdict") == "SETUP" and report.get("entry") is not None and report.get("stop") is not None \
            and report.get("tp1") is not None:
        kind = "long_position" if report.get("direction") == "long" else "short_position"
        items.append({"kind": kind, "points": [{"bars_ago": 0, "price": _p(report["entry"], price_offset)},
                                               {"bars_ago": 0, "price": _p(report["stop"], price_offset)},
                                               {"bars_ago": 0, "price": _p(report["tp1"], price_offset)}],
                      "text": "سێتەپی سام", "color": "entry"})
        items.append({"kind": "horizontal_ray", "points": [{"bars_ago": 0, "price": _p(report["stop"], price_offset)}],
                      "text": "ستۆپ", "color": "stop"})
        for index, key in enumerate(("tp1", "tp2", "tp3"), start=1):
            if report.get(key) is not None:
                items.append({"kind": "horizontal_ray",
                              "points": [{"bars_ago": 0, "price": _p(report[key], price_offset)}],
                              "text": f"ئامانجی {index}", "color": "target"})
    return items


def describe_plan(items: list[dict[str, Any]]) -> str:
    """Short English summary of a plan (for logs and dry runs)."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    return ", ".join(f"{n} {k}" for k, n in counts.items()) or "nothing"


__all__ = ["build_draw_plan", "describe_plan", "plan_zones"]
