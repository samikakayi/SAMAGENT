"""Theory execution and report assembly over per-timeframe analyses.

Ported from v1 ``sam_backend/trading/analyst.py`` (``MarketAnalyst``) without
its I/O: fetching moved to ``engine.core``/``mt5.py`` and persistence to
``analyze.py``. Audit fixes applied here (reports/trading-intelligence.json):

- v1 hard-coded ``"order_block": {"candidates": []}`` although detection found
  ~12 blocks per timeframe on live XAUUSD. The report now carries the real
  blocks (``order_blocks``) and uses them as zones.
- v1's drawing plan read ``report["setup"]``, which was never produced, so
  entry/stop/TP lines were never drawn. The report now always exposes
  ``entry, stop, invalidation, tp1, tp2, tp3, targets, rr`` at the top level
  and ``engine.drawplan`` reads exactly those keys.
"""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import fmean
from typing import Any

from .analysis import plan_from_levels, session_context
from .patterns import detect_harmonics, opening_range, rank_wave_counts
from .types import Candle, CapabilityState, Direction, SetupDecision, SetupState

TREND_WORD = {Direction.BULLISH.value: "up", Direction.BEARISH.value: "down", Direction.NEUTRAL.value: "range"}
HTF_ORDER = ("MN1", "W1", "D1", "H4", "H1", "M30", "M15", "M5", "M3", "M1")
LTF_ORDER = ("M1", "M3", "M5", "M15", "M30", "H1", "H4", "D1")
_TIMESTAMP_CHECKS = {"data_fresh", "timestamps_not_future", "quote_timestamps_verified"}


def _iso_to_ts(value: Any) -> int | None:
    try:
        return int(datetime.fromisoformat(str(value)).astimezone(UTC).timestamp())
    except (TypeError, ValueError):
        return None


def _r(value: Any, nd: int = 5) -> Any:
    return round(float(value), nd) if isinstance(value, (int, float)) else value


# --- self-check gate (v1 MarketAnalyst.analyze) -----------------------------------

def gate_setup(setup: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
    """Block entry output when timestamps/freshness failed, or when an
    ENTRY_READY setup fails any critical self-check."""
    failures = set(checks.get("critical_failures") or [])
    blocked = {"entry": None, "entry_zone": None, "technical_invalidation": None, "stop": None,
               "stop_distance": None, "targets": [], "rr": None, "missing_confirmation": sorted(failures)}
    if failures & _TIMESTAMP_CHECKS:
        return {**setup, **blocked, "decision": SetupDecision.WAIT.value, "state": SetupState.WATCH.value,
                "reasons": ["Provider timestamps or quote freshness could not be verified; setup output is "
                            "blocked until the feed clock is trustworthy."]}
    if setup.get("decision") == SetupDecision.ENTRY_READY.value and not checks.get("passed"):
        return {**setup, **blocked, "decision": SetupDecision.WAIT.value,
                "state": SetupState.WAITING_FOR_TRIGGER.value,
                "reasons": ["Self-check blocked ENTRY_READY because a critical verification failed."]}
    return setup


def confidence(setup: dict[str, Any], checks: dict[str, Any]) -> float:
    confirmations = setup.get("confirmations") or {}
    present = sum(bool(value) for value in confirmations.values())
    score = present / max(1, len(confirmations)) * 0.75 + (0.25 if checks.get("passed") else 0)
    return round(min(1.0, max(0.0, score)), 3)


# --- theories (v1 MarketAnalyst._execute_theory) ----------------------------------

def run_theory(theory: Any, analyses: dict[str, dict[str, Any]],
               candles_by_tf: dict[str, list[Candle]]) -> dict[str, Any]:
    """Run one catalogue theory over the analyses. ``theory`` is a
    ``theories.Theory`` (id, health, limitations, as_dict)."""
    if theory.health in (CapabilityState.UNAVAILABLE, CapabilityState.UNCONFIGURED):
        return {"theory": theory.as_dict(), "status": theory.health.value, "facts": [], "observations": [],
                "interpretation": None,
                "reason": theory.limitations[0] if theory.limitations else "Required data is unavailable."}
    facts: list[dict[str, Any]] = []
    observations: list[str] = []
    directions: list[Direction] = []
    for timeframe, analysis in analyses.items():
        direction = Direction(analysis["structure"]["trend"])
        directions.append(direction)
        facts.append({"timeframe": timeframe, "structure": direction.value, "current_price": analysis["current_price"]})
        active = len([gap for gap in analysis["fvg"] if gap["active"]])
        observations.append(f"{timeframe}: {direction.value.lower()} structure, {len(analysis['snr'])} scored S/R "
                            f"levels, {active} active FVG(s).")
    bullish = sum(d == Direction.BULLISH for d in directions)
    bearish = sum(d == Direction.BEARISH for d in directions)
    aggregate = Direction.BULLISH if bullish > bearish else Direction.BEARISH if bearish > bullish else Direction.NEUTRAL
    result: dict[str, Any] = {
        "theory": theory.as_dict(), "status": theory.health.value, "facts": facts, "observations": observations,
        "interpretation": {"direction": aggregate.value, "bullish_timeframes": bullish, "bearish_timeframes": bearish},
        "limitations": theory.limitations,
    }
    tid = theory.id
    per_tf = lambda key: {tf: a.get(key) for tf, a in analyses.items()}  # noqa: E731
    if tid == "snr":
        result["levels"] = per_tf("snr")
    elif tid in {"smc", "ict", "liquidity"}:
        result["structure"] = per_tf("structure")
        result["liquidity"] = per_tf("liquidity")
        result["imbalances"] = per_tf("fvg")
        if tid == "ict":
            result["session"] = session_context()
            result["unsupported_without_extra_data"] = ["SMT Divergence", "True Order Flow", "Economic Event Context"]
    elif tid == "wyckoff":
        result.update(wyckoff(analyses))
    elif tid == "volume_profile":
        result["profiles"] = per_tf("volume_profile")
    elif tid == "vwap":
        result["vwap"] = {tf: a["indicators"]["vwap"] for tf, a in analyses.items()}
    elif tid in {"price_action", "candlesticks"}:
        result["patterns"] = per_tf("candlestick_patterns")
    elif tid == "order_blocks":
        result["order_blocks"] = {tf: a.get("order_blocks", []) for tf, a in analyses.items()}
        fresh = sum(1 for blocks in result["order_blocks"].values() for block in blocks if block.get("fresh"))
        observations.append(f"{fresh} fresh order block(s) across the requested timeframes.")
    elif tid == "fibonacci":
        result["fibonacci"] = per_tf("fibonacci")
    elif tid == "harmonic":
        result["patterns"] = {tf: detect_harmonics(candles_by_tf.get(tf) or [], tf) for tf in analyses}
        if not any(result["patterns"].values()):
            observations.append("No XABCD structure passed ratio validation on the requested timeframes.")
    elif tid == "elliott":
        result["counts"] = {tf: rank_wave_counts(candles_by_tf.get(tf) or [], tf) for tf in analyses}
        observations.append("Alternate counts are ranked, never reduced to a single forced count.")
    elif tid == "sessions":
        result["opening_ranges"] = {
            tf: {session: opening_range(candles_by_tf[tf], minutes=15, session=session)
                 for session in ("london", "new_york")}
            for tf in analyses if candles_by_tf.get(tf)}
    return result


def wyckoff(analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = next((key for key in ("H1", "M30", "M15") if key in analyses), next(iter(analyses)))
    analysis = analyses[primary]
    zscore = float(analysis["statistics"].get("zscore") or 0)
    structure = analysis["structure"]["trend"]
    if abs(zscore) < 0.8 and structure == Direction.NEUTRAL.value:
        candidates, phase = ["ACCUMULATION", "DISTRIBUTION"], "B_OR_C_UNCONFIRMED"
    elif structure == Direction.BULLISH.value:
        candidates, phase = ["MARKUP", "REACCUMULATION"], "D_OR_E_UNCONFIRMED"
    elif structure == Direction.BEARISH.value:
        candidates, phase = ["MARKDOWN", "REDISTRIBUTION"], "D_OR_E_UNCONFIRMED"
    else:
        candidates, phase = ["UNCLASSIFIED_RANGE"], "UNCONFIRMED"
    return {"wyckoff": {"timeframe": primary, "alternate_interpretations": candidates, "phase_candidate": phase,
                        "events": [], "confidence": 0.35,
                        "warning": "A phase/event label is not confirmed without a validated range sequence and "
                                   "effort/result evidence."}}


# --- report pieces -----------------------------------------------------------------

def ordered_tfs(analyses: dict[str, Any], order: tuple[str, ...]) -> list[str]:
    known = [tf for tf in order if tf in analyses]
    return known + [tf for tf in analyses if tf not in known]


def execution_analysis(analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return next((analyses[k] for k in ("M5", "M3", "M1") if k in analyses), analyses[ordered_tfs(analyses, LTF_ORDER)[0]])


def merged_levels(analyses: dict[str, dict[str, Any]], price: float) -> list[dict[str, Any]]:
    """S/R from every timeframe, merged where two timeframes found the same
    price (within 0.2 ATR of the highest timeframe) so a drawing is one line
    per level, labelled with the strongest timeframe."""
    htf = analyses[ordered_tfs(analyses, HTF_ORDER)[0]]
    atr_value = float(htf["indicators"].get("atr14") or 0.0)
    tolerance = max(atr_value * 0.2, abs(price) * 0.0002)
    raw = sorted((dict(level) for a in analyses.values() for level in a.get("snr", [])), key=lambda l: l["price"])
    clusters: list[list[dict[str, Any]]] = []
    for level in raw:
        if clusters and abs(level["price"] - fmean(x["price"] for x in clusters[-1])) <= tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    top_score = max((l["score"] for l in raw), default=1.0) or 1.0
    merged = []
    for cluster in clusters:
        best = max(cluster, key=lambda l: l["score"])
        score = max(l["score"] for l in cluster) + 0.3 * (len({l["timeframe"] for l in cluster}) - 1)
        merged.append({
            "price": best["price"], "kind": "support" if best["price"] < price else "resistance",
            "tf": best["timeframe"], "tfs": sorted({l["timeframe"] for l in cluster}),
            "strength": round(min(1.0, score / top_score), 3), "score": round(score, 3),
            "reactions": sum(int(l.get("reactions") or 0) for l in cluster),
            "role_reversal": any(l.get("role_reversal") for l in cluster), "broken": all(l.get("broken") for l in cluster),
        })
    return sorted(merged, key=lambda l: abs(l["price"] - price))


def key_levels(levels: list[dict[str, Any]], analyses: dict[str, Any]) -> list[dict[str, Any]]:
    """Levels worth naming/drawing: found on M5 or higher (M1 swings are noise
    for a chart the user reads; measured live 2026-09-24: the nearest three
    'resistances' were all M1 levels within $13). Falls back to everything
    when only M1 was analysed."""
    wanted = {tf for tf in analyses if tf != "M1"}
    if not wanted:
        return levels
    return [level for level in levels if wanted & set(level.get("tfs") or [level.get("tf")])]


def collect_order_blocks(analyses: dict[str, dict[str, Any]], price: float, limit: int = 12) -> list[dict[str, Any]]:
    blocks = []
    for tf, analysis in analyses.items():
        for block in analysis.get("order_blocks") or []:
            blocks.append({"tf": tf, "kind": "bullish" if block["kind"] == "BULLISH_OB" else "bearish",
                           "low": block["bottom"], "high": block["top"], "state": block["state"],
                           "time": _iso_to_ts(block.get("time")), "displacement_atr": block.get("displacement_atr")})
    return sorted(blocks, key=lambda b: _distance(price, b["low"], b["high"]))[:limit]


def _distance(price: float, low: float, high: float) -> float:
    return 0.0 if low <= price <= high else min(abs(price - low), abs(price - high))


def collect_zones(analyses: dict[str, dict[str, Any]], order_blocks: list[dict[str, Any]], price: float,
                  limit: int = 12) -> list[dict[str, Any]]:
    """Active zones near price: FVGs, live order blocks, supply/demand, OTE."""
    zones: list[dict[str, Any]] = []
    for tf, analysis in analyses.items():
        for gap in analysis.get("fvg") or []:
            if gap.get("active"):
                zones.append({"kind": "fvg", "tf": tf, "low": gap["lower"], "high": gap["upper"],
                              "direction": gap["direction"].lower(), "state": "partial" if gap.get("partial_fill") else "fresh",
                              "time": _iso_to_ts(gap.get("time"))})
        for zone in analysis.get("supply_demand") or []:
            if not zone.get("invalidated"):
                zones.append({"kind": zone["type"].lower(), "tf": tf, "low": min(zone["proximal"], zone["distal"]),
                              "high": max(zone["proximal"], zone["distal"]),
                              "direction": "bullish" if zone["type"] == "DEMAND" else "bearish",
                              "state": "fresh" if zone.get("fresh") else "tested", "time": _iso_to_ts(zone.get("time"))})
    for block in order_blocks:
        if block["state"] != "BREAKER":
            zones.append({"kind": "order_block", "tf": block["tf"], "low": block["low"], "high": block["high"],
                          "direction": block["kind"], "state": block["state"].lower(), "time": block["time"]})
    htf_tf = ordered_tfs(analyses, HTF_ORDER)[0]
    fib = analyses[htf_tf].get("fibonacci")
    if fib and fib.get("ote"):
        zones.append({"kind": "ote", "tf": htf_tf, "low": fib["ote"]["low"], "high": fib["ote"]["high"],
                      "direction": fib["ote"]["direction"].lower(), "state": "fresh", "time": None})
    # The same block/gap is often found on two timeframes; keep the higher one.
    rank = {tf: i for i, tf in enumerate(HTF_ORDER)}
    zones.sort(key=lambda z: (rank.get(z["tf"], 99)))
    tolerance = abs(price) * 0.0002
    unique: list[dict[str, Any]] = []
    for zone in zones:
        if not any(z["kind"] == zone["kind"] and abs(z["low"] - zone["low"]) <= tolerance
                   and abs(z["high"] - zone["high"]) <= tolerance for z in unique):
            unique.append(zone)
    unique.sort(key=lambda z: _distance(price, z["low"], z["high"]))
    return unique[:limit]


def compact_structure(analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for tf, a in analyses.items():
        s = a["structure"]
        out[tf] = {"trend": TREND_WORD.get(s["trend"], "range"),
                   "bos": {"direction": s["bos"]["direction"], "level": s["bos"]["level"]} if s.get("bos") else None,
                   "choch": {"direction": s["choch"]["direction"], "level": s["choch"]["level"]} if s.get("choch") else None,
                   "last_swing_high": (s.get("last_swing_high") or {}).get("price"),
                   "last_swing_low": (s.get("last_swing_low") or {}).get("price")}
    return out


def compact_liquidity(analyses: dict[str, dict[str, Any]], price: float) -> dict[str, Any]:
    out = {}
    for tf, a in analyses.items():
        liq = a["liquidity"]
        out[tf] = {"buy_side": sorted((p for p in liq.get("buy_side_liquidity", []) if p > price))[:3],
                   "sell_side": sorted((p for p in liq.get("sell_side_liquidity", []) if p < price), reverse=True)[:3],
                   "sweeps": [{"type": s["type"], "price": s["price"]} for s in liq.get("sweeps", [])]}
    return out


def direction_word(value: str | None) -> str | None:
    return {"BULLISH": "long", "BEARISH": "short"}.get(str(value or ""))


def build_report(*, symbol: str, requested_symbol: str, broker_symbol: str, timeframes: list[str],
                 analyses: dict[str, dict[str, Any]], setup: dict[str, Any], checks: dict[str, Any],
                 errors: dict[str, str], theories: dict[str, Any] | None, data_source: str,
                 sources: dict[str, str], feed: str | None) -> dict[str, Any]:
    """The contract report (docs/CONTRACTS.md 3.5). Verdict: ENTRY_READY ->
    SETUP, NO_TRADE -> NO_TRADE, anything else -> WAIT (never 'buy now')."""
    current = execution_analysis(analyses)
    price = float(current["current_price"])
    metadata = current["metadata"]
    levels = merged_levels(analyses, price)
    order_blocks = collect_order_blocks(analyses, price)
    zones = collect_zones(analyses, order_blocks, price)
    htf_tf = ordered_tfs(analyses, HTF_ORDER)[0]
    decision = setup.get("decision")
    verdict = "SETUP" if decision == SetupDecision.ENTRY_READY.value else (
        "NO_TRADE" if decision == SetupDecision.NO_TRADE.value else "WAIT")
    targets = [{"price": t["price"], "rr": round(t["rr"], 2), "source": t.get("technical_source")}
               for t in setup.get("targets") or []]
    stale = any(a["metadata"].get("stale") for a in analyses.values())
    report: dict[str, Any] = {
        "symbol": symbol, "requested_symbol": requested_symbol, "broker_symbol": broker_symbol,
        "timeframes": list(timeframes), "price": price, "digits": metadata.get("precision"),
        "data_source": data_source, "sources": sources, "feed": feed,
        "at": datetime.now(UTC).timestamp(),
        "verdict": verdict, "decision": decision, "setup_state": setup.get("state"),
        "direction": direction_word(setup.get("direction")),
        "entry": setup.get("entry"), "stop": setup.get("stop"), "invalidation": setup.get("technical_invalidation"),
        "tp1": targets[0]["price"] if len(targets) > 0 else None,
        "tp2": targets[1]["price"] if len(targets) > 1 else None,
        "tp3": targets[2]["price"] if len(targets) > 2 else None,
        "targets": targets, "rr": round(setup["rr"], 2) if setup.get("rr") is not None else None,
        "entry_zone": setup.get("entry_zone"),
        "levels": levels[:16],
        "support": [l for l in key_levels(levels, analyses) if l["kind"] == "support"][:5],
        "resistance": [l for l in key_levels(levels, analyses) if l["kind"] == "resistance"][:5],
        "zones": zones, "order_blocks": order_blocks,
        "trend": {tf: TREND_WORD.get(analyses[tf]["structure"]["trend"], "range")
                  for tf in ordered_tfs(analyses, HTF_ORDER)},
        "htf_bias": TREND_WORD.get(analyses[htf_tf]["structure"]["trend"], "range"),
        "structure": compact_structure(analyses), "liquidity": compact_liquidity(analyses, price),
        "indicators": {tf: {k: _r(v) for k, v in a["indicators"].items()} for tf, a in analyses.items()},
        "session": session_context(),
        "confirmations": setup.get("confirmations"),
        "missing_confirmation": setup.get("missing_confirmation") or [],
        "reasons": setup.get("reasons") or [],
        "checks": checks.get("checks") or [], "self_check_passed": bool(checks.get("passed")),
        "stale": stale, "market_closed": any(a["metadata"].get("market_closed") for a in analyses.values()),
        "stale_timeframes": [tf for tf, a in analyses.items() if a["metadata"].get("stale")],
        "data_errors": errors, "confidence": confidence(setup, checks),
        "strategy": None, "theories": theories or None, "potential_plan": None, "drawn": [],
    }
    if verdict != "SETUP" and not stale and setup.get("direction") in ("BULLISH", "BEARISH"):
        plan = plan_from_levels(analyses, Direction(setup["direction"]), price, current)
        if plan.get("ok"):
            report["potential_plan"] = {"direction": direction_word(setup["direction"]), "entry": price,
                                        "stop": plan["stop"], "invalidation": plan["invalidation"],
                                        "targets": [{"price": t["price"], "rr": round(t["rr"], 2)} for t in plan["targets"]],
                                        "rr": round(plan["rr"], 2)}
    return report


__all__ = ["run_theory", "wyckoff", "gate_setup", "confidence", "build_report", "merged_levels",
           "collect_order_blocks", "collect_zones", "execution_analysis", "ordered_tfs", "direction_word",
           "TREND_WORD", "HTF_ORDER"]
