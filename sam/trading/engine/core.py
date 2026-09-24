"""``Engine``: bars in, report out (docs/CONTRACTS.md 3.5).

Measured on live XAUUSD with v1's engine (2026-09-24): 5 timeframes x 600
bars in 1.24 s total, of which fetch 120-500 ms and compute 35-48 ms per
timeframe. The fetch now uses one persistent MT5 session; the CPU part runs in
a worker thread (``asyncio.to_thread``) so the voice loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterable

from ..common import canonical_symbol
from ..predicates import PredicateContext, evaluate_card
from ..theories import THEORIES, find_theory
from .analysis import entry_hunter, plan_from_levels, self_check, timeframe_analysis
from .analyst import build_report, direction_word, execution_analysis, gate_setup, ordered_tfs, run_theory, HTF_ORDER
from .sorani import full_text, spoken_summary
from .types import TIMEFRAME_SECONDS, Candle, Direction, MarketDataBatch, build_batch, normalize_timeframe

log = logging.getLogger("sam.trading.engine")

DEFAULT_TIMEFRAMES = ["H1", "M15", "M5", "M1"]
DEFAULT_BARS = 600


def canonical_timeframes(timeframes: Iterable[str] | None, default: Iterable[str] = DEFAULT_TIMEFRAMES) -> list[str]:
    out: list[str] = []
    for value in list(timeframes or []) or list(default):
        tf = normalize_timeframe(str(value))
        if tf in TIMEFRAME_SECONDS and tf not in out:
            out.append(tf)
    if not out or len(out) > 8:
        raise ValueError("Choose between 1 and 8 distinct timeframes")
    return out


def closed_candles(batch: MarketDataBatch) -> list[Candle]:
    """The candles timeframe_analysis uses: the forming bar is dropped."""
    candles = batch.candles
    interval = TIMEFRAME_SECONDS.get(batch.timeframe)
    if interval and len(candles) > 20 and batch.fetched_at.timestamp() < candles[-1].time.timestamp() + interval:
        return candles[:-1]
    return candles


def resample_bars(bars: list[dict[str, Any]], to_tf: str) -> list[dict[str, Any]]:
    """Aggregate UTC bars into a higher timeframe (fallback when only the chart's
    own bars are available). Buckets align to UTC multiples of the interval."""
    interval = TIMEFRAME_SECONDS[normalize_timeframe(to_tf)]
    out: list[dict[str, Any]] = []
    for bar in sorted(bars, key=lambda b: b["time"]):
        start = int(bar["time"]) // interval * interval
        if out and out[-1]["time"] == start:
            last = out[-1]
            last["high"] = max(last["high"], bar["high"])
            last["low"] = min(last["low"], bar["low"])
            last["close"] = bar["close"]
            last["volume"] += float(bar.get("volume") or 0.0)
        else:
            out.append({"time": start, "open": bar["open"], "high": bar["high"], "low": bar["low"],
                        "close": bar["close"], "volume": float(bar.get("volume") or 0.0)})
    return out


def card_timeframes(card: dict[str, Any] | None) -> list[str]:
    frames = (card or {}).get("timeframes") or {}
    out: list[str] = []
    for role in ("bias", "setup", "entry"):
        tf = normalize_timeframe(str(frames.get(role) or "")) if frames.get(role) else None
        if tf in TIMEFRAME_SECONDS and tf not in out:
            out.append(tf)
    return out


def uses_predicate(card: dict[str, Any] | None, name: str) -> bool:
    return any(((rule.get("check") or {}).get("predicate") == name) for rule in (card or {}).get("rules") or [])


def strategy_context(card: dict[str, Any], analyses: dict[str, dict[str, Any]], candles: dict[str, list[Candle]],
                     price: float, setup: dict[str, Any], *, now: float, spread: float | None,
                     extra: dict[str, dict[str, Any]], settings: dict[str, Any]) -> PredicateContext:
    """Roles (bias/setup/entry timeframes) and the direction a card is judged in:
    the card's own direction, else the trend of its bias timeframe, else the
    engine's higher-timeframe direction."""
    order = ordered_tfs(analyses, HTF_ORDER)
    frames = card.get("timeframes") or {}
    roles = {}
    for role, fallback in (("bias", order[0]), ("setup", order[min(1, len(order) - 1)]),
                           ("entry", execution_analysis(analyses)["metadata"]["timeframe"])):
        tf = normalize_timeframe(str(frames.get(role))) if frames.get(role) else None
        roles[role] = tf if tf in analyses else fallback
    direction = str(card.get("direction") or "").lower()
    if direction not in ("long", "short"):
        bias_trend = analyses[roles["bias"]]["structure"]["trend"]
        direction = direction_word(bias_trend) or direction_word(setup.get("direction")) or ""
    return PredicateContext(analyses=analyses, candles=candles, price=price, direction=direction or None, now=now,
                            spread=spread, roles=roles, extra=extra, settings=settings)


def strategy_plan(card: dict[str, Any], ctx: PredicateContext) -> dict[str, Any] | None:
    """Stop/targets for the card's direction; ``risk.target_rr`` adds a fixed
    R-multiple first target (for cards that say 'TP at 1:2')."""
    if ctx.direction not in ("long", "short"):
        return None
    direction = Direction.BULLISH if ctx.direction == "long" else Direction.BEARISH
    ltf = ctx.analyses[ctx.roles["entry"]]
    plan = plan_from_levels(ctx.analyses, direction, ctx.price, ltf)
    target_rr = ((card.get("risk") or {}).get("target_rr"))
    fixed_rr = float(target_rr) if isinstance(target_rr, (int, float)) and target_rr > 0 else None
    # A card with its own R:R target does not need a level target, only a stop.
    if not plan.get("ok") and not (fixed_rr and plan.get("stop") is not None):
        return {"ok": False, "missing": plan.get("missing"), "reason": plan.get("reason")}
    targets = list(plan.get("targets") or [])
    if fixed_rr:
        # The card's own target comes first; only levels beyond it stay as TP2/TP3.
        sign = 1 if direction == Direction.BULLISH else -1
        fixed = ctx.price + sign * fixed_rr * plan["stop_distance"]
        targets = [{"price": fixed, "rr": fixed_rr, "technical_source": "FIXED_RR"},
                   *[t for t in targets if t["rr"] > fixed_rr]][:3]
    return {"ok": True, "direction": ctx.direction, "entry": ctx.price, "stop": plan["stop"],
            "invalidation": plan["invalidation"], "targets": targets, "rr": targets[0]["rr"]}


def apply_strategy(report: dict[str, Any], card_eval: dict[str, Any], plan: dict[str, Any] | None,
                   min_rr: float) -> dict[str, Any]:
    """The card decides the verdict (idempotent; re-run after vision verdicts).

    Failed filter/risk rule -> NO_TRADE. Every rule passed (predicate or
    vision) and a plan with rr >= min_rr -> SETUP. Anything unknown -> WAIT:
    an unclear rule never counts as a pass.
    """
    report["strategy"] = card_eval
    report.setdefault("engine_verdict", report.get("verdict"))
    report["strategy_plan"] = plan
    rules = card_eval.get("rules") or []
    card_eval["all_passed"] = bool(rules) and all(r.get("passed") is True for r in rules)
    card_eval["failed"] = [r["id"] for r in rules if r.get("passed") is False]
    card_eval["pending"] = [r["id"] for r in rules if r.get("passed") is None]
    clear = {"entry": None, "stop": None, "invalidation": None, "tp1": None, "tp2": None, "tp3": None,
             "targets": [], "rr": None}
    if report.get("stale"):
        report.update(clear, verdict="WAIT")
        return report
    if any(r.get("passed") is False and r.get("kind") in ("filter", "risk") for r in rules):
        report.update(clear, verdict="NO_TRADE")
    elif card_eval["all_passed"] and plan and plan.get("ok") and plan["rr"] >= min_rr:
        targets = [{"price": t["price"], "rr": round(t["rr"], 2), "source": t.get("technical_source")}
                   for t in plan["targets"]]
        report.update(verdict="SETUP", direction=plan["direction"], entry=plan["entry"], stop=plan["stop"],
                      invalidation=plan["invalidation"], targets=targets, rr=round(plan["rr"], 2),
                      tp1=targets[0]["price"], tp2=targets[1]["price"] if len(targets) > 1 else None,
                      tp3=targets[2]["price"] if len(targets) > 2 else None)
    else:
        report.update(clear, verdict="WAIT", direction=card_eval.get("direction") or report.get("direction"))
        if plan and plan.get("ok"):
            report["potential_plan"] = {k: plan[k] for k in ("direction", "entry", "stop", "invalidation", "rr")} | {
                "targets": [{"price": t["price"], "rr": round(t["rr"], 2)} for t in plan["targets"]]}
    return report


def finish_texts(report: dict[str, Any]) -> dict[str, Any]:
    report["summary_ckb"] = spoken_summary(report)
    report["text_ckb"] = full_text(report)
    return report


class Engine:
    """``app.trading.engine``. Keeps the latest report (``latest``) for the
    chart and for 'why?' follow-ups."""

    def __init__(self, app: Any = None) -> None:
        self.app = app
        self.latest: dict[str, Any] | None = None
        self.last_draw_tag: str | None = None  # SAM's previous analysis drawing (cleared before the next)

    def _setting(self, key: str, default: Any) -> Any:
        try:
            return self.app.config.get(key, default) if self.app is not None else default
        except Exception:  # noqa: BLE001
            return default

    @property
    def feed(self) -> Any:
        return getattr(getattr(self.app, "trading", None), "mt5", None)

    async def fetch_mt5(self, symbol: str, timeframes: list[str], count: int = DEFAULT_BARS
                        ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any] | None, dict[str, Any], dict[str, str]]:
        """(bars_by_tf, tick, meta, errors) from the MT5 feed."""
        feed = self.feed
        if feed is None:
            raise RuntimeError("MetaTrader 5 feed is not available")
        started = time.perf_counter()
        bars: dict[str, list[dict[str, Any]]] = {}
        errors: dict[str, str] = {}
        for tf in timeframes:
            try:
                bars[tf] = await feed.bars(symbol, tf, count)
            except Exception as exc:  # noqa: BLE001 - one missing timeframe is reported, not fatal
                errors[tf] = str(exc)[:200]
        tick = meta = None
        try:
            tick = await feed.tick(symbol)
            meta = {"name": tick.get("symbol"), "digits": tick.get("digits"), "point": tick.get("point")}
        except Exception as exc:  # noqa: BLE001
            errors["tick"] = str(exc)[:200]
        if self.app is not None:
            self.app.timing.record("mt5_fetch", (time.perf_counter() - started) * 1000.0, kind="analysis",
                                   symbol=symbol, timeframes=",".join(timeframes))
        return bars, tick, meta or {}, errors

    async def analyze(self, symbol: str, timeframes: list[str] | None = None, *,
                      bars_by_tf: dict[str, list[dict[str, Any]]] | None = None, strategy: dict[str, Any] | None = None,
                      theories: list[str] | None = None, data_source: str | None = None,
                      sources: dict[str, str] | None = None, tick: dict[str, Any] | None = None,
                      meta: dict[str, Any] | None = None, errors: dict[str, str] | None = None,
                      now: float | None = None) -> dict[str, Any]:
        """Analyse ``symbol`` on ``timeframes``. Without ``bars_by_tf`` the bars
        come from MT5. Raises RuntimeError when no timeframe has data."""
        started = time.perf_counter()
        canonical = canonical_symbol(symbol) or symbol
        tfs = canonical_timeframes(timeframes or card_timeframes(strategy) or None)
        errors = dict(errors or {})
        if bars_by_tf is None:
            bars_by_tf, tick, meta, fetch_errors = await self.fetch_mt5(canonical, tfs, DEFAULT_BARS)
            errors.update(fetch_errors)
            data_source = data_source or "mt5"
            sources = {tf: "mt5" for tf in bars_by_tf}
        extra: dict[str, dict[str, list[dict[str, Any]]]] = {}
        if strategy and uses_predicate(strategy, "usdx_trend") and self.feed is not None and canonical != "USDX":
            for tf in card_timeframes(strategy)[:1] or tfs[:1]:
                try:
                    extra.setdefault("USDX", {})[tf] = await self.feed.bars("USDX", tf, 300)
                except Exception as exc:  # noqa: BLE001
                    errors[f"USDX {tf}"] = str(exc)[:200]
        settings = {"min_rr": float(self._setting("trading.min_rr", 1.5)),
                    "news_blackouts": self._setting("trading.news_blackouts", [])}
        report = await asyncio.to_thread(
            self.analyze_sync, canonical, tfs, bars_by_tf, strategy=strategy, theories=theories,
            data_source=data_source or "provided", sources=sources or {tf: data_source or "provided" for tf in bars_by_tf},
            tick=tick, meta=meta or {}, errors=errors, extra_bars=extra, now=now, settings=settings,
            requested_symbol=symbol)
        report["engine_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        if self.app is not None:
            self.app.timing.record("analysis_engine", report["engine_ms"], kind="analysis", symbol=canonical,
                                   timeframes=",".join(tfs), source=report.get("data_source"))
        self.latest = report
        return report

    def analyze_sync(self, symbol: str, timeframes: list[str], bars_by_tf: dict[str, list[dict[str, Any]]], *,
                     strategy: dict[str, Any] | None = None, theories: list[str] | None = None,
                     data_source: str = "provided", sources: dict[str, str] | None = None,
                     tick: dict[str, Any] | None = None, meta: dict[str, Any] | None = None,
                     errors: dict[str, str] | None = None, extra_bars: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
                     now: float | None = None, settings: dict[str, Any] | None = None,
                     requested_symbol: str | None = None) -> dict[str, Any]:
        """CPU part (thread-safe, no I/O)."""
        settings = settings or {}
        meta = meta or {}
        errors = dict(errors or {})
        now_s = float(now if now is not None else time.time())
        broker = meta.get("name") or symbol
        digits = meta.get("digits")
        point = float(meta.get("point") or (10 ** -digits if isinstance(digits, int) else 0.01))
        analyses: dict[str, dict[str, Any]] = {}
        candles: dict[str, list[Candle]] = {}
        for tf in timeframes:
            bars = (bars_by_tf or {}).get(tf)
            if not bars:
                errors.setdefault(tf, "no bars")
                continue
            source = (sources or {}).get(tf, data_source)
            try:
                batch = build_batch(provider=source, requested_symbol=requested_symbol or symbol, resolved_symbol=broker,
                                    timeframe=tf, bars=bars, feed=f"{source}:{broker}",
                                    tick=tick if source == "mt5" else None, point=point,
                                    precision=digits if isinstance(digits, int) else 2, now=now_s)
                analyses[tf] = timeframe_analysis(batch)
                candles[tf] = closed_candles(batch)
            except Exception as exc:  # noqa: BLE001
                errors[tf] = f"{type(exc).__name__}: {exc}"[:200]
        if not analyses:
            raise RuntimeError("No requested timeframe returned usable bars: " +
                               "; ".join(f"{k}: {v}" for k, v in errors.items()))
        min_rr = float(((strategy or {}).get("risk") or {}).get("min_rr") or settings.get("min_rr", 1.5))
        setup = entry_hunter(analyses, minimum_rr=min_rr)
        checks = self_check(analyses, setup, symbol, [tf for tf in timeframes])
        setup = gate_setup(setup, checks)
        theory_out = None
        if theories:
            theory_out = {}
            for name in theories:
                theory = find_theory(name) or THEORIES.get(name)
                theory_out[theory.id if theory else name] = (
                    run_theory(theory, analyses, candles) if theory
                    else {"status": "UNAVAILABLE", "error": f"Unknown trading theory: {name}"})
        report = build_report(symbol=symbol, requested_symbol=requested_symbol or symbol, broker_symbol=broker,
                              timeframes=list(analyses), analyses=analyses, setup=setup, checks=checks, errors=errors,
                              theories=theory_out, data_source=data_source, sources=dict(sources or {}),
                              feed=f"{data_source}:{broker}")
        report["at"] = now_s
        if tick and tick.get("spread") is not None:
            report["spread"] = tick["spread"]
        if strategy:
            extra = {}
            for extra_symbol, per_tf in (extra_bars or {}).items():
                for tf, bars in per_tf.items():
                    try:
                        batch = build_batch(provider="mt5", requested_symbol=extra_symbol, resolved_symbol=extra_symbol,
                                            timeframe=tf, bars=bars, now=now_s)
                        extra.setdefault(extra_symbol, {})[tf] = timeframe_analysis(batch)
                    except Exception:  # noqa: BLE001
                        log.info("extra symbol %s %s unusable", extra_symbol, tf)
            ctx = strategy_context(strategy, analyses, candles, report["price"], setup, now=now_s,
                                   spread=report.get("spread"), extra=extra, settings=settings)
            plan = strategy_plan(strategy, ctx)
            ctx.plan = plan if plan and plan.get("ok") else None
            apply_strategy(report, evaluate_card(strategy, ctx), plan, min_rr)
        return finish_texts(report)


__all__ = ["Engine", "DEFAULT_TIMEFRAMES", "canonical_timeframes", "resample_bars", "apply_strategy",
           "finish_texts", "card_timeframes", "closed_candles", "strategy_context", "strategy_plan"]
