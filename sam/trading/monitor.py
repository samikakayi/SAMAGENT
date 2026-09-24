"""Background market watcher: price crosses, zone touches, candle closes,
volume spikes and strategy states -> spoken Sorani alerts.

No LLM on the hot path (reports/trading-intelligence.json): a persistent MT5
session answers a tick in 0.03 ms (measured), so every
``trading.monitor_interval_s`` (2 s) the watcher reads one tick (+ the last two
M1 bars, to catch wicks between polls) per symbol that has alerts. v1's setup
monitor only wrote transitions to the audit log (defect 3); here every fired
alert updates ``alerts``, publishes ``Alert`` (UI list + tray balloon) and
``SpeakRequest`` (voice) and logs an ``activity`` row.

Semantics (documented for the user in the tool description):
- price_cross, on "touch" (default): fires when price reaches the level from
  the armed side (up: from below, down: from above, any: either); on "close":
  when a closed candle of ``timeframe`` crosses it. Nothing fires for bars that
  closed before the alert existed.
- zone_touch: price enters [low, high] from outside.
- candle_close: a closed candle of ``timeframe`` beyond ``level``.
- volume_spike: a closed bar's volume >= k x the average of the previous n
  (tick volume on FX/metals: relative to the broker only).
- strategy_state: every predicate-checked rule of a card becomes true.
- once (default) -> status fired; repeat -> stays active, re-armed after price
  moves away by a hysteresis of max(spread, 0.01% of the level), min 60 s apart.

Repair review 2026-09-24 (monitor_harness.py): an M1 wick from BEFORE the alert
existed fired it on the second check (the whole current minute was used), and
a repeat alert fired again after a restart without a new cross (arming lived
in memory only). Now M1 wicks count only for bars that opened after the alert
was created or re-armed (the straddling bar: ticks only), and armed/direction/
fired_at/armed_at are kept in ``alerts.params`` like ``_last_bar``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from ..events import Alert, SpeakRequest
from .common import TIMEFRAME_SECONDS, canonical_symbol, normalize_timeframe
from .engine.sorani import fmt_price, spoken_price, symbol_ckb, tf_ckb
from .symbols import resolve_instrument, same_instrument

log = logging.getLogger("sam.trading.monitor")

KINDS = ("price_cross", "zone_touch", "candle_close", "volume_spike", "strategy_state")
DEFAULT_TF = {"price_cross": "M5", "candle_close": "M15", "volume_spike": "M5", "strategy_state": "M5"}
MIN_REPEAT_S = 60.0
STRATEGY_EVERY_S = 60.0


class AlertError(ValueError):
    """Invalid alert spec (message is shown to the model)."""


def _float(value: Any) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _direction(value: Any) -> str:
    text = str(value or "any").lower()
    if text in ("up", "above", "long", "bullish", "سەرەوە"):
        return "up"
    if text in ("down", "below", "short", "bearish", "خوارەوە"):
        return "down"
    return "any"


class Monitor:
    """``app.trading.monitor`` (docs/CONTRACTS.md 3.5)."""

    def __init__(self, app: Any, *, clock: Any = time.time) -> None:
        self.app = app
        self._clock = clock
        self._task: asyncio.Task[Any] | None = None
        self._state: dict[int, dict[str, Any]] = {}
        self._prices: dict[str, float] = {}
        self.last_error = ""

    # -- lifecycle ------------------------------------------------------------------------
    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = self.app.spawn(self._loop(), "trading-monitor")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the watcher must survive feed hiccups
                self.last_error = self.app.redact(f"{type(exc).__name__}: {exc}")[:200]
                log.warning("monitor cycle failed: %s", self.last_error)
            await asyncio.sleep(float(self.app.config.get("trading.monitor_interval_s", 2.0) or 2.0))

    # -- alerts CRUD ----------------------------------------------------------------------
    def add(self, spec: dict[str, Any], *, price: float | None = None) -> dict[str, Any]:
        """Validate and store an alert. ``price`` (current) arms touch alerts."""
        kind = str(spec.get("kind") or "").strip()
        if kind not in KINDS:
            raise AlertError(f"kind must be one of {', '.join(KINDS)}")
        wanted = str(spec.get("symbol") or self.app.config.get("trading.default_symbol", "XAUUSD"))
        symbol = resolve_instrument(wanted) or canonical_symbol(wanted)
        tf = normalize_timeframe(str(spec["timeframe"])) if spec.get("timeframe") else DEFAULT_TF.get(kind)
        params: dict[str, Any] = {"direction": _direction(spec.get("direction")),
                                  "on": "close" if kind == "candle_close" or spec.get("on") == "close" else "touch",
                                  "source": str(spec.get("source") or "mt5")}
        level, low, high = _float(spec.get("level")), _float(spec.get("low")), _float(spec.get("high"))
        if kind in ("price_cross", "candle_close"):
            if level is None:
                raise AlertError("a price level is required")
            params["level"] = level
            if params["direction"] == "any" and price is not None and kind == "candle_close":
                params["direction"] = "up" if price < level else "down"
        elif kind == "zone_touch":
            if low is None or high is None:
                raise AlertError("low and high of the zone are required")
            params["low"], params["high"] = min(low, high), max(low, high)
        elif kind == "volume_spike":
            params["k"] = _float(spec.get("k")) or 2.0
            params["n"] = int(_float(spec.get("n")) or 20)
        elif kind == "strategy_state":
            card = self.app.trading.strategies.get(spec.get("strategy_id")) if self.app.trading.strategies else None
            if card is None:
                raise AlertError("an existing strategy_id is required")
            if not any(rule.get("check") for rule in card.get("rules") or []):
                raise AlertError("this strategy has no rule SAM can check by numbers")
            spec = {**spec, "strategy_id": card["id"]}
            tf = tf or (card.get("timeframes") or {}).get("entry") or "M5"
        if price is not None:
            params["created_price"] = price
        hours = _float(spec.get("expires_in_hours")) or float(self.app.config.get("trading.alert_expiry_h", 72))
        now = self._clock()
        alert_id = self.app.db.insert("alerts", {
            "kind": kind, "symbol": symbol, "timeframe": tf, "params": params, "note": str(spec.get("note") or "")[:300],
            "strategy_id": spec.get("strategy_id"), "status": "active", "repeat": 1 if spec.get("repeat") else 0,
            "created_at": now, "expires_at": now + hours * 3600.0 if hours > 0 else None})
        return self.get(alert_id) or {}

    @staticmethod
    def _row(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        try:
            out["params"] = json.loads(row.get("params") or "{}")
        except json.JSONDecodeError:
            out["params"] = {}
        out["repeat"] = bool(row.get("repeat"))
        return out

    def get(self, alert_id: int) -> dict[str, Any] | None:
        row = self.app.db.query_one("SELECT * FROM alerts WHERE id=?", (int(alert_id),))
        return self._row(row) if row else None

    def list(self, status: str = "active") -> list[dict[str, Any]]:
        if status == "all":
            rows = self.app.db.query("SELECT * FROM alerts ORDER BY id DESC LIMIT 200")
        else:
            rows = self.app.db.query("SELECT * FROM alerts WHERE status=? ORDER BY id DESC LIMIT 200", (status,))
        return [self._row(r) for r in rows]

    def cancel(self, alert_id: int | str) -> int:
        if str(alert_id).strip().lower() in ("all", "هەموو", "هەمووی"):
            cursor = self.app.db.execute("UPDATE alerts SET status='cancelled' WHERE status='active'")
        else:
            cursor = self.app.db.execute("UPDATE alerts SET status='cancelled' WHERE id=? AND status='active'",
                                         (int(alert_id),))
        return int(cursor.rowcount or 0)

    # -- evaluation -----------------------------------------------------------------------
    async def check_once(self, now: float | None = None) -> list[dict[str, Any]]:
        """One pass over active alerts; returns the alerts that fired."""
        now = float(now if now is not None else self._clock())
        self.app.db.execute("UPDATE alerts SET status='expired' WHERE status='active' AND expires_at IS NOT NULL "
                            "AND expires_at < ?", (now,))
        alerts = self.list("active")
        if not alerts:
            return []
        fired: list[dict[str, Any]] = []
        groups = sorted({(a["symbol"], a["params"].get("source") or "mt5") for a in alerts})
        for symbol, source in groups:
            group = [a for a in alerts if a["symbol"] == symbol and (a["params"].get("source") or "mt5") == source]
            feed = await self._feed_for(symbol, source)
            if feed is None:
                self.last_error = "no market data feed is available"
                continue
            try:
                touch = any(a["kind"] in ("price_cross", "zone_touch") and a["params"].get("on") != "close" for a in group)
                snap = await self._snapshot(feed, symbol, need_bars=touch)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{symbol}: {exc}"[:200]
                continue
            for alert in group:
                try:
                    text = await self._evaluate(alert, snap, feed, now)
                except Exception as exc:  # noqa: BLE001 - one bad alert must not stop the rest
                    log.warning("alert %s failed: %s", alert["id"], exc)
                    continue
                if text:
                    fired.append(await self._fire(alert, text, snap["price"], now))
            self._prices[symbol] = snap["price"]
        return fired

    async def _feed_for(self, symbol: str, source: str) -> Any:
        """MT5 by default; the chart's own bars when the alert asked for
        TradingView and the chart shows this symbol (else MT5 as fallback)."""
        mt5 = self.app.trading.mt5
        tv = self.app.trading.tv
        if source == "tradingview" and tv is not None and getattr(tv, "connected", False):
            try:
                state = await asyncio.wait_for(tv.chart_state(), 4.0)
            except Exception:  # noqa: BLE001
                state = None
            if state and same_instrument(str(state.get("symbol") or state.get("canonical") or ""), symbol):
                return ChartFeed(tv, state, mt5)
        return mt5

    async def _snapshot(self, feed: Any, symbol: str, *, need_bars: bool) -> dict[str, Any]:
        tick = await feed.tick(symbol)
        bid, ask = tick.get("bid"), tick.get("ask")
        price = float(bid if bid else tick.get("last") or ask)
        snap = {"price": price, "spread": tick.get("spread") or 0.0, "digits": tick.get("digits"), "high": price,
                "low": price, "time": tick.get("time")}
        if need_bars:
            try:  # M1 wicks catch touches between polls; without them the tick path still works
                snap["bars"] = await feed.bars(symbol, "M1", 2)
            except Exception as exc:  # noqa: BLE001
                log.info("M1 bars for %s unavailable: %s", symbol, exc)
        return snap

    def _hysteresis(self, level: float, spread: float) -> float:
        return max(float(spread or 0.0), abs(level) * 0.0001)

    def _path(self, alert: dict[str, Any], snap: dict[str, Any], state: dict[str, Any]) -> tuple[float, float]:
        """Lowest/highest price since the last check (tick + M1 wicks). A bar's
        wick counts only if the bar OPENED after the alert was created or last
        re-armed: the minute that straddles that moment may hold prices from
        before (repair review: a 12:00:20 wick of 2700.2 fired an alert created
        at 12:00:40 with price 2698.5 that never reached 2700 afterwards)."""
        prev = state.get("prev_price", alert["params"].get("created_price", snap["price"]))
        low, high = min(prev, snap["price"]), max(prev, snap["price"])
        since = float(state.get("armed_at") or alert.get("created_at") or 0.0)
        if "checked_at" in state:
            for bar in snap.get("bars") or []:
                if bar["time"] + 60 >= state["checked_at"] and bar["time"] >= since:
                    low, high = min(low, bar["low"]), max(high, bar["high"])
        return low, high

    _PERSISTED = ("armed", "direction", "fired_at", "armed_at")

    def _state_of(self, alert: dict[str, Any]) -> dict[str, Any]:
        """In-memory state, restored from ``alerts.params`` after a restart."""
        state = self._state.get(alert["id"])
        if state is None:
            params = alert["params"]
            state = {key: params[f"_{key}"] for key in self._PERSISTED if f"_{key}" in params}
            self._state[alert["id"]] = state
        return state

    def _save_state(self, alert: dict[str, Any], state: dict[str, Any]) -> None:
        params = alert["params"]
        changed = {f"_{k}": state[k] for k in self._PERSISTED if k in state and params.get(f"_{k}") != state[k]}
        if changed:
            params.update(changed)
            self.app.db.execute("UPDATE alerts SET params=? WHERE id=?",
                                (json.dumps(params, ensure_ascii=False), alert["id"]))

    async def _evaluate(self, alert: dict[str, Any], snap: dict[str, Any], feed: Any, now: float) -> str | None:
        kind, params = alert["kind"], alert["params"]
        state = self._state_of(alert)
        price = snap["price"]
        text: str | None = None
        if alert["repeat"] and now - state.get("fired_at", -MIN_REPEAT_S) < MIN_REPEAT_S:
            # Cooldown after a repeat alert fired: do not evaluate (and so do not
            # consume/disarm a crossing that would then never be spoken).
            state["prev_price"], state["checked_at"] = price, now
            return None
        if kind == "price_cross" and params.get("on") != "close":
            text = self._touch_level(alert, snap, state)
        elif kind == "zone_touch":
            text = self._touch_zone(alert, snap, state)
        elif kind in ("price_cross", "candle_close", "volume_spike"):
            text = await self._on_close(alert, feed, state, snap)
        elif kind == "strategy_state":
            text = await self._strategy(alert, state, now)
        state["prev_price"], state["checked_at"] = price, now
        if text:
            state["fired_at"] = now
        if kind in ("price_cross", "zone_touch"):
            self._save_state(alert, state)
        return text

    def _touch_level(self, alert: dict[str, Any], snap: dict[str, Any], state: dict[str, Any]) -> str | None:
        params, price = alert["params"], snap["price"]
        level, wanted = float(params["level"]), params.get("direction", "any")
        h = self._hysteresis(level, snap["spread"])
        if "armed" not in state:  # first sight: armed unless already on the far side
            start = params.get("created_price", price)
            state["direction"] = wanted if wanted != "any" else ("up" if start < level else "down")
            state["armed"] = (start < level) if state["direction"] == "up" else (start > level)
        direction = state["direction"]
        low, high = self._path(alert, snap, state)
        if not state["armed"]:
            state["armed"] = price < level - h if direction == "up" else price > level + h
            if state["armed"]:
                state["armed_at"] = self._clock()
            return None
        if (direction == "up" and high >= level) or (direction == "down" and low <= level):
            state["armed"] = False
            if wanted == "any":  # a repeating "reaches" alert watches the next touch from the other side
                state["direction"] = "down" if direction == "up" else "up"
            sym, digits = symbol_ckb(alert["symbol"]), snap.get("digits")
            where = "گەیشتە سەرووی" if direction == "up" else "هاتە خوارووی"
            # A level with decimals (4265.36) is followed by the price at the same
            # precision: "reached 4265.36; the price is now 4265" read as a contradiction
            # in the live acceptance run (2026-09-24).
            now_text = fmt_price(price, digits) if float(level) != round(float(level)) else spoken_price(price)
            return self._with_note(alert, f"ئاگاداری: {sym} {where} {fmt_price(level, digits)}؛ نرخی ئێستا "
                                          f"{now_text}.")
        return None

    def _touch_zone(self, alert: dict[str, Any], snap: dict[str, Any], state: dict[str, Any]) -> str | None:
        params, price = alert["params"], snap["price"]
        low_z, high_z = float(params["low"]), float(params["high"])
        h = self._hysteresis(high_z, snap["spread"])
        if "armed" not in state:
            start = params.get("created_price", price)
            state["armed"] = not (low_z <= start <= high_z)
        low, high = self._path(alert, snap, state)
        if not state["armed"]:
            state["armed"] = price < low_z - h or price > high_z + h
            if state["armed"]:
                state["armed_at"] = self._clock()
            return None
        if low <= high_z and high >= low_z:
            state["armed"] = False
            sym, digits = symbol_ckb(alert["symbol"]), snap.get("digits")
            return self._with_note(alert, f"ئاگاداری: {sym} گەیشتە ناوچەی {fmt_price(low_z, digits)} تا "
                                          f"{fmt_price(high_z, digits)}؛ نرخی ئێستا {spoken_price(price)}.")
        return None

    async def _on_close(self, alert: dict[str, Any], feed: Any, state: dict[str, Any], snap: dict[str, Any]) -> str | None:
        params, tf = alert["params"], alert["timeframe"] or "M5"
        interval = TIMEFRAME_SECONDS.get(tf, 300)
        n = int(params.get("n") or 20) if alert["kind"] == "volume_spike" else 1
        bars = await feed.bars(alert["symbol"], tf, n + 3)
        if len(bars) < n + 2:
            return None
        closed, prior = bars[-2], bars[-3]  # bars[-1] is still forming
        last_seen = state.get("last_bar", params.get("_last_bar", 0))
        # Only bars that closed after the alert existed, each evaluated once.
        if closed["time"] <= last_seen or closed["time"] + interval <= alert["created_at"]:
            return None
        state["last_bar"] = closed["time"]
        self._persist_param(alert, "_last_bar", closed["time"])
        sym, digits = symbol_ckb(alert["symbol"]), snap.get("digits")
        if alert["kind"] == "volume_spike":
            base = [b["volume"] for b in bars[-2 - n:-2]]
            average = sum(base) / len(base) if base else 0.0
            k = float(params.get("k") or 2.0)
            if average > 0 and closed["volume"] >= k * average:
                return self._with_note(alert, f"ئاگاداری: ڤۆلیۆمی {sym} لە مۆمی {tf_ckb(tf)} "
                                              f"{closed['volume'] / average:.1f} هێندەی تێکڕا بەرز بووەوە (تیک ڤۆلیۆم).")
            return None
        level, direction = float(params["level"]), params.get("direction", "any")
        up = prior["close"] < level <= closed["close"]
        down = prior["close"] > level >= closed["close"]
        if (up and direction in ("up", "any")) or (down and direction in ("down", "any")):
            where = "سەرووی" if up else "خوارووی"
            return self._with_note(alert, f"ئاگاداری: مۆمێکی {tf_ckb(tf)}ی {sym} لە {where} {fmt_price(level, digits)} "
                                          f"داخرا؛ نرخی داخستن {fmt_price(closed['close'], digits)}.")
        return None

    async def _strategy(self, alert: dict[str, Any], state: dict[str, Any], now: float) -> str | None:
        if now - state.get("strategy_at", 0.0) < STRATEGY_EVERY_S:
            return None
        state["strategy_at"] = now
        card = self.app.trading.strategies.get(alert["strategy_id"])
        engine = self.app.trading.engine
        if card is None or engine is None:
            return None
        report = await engine.analyze(alert["symbol"], None, strategy=card, now=now)
        if report.get("stale"):  # closed market / frozen feed: no state change on old data
            return None
        rules = [r for r in (report.get("strategy") or {}).get("rules") or [] if r.get("how") == "predicate"]
        ready = bool(rules) and all(r.get("passed") is True for r in rules)
        was_ready = state.get("ready", bool(alert["params"].get("_ready")))
        state["ready"] = ready
        if ready == was_ready:
            return None
        self._persist_param(alert, "_ready", ready)
        if not ready:
            return None
        sym, digits = symbol_ckb(alert["symbol"]), report.get("digits")
        text = f"ئاگاداری: مەرجە ژمارەییەکانی ستراتیژی «{card.get('title_ckb')}» لە {sym} هەموویان جێبەجێ بوون."
        plan = report.get("strategy_plan") or {}
        if plan.get("ok"):
            text += (f" چوونەژوورەوە نزیکەی {fmt_price(plan['entry'], digits)}، ستۆپ {fmt_price(plan['stop'], digits)}، "
                     f"ئامانجی یەکەم {fmt_price(plan['targets'][0]['price'], digits)}.")
        pending = [r for r in (report.get("strategy") or {}).get("rules") or [] if r.get("how") == "llm"]
        if pending:
            text += " مەرجەکانی تر بە سەیرکردنی چارت بپشکنە."
        return self._with_note(alert, text + " بڕیار هی خۆتە.")

    @staticmethod
    def _with_note(alert: dict[str, Any], text: str) -> str:
        note = (alert.get("note") or "").strip()
        return f"{text} تێبینی: {note}" if note else text

    def _persist_param(self, alert: dict[str, Any], key: str, value: Any) -> None:
        alert["params"][key] = value
        self.app.db.execute("UPDATE alerts SET params=? WHERE id=?",
                            (json.dumps(alert["params"], ensure_ascii=False), alert["id"]))

    async def _fire(self, alert: dict[str, Any], text_ckb: str, price: float, now: float) -> dict[str, Any]:
        status = "active" if alert["repeat"] else "fired"
        self.app.db.execute("UPDATE alerts SET status=?, fired_at=?, fire_count=fire_count+1, last_value=?, "
                            "last_text_ckb=? WHERE id=?", (status, now, price, text_ckb, alert["id"]))
        self.app.bus.publish(Alert(alert_id=int(alert["id"]), kind=alert["kind"], symbol=alert["symbol"],
                                   text_ckb=text_ckb, price=price, timeframe=alert.get("timeframe") or ""))
        self.app.bus.publish(SpeakRequest(text_ckb=text_ckb, source="alert"))
        self.app.db.log_activity("alert", alert["kind"], ok=True, summary=text_ckb[:300],
                                 detail={"alert_id": alert["id"], "symbol": alert["symbol"], "price": price},
                                 source="monitor")
        return {**alert, "status": status, "text_ckb": text_ckb, "price": price}


class ChartFeed:
    """Feed interface over the TradingView chart the user sees: its last bar is
    the price; bars come from the chart when the timeframe matches the chart,
    otherwise from MT5 (the chart only holds its own timeframe)."""

    def __init__(self, tv: Any, state: dict[str, Any], fallback: Any) -> None:
        self.tv = tv
        self.state = state
        self.fallback = fallback

    async def tick(self, symbol: str) -> dict[str, Any]:
        last = self.state.get("last_bar") or {}
        return {"symbol": symbol, "bid": last.get("close"), "ask": None, "last": last.get("close"), "spread": 0.0,
                "time": last.get("time"), "digits": None}

    async def bars(self, symbol: str, timeframe: str, count: int = 500) -> list[dict[str, Any]]:
        if normalize_timeframe(str(self.state.get("timeframe") or "")) == normalize_timeframe(timeframe):
            return list(await asyncio.wait_for(self.tv.bars(count), 6.0))
        if self.fallback is None:
            raise RuntimeError(f"the chart is not on {timeframe} and MetaTrader 5 is not available")
        return await self.fallback.bars(symbol, timeframe, count)


__all__ = ["Monitor", "AlertError", "KINDS", "ChartFeed"]
