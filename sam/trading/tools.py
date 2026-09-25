"""Trading engine entry module (``sam.app.PACKAGES``): sets
``app.trading.mt5 / engine / theories / strategies / monitor`` and registers
the engine's tools (docs/CONTRACTS.md section 2): get_price, analyze_market,
set_alert, list_alerts, cancel_alert, strategy_save, strategy_list,
strategy_get, theory_info.

Every tool is analysis/alerts only. SAM 2 never places, modifies or closes
orders; the MT5 feed is wrapped read-only (``mt5.ReadOnlyMT5``).
"""

from __future__ import annotations

import logging
from typing import Any

from ..brain.confirm import classify_answer
from ..brain.llm import LLMError
from ..brain.tools import ToolContext, fail, ok, tool
from ..textnorm import normalize_ckb
from .analyze import TRADING_MIGRATIONS, analyze_market, compact_report
from .common import canonical_symbol
from .symbols import resolve_instrument, same_instrument
from .engine.core import DEFAULT_TIMEFRAMES, Engine
from .engine.sorani import fmt_price, spoken_price, symbol_ckb, tf_ckb
from .monitor import KINDS, AlertError, Monitor
from .mt5 import MT5Feed
from .strategies import StrategyStore
from .theories import THEORIES, find_theory, list_theories

log = logging.getLogger("sam.trading.tools")

SETTINGS = {
    "trading.analysis_timeframes": list(DEFAULT_TIMEFRAMES),  # analyze_market default (entry hunter needs M5/M1)
    "trading.alert_expiry_h": 72,
    "trading.mt5_offset_s": None,      # last VERIFIED broker offset (used while the market is closed)
    "trading.news_blackouts": [],      # [{"start": unix, "end": unix, "note": "..."}] for no_news_blackout
}


def _trading(ctx: ToolContext) -> Any:
    return ctx.app.trading


# --- prices ------------------------------------------------------------------------------

@tool("get_price",
      description="Current price of a market from MetaTrader 5 (bid/ask/spread), plus the TradingView chart price when "
                  "the chart shows it. Symbol accepts any name: XAUUSD, gold, زێڕ, OANDA:XAUUSD. Gold (زێڕ/گۆڵد) "
                  "is XAUUSD.",
      description_ckb="نرخی ئێستای بازاڕ",
      params={"type": "object", "properties": {"symbol": {
          "type": "string", "description": "as the user said it (Sorani is fine), e.g. زێڕ; empty = the user's main symbol"}}},
      examples_ckb=("نرخی زێڕ چەندە؟", "ئێستا گۆڵد لە چەندە؟"), risk="safe", blocking=True, timeout_s=15)
async def get_price(ctx: ToolContext, symbol: str = "") -> dict[str, Any]:
    trading = _trading(ctx)
    wanted = symbol or ctx.app.config.get("trading.default_symbol", "XAUUSD")
    canonical = resolve_instrument(wanted) or canonical_symbol(wanted)
    data: dict[str, Any] = {"symbol": canonical}
    if trading.mt5 is not None:
        try:
            tick = await trading.mt5.tick(canonical)
            data.update(broker_symbol=tick["symbol"], bid=tick["bid"], ask=tick["ask"], spread=tick["spread"],
                        time=tick["time"], digits=tick.get("digits"))
        except Exception as exc:  # noqa: BLE001
            data["mt5_error"] = str(exc)[:160]
    if trading.tv is not None and getattr(trading.tv, "connected", False):
        try:
            state = await trading.tv.chart_state()
            if same_instrument(str(state.get("symbol") or state.get("canonical") or ""), canonical):
                data["chart_price"] = (state.get("last_bar") or {}).get("close")
        except Exception:  # noqa: BLE001
            pass
    price = data.get("bid") or data.get("chart_price")
    if price is None:
        return fail(f"نەمتوانی نرخی {symbol_ckb(canonical)} بدۆزمەوە.",
                    reason=f"No price for {canonical}: {data.get('mt5_error') or 'no feed connected'}", **data)
    # Spoken: rounded the way a trader says it; the exact bid/ask stay in data.
    return ok(f"{symbol_ckb(canonical)} ئێستا لەسەر {spoken_price(price)} مامەڵە دەکرێت.", **data)


# --- analysis ----------------------------------------------------------------------------

@tool("analyze_market",
      description="Analyse a market like a professional trader: multi-timeframe structure, support/resistance, "
                  "liquidity, order blocks, FVGs, entry/stop/targets and a verdict WAIT, NO_TRADE or SETUP (analysis "
                  "only, never an order). With strategy_id it checks the user's saved strategy rule by rule. Draws the "
                  "result on the TradingView chart when it shows this symbol (draw: none, levels, full). The result has "
                  "a short Sorani summary to say.",
      description_ckb="شیکردنەوەی بازاڕ و کێشانی لەسەر چارت",
      params={"type": "object", "properties": {
          "symbol": {"type": "string", "description": "as the user said it (Sorani is fine); empty = main symbol"},
          "timeframes": {"type": "array", "items": {"type": "string"}, "description": "default H1,M15,M5,M1"},
          "strategy_id": {"type": "string", "description": "id of a saved strategy card"},
          "draw": {"type": "string", "enum": ["none", "levels", "full"], "description": "default full"},
          "vision": {"type": "boolean", "description": "look at the chart for rules numbers cannot check (default true)"},
          "theory": {"type": "string", "description": "optional catalogue theory to run too, e.g. elliott, wyckoff"}}},
      examples_ckb=("زێڕ شی بکەرەوە بە ستراتیژییەکەم", "بازاڕ چۆنە؟", "شیکاری گۆڵد بکە لەسەر پازدە خولەک"),
      risk="safe", blocking=False, timeout_s=90)
async def analyze_market_tool(ctx: ToolContext, symbol: str = "", timeframes: list[str] | None = None,
                              strategy_id: str = "", draw: str = "full", vision: bool = True, theory: str = "") -> dict[str, Any]:
    try:
        report = await analyze_market(ctx.app, symbol or None, timeframes or None, strategy_id or None, draw, vision,
                                      theory or None, progress=ctx.progress)
    except (ValueError, RuntimeError) as exc:
        return fail(f"Analysis failed: {exc}")
    result = compact_report(report)
    if report.get("theories") and theory:
        found = next(iter(report["theories"].values()))
        result["theory"] = {"status": found.get("status"), "observations": (found.get("observations") or [])[:4],
                            "interpretation": found.get("interpretation")}
    return ok(report["summary_ckb"], **result)


# --- alerts ------------------------------------------------------------------------------

def _alert_sentence(alert: dict[str, Any]) -> str:
    params, sym = alert["params"], symbol_ckb(alert["symbol"])
    kind = alert["kind"]
    if kind == "price_cross":
        where = {"up": "بەرەو سەرەوە تێپەڕی", "down": "بەرەو خوارەوە تێپەڕی"}.get(params.get("direction"), "گەیشتە")
        return f"کاتێک {sym} {where} {fmt_price(params['level'])}"
    if kind == "candle_close":
        return f"کاتێک مۆمێکی {tf_ckb(alert['timeframe'])} لە {fmt_price(params['level'])} تێپەڕی و داخرا"
    if kind == "zone_touch":
        return f"کاتێک {sym} گەیشتە ناوچەی {fmt_price(params['low'])} تا {fmt_price(params['high'])}"
    if kind == "volume_spike":
        return f"کاتێک ڤۆلیۆمی {sym} لە {tf_ckb(alert['timeframe'])} {params['k']:g} هێندەی تێکڕا بەرز بووەوە"
    return f"کاتێک ستراتیژی {alert.get('strategy_id')} لە {sym} ئامادە بوو"


@tool("set_alert",
      description="Watch the market in the background and SPEAK an alert. kind: price_cross (level; direction up/down/any; "
                  "on touch), candle_close (a candle of timeframe closes beyond level), zone_touch (low/high), "
                  "volume_spike (k x average of n bars; tick volume), strategy_state (all numeric rules of strategy_id "
                  "become true). repeat=false fires once. expires_in_hours default 72.",
      description_ckb="دانانی ئاگادارکردنەوە",
      params={"type": "object", "properties": {
          "kind": {"type": "string", "enum": list(KINDS)},
          "symbol": {"type": "string", "description": "as the user said it (Sorani is fine), e.g. زێڕ"},
          "level": {"type": "number"}, "low": {"type": "number"}, "high": {"type": "number"},
          "direction": {"type": "string", "enum": ["up", "down", "any"]}, "timeframe": {"type": "string"},
          "k": {"type": "number"}, "n": {"type": "integer"}, "strategy_id": {"type": "string"},
          "repeat": {"type": "boolean"}, "note": {"type": "string"}, "expires_in_hours": {"type": "number"},
          "source": {"type": "string", "enum": ["mt5", "tradingview"], "description": "price feed (default mt5)"}},
          "required": ["kind"]},
      examples_ckb=("ئەگەر زێڕ گەیشتە [نرخ] ئاگادارم بکەرەوە", "ئەگەر مۆمی پازدە خولەک لە سەرووی [نرخ] داخرا پێم بڵێ"),
      risk="safe", blocking=True, timeout_s=15)
async def set_alert(ctx: ToolContext, kind: str, **spec: Any) -> dict[str, Any]:
    trading = _trading(ctx)
    if trading.monitor is None:
        return fail("The market monitor is not running.")
    wanted = spec.get("symbol") or ctx.app.config.get("trading.default_symbol", "XAUUSD")
    symbol = resolve_instrument(wanted) or canonical_symbol(wanted)
    price = None
    if trading.mt5 is not None:
        try:
            tick = await trading.mt5.tick(symbol)
            price = tick.get("bid") or tick.get("last")
        except Exception:  # noqa: BLE001 - the alert still works; arming happens on the first check
            price = None
    try:
        alert = trading.monitor.add({**spec, "kind": kind, "symbol": symbol}, price=price)
    except AlertError as exc:
        return fail(f"Alert not set: {exc}.")
    return ok(f"ئاگادارکردنەوەکە دانرا: {_alert_sentence(alert)} پێت دەڵێم.", alert_id=alert["id"], kind=alert["kind"],
              symbol=alert["symbol"], timeframe=alert["timeframe"], params={k: v for k, v in alert["params"].items()
                                                                           if not k.startswith("_")},
              price_now=price, expires_at=alert["expires_at"], repeat=alert["repeat"])


@tool("list_alerts", description="List the user's market alerts (active, fired or all).", description_ckb="لیستی ئاگادارکردنەوەکان",
      params={"type": "object", "properties": {"status": {"type": "string", "enum": ["active", "fired", "all"]}}},
      examples_ckb=("چ ئاگادارکردنەوەیەکم هەیە؟",), risk="safe", blocking=True, timeout_s=10)
async def list_alerts(ctx: ToolContext, status: str = "active") -> dict[str, Any]:
    monitor = _trading(ctx).monitor
    if monitor is None:
        return fail("The market monitor is not running.")
    alerts = monitor.list(status or "active")
    items = [{"id": a["id"], "kind": a["kind"], "symbol": a["symbol"], "timeframe": a["timeframe"],
              "status": a["status"], "fired": a["fire_count"], "note": a["note"],
              "what_ckb": _alert_sentence(a), "last_text_ckb": a.get("last_text_ckb") or None} for a in alerts[:30]]
    return ok(f"{len(alerts)} ئاگادارکردنەوە ({status}).", alerts=items)


_ALL_WORDS = ("all", "هەموو", "هەمووی")
_EASTERN = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


@tool("cancel_alert", description="Cancel one alert by id, or all active alerts with alert_id='all' (the user is "
                                  "asked first when more than one is active).",
      description_ckb="هەڵوەشاندنەوەی ئاگادارکردنەوە",
      params={"type": "object", "properties": {"alert_id": {"type": "string"}}, "required": ["alert_id"]},
      examples_ckb=("هەموو ئاگادارکردنەوەکان هەڵبوەشێنەوە",), risk="safe", blocking=True, timeout_s=30)
async def cancel_alert(ctx: ToolContext, alert_id: str) -> dict[str, Any]:
    monitor = _trading(ctx).monitor
    if monitor is None:
        return fail("The market monitor is not running.")
    if str(alert_id).strip().lower() in _ALL_WORDS:
        # Cancelled alerts cannot be brought back: several at once need a yes (the
        # review found «cancel the alert» / «stop the alarm» cancelling every one).
        active = monitor.list("active")
        if len(active) > 1:
            question = f"هەر {str(len(active)).translate(_EASTERN)} ئاگادارکردنەوە چالاکەکەت هەڵبوەشێنمەوە؟"
            detail = "\n".join(f"#{a['id']}: {_alert_sentence(a)}" for a in active[:12])
            if not await ctx.confirm(question, detail):
                return fail("The user did not approve cancelling all alerts; nothing was cancelled.", declined=True)
    try:
        count = monitor.cancel(alert_id)
    except (TypeError, ValueError):
        return fail("alert_id must be a number or 'all'.")
    if not count:
        return fail(f"No active alert {alert_id}.")
    return ok(f"{count} ئاگادارکردنەوە هەڵوەشێنرایەوە.", cancelled=count)


# --- strategies --------------------------------------------------------------------------

def _is_confirmation(text: str) -> bool:
    short = len(normalize_ckb(text, strip_punct=True)) <= 40
    return short and (classify_answer(text) is True or normalize_ckb(text, strip_punct=True) in ("activate", "confirm", "چالاکی بکە"))


@tool("strategy_save",
      description="Save the user's trading strategy (text exactly as the user said/pasted it, Sorani or English) as a "
                  "strategy card with checkable rules; returns a Sorani read-back to say and what is missing. To "
                  "update a card pass strategy_id. After the user confirms the read-back, call again with "
                  "strategy_id, status='active' and text='confirm'.",
      description_ckb="پاشەکەوتکردنی ستراتیژی",
      params={"type": "object", "properties": {
          "text": {"type": "string", "description": "the strategy verbatim"},
          "strategy_id": {"type": "string"}, "status": {"type": "string", "enum": ["draft", "active"]}},
          "required": ["text"]},
      examples_ckb=("ئەم ستراتیژییەم پاشەکەوت بکە", "بەڵێ چالاکی بکە"), risk="safe", blocking=True, timeout_s=45)
async def strategy_save(ctx: ToolContext, text: str, strategy_id: str = "", status: str = "") -> dict[str, Any]:
    store = _trading(ctx).strategies
    if store is None:
        return fail("Strategy memory is not available.")
    # Writes and activation use the EXACT id: a guessed id must never pick another card.
    existing = store.get(strategy_id, fuzzy=False) if strategy_id else None
    if strategy_id and existing is None:
        return fail(f"No strategy with the id '{strategy_id}'. Use strategy_list for the exact id.")
    if existing and existing.get("status") == "archived":
        return fail("That card is archived (imported from SAM v1 or retired); it can be changed or activated only "
                    "from the strategies panel.", strategy_id=existing["id"], status="archived")
    if existing and status and _is_confirmation(text):
        card = store.set_status(existing["id"], status)
        word = "چالاک کرا" if status == "active" else "کرایە ڕەشنووس"
        return ok(f"ستراتیژی «{card['title_ckb']}» {word}.", strategy_id=card["id"], status=card["status"],
                  version=card["version"])
    try:
        result = await store.ingest(text, strategy_id=existing["id"] if existing else None, status=status or None)
    except LLMError as exc:
        return fail(f"Could not read the strategy (model unavailable: {exc.kind}). Nothing was saved.")
    except ValueError as exc:
        return fail(f"Could not read the strategy: {exc}. Nothing was saved.")
    card = result["card"]
    return ok(result["readback_ckb"], strategy_id=card["id"], status=card["status"], version=card["version"],
              title_ckb=card["title_ckb"], rules=len(card["rules"]),
              checked_by_numbers=sum(1 for r in card["rules"] if r.get("check")), missing=result["missing"])


@tool("strategy_list", description="List saved strategy cards (id, Sorani title, status, one-line summary).",
      description_ckb="لیستی ستراتیژییەکان",
      params={"type": "object", "properties": {"status": {"type": "string", "enum": ["draft", "active", "archived", "all"]}}},
      examples_ckb=("ستراتیژییەکانم چین؟",), risk="safe", blocking=True, timeout_s=10)
async def strategy_list(ctx: ToolContext, status: str = "all") -> dict[str, Any]:
    store = _trading(ctx).strategies
    if store is None:
        return fail("Strategy memory is not available.")
    cards = store.list(status or "all")
    return ok(f"{len(cards)} ستراتیژی.", strategies=[{k: c[k] for k in ("id", "title_ckb", "status", "version",
                                                                           "summary_ckb", "rules")} for c in cards[:30]])


@tool("strategy_get", description="Full rules of one strategy card (by id or by name).", description_ckb="وردەکاری ستراتیژی",
      params={"type": "object", "properties": {"strategy_id": {"type": "string"}}, "required": ["strategy_id"]},
      examples_ckb=("ستراتیژیی ئاسیا سویپ چییە؟",), risk="safe", blocking=True, timeout_s=10)
async def strategy_get(ctx: ToolContext, strategy_id: str) -> dict[str, Any]:
    store = _trading(ctx).strategies
    card = store.get(strategy_id) if store is not None else None
    if card is None:
        return fail(f"No strategy '{strategy_id}'.")
    rules = [{"kind": r["kind"], "text_ckb": r["text_ckb"], "check": (r.get("check") or {}).get("predicate") or "chart"}
             for r in card.get("rules") or []]
    return ok(card.get("summary_ckb") or card["title_ckb"], id=card["id"], title_ckb=card["title_ckb"],
              status=card["status"], version=card["version"], markets=card.get("markets"),
              timeframes=card.get("timeframes"), sessions=card.get("sessions"), risk=card.get("risk"), rules=rules)


# --- theories ----------------------------------------------------------------------------

@tool("theory_info",
      description="Explain a trading theory from SAM's 40-theory catalogue (ICT, SMC, Wyckoff, Elliott, harmonics, "
                  "order blocks, ...): description, components, limits, and whether SAM can compute it. Without a name, "
                  "lists the catalogue.",
      description_ckb="زانیاری تیۆرییەکانی ترەیدینگ",
      params={"type": "object", "properties": {"name": {"type": "string", "description": "English or Sorani name"}}},
      examples_ckb=("تیۆری وایکۆف چییە؟", "چ تیۆرییەکت دەزانیت؟"), risk="safe", blocking=True, timeout_s=10)
async def theory_info(ctx: ToolContext, name: str = "") -> dict[str, Any]:
    if not name:
        items = list_theories(_trading(ctx).theories or THEORIES)
        return ok(f"{len(items)} تیۆری لە کەتەلۆگدایە.", theories=items)
    theory = find_theory(name, _trading(ctx).theories or THEORIES)
    if theory is None:
        return fail(f"No theory called '{name}'.", known=[t["id"] for t in list_theories()])
    return ok(f"{theory.name_ckb}: {theory.description_ckb}", **theory.summary())


# --- package lifecycle -------------------------------------------------------------------

def register(app: Any) -> None:
    app.config.register_defaults(SETTINGS)
    app.db.ensure_schema("trading", TRADING_MIGRATIONS)
    app.trading.mt5 = MT5Feed(app)
    app.trading.engine = Engine(app)
    app.trading.theories = THEORIES
    app.trading.strategies = StrategyStore(app)
    app.trading.monitor = Monitor(app)
    for fn in (get_price, analyze_market_tool, set_alert, list_alerts, cancel_alert, strategy_save, strategy_list,
               strategy_get, theory_info):
        app.tools.add(fn, owner="trading")


async def start(app: Any) -> None:
    if app.trading.mt5 is not None:
        app.spawn(app.trading.mt5.connect(), "mt5-connect")
    if app.trading.monitor is not None:
        await app.trading.monitor.start()


async def stop(app: Any) -> None:
    if app.trading.monitor is not None:
        await app.trading.monitor.stop()
    if app.trading.mt5 is not None:
        await app.trading.mt5.close()


__all__ = ["register", "start", "stop", "SETTINGS"]
