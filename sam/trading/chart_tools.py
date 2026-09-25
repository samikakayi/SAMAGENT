"""Entry module of the chart bridge (``sam.app.PACKAGES``): ``app.trading.tv`` +
the tools tv_open, tv_set_chart, chart_state, draw_on_chart, clear_my_drawings.

Summaries are short Sorani sentences the model can rephrase; they only claim
what the bridge verified (read-back symbol/resolution, shape ids that exist).
Chart text the user or indicators wrote (study names) is returned under
``untrusted``: it is data, never instructions.

The chart screenshot used by analyze_market is the bridge method
``app.trading.tv.screenshot()`` (no separate tool; see docs/CONTRACTS.md 3.4).
"""

from __future__ import annotations

import logging
from typing import Any

import re

from ..brain.tools import ToolContext, fail, ok, tool
from ..textnorm import normalize_ckb
from .common import DRAWING_KINDS, canonical_symbol
from .engine.sorani import spoken_price, symbol_ckb
from .symbols import (chart_symbol, mentioned_instruments, recent_user_text, resolve_instrument, same_instrument,
                      user_named)
from .tradingview import TradingViewBridge, TvError
from .tv_parse import kinds_summary_ckb, resolution_label_ckb

# draw_on_chart: a model-sent price further than this from the chart price is
# refused unless the user said the number (review 2026-09-24: SAM offered gold
# S/R at 2650/2700 from its prompt examples with gold near 4270; the old guard
# only refused prices outside /10..x10 of the chart price).
FAR_SHARE = 0.15

log = logging.getLogger("sam.trading.chart_tools")

OWNER = "trading.chart"
TV_CKB = "ترەیدینگ ڤیو"
_STATE_TEXT_CKB = {
    "connected": f"{TV_CKB} ئامادەیە و پەیوەستم پێیەوە.",
    "started": f"{TV_CKB}م کردەوە و پەیوەست بووم پێیەوە.",
    "restarted": f"{TV_CKB}م داخست و دووبارە کردمەوە؛ ئێستا دەتوانم لەسەر چارتەکە کار بکەم.",
    "needs_restart": (f"{TV_CKB} کراوەیە بەڵام بێ دەرگای پەیوەندی، بۆیە ناتوانم لەسەری کار بکەم. "
                      "دەبێت جارێک دابخرێت و دووبارە بکرێتەوە."),
    "declined": f"باشە، {TV_CKB}م دانەخست.",
    "not_installed": f"{TV_CKB}ی دیسکتۆپ لەم کۆمپیوتەرەدا دانەمەزراوە.",
    "failed": f"نەمتوانی پەیوەندی بە {TV_CKB}ەوە بکەم.",
}


def _bridge(ctx: ToolContext) -> TradingViewBridge | None:
    trading = getattr(ctx.app, "trading", None)
    return getattr(trading, "tv", None)


def _state_text(result: dict[str, Any]) -> str:
    key = "declined" if result.get("declined") else str(result.get("state", "failed"))
    return _STATE_TEXT_CKB.get(key, _STATE_TEXT_CKB["failed"])


async def _ready(ctx: ToolContext, *, start: bool, focus: bool = False) -> dict[str, Any] | None:
    """None when the chart is usable, else a failed tool result.

    ``start``: open TradingView (and, after the user agrees, restart it with the
    port) because the user asked to act on the chart. Read-only tools do not
    start apps behind the user's back.
    """
    tv = _bridge(ctx)
    if tv is None:
        return fail("بەشی چارتی ترەیدینگ ڤیو ئامادە نییە.", unavailable=True)
    if tv.connected and not focus and await tv.connect(wait_ready_s=5):
        return None
    if start:
        result = await tv.ensure_running(allow_restart=True, confirm=ctx.confirm_routine, focus=focus)
        return None if result.get("ok") else fail(_state_text(result), **_data(result))
    if await tv.connect(wait_ready_s=10):
        return None
    return fail(f"{TV_CKB} کراوە نییە یان پەیوەستی نیم؛ ئەگەر بتەوێت دەیکەمەوە.", connected=False,
                hint="call tv_open to start TradingView")


# Said in front of a chart answer when SAM restarted TradingView without asking (full authority).
RESTARTED_NOTE_CKB = "ترەیدینگ ڤیوم دووبارە کردەوە بۆ ئەوەی کار لەسەر چارتەکە بکەم."


def _acted_note(ctx: ToolContext) -> list[str]:
    """[RESTARTED_NOTE_CKB] when this call restarted TradingView without a question."""
    return [RESTARTED_NOTE_CKB] if getattr(ctx, "acted", None) else []


def _name(state: dict[str, Any]) -> str:
    """The instrument as it is said («زێڕ»), not the broker ticker; the ticker stays in the data."""
    return symbol_ckb(str(state.get("canonical") or "")) or str(state.get("symbol") or "")


def _data(result: dict[str, Any]) -> dict[str, Any]:
    """Bridge result without its own ``ok`` (the tool result already has one)."""
    return {k: v for k, v in result.items() if k != "ok"}


def _chart_error(action_ckb: str, exc: Exception) -> dict[str, Any]:
    code = getattr(exc, "code", "error")
    return fail(f"نەمتوانی {action_ckb}: {exc}", error=code)


@tool("tv_open",
      description="Open TradingView Desktop (or bring it to the front) with SAM's local chart connection. If it runs "
                  "without that connection SAM asks the user once and restarts it. Use for 'open TradingView'.",
      description_ckb="کردنەوەی ترەیدینگ ڤیو",
      examples_ckb=("ترەیدینگ ڤیو بکەرەوە", "چارتەکەم بۆ بکەرەوە"),
      risk="safe", blocking=True, timeout_s=150)
async def tv_open(ctx: ToolContext) -> dict[str, Any]:
    tv = _bridge(ctx)
    if tv is None:
        return fail("بەشی چارتی ترەیدینگ ڤیو ئامادە نییە.", unavailable=True)
    # A TradingView without SAM's port is restarted: routine (with full authority SAM does it without
    # asking and says so -- the user, 2026-09-25: "don't ask me yes or no").
    result = await tv.ensure_running(allow_restart=True, confirm=ctx.confirm_routine, focus=True)
    if not result.get("ok"):
        return fail(_state_text(result), **_data(result))
    try:
        state = await tv.chart_state()
    except TvError:
        return ok(_state_text(result), **_data(result))
    text = f"{_state_text(result)} چارت: {_name(state)}، {state['timeframe_ckb']}."
    return ok(text, **_data(result), symbol=state["symbol"], resolution=state["resolution"])


@tool("tv_set_chart",
      description="Change the TradingView chart's symbol and/or timeframe. Accepts any alias (gold/XAUUSD/زێڕ/"
                  "گۆڵد, بیتکۆین, OANDA:XAUUSD) and any timeframe form (15, 15m, M15, H1, 4h, D, ١٥ خولەک, "
                  "کاتژمێرێک, ڕۆژانە). Gold keeps the user's own gold symbol when the chart already shows gold.",
      params={"type": "object", "properties": {
          "symbol": {"type": "string", "description": "Only when the user named an instrument: as the user said "
                                                      "it (Sorani is fine), e.g. زێڕ, گۆڵد, گوڵت, بیتکۆین, XAUUSD. "
                                                      "Never a number or a word you are unsure of: leave it out"},
          "timeframe": {"type": "string", "description": "Timeframe in any form, e.g. '15', 'H1', '١٥ خولەک'"}}},
      description_ckb="گۆڕینی سیمبۆڵ و کاتی چارت",
      examples_ckb=("گۆڵد لەسەر ١٥ خولەک پیشان بدە", "بیکە بە کاتژمێرێک", "بیتکۆین بکەرەوە"),
      risk="safe", blocking=True, timeout_s=150)
async def tv_set_chart(ctx: ToolContext, symbol: str | None = None, timeframe: str | None = None) -> dict[str, Any]:
    if not (symbol or "").strip() and not (timeframe or "").strip():
        return fail("Give a symbol or a timeframe.")
    problem = await _ready(ctx, start=True, focus=True)
    if problem:
        return problem
    tv = _bridge(ctx)
    assert tv is not None
    data: dict[str, Any] = {}
    parts: list[str] = _acted_note(ctx)
    if (symbol or "").strip():
        symbol, refusal = await _checked_symbol(ctx, tv, symbol.strip(), bool((timeframe or "").strip()), data)
        if refusal is not None:
            return refusal
    try:
        if (symbol or "").strip():
            res = await tv.set_symbol(symbol)  # type: ignore[arg-type]
            data["symbol_result"] = res
            if not res.get("ok"):
                why = "ئەم سیمبۆڵە نەناسرایەوە" if res.get("error") == "unknown_symbol" else "ترەیدینگ ڤیو قبووڵی نەکرد"
                back = "، چارتەکە گەڕایەوە بۆ پێشوو" if res.get("restored") else ""
                return fail(f"نەمتوانی سیمبۆڵ بگۆڕم بۆ «{symbol}»: {why}{back}.", **data)
        if (timeframe or "").strip():
            res_tf = await tv.set_timeframe(timeframe)  # type: ignore[arg-type]
            data["timeframe_result"] = res_tf
            if not res_tf.get("ok"):
                why = "ئەم کاتە نەناسرایەوە" if res_tf.get("error") == "unknown_timeframe" else "ترەیدینگ ڤیو قبووڵی نەکرد"
                return fail(f"نەمتوانی کاتی چارت بگۆڕم بۆ «{timeframe}»: {why}.", **data)
        state = await tv.chart_state()
    except TvError as exc:
        return _chart_error("چارتەکە بگۆڕم", exc)
    changed = any((data.get(k) or {}).get("changed") for k in ("symbol_result", "timeframe_result"))
    # Spoken: the Kurdish name, not the broker ticker («زێڕ», not "PEPPERSTONE:XAUUSD").
    where = f"{_name(state)} لەسەر {state['timeframe_ckb']}"
    parts.append(f"چارتەکە گۆڕا بۆ {where}." if changed else f"چارتەکە پێشتر {where} بوو.")
    ignored = data.get("symbol_ignored") or {}
    if ignored.get("why") == "not a known instrument":
        # «مەبەستت زێڕە؟»: the timeframe was clear, the instrument was not (live 2026-09-25: '100')
        parts.append(f"«{ignored['requested']}» وەک بازاڕێک نەناسرایەوە، بۆیە بازاڕەکەم نەگۆڕی؛ "
                     f"مەبەستت {ignored.get('guess_ckb') or 'زێڕ'}ە؟")
    return ok(" ".join(parts), symbol=state["symbol"], canonical=state["canonical"], resolution=state["resolution"],
              timeframe=state["timeframe"], price=state["price"], changed=changed, **data)


def _feeds(app: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The user's own TradingView feeds: (learned, mapped) settings."""
    try:
        learned = dict(app.config.get("trading.tv_learned_symbols", {}) or {})
        mapped = dict(app.config.get("trading.symbol_map", {}) or {})
    except Exception:  # noqa: BLE001
        return {}, {}
    return learned, mapped


def _guess_ckb(app: Any) -> str:
    """The instrument the user most likely meant: his main one (setting trading.default_symbol)."""
    try:
        main = resolve_instrument(str(app.config.get("trading.default_symbol", "XAUUSD") or "XAUUSD"))
    except Exception:  # noqa: BLE001
        main = None
    return symbol_ckb(main or "XAUUSD") or "زێڕ"


async def _checked_symbol(ctx: ToolContext, tv: Any, symbol: str, has_timeframe: bool,
                          data: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """(symbol to set or None, refusal).

    Only a KNOWN instrument (``symbols.chart_symbol``: aliases incl. «گوڵت»/«گوڵ»,
    or the user's own learned/mapped feed) is ever applied -- never a bare number,
    one or two letters or an unknown word. Live 2026-09-25 «بڕۆ 100 چار 3 خولەکی
    یەکسەر لە گوڵت» ("the gold chart, 3 minutes"; STT wrote "100") became
    tv_set_chart(symbol='100') and the chart switched to a symbol named "100".
    When the user's own last words name exactly one instrument, that one is used
    instead of the model's garbage; else the chart keeps its symbol (the
    timeframe is still applied) and the answer asks «مەبەستت زێڕە؟».
    A known symbol the user did not name is not applied either: live
    2026-09-24 the user asked only for the 1-minute timeframe, the model sent
    symbol='z' and the chart became BATS:Z (a Nasdaq stock)."""
    learned, mapped = _feeds(ctx.app)
    known = chart_symbol(symbol, learned=learned, mapped=mapped)
    if not known:
        latest = recent_user_text(ctx.app, turns=1) if ctx.source not in ("worker", "ui") else None
        spoken = mentioned_instruments(latest, context=True) if latest else set()
        if len(spoken) == 1:
            known = next(iter(spoken))
            data["symbol_from_words"] = {"requested": symbol, "used": known}
            return known, None
        guess = _guess_ckb(ctx.app)
        if has_timeframe:  # e.g. symbol='z' / '100' next to the timeframe the user asked for: apply only that
            data["symbol_ignored"] = {"requested": symbol, "why": "not a known instrument", "guess_ckb": guess}
            return None, None
        return None, fail(f"«{symbol}» وەک بازاڕێک نەناسرایەوە، بۆیە چارتەکەم نەگۆڕی. مەبەستت {guess}ە؟",
                          error="unknown_symbol", requested=symbol,
                          hint="ask the user which instrument; never invent a ticker or copy a number as a symbol")
    # An explicit feed ('PEPPERSTONE:XAUUSD') goes as is; anything else as the instrument,
    # so the bridge picks the user's own feed for it (tv_symbol_for: same / learned / mapped).
    target = symbol if ":" in symbol else known
    named = user_named(ctx.app, symbol, ctx.source)
    if named is not False:
        return target, None
    try:
        current = str((await tv.chart_state()).get("symbol") or "")
    except TvError:
        current = ""
    if current and same_instrument(current, known):
        return target, None
    if has_timeframe:
        data["symbol_ignored"] = {"requested": symbol, "why": "the user did not name this instrument"}
        return None, None
    return None, fail("چارتەکەم نەگۆڕی، چونکە ناوی ئەو بازاڕەت نەهێنا. کام بازاڕت دەوێت؟", error="symbol_not_named",
                      requested=symbol, hint="the user did not name this instrument: ask before changing the chart")


@tool("chart_state",
      description="Read what the TradingView chart shows now: symbol, timeframe, last price and bar, visible range, "
                  "indicators, and how many drawings are SAM's vs the user's. Does not open TradingView.",
      description_ckb="خوێندنەوەی باری چارت",
      examples_ckb=("چارتەکە چی پیشان دەدات؟", "ئێستا لەسەر چ کاتێکە؟"),
      risk="safe", blocking=True, timeout_s=30)
async def chart_state(ctx: ToolContext) -> dict[str, Any]:
    problem = await _ready(ctx, start=False)
    if problem:
        return problem
    tv = _bridge(ctx)
    assert tv is not None
    try:
        state = await tv.chart_state()
    except TvError as exc:
        return _chart_error("چارتەکە بخوێنمەوە", exc)
    price = f"، دوایین نرخ {spoken_price(state['price'])}" if state.get("price") is not None else ""
    text = (f"چارت: {_name(state)}، {state['timeframe_ckb']}{price}. "
            f"{state['my_drawings']} نیشانەی سام و {state['user_drawings']} نیشانەی تۆ لەسەر چارتەکەیە.")
    studies = state.pop("studies", [])
    return ok(text, **state, untrusted={"indicator_names": studies})


_POINT_SCHEMA = {"type": "object", "properties": {
    "price": {"type": "number", "description": "Price level"},
    "time": {"type": "integer", "description": "Bar time, UTC unix seconds (optional)"},
    "bars_ago": {"type": "integer", "description": "Instead of time: this many bars before the last bar"}},
    "required": ["price"]}
_ITEM_SCHEMA = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": list(DRAWING_KINDS)},
    "points": {"type": "array", "items": _POINT_SCHEMA,
               "description": "horizontal_line/ray, text, arrows: 1 point; trend_line, rectangle, fib_retracement: "
                              "2 points; long_position/short_position: 3 points = entry, stop, target"},
    "text": {"type": "string", "description": "Short label shown on the chart, e.g. 'پشتگیری' or 'Resistance'"},
    "color": {"type": "string", "description": "Hex or a meaning: support, resistance, entry, stop, target, zone, "
                                               "info, liquidity"}},
    "required": ["kind", "points"]}


@tool("draw_on_chart",
      description="Draw labelled lines, zones, fibs, arrows or long/short position tools on the TradingView chart at "
                  "exact prices. Every drawing is recorded as SAM's own, so clear_my_drawings can remove only these. "
                  "Use real levels (from chart_state/analyze_market or the user's words), never invented prices.",
      params={"type": "object", "properties": {
          "items": {"type": "array", "items": _ITEM_SCHEMA, "description": "Drawings to create (max 30)"},
          "tag": {"type": "string", "description": "Group name to clear later, e.g. 'levels'"}},
          "required": ["items"]},
      description_ckb="کێشانی هێڵ و ناوچە لەسەر چارت",
      examples_ckb=("هێڵی پشتگیری و بەرگری بکێشە", "هێڵێک لەسەر نرخی ئێستا بکێشە"),
      risk="safe", blocking=True, timeout_s=150)
async def draw_on_chart(ctx: ToolContext, items: list[dict[str, Any]], tag: str | None = None) -> dict[str, Any]:
    problem = await _ready(ctx, start=True)
    if problem:
        return problem
    tv = _bridge(ctx)
    assert tv is not None
    try:
        price = (await tv.chart_state()).get("price")
    except TvError:
        price = None
    items, far = _near_price(items, price, recent_user_text(ctx.app) or "", ctx.source)
    if not items:
        return fail(f"ئەو ئاستانەم نەکێشا چونکە زۆر دوورن لە نرخی ئێستای چارت ({price:g}).", far_prices=far,
                    chart_price=price, hint="use real levels from analyze_market/get_price or the user's own numbers")
    try:
        result = await tv.draw_many(items, tag=tag or "user-request")
    except TvError as exc:
        return _chart_error("لەسەر چارتەکە بکێشم", exc)
    errors = result.get("errors") or []
    if not result.get("drawn"):
        reason = errors[0]["error"] if errors else "unknown"
        return fail(f"هیچ شتێکم نەکێشا لەسەر چارتەکە ({reason}).", **_data(result))
    text = " ".join([*_acted_note(ctx), f"لەسەر چارتی {symbol_ckb(canonical_symbol(str(result.get('symbol') or '')))} "
                                        f"کێشام: {kinds_summary_ckb(result['kinds'])}."])
    if errors:
        text += f" {len(errors)} دانەیان نەکێشران."
    return ok(text, **_data(result))


@tool("clear_my_drawings",
      description="Remove the drawings SAM made on the TradingView chart (optionally one group/tag). Never touches the "
                  "user's own drawings.",
      params={"type": "object", "properties": {
          "tag": {"type": "string", "description": "Leave empty to remove all of SAM's drawings (what the user "
                                                   "usually means). Only a group SAM drew earlier, e.g. 'analysis'"}}},
      description_ckb="سڕینەوەی هێڵەکانی سام",
      examples_ckb=("هێڵەکانت بسڕەوە", "بیانسڕەوە"),
      risk="safe", blocking=True, timeout_s=60)
async def clear_my_drawings(ctx: ToolContext, tag: str | None = None) -> dict[str, Any]:
    problem = await _ready(ctx, start=False)
    if problem:
        return problem
    tv = _bridge(ctx)
    assert tv is not None
    wanted = _tag_filter(tag)
    try:
        result = await tv.clear(tag=wanted)
        if wanted and not any(result.get(k) for k in ("removed", "pruned", "other_symbol", "failed")):
            # Integration smoke 2026-09-24: for "هێڵەکانت بسڕەوە" models sent tags naming
            # no group SAM drew ("levels", "none") and 7 SAM drawings stayed on the chart.
            # Every row here is SAM's own, so clearing all of them never touches the user's.
            result = {**await tv.clear(tag=None), "tag_ignored": wanted}
    except TvError as exc:
        return _chart_error("هێڵەکانم بسڕمەوە", exc)
    removed = int(result.get("removed", 0))
    if result.get("failed"):
        return fail(f"{removed} نیشانەم سڕییەوە بەڵام {result['failed']} دانەیان نەسڕانەوە.", **result)
    if removed == 0:
        text = "هیچ نیشانەیەکی خۆم لەسەر ئەم چارتە نەبوو."
    else:
        text = f"{removed} نیشانەی خۆمم سڕییەوە؛ دەستم لە هێڵەکانی تۆ نەدا."
    if result.get("other_symbol"):
        text += f" {result['other_symbol']} نیشانەی ترم لەسەر سیمبۆڵێکی تر ماون."
    return ok(text, **result)


def _numbers(text: str) -> list[float]:
    out = []
    for token in re.findall(r"\d+(?:[.,]\d+)?", normalize_ckb(text)):
        try:
            out.append(float(token.replace(",", ".")))
        except ValueError:
            continue
    return out


def _near_price(items: Any, price: Any, said: str, source: str) -> tuple[list[Any], list[float]]:
    """Items whose prices are within FAR_SHARE of the chart price, or a number the
    user said; (kept, refused prices). Worker/UI requests are not filtered."""
    if not isinstance(items, list) or not isinstance(price, (int, float)) or price <= 0 or source in ("worker", "ui"):
        return list(items or []), []
    spoken = _numbers(said)
    kept, far = [], []
    for item in items:
        prices = [p.get("price") for p in (item.get("points") or []) if isinstance(p, dict)] if isinstance(item, dict) else []
        bad = [float(p) for p in prices if isinstance(p, (int, float)) and abs(float(p) - price) > price * FAR_SHARE
               and not any(abs(float(p) - n) <= max(1.0, float(p) * 0.001) for n in spoken)]
        if bad:
            far.extend(bad)
        else:
            kept.append(item)
    return kept, far


_ALL_TAGS = frozenset({"", "none", "null", "all", "any", "*", "everything", "هەموو", "هەمووی"})


def _tag_filter(tag: str | None) -> str | None:
    """A group name, or None for "all of SAM's drawings" (models write 'none'/'all')."""
    text = (tag or "").strip()
    return None if text.lower() in _ALL_TAGS else text


TOOLS = (tv_open, tv_set_chart, chart_state, draw_on_chart, clear_my_drawings)


def register(app: Any) -> None:
    """Sync + fast: create the bridge (no network) and add the tools."""
    app.trading.tv = TradingViewBridge(app)
    for handler in TOOLS:
        app.tools.add(handler, owner=OWNER)


async def start(app: Any) -> None:
    """Attach in the background when TradingView already has the port; never launches it."""
    if app.trading.tv is not None:
        app.spawn(app.trading.tv.probe(), "tradingview-probe")


async def stop(app: Any) -> None:
    if app.trading.tv is not None:
        await app.trading.tv.close()


__all__ = ["register", "start", "stop", "TOOLS", "tv_open", "tv_set_chart", "chart_state", "draw_on_chart",
           "clear_my_drawings", "resolution_label_ckb"]
