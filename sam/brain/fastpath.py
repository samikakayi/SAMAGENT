"""No-AI fast path: run a matched common command (intents.py) straight
through the tool registry and answer with the tool's own Sorani result.

No model request, so no quota and no model latency: the tool itself is the
whole turn (get_price, tv_open, open_app, tv_set_chart: well under a second
when the app is up). Tools still go through ``app.tools.dispatch`` (risk
classes, confirmations, timings, activity rows, taint), exactly as when a
model picks them. Analysis runs with ``vision=False``: the vision look is a
model call too, and only strategy rules need it.

Every fast turn is counted in ``usage_counters`` (provider ``fastpath``,
model = intent name), so the panel's activity numbers show what it saved.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable

from ..textnorm import is_arabic_script
from .intents import Intent, match
from .outcome import alerts_sentence, tool_sentence

log = logging.getLogger("sam.fastpath")

ACK_LOOK = "با سەیری بکەم."
_DIGITS_CKB = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def enabled(app: Any) -> bool:
    try:
        return bool(app.config.get("brain.fastpath.enabled", True))
    except Exception:  # noqa: BLE001
        return True


def intent_for(app: Any, text: str) -> Intent | None:
    """The fast-path intent of ``text`` when the fast path is on and the tool exists."""
    if not enabled(app):
        return None
    try:
        intent = match(text)
    except Exception:  # noqa: BLE001 - a matcher bug must never cost the user an answer
        log.exception("fast-path matcher failed")
        return None
    if intent is None or app.tools.get(intent.tool) is None:
        return None
    return intent


async def run(app: Any, intent: Intent, *, source: str,
              proceed: Callable[[], bool] | None = None) -> AsyncIterator[tuple[str, str]]:
    """Yield ("ack", text) for slow tools, then ("answer", text).

    ``proceed`` is asked right before the tool runs: False = a newer user turn
    replaced this one (conversation.py), so nothing runs and nothing is said."""
    args = dict(intent.args)
    if intent.chart_symbol:
        symbol = await _chart_symbol(app)
        if symbol:
            args["symbol"] = symbol
    if intent.slow:
        yield "ack", ACK_LOOK
    if proceed is not None and not proceed():
        return
    result = await app.tools.dispatch(intent.tool, args, source=source)
    _count(app, intent, bool(result.get("ok")))
    if intent.name == "quiet" and source in ("cascade", "live") and result.get("ok"):
        yield "answer", ""          # the user asked for quiet: SAM says nothing more
        return
    yield "answer", reply(intent, args, result)


async def _chart_symbol(app: Any) -> str:
    """The instrument on the user's chart («هێڵی پشتگیری و بەرگری بکێشە» means
    the chart they look at), or "" (the analysis then uses the main symbol)."""
    tv = getattr(getattr(app, "trading", None), "tv", None)
    if tv is None or not getattr(tv, "connected", False):
        return ""
    try:
        state = await tv.chart_state()
    except Exception:  # noqa: BLE001
        return ""
    return str(state.get("canonical") or state.get("symbol") or "")


def _count(app: Any, intent: Intent, ok: bool) -> None:
    try:
        app.db.bump_usage("fastpath", intent.name, kind="text", errors=0 if ok else 1)
    except Exception:  # noqa: BLE001
        log.debug("fast-path usage count failed", exc_info=True)


# --- the spoken answer ----------------------------------------------------------------------------------
def reply(intent: Intent, args: dict[str, Any], result: dict[str, Any]) -> str:
    """One honest Sorani sentence for ``result`` (the tools already speak
    Sorani; open_app, alerts and a few failures get their own wording)."""
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    special = next((k for k in ("declined", "blocked", "timeout", "cancelled") if data.get(k) is True), None)
    if special:
        return tool_sentence(intent.tool, args, result)
    good = bool(result.get("ok"))
    if intent.name == "open_app":
        return _open_app_reply(intent, data, good)
    if intent.name == "list_alerts" and good:
        return _alerts_reply(data.get("alerts") or [])
    if intent.name == "cancel_alerts" and not good:
        if str(args.get("alert_id")) == "all":
            return "هیچ ئاگادارکردنەوەیەکی چالاکت نەبوو."
        return f"ئاگادارکردنەوەی ژمارە {str(args.get('alert_id')).translate(_DIGITS_CKB)} نەدۆزرایەوە."
    if intent.name == "stop":
        return "ڕاگیرا."
    if intent.name == "quiet":
        return "باشە، بێدەنگ بووم." if good else "نەمتوانی دەنگم ببڕم."
    return tool_sentence(intent.tool, args, result)


def _app_name(intent: Intent) -> str:
    if intent.said and is_arabic_script(intent.said):
        return intent.said
    try:
        from ..hands.aliases import ALIASES

        name = str(intent.args.get("name") or "")
        for spec in ALIASES:
            if spec.display == name:
                spoken = next((a for a in spec.aliases if is_arabic_script(a)), "")
                return spoken or name
    except Exception:  # noqa: BLE001
        pass
    return str(intent.args.get("name") or "بەرنامەکە")


def _open_app_reply(intent: Intent, data: dict[str, Any], good: bool) -> str:
    name = _app_name(intent)
    state = str(data.get("state") or "")
    if good:
        return f"{name} پێشتر کرابووەوە؛ هێنامە پێشەوە." if state == "focused" else f"{name} کرایەوە."
    if state == "not_found":
        return f"{name} لەسەر ئەم کۆمپیوتەرە نەدۆزرایەوە."
    return f"نەمتوانی {name} بکەمەوە."


def _alerts_reply(alerts: list[dict[str, Any]]) -> str:
    return alerts_sentence(alerts)


__all__ = ["intent_for", "run", "reply", "enabled", "ACK_LOOK"]
