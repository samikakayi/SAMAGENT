"""What SAM says about a tool result when no model can word it.

The integration smoke (2026-09-24) heard a bare «تەواو بوو.» for a price
question and «ببورە، نەکرا.» with no reason, because every wording rung was
rate limited and the hands/memory tools summarise in English ("Notepad is
open.", "Saved to memory."). This module turns any tool result into one
short, honest Sorani sentence without a model:

1. ``data.summary_ckb`` or an Arabic-script ``summary`` (the trading tools
   already speak Sorani) wins;
2. otherwise a per-tool template (ok / failed), with the few arguments that
   make it specific (the app name, the window action);
3. declined / blocked / timeout / cancelled results have their own sentence.
"""

from __future__ import annotations

from typing import Any

from ..textnorm import is_arabic_script

DONE = "تەواو بوو."
NOT_DONE = "ببورە، نەکرا."
NO_MODEL_YET = "ببورە، ئێستا ناتوانم ئەوە بکەم؛ کەمێکی تر هەوڵ بدەرەوە."
NO_RESULTS = "هیچ ئەنجامێکم نەدۆزییەوە."

_SPECIAL = {
    "declined": "باشە، نەمکرد.",
    "blocked": "ئەمە ڕێگەپێدراو نییە، بۆیە نەمکرد.",
    "timeout": "کارەکە زۆری خایاند و ڕاگیرا.",
    "cancelled": "ڕاگیرا.",
}

# tool -> (sentence when ok, sentence when it failed). ``{name}`` etc. come from the arguments.
_TEMPLATES: dict[str, tuple[str, str]] = {
    "open_app": ("{name} کرایەوە.", "نەمتوانی {name} بکەمەوە."),
    "type_text": ("نووسیم.", "نەمتوانی بینووسم."),
    "press_keys": ("کلیلەکانم داگرت.", "نەمتوانی کلیلەکان دابگرم."),
    "click": ("کلیکم کرد.", "نەمتوانی کلیک بکەم؛ ئەو شتەم نەدۆزییەوە."),
    "screen_look": ("سەیری شاشەکەم کرد.", "نەمتوانی سەیری شاشەکە بکەم."),
    "screen_act": ("کارەکەم لەسەر شاشەکە کرد.", "نەمتوانی کارەکە لەسەر شاشەکە تەواو بکەم."),
    "run_powershell": ("فەرمانەکە جێبەجێ کرا.", "فەرمانەکە سەرکەوتوو نەبوو."),
    "files": ("کارەکە لەسەر فایلەکە کرا.", "نەمتوانی ئەو کارە لەسەر فایلەکە بکەم."),
    "open_url": ("ماڵپەڕەکە کرایەوە.", "نەمتوانی ماڵپەڕەکە بکەمەوە."),
    "web_search": ("گەڕام، بەڵام ئێستا ناتوانم ئەنجامەکان بخوێنمەوە.", "گەڕانەکە سەرکەوتوو نەبوو."),
    "fetch_page": ("پەڕەکەم خوێندەوە.", "نەمتوانی پەڕەکە بخوێنمەوە."),
    "build_project": ("پرۆژەکە دروست کرا.", "نەمتوانی پرۆژەکە تەواو بکەم."),
    "system_control": ("کرا.", "نەمتوانی ئەوە بکەم."),
    "remember": ("باشە، لەبیرم دەبێت.", "نەمتوانی لەبیری بکەم."),
    "recall": ("شتێکم لەبیرە، بەڵام ئێستا ناتوانم بیڵێم.", "هیچم لەبیر نییە دەربارەی ئەوە."),
    "forget": ("لەبیرم بردەوە.", "شتێکی وام لەبیر نەبوو."),
    "delegate_task": ("دەستم پێکرد، کە تەواو بوو پێت دەڵێم.", "نەمتوانی ئەرکەکە دەست پێ بکەم."),
    "tv_open": ("ترەیدینگ ڤیو ئامادەیە.", "نەمتوانی ترەیدینگ ڤیو بکەمەوە."),
    "tv_set_chart": ("چارتەکە گۆڕا.", "نەمتوانی چارتەکە بگۆڕم."),
    "chart_state": ("چارتەکەم خوێندەوە.", "نەمتوانی چارتەکە بخوێنمەوە."),
    "draw_on_chart": ("لەسەر چارتەکە کێشام.", "نەمتوانی لەسەر چارتەکە بکێشم."),
    "clear_my_drawings": ("هێڵەکانی خۆمم سڕییەوە.", "نەمتوانی هێڵەکانم بسڕمەوە."),
    "get_price": ("نرخەکەم هێنا.", "نەمتوانی نرخەکە بدۆزمەوە."),
    "analyze_market": ("شیکارییەکە تەواو بوو.", "نەمتوانی شیکارییەکە بکەم."),
    "set_alert": ("ئاگادارکردنەوەکە دانرا.", "نەمتوانی ئاگادارکردنەوەکە دابنێم."),
    "list_alerts": ("لیستەکەم هێنا، بەڵام ئێستا ناتوانم بیخوێنمەوە.", "نەمتوانی لیستەکە بهێنم."),
    "cancel_alert": ("ئاگادارکردنەوەکە هەڵوەشێنرایەوە.", "هیچ ئاگادارکردنەوەیەکی وا نەبوو."),
    "strategy_save": ("ستراتیژییەکە پاشەکەوت کرا.", "نەمتوانی ستراتیژییەکە پاشەکەوت بکەم."),
    "strategy_list": ("ستراتیژییەکانم هێنا، بەڵام ئێستا ناتوانم بیانخوێنمەوە.", "نەمتوانی ستراتیژییەکان بهێنم."),
    "strategy_get": ("ستراتیژییەکەم دۆزییەوە، بەڵام ئێستا ناتوانم بیخوێنمەوە.", "ئەو ستراتیژییەم نەدۆزییەوە."),
    "theory_info": ("زانیارییەکەم هەیە، بەڵام ئێستا ناتوانم بیخوێنمەوە.", "ئەو تیۆرییەم نەدۆزییەوە."),
    # more_tools only attaches tools: nothing was done yet (acceptance review:
    # "cancel my gold alert" answered «تەواو بوو.» while the alert stayed).
    "more_tools": (NO_MODEL_YET, NO_MODEL_YET),
    "stop_all": ("هەموو شتێکم ڕاگرت.", "نەمتوانی ڕایبگرم."),
    "stop_speaking": ("باشە، بێدەنگ بووم.", "نەمتوانی دەنگم ببڕم."),
}

_WINDOW_OK = {"focus": "پەنجەرەکە هێنرایە پێشەوە.", "minimize": "پەنجەرەکە بچووک کرایەوە.",
              "maximize": "پەنجەرەکە گەورە کرا.", "restore": "پەنجەرەکە گەڕایەوە بارە ئاساییەکەی.",
              "close": "پەنجەرەکە داخرا.", "snap_left": "پەنجەرەکە بردرا بۆ لای چەپ.",
              "snap_right": "پەنجەرەکە بردرا بۆ لای ڕاست.",
              "list": "لیستی پەنجەرەکانم هێنا، بەڵام ئێستا ناتوانم بیخوێنمەوە."}


def _app_name(args: dict[str, Any]) -> str:
    name = str(args.get("name") or "").strip()
    return name if name and is_arabic_script(name) else "بەرنامەکە"


def tool_sentence(name: str, args: dict[str, Any] | None, result: dict[str, Any] | None) -> str:
    """One honest Sorani sentence for ``result`` of tool ``name``."""
    if not result:
        return NOT_DONE
    args = args or {}
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    good = bool(result.get("ok"))
    for key in ("declined", "blocked", "timeout", "cancelled"):
        if data.get(key) is True:   # the registry's flags; cancel_alert's data.cancelled is a COUNT
            return _SPECIAL[key]
    own = str(data.get("summary_ckb") or "").strip()
    if own and is_arabic_script(own):
        return own
    summary = str(result.get("summary") or "").strip()
    if summary and len(summary) <= 240 and is_arabic_script(summary):
        return summary
    if name == "web_search" and good and summary.startswith("No results"):
        return NO_RESULTS
    if name == "window_control":
        if good:
            return _WINDOW_OK.get(str(args.get("action") or ""), DONE)
        return "نەمتوانی ئەو پەنجەرەیە بدۆزمەوە." if "No open window" in summary else NOT_DONE
    template = _TEMPLATES.get(name)
    if template is None:
        return DONE if good else NOT_DONE
    return template[0 if good else 1].format(name=_app_name(args))


_DIGITS_CKB = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
_COUNT_CKB = {1: "یەک", 2: "دوو", 3: "سێ", 4: "چوار", 5: "پێنج", 6: "شەش", 7: "حەوت", 8: "هەشت", 9: "نۆ",
              10: "دە"}


def alerts_sentence(alerts: list[dict[str, Any]]) -> str:
    """The active alerts read out in Sorani (list_alerts' own ``what_ckb``
    lines). qwen3:8b worded an empty list as «هیچ ئاگادارکردنەوەکەی نەدەرە. بۆ
    چی نەدەرە؟» in the live run (2026-09-25), so no model words this."""
    if not alerts:
        return "هیچ ئاگادارکردنەوەیەکی چالاکت نییە."
    count = len(alerts)
    spoken = _COUNT_CKB.get(count) or str(count).translate(_DIGITS_CKB)
    items = [str(a.get("what_ckb") or "").strip() for a in alerts[:3] if isinstance(a, dict)]
    items = [i for i in items if i]
    text = f"{spoken} ئاگادارکردنەوەی چالاکت هەیە"
    if items:
        text += ": " + "؛ ".join(items)
    if count > 3:
        text += f"؛ و {str(count - 3).translate(_DIGITS_CKB)}ی تر"
    return text + "."


# Results the user wants READ (a list, a page, what SAM remembers): a template
# would only say «I have it but cannot read it now», so a model words these.
_READ_TOOLS = frozenset({"list_alerts", "strategy_list", "strategy_get", "recall", "chart_state", "screen_look",
                         "web_search", "fetch_page", "theory_info", "more_tools", "files", "run_powershell"})


def own_sentence(name: str, args: dict[str, Any] | None, result: dict[str, Any] | None) -> str | None:
    """The tool's own honest Sorani sentence when that is a complete answer
    (the local brain then skips its wording round), else None."""
    if not result:
        return None
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    for key in ("declined", "blocked", "timeout", "cancelled"):
        if data.get(key) is True:   # the registry's flags; cancel_alert's data.cancelled is a COUNT
            return _SPECIAL[key]
    if name == "list_alerts" and result.get("ok") and str((args or {}).get("status") or "active") == "active":
        return alerts_sentence(data.get("alerts") or [])
    if name in _READ_TOOLS or (name == "window_control" and str((args or {}).get("action")) == "list"):
        return None
    own = str(data.get("summary_ckb") or "").strip()
    if own and is_arabic_script(own):
        return own
    summary = str(result.get("summary") or "").strip()
    if summary and len(summary) <= 240 and is_arabic_script(summary):
        return summary
    if result.get("ok") and (name in _TEMPLATES or name == "window_control"):
        return tool_sentence(name, args, result)
    return None


__all__ = ["tool_sentence", "own_sentence", "alerts_sentence", "DONE", "NOT_DONE", "NO_MODEL_YET", "NO_RESULTS"]
