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
    "web_search": ("گەڕانەکەم کرد، ئەنجامەکان لەسەر شاشەن.", "گەڕانەکە سەرکەوتوو نەبوو."),
    "fetch_page": ("پەڕەکەم خوێندەوە.", "نەمتوانی پەڕەکە بخوێنمەوە."),
    "build_project": ("پرۆژەکە دروست کرا.", "نەمتوانی پرۆژەکە تەواو بکەم."),
    "system_control": ("کرا.", "نەمتوانی ئەوە بکەم."),
    "remember": ("باشە، لەبیرم دەبێت.", "نەمتوانی لەبیری بکەم."),
    "recall": ("ئەوەی لەبیرم بوو لەسەر شاشەیە.", "هیچم لەبیر نییە دەربارەی ئەوە."),
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
    "list_alerts": ("لیستی ئاگادارکردنەوەکان لەسەر شاشەیە.", "نەمتوانی لیستەکە بهێنم."),
    "cancel_alert": ("ئاگادارکردنەوەکە هەڵوەشێنرایەوە.", "هیچ ئاگادارکردنەوەیەکی وا نەبوو."),
    "strategy_save": ("ستراتیژییەکە پاشەکەوت کرا.", "نەمتوانی ستراتیژییەکە پاشەکەوت بکەم."),
    "strategy_list": ("ستراتیژییەکان لەسەر شاشەن.", "نەمتوانی ستراتیژییەکان بهێنم."),
    "strategy_get": ("ستراتیژییەکە لەسەر شاشەیە.", "ئەو ستراتیژییەم نەدۆزییەوە."),
    "theory_info": ("زانیارییەکە لەسەر شاشەیە.", "ئەو تیۆرییەم نەدۆزییەوە."),
    "stop_all": ("هەموو شتێکم ڕاگرت.", "نەمتوانی ڕایبگرم."),
}

_WINDOW_OK = {"focus": "پەنجەرەکە هێنرایە پێشەوە.", "minimize": "پەنجەرەکە بچووک کرایەوە.",
              "maximize": "پەنجەرەکە گەورە کرا.", "restore": "پەنجەرەکە گەڕایەوە بارە ئاساییەکەی.",
              "close": "پەنجەرەکە داخرا.", "snap_left": "پەنجەرەکە بردرا بۆ لای چەپ.",
              "snap_right": "پەنجەرەکە بردرا بۆ لای ڕاست.", "list": "لیستی پەنجەرەکان لەسەر شاشەیە."}


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
        if data.get(key):
            return _SPECIAL[key]
    own = str(data.get("summary_ckb") or "").strip()
    if own and is_arabic_script(own):
        return own
    summary = str(result.get("summary") or "").strip()
    if summary and len(summary) <= 240 and is_arabic_script(summary):
        return summary
    if name == "window_control":
        if good:
            return _WINDOW_OK.get(str(args.get("action") or ""), DONE)
        return "نەمتوانی ئەو پەنجەرەیە بدۆزمەوە." if "No open window" in summary else NOT_DONE
    template = _TEMPLATES.get(name)
    if template is None:
        return DONE if good else NOT_DONE
    return template[0 if good else 1].format(name=_app_name(args))


__all__ = ["tool_sentence", "DONE", "NOT_DONE"]
