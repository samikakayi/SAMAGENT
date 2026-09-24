"""Labelled utterances for the no-AI fast path (sam/brain/intents.py).

Each row: (utterance, expected intent name or None, expected argument subset).
``None`` = must NOT be answered without a model (a question, a statement, a
negation, a condition, two actions, another tool, small talk). Labels say what
a person would mean, not what the matcher happens to do; a command the
matcher misses only lowers recall, a wrong firing lowers precision.
Written 2026-09-24 from the user's own phrasings (docs/CONTRACTS.md examples,
the evening's transcripts, KurdishTTS STT spellings) plus English forms.
"""

from __future__ import annotations

from typing import Any

Row = tuple[str, str | None, dict[str, Any]]

POSITIVE: list[Row] = [
    # --- price ---
    ("نرخی زێڕ چەندە", "price", {"symbol": "زێڕ"}),
    ("نرخی زێڕ چەندە؟", "price", {}),
    ("نەخنەشکی زێڕ چەندە", "price", {}),                   # KurdishTTS STT, 2026-09-24
    ("زێڕ چەندە", "price", {}),
    ("زێڕ ئێستا بە چەندە؟", "price", {}),
    ("گۆڵد لە چەندە", "price", {"symbol": "گۆڵد"}),
    ("نرخی گۆڵد", "price", {}),
    ("نرخی ئێستای زێڕ چەندە", "price", {}),
    ("نرخی بیتکۆین چەندە", "price", {"symbol": "بیتکۆین"}),
    ("بیتکۆین بە چەندە", "price", {}),
    ("سام، نرخی زێڕ چەندە؟", "price", {}),
    ("نرخی زیو چەندە", "price", {}),
    ("نرخی نەوت چەندە", "price", {}),
    ("گۆڵت چەندە", "price", {}),
    ("نرخی زێڕ چییە", "price", {}),
    ("نرخی زێڕەکە چەندە", "price", {}),
    ("نرخی زێڕ ئەمڕۆ چەندە", "price", {}),
    ("تکایە نرخی زێڕم پێ بڵێ", "price", {}),
    ("قیمەتی زێڕ چەندە", "price", {}),
    ("نرخي زيڕ چەندە", "price", {}),                        # Arabic ي in STT output
    ("how much is gold", "price", {"symbol": "gold"}),
    ("gold price", "price", {}),
    ("what's the gold price?", "price", {}),
    ("price of bitcoin", "price", {"symbol": "bitcoin"}),
    ("what is the price of gold now", "price", {}),
    # --- open TradingView ---
    ("ترەیدینگ ڤیو بکەرەوە", "open_tradingview", {}),
    ("تریدینگ ڤیو بکەرەوە", "open_tradingview", {}),
    ("ترێدینگ ڤیو بکەرەوە تکایە", "open_tradingview", {}),
    ("سام ترەیدینگ ڤیو بکەرەوە", "open_tradingview", {}),
    ("ترەیدینگ ڤیوم بۆ بکەرەوە", "open_tradingview", {}),
    ("ترەیدینگڤیو بکەرەوە", "open_tradingview", {}),
    ("ترەیدینگ ڤیو بکەرەوه", "open_tradingview", {}),       # word-final Arabic heh
    ("چارتەکەم بکەرەوە", "open_tradingview", {}),
    ("open tradingview", "open_tradingview", {}),
    ("open trading view please", "open_tradingview", {}),
    # --- open an app ---
    ("کرۆم بکەرەوە", "open_app", {"name": "Google Chrome"}),
    ("نۆتپاد بکەرەوە", "open_app", {"name": "Notepad"}),
    ("تێلێگرام بکەرەوە", "open_app", {"name": "Telegram"}),
    ("مێتاترەیدەر بکەرەوە", "open_app", {"name": "MetaTrader 5"}),
    ("ئێج بکەرەوە", "open_app", {"name": "Microsoft Edge"}),
    ("ڤی ئێس کۆد بکەرەوە", "open_app", {"name": "Visual Studio Code"}),
    ("واتساپ بکەرەوە", "open_app", {"name": "WhatsApp"}),
    ("یوتیوب بکەرەوە", "open_app", {"name": "YouTube"}),
    ("ئێکسڵ بکەرەوە", "open_app", {"name": "Excel"}),
    ("کرۆمەکە بکەرەوە", "open_app", {"name": "Google Chrome"}),
    ("سام، تکایە کرۆم بکەرەوە", "open_app", {}),
    ("باشە، کرۆم بکەرەوە", "open_app", {}),
    ("دەتوانیت کرۆم بکەیتەوە؟", "open_app", {}),
    ("open chrome", "open_app", {"name": "Google Chrome"}),
    ("open notepad", "open_app", {"name": "Notepad"}),
    ("launch telegram", "open_app", {}),
    ("can you open chrome please", "open_app", {}),
    # --- chart symbol / timeframe ---
    ("گۆڵد لەسەر ١٥ خولەک پیشان بدە", "set_chart", {"symbol": "گۆڵد", "timeframe": "M15"}),
    ("گۆڵد لەسەر پازدە خولەک پیشان بدە", "set_chart", {"timeframe": "M15"}),
    ("بیکە بە کاتژمێرێک", "set_chart", {"timeframe": "H1"}),
    ("چارتەکە بکە بە چوار کاتژمێر", "set_chart", {"timeframe": "H4"}),
    ("کاتی چارتەکە بگۆڕە بۆ یەک کاتژمێر", "set_chart", {"timeframe": "H1"}),
    ("بیتکۆین پیشان بدە", "set_chart", {"symbol": "بیتکۆین"}),
    ("زێڕ لەسەر ڕۆژانە پیشان بدە", "set_chart", {"timeframe": "D1"}),
    ("گۆڵد لەسەر ١٥ خولەک دابنێ", "set_chart", {"timeframe": "M15"}),
    ("چارتی زێڕ بکەرەوە", "set_chart", {"symbol": "زێڕ"}),
    ("زێڕ لەسەر پێنج خولەک", "set_chart", {"timeframe": "M5"}),
    ("switch the chart to 15 minutes", "set_chart", {"timeframe": "M15"}),
    ("show gold on the 1 hour chart", "set_chart", {"timeframe": "H1"}),
    ("change timeframe to 4h", "set_chart", {"timeframe": "H4"}),
    # --- analysis (+ drawing) ---
    ("شیکاری زێڕ بکە و ئاستەکان بکێشە", "analyze", {"symbol": "زێڕ", "draw": "full"}),
    ("زێڕ شی بکەرەوە", "analyze", {}),
    ("شیکاری گۆڵد بکە", "analyze", {}),
    ("شیکاری زێڕ بکە لەسەر ١٥ خولەک", "analyze", {"timeframes": ["M15"]}),
    ("بازاڕەکە شی بکەرەوە", "analyze", {}),
    ("شیکاری بیتکۆین بکە", "analyze", {"symbol": "بیتکۆین"}),
    ("analyze gold", "analyze", {}),
    ("analyze gold and draw the levels", "analyze", {}),
    # --- support / resistance levels ---
    ("هێڵی پشتگیری و بەرگری بکێشە", "draw_levels", {"draw": "levels"}),
    ("ئاستەکان بکێشە", "draw_levels", {}),
    ("ئاستەکانی پشتگیری و بەرگری لەسەر چارت بکێشە", "draw_levels", {}),
    ("هێڵی پشتگیری و بەرگری زێڕ بکێشە", "draw_levels", {"symbol": "زێڕ"}),
    ("draw support and resistance", "draw_levels", {}),
    ("draw the levels", "draw_levels", {}),
    # --- remove SAM's drawings ---
    ("هێڵەکانت بسڕەوە", "clear_drawings", {}),
    ("هێڵەکان بسڕەوە", "clear_drawings", {}),
    ("هەموو هێڵەکانت بسڕەوە", "clear_drawings", {}),
    ("کێشراوەکانت لابە", "clear_drawings", {}),
    ("نیشانەکانت بسڕەوە", "clear_drawings", {}),
    ("چارتەکە پاک بکەرەوە", "clear_drawings", {}),
    ("clear your drawings", "clear_drawings", {}),
    ("remove the lines", "clear_drawings", {}),
    # --- alerts ---
    ("ئاگادارکردنەوەکانم پیشان بدە", "list_alerts", {}),
    ("چ ئاگادارکردنەوەیەکم هەیە؟", "list_alerts", {}),
    ("ئاگادارکردنەوەکانم چین", "list_alerts", {}),
    ("لیستی ئاگادارکردنەوەکان", "list_alerts", {}),
    ("list my alerts", "list_alerts", {}),
    ("show my alerts", "list_alerts", {}),
    ("what alerts do i have", "list_alerts", {}),
    ("هەموو ئاگادارکردنەوەکان هەڵبوەشێنەوە", "cancel_alerts", {"alert_id": "all"}),
    ("ئاگادارکردنەوەکان بسڕەوە", "cancel_alerts", {"alert_id": "all"}),
    ("ئاگادارکردنەوەی ژمارە ٣ هەڵبوەشێنەوە", "cancel_alerts", {"alert_id": "3"}),
    ("cancel all alerts", "cancel_alerts", {"alert_id": "all"}),
    ("delete all my alerts", "cancel_alerts", {"alert_id": "all"}),
    ("cancel alert 3", "cancel_alerts", {"alert_id": "3"}),
    # --- stop ---
    ("بوەستە", "stop", {}),
    ("ڕاوەستە", "stop", {}),
    ("بەسە", "stop", {}),
    ("هەمووی ڕابگرە", "stop", {}),
    ("بێدەنگ بە", "stop", {}),
    ("سام بوەستە", "stop", {}),
    ("stop", "stop", {}),
    ("stop everything", "stop", {}),
]

NEGATIVE: list[Row] = [
    # questions and statements about gold (not commands)
    ("بۆچی زێڕ دابەزی", None, {}),
    ("زێڕ چییە", None, {}),
    ("پێت وایە زێڕ بەرز دەبێتەوە", None, {}),
    ("نرخی زێڕ بەرز دەبێتەوە؟", None, {}),
    ("ئایا ئێستا کاتی باشە بۆ کڕینی زێڕ", None, {}),
    ("زێڕ باشە بۆ کڕین؟", None, {}),
    ("زێڕ لە ٢٠٢٠ چەندە بوو", None, {}),
    ("دوێنێ زێڕ چەندە بوو", None, {}),
    ("ئەگەر زێڕ گەیشتە ٤٣٠٠ ئاگادارم بکەرەوە", None, {}),
    ("ئەگەر نرخی زێڕ دابەزی پێم بڵێ", None, {}),
    ("زێڕ چی بەسەر هات", None, {}),
    ("دەربارەی زێڕ قسە بکە", None, {}),
    ("زێڕ بە چ هۆیەک دابەزیوە", None, {}),
    ("گۆڵد چۆنە ئەمڕۆ", None, {}),
    ("بازاڕ چۆنە", None, {}),
    ("پێشبینی نرخی زێڕ بکە", None, {}),
    ("چەند زێڕم هەیە", None, {}),
    ("ماوەی زێڕ چەندە", None, {}),
    ("نرخی زێڕ لە بەغدا چەندە", None, {}),
    ("نرخی زێڕی ٢١ چەندە", None, {}),
    ("ئێستا زێڕ لە ٤٢٩٠ە", None, {}),
    ("زێڕ بەرزبووەوە", None, {}),
    ("زێڕ", None, {}),
    ("what is gold", None, {}),
    ("why is gold falling", None, {}),
    ("will gold go up", None, {}),
    ("should i buy gold", None, {}),
    ("is gold a good investment", None, {}),
    ("how is gold doing today", None, {}),
    ("tell me about gold", None, {}),
    ("what do you think about the gold price", None, {}),
    ("gold is at 4290", None, {}),
    ("i think gold will fall", None, {}),
    # trading orders are never run
    ("زێڕ بکڕە", None, {}),
    ("زێڕ بفرۆشە", None, {}),
    ("buy gold now", None, {}),
    ("sell bitcoin", None, {}),
    # other tools / multi-step requests the model must handle
    ("دەنگەکە بەرز بکەرەوە", None, {}),
    ("پەنجەرەکە دابخە", None, {}),
    ("ئیمەیڵێک بنووسە", None, {}),
    ("یوتیوب بکەرەوە و گۆرانییەک لێبدە", None, {}),
    ("کرۆم بکەرەوە و بڕۆ بۆ گووگڵ", None, {}),
    ("ترەیدینگ ڤیو بکەرەوە و زێڕ پیشان بدە", None, {}),
    ("شیکاری زێڕ بکە بە ستراتیژییەکەم", None, {}),
    ("ستراتیژییەکانم پیشان بدە", None, {}),
    ("لەبیرت بێت من تەنها کاتی لەندەن ترەید دەکەم", None, {}),
    ("وێبسایتێک دروست بکە", None, {}),
    ("فایلێک بسڕەوە", None, {}),
    ("وێنەیەک بکێشە", None, {}),
    ("هێڵێک بکێشە لە ٤٣٠٠", None, {}),
    ("ئاگادارم بکەرەوە کاتێک زێڕ گەیشتە ٤٣٠٠", None, {}),
    ("ئاگادارکردنەوەیەک دابنێ بۆ زێڕ", None, {}),
    ("ئاگادارکردنەوەکە هەڵبوەشێنەوە", None, {}),
    ("ئاگادارکردنەوەی زێڕ هەڵبوەشێنەوە", None, {}),
    ("بیانسڕەوە", None, {}),
    ("بیسڕەوە", None, {}),
    ("کرۆم دابخە", None, {}),
    ("پەنجەرەی کرۆم بهێنە پێشەوە", None, {}),
    ("ماڵپەڕی گووگڵ بکەرەوە", None, {}),
    ("ئەم فایلە بکەرەوە", None, {}),
    ("ستۆپ لۆس لە کوێ دابنێم", None, {}),
    ("draw a cat", None, {}),
    ("draw a line at 4300", None, {}),
    ("set an alert for gold at 4300", None, {}),
    ("remove the last alert", None, {}),
    ("clear the cache", None, {}),
    ("delete the file", None, {}),
    ("open the door", None, {}),
    ("open a new tab", None, {}),
    ("open my email", None, {}),
    ("close chrome", None, {}),
    ("list my files", None, {}),
    ("show me the news", None, {}),
    ("show me a joke", None, {}),
    ("show my strategies", None, {}),
    ("can you analyze my strategy", None, {}),
    ("stop the music", None, {}),
    ("stop loss چییە", None, {}),
    # negations, past tense, questions about SAM's own actions
    ("کرۆم مەکەرەوە", None, {}),
    ("ترەیدینگ ڤیو مەکەرەوە", None, {}),
    ("هێڵەکان مەسڕەوە", None, {}),
    ("دوێنێ کرۆمم کردەوە", None, {}),
    ("کرۆم کراوەیە؟", None, {}),
    ("کرۆم باشترە یان ئێج", None, {}),
    ("بۆچی ئەو هێڵانەت کێشا", None, {}),
    ("i opened tradingview yesterday", None, {}),
    ("don't open chrome", None, {}),
    ("did you open chrome", None, {}),
    ("why did you draw those lines", None, {}),
    ("are my alerts active", None, {}),
    ("how many alerts fired today", None, {}),
    ("chrome is slow", None, {}),
    ("the chart looks weird", None, {}),
    ("my favorite app is chrome", None, {}),
    ("chrome", None, {}),
    # fragments
    ("١٥ خولەک", None, {}),
    ("لەسەر ١٥ خولەک", None, {}),
    ("ترەیدینگ ڤیو", None, {}),
    ("ترەیدینگ ڤیو چییە", None, {}),
    ("سام", None, {}),
    ("زێڕ بکە", None, {}),
    ("بەسە بۆ ئەمڕۆ؟", None, {}),
    # small talk and other questions
    ("سڵاو سام چۆنی", None, {}),
    ("سڵاو", None, {}),
    ("سوپاس", None, {}),
    ("ئەمڕۆ هەوا چۆنە", None, {}),
    ("کاتژمێر چەندە", None, {}),
    ("ئێستا کاتژمێر چەندە", None, {}),
    ("چەند ساڵتە", None, {}),
    ("نوکتەیەکم بۆ بڵێ", None, {}),
    ("گوێ بگرە", None, {}),
    ("what's the time", None, {}),
    ("what is the best timeframe", None, {}),
    ("hello sam", None, {}),
    ("thank you", None, {}),
]

# Written after the grammar, BEFORE any tuning on it (first run 2026-09-24:
# see HELD_OUT_FIRST_RUN); kept in the corpus as is.
HELD_OUT: list[Row] = [
    ("زێڕ بە چەندە ئێستا", "price", {}),
    ("نرخی ئاڵتوون چەندە", "price", {}),
    ("سام گیان نرخی زێڕ چەندە", "price", {}),
    ("نرخی یۆرۆ دۆلار چەندە", "price", {}),
    ("bitcoin price now", "price", {}),
    ("what's bitcoin at", "price", {}),
    ("سڵاو، نرخی زێڕ چەندە؟", "price", {}),
    ("ترەیدینگ ڤیوەکە بکەرەوە", "open_tradingview", {}),
    ("ترەیدینگ ڤیو هەڵبکە", "open_tradingview", {}),
    ("کرۆم هەڵبکە", "open_app", {}),
    ("تێلێگرامم بۆ بکەرەوە", "open_app", {}),
    ("ئێکسڵ بکەرەوە بۆم", "open_app", {}),
    ("ئەگەر دەکرێت کرۆم بکەرەوە", "open_app", {}),
    ("open whatsapp", "open_app", {}),
    ("start spotify", "open_app", {}),
    ("گۆڵد بکە بە پازدە خولەک", "set_chart", {"timeframe": "M15"}),
    ("چارتەکە بکە بە ڕۆژانە", "set_chart", {"timeframe": "D1"}),
    ("timeframe 15 minutes", "set_chart", {"timeframe": "M15"}),
    ("show bitcoin", "set_chart", {}),
    ("put the chart on 4 hours", "set_chart", {"timeframe": "H4"}),
    ("زێڕ لەسەر ٤ کاتژمێر پیشان بدە", "set_chart", {"timeframe": "H4"}),
    ("شیکاری زێڕ بکە و هێڵەکان بکێشە", "analyze", {}),
    ("شیکاری گۆڵد بکە لەسەر کاتژمێرێک", "analyze", {"timeframes": ["H1"]}),
    ("analyse bitcoin", "analyze", {}),
    ("ئاستەکانی پشتگیری و بەرگری بکێشە", "draw_levels", {}),
    ("draw support and resistance levels for gold", "draw_levels", {}),
    ("هێڵەکانت لەسەر چارت بسڕەوە", "clear_drawings", {}),
    ("هەموو کێشراوەکانت بسڕەوە", "clear_drawings", {}),
    ("delete your lines", "clear_drawings", {}),
    ("ئاگادارییەکانم پیشان بدە", "list_alerts", {}),
    ("ئالێرتەکانم پیشان بدە", "list_alerts", {}),
    ("show all alerts", "list_alerts", {}),
    ("هەموو ئالێرتەکان بسڕەوە", "cancel_alerts", {"alert_id": "all"}),
    ("remove all alerts", "cancel_alerts", {"alert_id": "all"}),
    ("ڕابگرە", "stop", {}),
    ("بەسە ئیتر", "stop", {}),
    ("be quiet", "stop", {}),
    ("cancel", "stop", {}),
    ("نرخی زێڕ لە بازاڕی سلێمانی چەندە", None, {}),
    ("بۆچی بیتکۆین بەرز بووەوە", None, {}),
    ("زێڕ دابەزیوە؟", None, {}),
    ("چ کاتێک زێڕ بکڕم", None, {}),
    ("ئایا زێڕ دەگاتە ٥٠٠٠", None, {}),
    ("زێڕ چەند دابەزی ئەمڕۆ", None, {}),
    ("نرخی زێڕ چەندە بوو دوێنێ", None, {}),
    ("کەی کرۆمم کردەوە", None, {}),
    ("ئایا ترەیدینگ ڤیو کراوەیە", None, {}),
    ("پێویست ناکات کرۆم بکەیتەوە", None, {}),
    ("کرۆم نەکەیتەوە", None, {}),
    ("کرۆم و ئێج بکەرەوە", None, {}),
    ("ئاگادارکردنەوەکانم زۆرن", None, {}),
    ("ئاگادارکردنەوەکان کار ناکەن", None, {}),
    ("بۆچی هێڵەکانت سڕییەوە", None, {}),
    ("هێڵەکانت جوانن", None, {}),
    ("چارتەکە زۆر هێواشە", None, {}),
    ("پێم بڵێ زێڕ بۆچی دابەزی", None, {}),
    ("نرخی زێڕ بنووسە لە فایلێک", None, {}),
    ("نرخی زێڕ بنێرە بۆ تێلێگرام", None, {}),
    ("stop talking about gold", None, {}),
    ("open chrome and search for gold price", None, {}),
    ("what's the price target for gold", None, {}),
    ("gold price prediction", None, {}),
    ("is tradingview open", None, {}),
    ("why is my chart on 15 minutes", None, {}),
    ("the price of gold is too high", None, {}),
    ("زێڕ و زیو چەندەن", None, {}),
    ("گۆڵد", None, {}),
]
# The held-out rows as the matcher first scored them (before the recall fixes
# listed in the fast-path report): precision / recall on HELD_OUT alone.
HELD_OUT_FIRST_RUN: dict[str, Any] = {"rows": 67, "positive": 38, "negative": 29, "tp": 33, "fp": 0, "fn": 5,
                                      "precision": 1.0, "recall": 0.8684}

# A second held-out set: the independent reviewer's adversarial rows (2026-09-25),
# written before running them. 64 non-commands (or commands whose fast-path action
# would be wrong) and 13 commands. First run on the matcher as built: 21 of the 64
# negatives fired (every singular alert cancelled ALL alerts, prices were read as
# timeframes, «نەخۆشی/نەرمی زێڕ چەندە» passed as a mangled price word).
REVIEW_HELD_OUT: list[Row] = [
    # a single alert never cancels them all without a number / an explicit "all"
    ("cancel the alert", None, {}), ("remove my alert", None, {}), ("delete the alert", None, {}),
    ("clear the alert", None, {}), ("cancel my alarm", None, {}), ("stop the alarm", None, {}),
    ("stop the alert", None, {}), ("delete my alarm", None, {}), ("stop alerting me", None, {}),
    # a price is not a timeframe
    ("draw support at 60", None, {}), ("draw resistance at 30", None, {}), ("draw lines at 240", None, {}),
    ("put support at 60", None, {}), ("هێڵی پشتگیری لە ٦٠ بکێشە", None, {}), ("هێڵی بەرگری لە ٣٠ بکێشە", None, {}),
    ("draw support for oil at 60", None, {}), ("هێڵی پشتگیری نەوت لە ٦٠ بکێشە", None, {}),
    # words that are not a price word before an instrument
    ("نەخۆشی زێڕ چەندە", None, {}), ("نەرمی زێڕ چەندە", None, {}), ("نزیکترین زێڕ چەندە", None, {}),
    # fragments / questions
    ("چەند زێڕ", None, {}), ("how much gold", None, {}), ("how much gold can i buy", None, {}),
    ("what is gold price doing", None, {}), ("check gold", None, {}), ("gold now", None, {}),
    ("gold today", None, {}), ("open", None, {}), ("launch", None, {}), ("زێڕ چەندی ماوە", None, {}),
    ("زێڕ چەند پۆینت ڕۆیشت", None, {}), ("نرخی زێڕ بە دینار چەندە", None, {}),
    # negation / condition / reported speech / sequencing
    ("کرۆم نەکرایەوە", None, {}), ("نامەوێت کرۆم بکەیتەوە", None, {}), ("no need to open chrome", None, {}),
    ("never open chrome", None, {}), ("please don't clear the lines", None, {}), ("do not delete my alerts", None, {}),
    ("unless gold falls clear the lines", None, {}), ("once gold hits 4300 clear the lines", None, {}),
    ("ئەگەرنا هێڵەکان بسڕەوە", None, {}), ("تا زێڕ دەگاتە ٤٣٠٠ بوەستە", None, {}),
    ("هێڵەکانت بسڕەوە یان نا", None, {}), ("بیرم چووە کرۆم بکەمەوە", None, {}),
    ("هەموو ئاگادارکردنەوەکان بسڕەوە جگە لە زێڕ", None, {}), ("ئاگادارکردنەوەکان بسڕەوە بەڵام زێڕ نا", None, {}),
    ("ئەحمەد گوتی کرۆم بکەرەوە", None, {}), ("he said open chrome", None, {}), ("my friend said stop", None, {}),
    ("someone told me to delete all alerts", None, {}), ("open chrome also", None, {}),
    ("open chrome plus notepad", None, {}), ("کرۆم بکەرەوە هەروەها نۆتپاد", None, {}),
    ("کرۆم بکەرەوە لەگەڵ نۆتپاد", None, {}), ("کرۆم نۆتپاد بکەرەوە", None, {}), ("cancel alert 3 and 4", None, {}),
    ("delete everything on the chart", None, {}), ("delete all lines and alerts", None, {}),
    ("clear everything", None, {}), ("stop listening", None, {}), ("بوەستە بۆ ماوەیەک", None, {}),
    ("زێڕی ٢٤ چەندە", None, {}), ("هێڵەکانی زێڕ بسڕەوە", None, {}), ("how much is gold worth in dinars", None, {}),
    # true commands (recall)
    ("cancel all my alerts", "cancel_alerts", {"alert_id": "all"}), ("delete alert 7", "cancel_alerts", {"alert_id": "7"}),
    ("stop", "stop", {}), ("cancel that", "stop", {}), ("کپ بە", "stop", {}),
    ("open task manager", "open_app", {"name": "Task Manager"}),
    ("زیو لەسەر ٣٠ خولەک پیشان بدە", "set_chart", {"timeframe": "M30"}),
    ("show gold on 4 hours", "set_chart", {"timeframe": "H4"}), ("نرخی نەوت چەندە", "price", {}),
    ("analyze silver", "analyze", {}), ("هێڵەکانت لابە", "clear_drawings", {}),
    ("ئالێرتەکانم پیشان بدە", "list_alerts", {}), ("ترەیدینگ ڤیو بکەرەوە سام", "open_tradingview", {}),
]
REVIEW_FIRST_RUN: dict[str, Any] = {"rows": 77, "positive": 13, "negative": 64, "tp": 13, "fp": 21, "fn": 0,
                                    "precision": 0.38, "recall": 1.0}

# Written with the fixes for the reviewer's rows (so not held out): the price-as-
# timeframe and bare-noun probes of the review's wider run, and the commands that
# must keep working next to them.
REVIEW_PROBES: list[Row] = [
    ("gold 240", None, {}), ("بیتکۆین ٦٠", None, {}), ("show me 30", None, {}), ("switch to 240", None, {}),
    ("زێڕ لەسەر ٦٠", None, {}), ("gold on 60", None, {}), ("چارتەکە ٦٠", None, {}), ("gold 15", None, {}),
    ("draw support levels at 120", None, {}), ("analysis", None, {}), ("analyze", None, {}),
    ("market analysis", None, {}), ("delete the chart", None, {}), ("remove the chart", None, {}),
    ("stop alerts", None, {}), ("alerts stop", None, {}), ("stop all alerts", None, {}),
    ("change the timeframe to 60", "set_chart", {"timeframe": "H1"}),
    ("تایمفرەیمەکە بکە بە ٢٤٠", "set_chart", {"timeframe": "H4"}),
    ("analyze the market", "analyze", {}), ("چارتەکە شی بکەرەوە", "analyze", {}),
    ("clear the chart", "clear_drawings", {}), ("چارتەکە پاک بکەرەوە", "clear_drawings", {}),
    ("delete my alerts", "cancel_alerts", {"alert_id": "all"}),
    ("cancel the alert number 4", "cancel_alerts", {"alert_id": "4"}),
    ("draw support and resistance on 4 hours", "draw_levels", {"timeframes": ["H4"]}),
    # the phrasing qwen3:8b answered from memory in the live run (2026-09-25)
    ("پێم بڵێ زێڕ ئێستا بە چەند مامەڵە دەکرێت", "price", {"symbol": "زێڕ"}),
    ("بیتکۆین لە چ نرخێک مامەڵە دەکرێت؟", "price", {"symbol": "بیتکۆین"}),
    ("زێڕ بە چەند مامەڵە بکەم", None, {}),
    ("زێڕ بە چەند مامەڵە دەکرێت لە سلێمانی", None, {}),
]

CORPUS: list[Row] = POSITIVE + NEGATIVE + HELD_OUT + REVIEW_HELD_OUT + REVIEW_PROBES


def evaluate(match: Any, rows: list[Row] | None = None) -> dict[str, Any]:
    """Precision / recall of ``match`` (text -> Intent | None) over ``rows`` (default CORPUS)."""
    rows = CORPUS if rows is None else rows
    tp = fp = fn = 0
    wrong: list[tuple[str, str | None, str | None]] = []
    for text, expected, args in rows:
        intent = match(text)
        got = intent.name if intent is not None else None
        good_args = intent is not None and all(intent.args.get(k) == v for k, v in args.items())
        if got is not None and got == expected and good_args:
            tp += 1
        elif got is not None:
            fp += 1
            wrong.append((text, expected, got))
            if expected is not None:
                fn += 1
        elif expected is not None:
            fn += 1
            wrong.append((text, expected, None))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    positive = sum(1 for _, expected, _ in rows if expected is not None)
    return {"rows": len(rows), "positive": positive, "negative": len(rows) - positive, "tp": tp, "fp": fp,
            "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4), "wrong": wrong}


__all__ = ["CORPUS", "POSITIVE", "NEGATIVE", "HELD_OUT", "HELD_OUT_FIRST_RUN", "REVIEW_HELD_OUT", "REVIEW_FIRST_RUN",
           "REVIEW_PROBES", "evaluate"]
