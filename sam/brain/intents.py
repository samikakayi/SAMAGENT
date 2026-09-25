"""Deterministic Sorani + English command matcher for the no-AI fast path.

Why: free quotas are tiny (the user's evening test on 2026-09-24 used every
free model up; «نرخی زێڕ چەندە» then got «ناتوانم پەیوەندی بە مۆدێلەکانەوە
بکەم») and even a healthy model round costs 1-6 s. The most common commands
map to exactly one tool with obvious arguments, so they need no model at all.

Precision over recall: a command is matched only when EVERY word of the
utterance is accounted for by one intent's grammar (verbs, objects, fillers,
one instrument / timeframe / app name). Anything else -- a question about
gold, a negation, a past tense, a condition («ئەگەر ...»), two actions joined
by «و», a strategy name, an unknown word -- goes to the model as before.
The labelled corpus (tests/fastpath_corpus.py, ~400 utterances including an
independent reviewer's adversarial rows) measures it -- on that corpus only.
Rules the review added (2026-09-25): one alert never means every alert, a
bare number is not a timeframe (it may be a price), mangled price words are
only spellings heard from STT, and bare nouns ("analysis") need an object.

Spellings: text is normalised with ``normalize_ckb`` (Arabic ي/ك, digits,
punctuation), a word-final Arabic heh counts as ە (STT writes «بکەرەوه»), and
a mangled price word right before an instrument is accepted (KurdishTTS STT
wrote «نرخی زێڕ» as «نەخنەشکی زێڕ» on 2026-09-24).

The user's live session (2026-09-25) added: STT spellings of TradingView
(«ترێیت ملیۆم») and of the chart («چار», «چاوتی»), «لۆ» for «بۆ», «بڕۆ سەر ...»
("go to ..."), leading/trailing fillers and insults a frustrated user says
(«وەڵاهی», «جارێ», «کوڕە», «قەشمەر», «... چی دەکەی؟») -- they carry no command and
are skipped only at the edges --, TradingView's own 3/45-minute and 2/3-hour
intervals for the chart, «گوڵ» as gold only next to a chart/price/timeframe
word, and "SAM, be quiet" («دەنگت بنەکەرە», «بێدەنگ بە», «دەنگ مەکە») as SAM's
own voice (``quiet`` -> stop_speaking), never the Windows volume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..textnorm import normalize_ckb

MAX_WORDS = 12


@dataclass(frozen=True)
class Intent:
    """A matched command: one tool call with its arguments."""

    name: str                     # price|open_tradingview|open_app|set_chart|analyze|draw_levels|clear_drawings|list_alerts|cancel_alerts|stop|quiet
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    said: str = ""                # the user's word for the app/instrument (for the reply)
    slow: bool = False            # worth a spoken acknowledgement first (analysis)
    chart_symbol: bool = False    # no instrument named: use the one on the chart


def _n(words: Iterable[str]) -> frozenset[str]:
    return frozenset(_norm_token(w) for w in words)


def _norm_token(word: str) -> str:
    value = normalize_ckb(word, strip_punct=True)
    return re.sub("ه\\b", "ە", value.replace("ھ", "ه"))


def _phrases(items: Iterable[str]) -> list[tuple[str, ...]]:
    out = {tuple(_norm_token(w) for w in item.split()) for item in items}
    return sorted(out, key=len, reverse=True)


# --- vocabulary (compared after normalisation) --------------------------------------------------------
VOCATIVE = _phrases(["سام", "sam", "هەی سام", "ئەی سام", "hey sam", "hi sam", "سام گیان", "کاکە سام", "ok sam",
                     "okay sam"])
POLITE = _phrases(["تکایە", "تکایه", "please", "pls", "plz", "زەحمەت نەبێت", "لە زەحمەت", "ئەگەر دەکرێت",
                   "can you", "could you", "would you", "can u", "دەتوانیت", "دەتوانی", "دەکرێت", "بۆم", "for me",
                   "now please", "ئێستا تکایە", "لۆم", "بۆ من", "لۆ من", "یەکسەر", "خێرا", "بە خێرایی", "زوو",
                   "right now", "quickly"])
# Words that make an utterance something other than a plain command.
REJECT = _n([
    # questions / opinions
    "بۆچی", "چۆن", "چۆنە", "چۆنی", "کەی", "ئایا", "پێت", "پێتوایە", "وایە", "دەزانیت", "دەزانی", "بزانم", "بزانین",
    "باشترین", "خراپە", "دەبێت", "دەبێتەوە", "بەرز", "نزم", "دادەبەزێت", "بەرزدەبێتەوە", "پێشبینی",
    "why", "when", "should", "would", "will", "think", "maybe", "predict", "forecast", "going", "rise",
    "fall", "drop", "buy", "sell", "better", "good", "bad",
    # conditions / time / sequence
    "ئەگەر", "کاتێک", "کە", "دوای", "دواتر", "پاشان", "پێش", "پێشتر", "دوێنێ", "سبەی", "سبەینێ", "بەیانی", "ئێوارە",
    "if", "then", "after", "before", "later", "yesterday", "tomorrow", "until",
    # negation
    "نا", "نەخێر", "مەکە", "نەکە", "نەکەیت", "نییە", "نیە", "don", "dont", "not", "never", "no", "stopped",
    # past / reported
    "کرد", "کردەوە", "کرایەوە", "کردبوو", "بوو", "بوون", "بووە", "کێشا", "سڕییەوە", "سڕیەوە", "گوت", "وتی", "وت",
    "opened", "closed", "was", "were", "did", "had", "said",
    # trading orders are never a command SAM runs
    "بکڕە", "بفرۆشە", "کڕین", "فرۆشتن", "order", "trade",
])
AND = _n(["و", "and", "&"])
# Words a frustrated user puts around a command («کوڕە ...», «قەشمەر ...», live session
# 2026-09-25): interjections, address words and insults. They carry no command, so they are
# skipped -- only at the start or the end of the utterance -- and the command is done calmly.
INTERJECTIONS = ["وەڵاهی", "وەڵڵاهی", "وەڵا", "وەڵڵا", "واڵا", "واڵڵا", "والله", "جارێ", "جاری", "کوڕە", "کورە",
                 "کوڕی", "برا", "براکەم", "برام", "هاوڕێ", "دەی", "دەسا", "یەڵا", "یاڵا", "یالا", "ئەرێ", "ها",
                 "قەشمەر", "کڕۆکڕۆک", "گەمژە", "گێل", "گەوج", "حیمار", "حمار", "کەر", "ئەحمەق", "احمق", "بێ ئەقڵ",
                 "بێئەقڵ", "ئەی کەر", "ئەی گەمژە", "stupid", "idiot", "dude", "man", "bro", "come on"]
# Leading words that carry no meaning in a spoken command («باشە، کرۆم بکەرەوە»).
LEAD = _n(["باشە", "ئا", "ئەی", "ئەها", "ئێستا", "ok", "okay", "so", "well", "now", "yes", "بەڵێ", "سڵاو",
           "hello", "hi", *[w for w in INTERJECTIONS if " " not in w]])
LEAD_PHRASES = _phrases([w for w in INTERJECTIONS if " " in w])
# Trailing words: the same, plus the frustrated «... چی دەکەی؟» ("what are you doing?!") after a command.
TAIL = _phrases([*INTERJECTIONS, "چی دەکەی", "چی دەکەیت", "چی دەکەیتن", "ئیتر", "what are you doing"])

OPEN = _phrases(["بکەرەوە", "بکەوە", "بکەرەو", "بیکەرەوە", "بکەیتەوە", "بکرێتەوە", "بکەیتەوە", "هەڵبکە", "هەلبکە",
                 "بکرەوە", "بکەروە", "بکەو", "بیکەوە", "بیکرەوە",
                 "بێنە", "بهێنە", "بێنە پێشەوە", "بهێنە پێشەوە", "بکە بەرز", "open", "launch", "start", "run",
                 "open up", "bring up", "fire up"])
APP_FILLER = _n(["ئەپی", "ئەپ", "ئەپەکە", "ئاپی", "ئاپ", "ئاپەکە", "بەرنامەی", "بەرنامە", "بەرنامەکە", "پرۆگرامی",
                 "پرۆگرام", "پرۆگرامەکە", "ئەپلیکەیشنی", "ئەپلیکەیشن", "the", "app", "application", "program", "up",
                 "a", "new", "me", "بۆ", "لۆ"])
SHOW = _phrases(["پیشان بدە", "پیشانم بدە", "پیشانبدە", "پیشانمبدە", "نیشان بدە", "نیشانم بدە", "نیشانبدە",
                 "دابنێ", "بیکە بە", "بکە بە", "بیکە", "بکە", "بگۆڕە بۆ", "بیگۆڕە بۆ", "بگۆڕە", "بیگۆڕە",
                 "بگۆڕە لۆ", "بیگۆڕە لۆ", "بیکە لۆ", "بکە لۆ",
                 "بکەرەوە", "بکەوە", "بهێنە", "بێنە", "show", "show me", "switch to", "switch", "change to",
                 "change", "set", "set to", "put", "open", "go to", "load", "display", "make it",
                 # «بڕۆ سەر چارتی زێڕ» / «بچۆ سەر ...» ("go to ...") and «بیخەرە سەر ...» ("put it on ...")
                 "بڕۆ سەر", "بڕۆ بۆ", "بڕۆ لۆ", "بڕۆرە سەر", "برۆ سەر", "برۆ بۆ", "بچۆ سەر", "بچۆ بۆ", "بچۆرە سەر",
                 "بڕۆ", "برۆ", "بچۆ", "بیخەرە سەر", "بخەرە سەر", "بیخە سەر", "بخە سەر", "بیبە سەر", "ببە سەر",
                 "بیبە بۆ", "ببە بۆ"])
WEAK_SHOW = frozenset(_phrases(["دابنێ", "بیکە", "بکە", "set", "put", "change", "make it", "load", "بڕۆ", "برۆ",
                                "بچۆ"]))
# The chart as the user and KurdishTTS STT say it: «چار», «چارت», «چاوت», «چارتی», «چاوتی» (2026-09-25).
CHART_NOUNS = _n(["چارت", "چارتەکە", "چارتەکەم", "چارتەکەی", "چارتی", "چار", "چاوت", "چاوتی", "چاوتەکە",
                  "چاوتەکەم", "chart"])
CHART = CHART_NOUNS | _n(["کاتی", "تایمفرەیم", "تایمفرەیمی", "تایمفرەیمەکە", "سیمبۆڵ", "سیمبۆڵەکە", "سیمبۆڵی", "the",
                          "timeframe", "time", "frame", "symbol", "my", "tradingview", "a", "سەر"])
TF_PREP = _n(["لەسەر", "بۆ", "لۆ", "بە", "لە", "on", "to", "at", "in", "into"])
# "go to the chart" after opening TradingView («ترێیت ملیۆم لۆ بکەوە بڕۆ سەر چار»).
GO = _phrases(["بڕۆ سەر", "بڕۆ بۆ", "بڕۆ لۆ", "بڕۆرە سەر", "برۆ سەر", "بچۆ سەر", "بچۆ بۆ", "بچۆرە سەر", "go to",
               "switch to"])
# Words that make a bare number a timeframe (set_chart only).
TF_WORDS = _n(["تایمفرەیم", "تایمفرەیمی", "تایمفرەیمەکە", "کاتی", "timeframe", "frame"])
PRICE = _n(["نرخی", "نرخ", "نەرخی", "نەرخ", "نرخەکەی", "نرخەکە", "نرخێ", "price", "prices", "rate", "quote",
            "the", "of", "is", "current", "s"])
HOW_MUCH = _phrases(["چەندە", "چەند", "بە چەندە", "لە چەندە", "لە چەندایە", "بەچەندە", "چ نرخێکە",
                     "لە چ نرخێکە", "لە چ نرخێکدایە", "چ نرخێکدایە", "how much", "how much is",
                     # «زێڕ بە چەند مامەڵە دەکرێت» ("what is gold traded at"; the local brain
                     # answered this phrasing from memory with a made-up price, 2026-09-25)
                     # («دەکرێت» itself is consumed as a politeness word, see POLITE)
                     "بە چەند مامەڵە", "بە چەندە مامەڵە", "لە چ نرخێک مامەڵە", "لە چەند مامەڵە"])
# "tell me ..." before a price question (with or without a price word).
TELL_ME = _phrases(["پێم بڵێ", "پێ بڵێ", "بۆم بڵێ", "بڵێ", "tell me"])
# "what is ..." / «... چییە» ask for a price only next to a price word («زێڕ چییە» = what is gold?).
WHAT_IS = _phrases(["چییە", "چیە", "چی یە", "پێم بڵێ", "پێ بڵێ", "بڵێ", "بزانە", "what is", "whats", "what s",
                    "what", "tell me", "give me", "check", "get"])
# The weak forms ask a quantity unless a price word is there («چەند زێڕ», "how much gold").
WEAK_HOW_MUCH = frozenset(_phrases(["چەند", "how much"]))
# KurdishTTS STT spellings of «نرخی» heard on this PC (2026-09-24). A new one is added
# here when it is seen in a transcript, never guessed from its shape.
MANGLED_PRICE = _n(["نەخنەشکی"])
PRICE_CORE = _n(["نرخی", "نرخ", "نەرخی", "نەرخ", "نرخەکەی", "نرخەکە", "قیمەتی", "قیمەت", "price", "prices", "quote"])
NOW = _n(["ئێستا", "ئێستای", "ئەمڕۆ", "ئەمڕۆی", "now", "right", "today", "currently", "live"])
ANALYZE = _phrases(["شیکاری", "شیکار", "شیکردنەوە", "شیکردنەوەی", "شی", "analyze", "analyse", "analysis",
                    "analysis of", "analyze the", "analyse the", "do an analysis of", "do analysis on"])
ANALYSIS_NOUNS = frozenset(_phrases(["analysis", "analysis of"]))
# What "analyze ..." may take without an instrument ("analyze the market", «چارتەکە شی بکەرەوە»).
ANALYZE_OBJECT = _n(["market", "chart", "بازاڕ", "بازاڕەکە", "بازار", "بازارەکە", "چارت", "چارتەکە", "چارتەکەم"])
ANALYZE_VERB = _phrases(["بکە", "بکەرەوە", "بکەوە", "بکەیت", "بکەیتەوە", "بۆ بکە", "شیبکەرەوە"])
ANALYZE_ONE = _phrases(["شیبکەرەوە", "شیکاربکە", "شیکاریبکە"])
MARKET = _n(["بازاڕ", "بازاڕەکە", "بازار", "بازارەکە", "market", "the"])
DRAW = _phrases(["بکێشە", "بیکێشە", "بیانکێشە", "بکێشە لەسەر چارت", "بکێشە لەسەر چارتەکە", "لەسەر چارت بکێشە",
                 "لەسەر چارتەکە بکێشە", "لەسەر چارت بیانکێشە", "لەسەر چارتەکە بیانکێشە", "دیاری بکە", "دابنێ",
                 "draw", "draw them", "draw it", "mark", "plot", "put", "on the chart", "on chart"])
LEVELS = _n(["هێڵی", "هێڵەکانی", "هێڵەکان", "هێڵ", "ئاستی", "ئاستەکانی", "ئاستەکان", "ئاست", "ئاستە", "ناوچەکانی",
             "پشتگیری", "بەرگری", "سەپۆرت", "ڕەزستەنس", "ڕێزیستەنس", "گرنگەکان", "سەرەکییەکان", "levels", "level",
             "lines", "line", "support", "resistance", "key", "the", "sr", "s", "r"])
CLEAR = _phrases(["بسڕەوە", "بیسڕەوە", "بیانسڕەوە", "بسرەوە", "لابە", "لاببە", "لایانبە", "لایببە", "پاک بکەرەوە",
                  "پاکی بکەرەوە", "پاکبکەرەوە", "clear", "remove", "delete", "erase", "wipe", "clean"])
DRAWINGS = _n(["هێڵەکانت", "هێڵەکان", "هێڵەکانم", "هێڵەکانی", "کێشراوەکانت", "کێشراوەکان", "کێشراوەکانی",
               "نیشانەکانت", "نیشانەکان", "ئاستەکانت", "ئاستەکان", "کێشانەکانت", "کێشانەکان", "drawings",
               "your", "lines", "the", "levels", "all", "my", "marks", "chart"])
DRAWINGS_CORE = _n(["هێڵەکانت", "هێڵەکان", "هێڵەکانم", "هێڵەکانی", "کێشراوەکانت", "کێشراوەکان", "کێشراوەکانی",
                    "نیشانەکانت", "نیشانەکان", "ئاستەکانت", "ئاستەکان", "کێشانەکانت", "کێشانەکان", "drawings",
                    "lines", "levels", "marks"])
# Verbs that clear a chart without deleting it («چارتەکە پاک بکەرەوە», "clear the chart").
CLEAN_VERBS = frozenset(_phrases(["پاک بکەرەوە", "پاکی بکەرەوە", "پاکبکەرەوە", "clear", "clean", "wipe"]))
CLEAR_EXTRA = _n(["هەموو", "هەمووی", "لەسەر", "چارتەکە", "چارت", "سەر", "off", "from"])
ALERTS = _n(["ئاگادارکردنەوەکانم", "ئاگادارکردنەوەکان", "ئاگادارکردنەوەکانی", "ئاگادارییەکانم", "ئاگادارییەکان",
             "ئاگاداریەکانم", "ئاگاداریەکان", "ئالێرتەکانم", "ئالێرتەکان", "ئەلێرتەکانم", "ئەلێرتەکان", "alerts",
             "alarms"])
ALERT_ONE = _n(["ئاگادارکردنەوەی", "ئاگادارکردنەوە", "ئاگاداری", "ئالێرتی", "ئالێرت", "alert", "alarm"])
LIST = _phrases(["پیشان بدە", "پیشانم بدە", "پیشانبدە", "نیشان بدە", "نیشانم بدە", "بخوێنەرەوە", "بۆم بخوێنەرەوە",
                 "چین", "کامانەن", "کامەن", "لیست بکە", "لیستی", "list", "show", "show me", "read", "what are",
                 "which", "my", "all", "active", "the", "چالاکەکان", "چالاکەکانم", "هەموو"])
HAVE = _phrases(["چ ئاگادارکردنەوەیەکم هەیە", "چ ئاگادارییەکم هەیە", "ئاگادارکردنەوەم هەیە", "چەند ئاگادارکردنەوەم هەیە",
                 "what alerts do i have", "do i have alerts", "do i have any alerts", "how many alerts do i have",
                 "any alerts"])
# No "stop": «stop the alarm» is what the user says while an alert is being
# spoken (review 2026-09-25: it cancelled EVERY active alert, with no question).
CANCEL = _phrases(["هەڵبوەشێنەوە", "هەڵوەشێنەوە", "هەڵیبوەشێنەوە", "هەڵیانبوەشێنەوە", "بسڕەوە", "بیانسڕەوە",
                   "لابە", "لاببە", "ڕەتبکەرەوە", "بکوژێنەوە", "cancel", "delete", "remove", "clear"])
# Only an explicit "all" (or the plural alert word) means every alert: "the"
# and "my" made «cancel the alert» / «remove my alert» cancel all of them.
ALL = _n(["هەموو", "هەمووی", "all", "every"])
ALERT_FILLER = _n(["my", "the", "active", "چالاکەکان", "چالاکەکانم"])
NUMBER_WORD = _n(["ژمارە", "ژمارەی", "number", "no", "id"])
STOP_CORE = _n(["بوەستە", "ڕاوەستە", "ڕابوەستە", "وەستە", "بەسە", "ڕایگرە", "ڕابگرە", "ستۆپ", "stop", "enough",
                "halt", "cancel"])
STOP_FILLER = _n(["هەمووی", "هەموو", "شتێک", "شت", "بە", "ئیتر", "ئێستا", "یەکسەر", "it", "everything", "all", "be",
                  "now", "right", "that"])
# "SAM, be quiet": SAM's OWN voice (tool stop_speaking), never the Windows volume. Live session
# 2026-09-25: «کوڕە دەنگی بنەکەرە!» muted the computer through system_control. These contain
# «مەکە» (a negative imperative), so they are matched before the negation filter.
QUIET = _phrases(["بێدەنگ بە", "بێدەنگبە", "بێدەنگ", "بیدەنگ بە", "بێ دەنگ بە", "کپ بە", "کپبە", "کپ",
                  "دەنگت بنەکەرە", "دەنگی بنەکەرە", "دەنگ بنەکەرە", "دەنگت بنە", "دەنگت ببڕە", "دەنگت بڕە",
                  "دەنگت کپ کە", "دەنگت کپ بکە", "دەنگت کپکە", "دەنگت بکوژێنەوە", "دەنگ مەکە", "دەنگت مەکە",
                  "قسە مەکە", "قسە مەکەن", "ئیتر قسە مەکە", "shut up", "be quiet", "quiet", "keep quiet",
                  "stop talking", "stop speaking", "silence", "shush", "hush", "enough talking"])
QUIET_FILLER = _n(["ئیتر", "ئێستا", "یەکسەر", "تکایە", "بە", "please", "now", "right", "just", "sam", "سام"])

# Short or generic app aliases that are not safe without a model (a folder, SAM's own settings, "code").
_APP_SKIP = _n(["browser", "web browser", "براوزەر", "براوسەر", "وێبگەڕ", "گەڕۆک", "code", "کۆد", "files",
                "explorer", "فایلەکان", "my computer", "this pc", "کۆمپیوتەرەکەم", "settings", "ڕێکخستن",
                "ڕێکخستنەکان", "سێتینگ", "سێتینگز", "سێتینگەکان", "run", "start", "snip", "word", "paint", "calc",
                "terminal", "power point"])
_NUMBERS_EN = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15, "thirty": 30}
# «چار» is how people (and STT) say «چوار» (four) before a unit («چار سەعات»); alone it is the chart.
_SORANI_NUMBERS = {_norm_token(k): v for k, v in {
    "یەک": 1, "دوو": 2, "سێ": 3, "چوار": 4, "چار": 4, "پێنج": 5, "شەش": 6, "دە": 10, "پازدە": 15, "پانزە": 15,
    "پانزدە": 15, "بیست": 20, "سی": 30, "سیی": 30}.items()}
_UNIT = {_norm_token(k): v for k, v in {
    "خولەک": "m", "خولەکی": "m", "خولەکە": "m", "خولەکێ": "m", "دەقیقە": "m", "دەقیقەی": "m", "دەقە": "m",
    "دەقەی": "m", "دقیقە": "m", "m": "m", "min": "m", "mins": "m",
    "minute": "m", "minutes": "m", "کاتژمێر": "h", "کاتژمێری": "h", "کاتژمێرە": "h", "سەعات": "h", "سەعاتی": "h",
    "سەعاتە": "h", "ساعات": "h", "ساعەت": "h", "سعات": "h", "h": "h", "hr": "h",
    "hour": "h", "hours": "h", "ڕۆژ": "d", "ڕۆژی": "d", "day": "d", "هەفتە": "w", "هەفتەی": "w", "week": "w",
    "مانگ": "mn", "مانگی": "mn", "month": "mn"}.items()}
_UNIT_ONE = {_norm_token(k): v for k, v in {
    "خولەکێک": "M1", "کاتژمێرێک": "H1", "کاتژمێرێ": "H1", "سەعاتێک": "H1", "سەعاتێ": "H1", "ڕۆژێک": "D1",
    "هەفتەیەک": "W1", "مانگێک": "MN1", "ڕۆژانە": "D1", "هەفتانە": "W1", "مانگانە": "MN1", "daily": "D1",
    "weekly": "W1", "monthly": "MN1", "hourly": "H1"}.items()}
# TradingView's own intervals that are not engine timeframes: the chart takes them («٣ خولەکی»).
_CHART_ONLY_TF = {3: "M3", 45: "M45", 120: "H2", 180: "H3"}
_COMPACT_TF = re.compile(r"^(?:(m|h|d|w)(\d{1,3})|(\d{1,3})(m|h|d|w|min)?)$")


# --- token bookkeeping --------------------------------------------------------------------------------
class Words:
    """Tokens of one utterance with a 'used' mark per token."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.used = [False] * len(tokens)

    def left(self) -> list[str]:
        return [t for t, u in zip(self.tokens, self.used) if not u]

    def free_runs(self) -> list[tuple[int, int]]:
        runs, start = [], None
        for i, used in enumerate(self.used + [True]):
            if not used and start is None:
                start = i
            elif used and start is not None:
                runs.append((start, i))
                start = None
        return runs

    def take(self, phrases: list[tuple[str, ...]], *, limit: int = 1) -> list[tuple[str, ...]]:
        """Mark up to ``limit`` occurrences of the longest matching phrases."""
        found: list[tuple[str, ...]] = []
        for phrase in phrases:
            size = len(phrase)
            i = 0
            while i + size <= len(self.tokens) and len(found) < limit:
                window = self.tokens[i:i + size]
                if tuple(window) == phrase and not any(self.used[i:i + size]):
                    for k in range(i, i + size):
                        self.used[k] = True
                    found.append(phrase)
                    i += size
                else:
                    i += 1
            if len(found) >= limit:
                break
        return found

    def take_words(self, words: frozenset[str]) -> int:
        count = 0
        for i, token in enumerate(self.tokens):
            if not self.used[i] and token in words:
                self.used[i] = True
                count += 1
        return count

    def take_window(self, finder: Callable[[list[str]], Any], sizes: tuple[int, ...] = (3, 2, 1)) -> Any:
        """First value ``finder`` returns for an unused window (longest first)."""
        for size in sizes:
            for start, end in self.free_runs():
                for i in range(start, end - size + 1):
                    value = finder(self.tokens[i:i + size])
                    if value is not None:
                        for k in range(i, i + size):
                            self.used[k] = True
                        return value
        return None

    def done(self) -> bool:
        return all(self.used)


# --- slots ------------------------------------------------------------------------------------------------
def _instrument_table() -> dict[str, str]:
    from ..trading.common import SYMBOL_ALIASES
    from ..trading.symbols import EXTRA_ALIASES, KNOWN, LATIN

    table: dict[str, str] = {}
    for key, value in [*SYMBOL_ALIASES.items(), *EXTRA_ALIASES.items(), *LATIN.items()]:
        norm = " ".join(_norm_token(w) for w in key.split())
        if len(norm.replace(" ", "")) >= 3:
            table[norm] = value
    for ticker in KNOWN:
        table.setdefault(ticker.lower(), ticker)
    table.update({"زێر": "XAUUSD", "گۆڵدی": "XAUUSD", "گۆڵت": "XAUUSD"})
    return table


_INSTRUMENTS: dict[str, str] | None = None
_SPELLED: dict[str, str] | None = None
_CONTEXT: dict[str, str] | None = None
# Sorani endings on an instrument word: «زێڕەکە», «زێڕی», «زێڕم» (object clitic: «نرخی زێڕم پێ بڵێ»).
_SUFFIXES = ("یەکە", "ەکەی", "ەکە", "ی", "ە", "م")


def instrument(window: list[str], *, context: bool = False) -> tuple[str, str] | None:
    """(canonical symbol, words as said) for an exact instrument name.
    ``context``: the utterance is about the chart / a price / a timeframe, so
    «گوڵ» / «گۆڵ» (also "flower" / "goal") mean gold too."""
    global _INSTRUMENTS, _SPELLED, _CONTEXT
    if _INSTRUMENTS is None:
        from ..trading.symbols import CONTEXT_ALIASES, SPELLED

        _INSTRUMENTS = _instrument_table()
        _SPELLED = dict(SPELLED)
        _CONTEXT = {_norm_token(k): v for k, v in CONTEXT_ALIASES.items()}
    tables = [_INSTRUMENTS, _CONTEXT or {}] if context else [_INSTRUMENTS]
    text = " ".join(window)
    found = _INSTRUMENTS.get(text) or (_SPELLED or {}).get(text.replace(" ", ""))
    if found is None and context and len(window) == 1:
        found = (_CONTEXT or {}).get(text)
    if found is None and len(window) == 1 and not text.isascii():
        for suffix in _SUFFIXES:
            if text.endswith(suffix) and len(text) - len(suffix) >= 3:
                found = next((t[text[: -len(suffix)]] for t in tables if text[: -len(suffix)] in t), None)
                if found:
                    break
    return (found, text) if found else None


def _context(tokens: list[str]) -> bool:
    """The utterance names the chart, a price or a timeframe (then «گوڵ» is gold)."""
    if any(t in CHART_NOUNS or t in TF_WORDS or t in PRICE_CORE for t in tokens):
        return True
    return any(timeframe(tokens[i:i + size], extra=True) for size in (1, 2) for i in range(len(tokens) - size + 1))


def timeframe(window: list[str], *, bare: bool = False, extra: bool = False) -> str | None:
    """Canonical timeframe for an exact timeframe expression (every word must
    be a number or a unit): «١٥ خولەک», «چوار کاتژمێر», «کاتژمێرێک», "15m", "H4".

    A bare number ("60", «٦٠») counts only with ``bare=True``: the review
    (2026-09-25) saw "draw support at 60" draw engine levels on H1 instead of a
    line at 60, and "gold 240" switch the chart to H4 -- prices of silver/oil
    look exactly like minutes.

    ``extra``: TradingView's own 3/45-minute and 2/3-hour intervals count too
    ('M3', 'M45', 'H2', 'H3': the chart takes them, the engine does not)."""
    from ..trading.common import TIMEFRAMES, normalize_timeframe

    if len(window) == 1:
        word = window[0]
        if word in _UNIT_ONE:
            return _UNIT_ONE[word]
        compact = _COMPACT_TF.match(word)
        if compact:
            if not bare and compact.group(3) and not compact.group(4):
                return None
            value = normalize_timeframe(word)
            if value in TIMEFRAMES:
                return value
            return _chart_only(word) if extra else None
        return None
    if len(window) != 2:
        return None
    count, unit = window
    if count in _n(["نیو", "نیوە", "half"]) and unit in _UNIT and _UNIT[unit] == "h":
        return "M30"                                       # «نیو کاتژمێر» = half an hour
    number = int(count) if count.isdigit() else _SORANI_NUMBERS.get(count) or _NUMBERS_EN.get(count)
    if number is None or unit not in _UNIT:
        return None
    value = normalize_timeframe(f"{number} {_unit_word(_UNIT[unit])}")
    if value in TIMEFRAMES:
        return value
    if extra and _UNIT[unit] in ("m", "h"):
        return _CHART_ONLY_TF.get(number * (60 if _UNIT[unit] == "h" else 1))
    return None


def _chart_only(word: str) -> str | None:
    """'3m' / 'm3' / 'h2' -> 'M3' / 'H2' (TradingView-only intervals)."""
    found = re.fullmatch(r"(?:(m|h)(\d{1,3})|(\d{1,3})(m|min|h))", word)
    if not found:
        return None
    unit = (found.group(1) or found.group(4))[0]
    count = int(found.group(2) or found.group(3))
    return _CHART_ONLY_TF.get(count * (60 if unit == "h" else 1))


def _unit_word(unit: str) -> str:
    return {"m": "minutes", "h": "hours", "d": "days", "w": "weeks", "mn": "months"}[unit]


_APPS: dict[str, Any] | None = None


def _app_table() -> dict[str, Any]:
    """normalised alias (and its joined form) -> AppAlias, built once."""
    from ..hands.aliases import ALIASES

    table: dict[str, Any] = {}
    for spec in ALIASES:
        for alias in spec.aliases:
            norm = " ".join(_norm_token(w) for w in alias.split())
            if not norm or norm in _APP_SKIP:
                continue
            table.setdefault(norm, spec)
            table.setdefault(norm.replace(" ", ""), spec)
    return table


def app_alias(window: list[str]) -> tuple[Any, str] | None:
    """(AppAlias, words as said) for an exact app name from the hands alias table."""
    global _APPS
    from ..hands.aliases import strip_suffix

    if _APPS is None:
        _APPS = _app_table()
    text = " ".join(window)
    if text in _APP_SKIP:
        return None
    spec = _APPS.get(text)
    if spec is None and len(window) == 1 and not text.isascii():
        spec = _APPS.get(strip_suffix(text))
    if spec is None and not text.isascii() and text.endswith("م") and len(window[-1]) > 3:
        spec = _APPS.get(text[:-1])      # «ترەیدینگ ڤیوم بۆ بکەرەوە»: the object clitic on the name
    if spec is None and len(window) > 1 and not text.isascii():
        spec = _APPS.get(" ".join([*window[:-1], strip_suffix(window[-1])]))   # «ترەیدینگ ڤیوەکە»
    return (spec, text) if spec is not None else None


# --- the matcher --------------------------------------------------------------------------------------------
def tokens_of(text: str) -> list[str]:
    return [_norm_token(t) for t in normalize_ckb(text or "", strip_punct=True).split()]


def _strip_edges(words: Words) -> None:
    """Vocative («سام», "hey SAM") at the start or end, politeness anywhere."""
    for phrase in VOCATIVE:
        size = len(phrase)
        if tuple(words.tokens[:size]) == phrase:
            for k in range(size):
                words.used[k] = True
            break
    for phrase in VOCATIVE:
        size = len(phrase)
        if size <= len(words.tokens) and tuple(words.tokens[-size:]) == phrase and not any(words.used[-size:]):
            for k in range(len(words.tokens) - size, len(words.tokens)):
                words.used[k] = True
            break
    _strip_lead(words)
    _strip_tail(words)
    words.take(POLITE, limit=4)


def _strip_lead(words: Words) -> None:
    i = 0
    while i < len(words.tokens):
        if words.used[i]:
            i += 1
            continue
        phrase = next((p for p in LEAD_PHRASES if tuple(words.tokens[i:i + len(p)]) == p), None)
        if phrase is not None:
            for k in range(i, i + len(phrase)):
                words.used[k] = True
            i += len(phrase)
            continue
        if words.tokens[i] in LEAD:
            words.used[i] = True
            i += 1
            continue
        break


def _strip_tail(words: Words) -> None:
    """Interjections, insults and «چی دەکەی» after the command («... بڕۆ سەر چار چی دەکەی؟»)."""
    end = len(words.tokens)
    while end > 0:
        while end > 0 and words.used[end - 1]:
            end -= 1
        phrase = next((p for p in TAIL if len(p) <= end and tuple(words.tokens[end - len(p):end]) == p
                       and not any(words.used[end - len(p):end])), None)
        if phrase is None:
            return
        for k in range(end - len(phrase), end):
            words.used[k] = True
        end -= len(phrase)


def _prepared(text: str, *, reject: bool = True) -> Words | None:
    tokens = tokens_of(text)
    if not tokens or len(tokens) > MAX_WORDS:
        return None
    words = Words(tokens)
    _strip_edges(words)
    left = words.left()
    if not left:
        return None
    if reject:
        for token in left:
            if token in REJECT or _negative_verb(token):
                return None
    return words


def _negative_verb(token: str) -> bool:
    """«مەکەرەوە», «مەیسڕەوە», «مەیکێشە»: a negative imperative."""
    return token.startswith("مە") and len(token) > 3 and (token.endswith("ەوە") or token.endswith("ە")) \
        and token not in {"مەنەجەر"}


def match(text: str) -> Intent | None:
    """The fast-path intent of ``text``, or None (then the model answers)."""
    words = _prepared(text, reject=False)
    if words is not None:
        intent = _quiet(words)       # «دەنگ مەکە» / «قسە مەکە»: before the negation filter
        if intent is not None and words.done():
            return intent
    for grammar in (_stop, _open_tradingview, _open_app, _list_alerts, _cancel_alerts, _clear_drawings,
                    _analyze, _draw_levels, _price, _set_chart):
        words = _prepared(text)
        if words is None:
            return None
        intent = grammar(words)
        if intent is not None and words.done():
            return intent
    return None


def _quiet(words: Words) -> Intent | None:
    """SAM's own voice: «بێدەنگ بە», «دەنگت بنەکەرە», «دەنگ مەکە», "be quiet"."""
    if not words.take(QUIET):
        return None
    words.take_words(QUIET_FILLER)
    return Intent("quiet", "stop_speaking", {})


def _stop(words: Words) -> Intent | None:
    if not words.take_words(STOP_CORE):
        return None
    words.take_words(STOP_FILLER)
    return Intent("stop", "stop_all", {})


def _open_tradingview(words: Words) -> Intent | None:
    found = words.take_window(lambda w: _tv_alias(w))
    if found is None:
        return None
    # «بڕۆ سەر چارت» ("go to the chart") brings TradingView to the front; after «... بکەوە»
    # («ترێیت ملیۆم لۆ بکەوە بڕۆ سەر چار») it is part of opening it.
    go = bool(words.take(GO))
    if go:
        words.take_words(CHART_NOUNS)
    if not words.take(OPEN) and not go:
        return None
    words.take_words(APP_FILLER)
    return Intent("open_tradingview", "tv_open", {}, said=found)


def _tv_alias(window: list[str]) -> str | None:
    text = " ".join(window)
    if text in _n(["چارتەکەم", "چارتەکە", "چارت", "چار", "چارتی"]) and len(window) == 1:
        return text
    hit = app_alias(window)
    if hit is not None and hit[0].key == "tradingview":
        return hit[1]
    return None


def _open_app(words: Words) -> Intent | None:
    found = words.take_window(lambda w: _other_app(w))
    if found is None:
        return None
    spec, said = found
    if not words.take(OPEN):
        return None
    words.take_words(APP_FILLER)
    return Intent("open_app", "open_app", {"name": spec.display}, said=said)


def _other_app(window: list[str]) -> tuple[Any, str] | None:
    hit = app_alias(window)
    return hit if hit is not None and hit[0].key != "tradingview" else None


def _list_alerts(words: Words) -> Intent | None:
    if words.take(HAVE):
        words.take_words(_n(["ئێستا", "now", "active", "چالاک"]))
        return Intent("list_alerts", "list_alerts", {"status": "active"})
    if not words.take_words(ALERTS):
        return None
    if not words.take(LIST, limit=4):
        return None
    return Intent("list_alerts", "list_alerts", {"status": "active"})


def _cancel_alerts(words: Words) -> Intent | None:
    plural = words.take_words(ALERTS)
    single = 0 if plural else words.take_words(ALERT_ONE)
    if not (plural or single):
        return None
    if not words.take(CANCEL):
        return None
    everything = words.take_words(ALL)
    if plural:
        # «ئاگادارکردنەوەکان بسڕەوە» / "delete my alerts": every alert (cancel_alert
        # itself asks first when more than one is active).
        words.take_words(ALERT_FILLER)
        return Intent("cancel_alerts", "cancel_alert", {"alert_id": "all"})
    words.take_words(NUMBER_WORD)
    number = words.take_window(lambda w: w[0] if w[0].isdigit() else None, sizes=(1,))
    if number is None:
        # one alert without a number («cancel the alert»): the model asks which
        if not everything:
            return None
        words.take_words(ALERT_FILLER)
        return Intent("cancel_alerts", "cancel_alert", {"alert_id": "all"})
    words.take_words(ALERT_FILLER)
    return Intent("cancel_alerts", "cancel_alert", {"alert_id": str(int(number))})


def _clear_drawings(words: Words) -> Intent | None:
    verb = words.take(CLEAR)
    if not verb:
        return None
    if not words.take_words(DRAWINGS_CORE):
        # «چارتەکە پاک بکەرەوە» / "clear the chart": only SAM's drawings go (never the
        # user's). "delete/remove the chart" may mean the chart window: the model decides.
        if verb[0] not in CLEAN_VERBS or not words.take_words(_n(["چارتەکە", "چارت", "chart"])):
            return None
    words.take_words(DRAWINGS)
    words.take_words(CLEAR_EXTRA)
    return Intent("clear_drawings", "clear_my_drawings", {})


def _analyze(words: Words) -> Intent | None:
    one = words.take(ANALYZE_ONE)
    english = noun = False
    if not one:
        found = words.take(ANALYZE)
        if not found:
            return None
        english = found[0][0].isascii()
        noun = found[0] in ANALYSIS_NOUNS
        if not english and not words.take(ANALYZE_VERB):
            return None
    obj = words.take_words(ANALYZE_OBJECT)
    words.take_words(MARKET)
    symbol = words.take_window(lambda w: instrument(w, context=True))
    if english and symbol is None and (noun or not obj):
        # bare "analysis" / "analyze" / "market analysis" ran a full analyse-and-draw on the
        # chart (review 2026-09-25): without an instrument English needs a verb and an object.
        return None
    tf = _take_timeframe(words)
    if words.take_words(AND):
        words.take_words(LEVELS)
        if not words.take(DRAW, limit=2):
            return None
    elif words.take(DRAW, limit=2):
        words.take_words(LEVELS)
    args: dict[str, Any] = {"draw": "full", "vision": False}
    if symbol:
        args["symbol"] = symbol[1]
    if tf:
        args["timeframes"] = [tf]
    return Intent("analyze", "analyze_market", args, said=symbol[1] if symbol else "", slow=True,
                  chart_symbol=symbol is None)


def _draw_levels(words: Words) -> Intent | None:
    if not words.take(DRAW, limit=2):
        return None
    core = _n(["هێڵی", "هێڵەکانی", "هێڵەکان", "ئاستی", "ئاستەکانی", "ئاستەکان", "پشتگیری", "بەرگری", "سەپۆرت",
               "ڕەزستەنس", "levels", "lines", "support", "resistance"])
    if not words.take_words(core):
        return None
    words.take_words(LEVELS)
    words.take_words(AND)
    symbol = words.take_window(lambda w: instrument(w, context=True))
    if symbol:
        words.take_words(_n(["for", "of", "بۆ"]))
    tf = _take_timeframe(words)
    args: dict[str, Any] = {"draw": "levels", "vision": False}
    if symbol:
        args["symbol"] = symbol[1]
    if tf:
        args["timeframes"] = [tf]
    return Intent("draw_levels", "analyze_market", args, said=symbol[1] if symbol else "", slow=True,
                  chart_symbol=symbol is None)


def _take_timeframe(words: Words, *, bare: bool = False, extra: bool = False) -> str | None:
    """A timeframe with a unit («١٥ خولەک», "H4", "15m"); a bare number only
    when ``bare`` (set_chart next to an explicit timeframe word); TradingView's
    own intervals (M3, H2 ...) only when ``extra`` (the chart)."""
    before = list(words.used)
    tf = words.take_window(lambda window: timeframe(window, bare=bare, extra=extra), sizes=(2, 1))
    if tf is None:
        return None
    # a preposition right before it belongs to it («لەسەر ١٥ خولەک»)
    first = next(i for i, (a, b) in enumerate(zip(before, words.used)) if a != b)
    if first > 0 and not words.used[first - 1] and words.tokens[first - 1] in TF_PREP:
        words.used[first - 1] = True
    return tf


def _price(words: Words) -> Intent | None:
    how = words.take(HOW_MUCH)
    price = words.take_words(PRICE_CORE)
    if not (how or price):
        return None
    if not price and how[0] in WEAK_HOW_MUCH:
        return None      # «چەند زێڕ» / "how much gold": a quantity, not a price question
    if price:
        words.take(WHAT_IS)
    else:
        words.take(TELL_ME)
    words.take_words(PRICE)
    words.take_words(NOW)
    symbol = words.take_window(lambda w: instrument(w, context=bool(price)))
    if symbol is None:
        return None
    left = [i for i, used in enumerate(words.used) if not used]
    if len(left) == 1 and not price:
        # a mangled price word right before the instrument («نەخنەشکی زێڕ چەندە»)
        token = words.tokens[left[0]]
        nxt = left[0] + 1
        if nxt < len(words.tokens) and instrument([words.tokens[nxt]]) and _mangled_price(token):
            words.used[left[0]] = True
    return Intent("price", "get_price", {"symbol": symbol[1]}, said=symbol[1])


def _mangled_price(token: str) -> bool:
    """A price word STT mangled: a spelling heard on this PC, or one letter
    away from «نرخی». The old shape rule (any ن-word with خ/ر) took «نەخۆشی»,
    «نەرمی» and «نزیکترین» for a price word (review 2026-09-25)."""
    return not token.isascii() and (token in MANGLED_PRICE or _within_one(token, "نرخی"))


def _within_one(a: str, b: str) -> bool:
    """Edit distance <= 1 (one letter added, dropped or changed)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _set_chart(words: Words) -> Intent | None:
    # "change the timeframe to 60" names a timeframe; "gold 240" / «بیتکۆین ٦٠» may be a price.
    bare = any(token in TF_WORDS for token in words.tokens)
    context = _context(words.tokens)
    # the timeframe first: «چار سەعات» is four hours, a lone «چار» is the chart
    tf = _take_timeframe(words, bare=bare, extra=True)
    verb = words.take(SHOW)
    words.take_words(CHART)
    symbol = words.take_window(lambda w: instrument(w, context=context))
    if not (symbol or tf):
        return None
    if verb and verb[0] in WEAK_SHOW and not tf:
        return None       # «زێڕ بکە» / "set gold": only with a timeframe
    if not verb:
        # «گۆڵد لەسەر ١٥ خولەک» / "timeframe 15 minutes" without a verb: instrument or a
        # timeframe word next to the timeframe
        if not (tf and (symbol or bare)):
            return None
    words.take_words(TF_PREP)
    words.take_words(CHART)
    args: dict[str, Any] = {}
    if symbol:
        args["symbol"] = symbol[1]
    if tf:
        args["timeframe"] = tf
    return Intent("set_chart", "tv_set_chart", args, said=symbol[1] if symbol else "")


__all__ = ["Intent", "match", "instrument", "timeframe", "app_alias", "tokens_of", "MAX_WORDS"]
