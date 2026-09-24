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
The labelled corpus (tests/fastpath_corpus.py, >= 150 utterances) measures it.

Spellings: text is normalised with ``normalize_ckb`` (Arabic ي/ك, digits,
punctuation), a word-final Arabic heh counts as ە (STT writes «بکەرەوه»), and
a mangled price word right before an instrument is accepted (KurdishTTS STT
wrote «نرخی زێڕ» as «نەخنەشکی زێڕ» on 2026-09-24).
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

    name: str                     # price|open_tradingview|open_app|set_chart|analyze|draw_levels|clear_drawings|list_alerts|cancel_alerts|stop
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
                   "now please", "ئێستا تکایە"])
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
# Leading words that carry no meaning in a spoken command («باشە، کرۆم بکەرەوە»).
LEAD = _n(["باشە", "ئا", "ئەی", "ئەها", "ئێستا", "ok", "okay", "so", "well", "now", "yes", "بەڵێ", "سڵاو",
           "hello", "hi"])

OPEN = _phrases(["بکەرەوە", "بکەوە", "بکەرەو", "بیکەرەوە", "بکەیتەوە", "بکرێتەوە", "بکەیتەوە", "هەڵبکە", "هەلبکە",
                 "بێنە", "بهێنە", "بێنە پێشەوە", "بهێنە پێشەوە", "بکە بەرز", "open", "launch", "start", "run",
                 "open up", "bring up", "fire up"])
APP_FILLER = _n(["ئەپی", "ئەپ", "ئەپەکە", "ئاپی", "ئاپ", "ئاپەکە", "بەرنامەی", "بەرنامە", "بەرنامەکە", "پرۆگرامی",
                 "پرۆگرام", "پرۆگرامەکە", "ئەپلیکەیشنی", "ئەپلیکەیشن", "the", "app", "application", "program", "up",
                 "a", "new", "me", "بۆ"])
SHOW = _phrases(["پیشان بدە", "پیشانم بدە", "پیشانبدە", "پیشانمبدە", "نیشان بدە", "نیشانم بدە", "نیشانبدە",
                 "دابنێ", "بیکە بە", "بکە بە", "بیکە", "بکە", "بگۆڕە بۆ", "بیگۆڕە بۆ", "بگۆڕە", "بیگۆڕە",
                 "بکەرەوە", "بکەوە", "بهێنە", "بێنە", "show", "show me", "switch to", "switch", "change to",
                 "change", "set", "set to", "put", "open", "go to", "load", "display", "make it"])
WEAK_SHOW = frozenset(_phrases(["دابنێ", "بیکە", "بکە", "set", "put", "change", "make it", "load"]))
CHART = _n(["چارت", "چارتەکە", "چارتەکەم", "چارتی", "کاتی", "تایمفرەیم", "تایمفرەیمی", "سیمبۆڵ", "سیمبۆڵەکە",
            "سیمبۆڵی", "chart", "the", "timeframe", "time", "frame", "symbol", "my", "tradingview", "a"])
TF_PREP = _n(["لەسەر", "بۆ", "بە", "لە", "on", "to", "at", "in", "into"])
PRICE = _n(["نرخی", "نرخ", "نەرخی", "نەرخ", "نرخەکەی", "نرخەکە", "نرخێ", "price", "prices", "rate", "quote",
            "the", "of", "is", "current", "s"])
HOW_MUCH = _phrases(["چەندە", "چەند", "بە چەندە", "لە چەندە", "لە چەندایە", "بەچەندە", "چ نرخێکە",
                     "لە چ نرخێکە", "لە چ نرخێکدایە", "چ نرخێکدایە", "how much", "how much is"])
# "what is ..." / «... چییە» ask for a price only next to a price word («زێڕ چییە» = what is gold?).
WHAT_IS = _phrases(["چییە", "چیە", "چی یە", "پێم بڵێ", "پێ بڵێ", "بڵێ", "بزانە", "what is", "whats", "what s",
                    "what", "tell me", "give me", "check", "get"])
PRICE_CORE = _n(["نرخی", "نرخ", "نەرخی", "نەرخ", "نرخەکەی", "نرخەکە", "قیمەتی", "قیمەت", "price", "prices", "quote"])
NOW = _n(["ئێستا", "ئێستای", "ئەمڕۆ", "ئەمڕۆی", "now", "right", "today", "currently", "live"])
ANALYZE = _phrases(["شیکاری", "شیکار", "شیکردنەوە", "شیکردنەوەی", "شی", "analyze", "analyse", "analysis",
                    "analysis of", "analyze the", "analyse the", "do an analysis of", "do analysis on"])
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
CANCEL = _phrases(["هەڵبوەشێنەوە", "هەڵوەشێنەوە", "هەڵیبوەشێنەوە", "هەڵیانبوەشێنەوە", "بسڕەوە", "بیانسڕەوە",
                   "لابە", "لاببە", "ڕەتبکەرەوە", "بکوژێنەوە", "cancel", "delete", "remove", "clear", "stop"])
ALL = _n(["هەموو", "هەمووی", "all", "every", "my", "the"])
NUMBER_WORD = _n(["ژمارە", "ژمارەی", "number", "no", "id"])
STOP_CORE = _n(["بوەستە", "ڕاوەستە", "ڕابوەستە", "وەستە", "بەسە", "ڕایگرە", "ڕابگرە", "ستۆپ", "بێدەنگ", "بێدەنگبە",
                "کپ", "stop", "enough", "quiet", "silence", "halt", "shush", "cancel"])
STOP_FILLER = _n(["هەمووی", "هەموو", "شتێک", "شت", "بە", "ئیتر", "ئێستا", "یەکسەر", "it", "everything", "all", "be",
                  "now", "right", "that", "talking", "speaking"])

# Short or generic app aliases that are not safe without a model (a folder, SAM's own settings, "code").
_APP_SKIP = _n(["browser", "web browser", "براوزەر", "براوسەر", "وێبگەڕ", "گەڕۆک", "code", "کۆد", "files",
                "explorer", "فایلەکان", "my computer", "this pc", "کۆمپیوتەرەکەم", "settings", "ڕێکخستن",
                "ڕێکخستنەکان", "سێتینگ", "سێتینگز", "سێتینگەکان", "run", "start", "snip", "word", "paint", "calc",
                "terminal", "power point"])
_NUMBERS_EN = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15, "thirty": 30}
_SORANI_NUMBERS = {_norm_token(k): v for k, v in {
    "یەک": 1, "دوو": 2, "سێ": 3, "چوار": 4, "پێنج": 5, "شەش": 6, "دە": 10, "پازدە": 15, "پانزە": 15, "بیست": 20,
    "سی": 30}.items()}
_UNIT = {_norm_token(k): v for k, v in {
    "خولەک": "m", "خولەکی": "m", "خولەکە": "m", "دەقیقە": "m", "دەقە": "m", "m": "m", "min": "m", "mins": "m",
    "minute": "m", "minutes": "m", "کاتژمێر": "h", "کاتژمێری": "h", "سەعات": "h", "سەعاتی": "h", "h": "h", "hr": "h",
    "hour": "h", "hours": "h", "ڕۆژ": "d", "ڕۆژی": "d", "day": "d", "هەفتە": "w", "هەفتەی": "w", "week": "w",
    "مانگ": "mn", "مانگی": "mn", "month": "mn"}.items()}
_UNIT_ONE = {_norm_token(k): v for k, v in {
    "خولەکێک": "M1", "کاتژمێرێک": "H1", "سەعاتێک": "H1", "ڕۆژێک": "D1", "هەفتەیەک": "W1", "مانگێک": "MN1",
    "ڕۆژانە": "D1", "هەفتانە": "W1", "مانگانە": "MN1", "daily": "D1", "weekly": "W1", "monthly": "MN1",
    "hourly": "H1"}.items()}
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
# Sorani endings on an instrument word: «زێڕەکە», «زێڕی», «زێڕم» (object clitic: «نرخی زێڕم پێ بڵێ»).
_SUFFIXES = ("یەکە", "ەکەی", "ەکە", "ی", "ە", "م")


def instrument(window: list[str]) -> tuple[str, str] | None:
    """(canonical symbol, words as said) for an exact instrument name."""
    global _INSTRUMENTS, _SPELLED
    if _INSTRUMENTS is None:
        from ..trading.symbols import SPELLED

        _INSTRUMENTS = _instrument_table()
        _SPELLED = dict(SPELLED)
    text = " ".join(window)
    found = _INSTRUMENTS.get(text) or (_SPELLED or {}).get(text.replace(" ", ""))
    if found is None and len(window) == 1 and not text.isascii():
        for suffix in _SUFFIXES:
            if text.endswith(suffix) and len(text) - len(suffix) >= 3:
                found = _INSTRUMENTS.get(text[: -len(suffix)])
                if found:
                    break
    return (found, text) if found else None


def timeframe(window: list[str]) -> str | None:
    """Canonical timeframe for an exact timeframe expression (every word must
    be a number or a unit): «١٥ خولەک», «چوار کاتژمێر», «کاتژمێرێک», "15m", "H4"."""
    from ..trading.common import TIMEFRAMES, normalize_timeframe

    if len(window) == 1:
        word = window[0]
        if word in _UNIT_ONE:
            return _UNIT_ONE[word]
        compact = _COMPACT_TF.match(word)
        if compact:
            value = normalize_timeframe(word)
            return value if value in TIMEFRAMES else None
        return None
    if len(window) != 2:
        return None
    count, unit = window
    number = int(count) if count.isdigit() else _SORANI_NUMBERS.get(count) or _NUMBERS_EN.get(count)
    if number is None or unit not in _UNIT:
        return None
    value = normalize_timeframe(f"{number} {_unit_word(_UNIT[unit])}")
    return value if value in TIMEFRAMES else None


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
    for i, token in enumerate(words.tokens):
        if words.used[i]:
            continue
        if token in LEAD:
            words.used[i] = True
            continue
        break
    words.take(POLITE, limit=3)


def _prepared(text: str) -> Words | None:
    tokens = tokens_of(text)
    if not tokens or len(tokens) > MAX_WORDS:
        return None
    words = Words(tokens)
    _strip_edges(words)
    left = words.left()
    if not left:
        return None
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
    for grammar in (_stop, _open_tradingview, _open_app, _list_alerts, _cancel_alerts, _clear_drawings,
                    _analyze, _draw_levels, _price, _set_chart):
        words = _prepared(text)
        if words is None:
            return None
        intent = grammar(words)
        if intent is not None and words.done():
            return intent
    return None


def _stop(words: Words) -> Intent | None:
    if not words.take_words(STOP_CORE):
        return None
    words.take_words(STOP_FILLER)
    return Intent("stop", "stop_all", {})


def _open_tradingview(words: Words) -> Intent | None:
    found = words.take_window(lambda w: _tv_alias(w))
    if found is None:
        return None
    if not words.take(OPEN):
        return None
    words.take_words(APP_FILLER)
    return Intent("open_tradingview", "tv_open", {}, said=found)


def _tv_alias(window: list[str]) -> str | None:
    text = " ".join(window)
    if text in _n(["چارتەکەم", "چارتەکە", "چارت"]) and len(window) == 1:
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
        return Intent("cancel_alerts", "cancel_alert", {"alert_id": "all"})
    words.take_words(NUMBER_WORD)
    number = words.take_window(lambda w: w[0] if w[0].isdigit() else None, sizes=(1,))
    if number is None:
        return Intent("cancel_alerts", "cancel_alert", {"alert_id": "all"}) if everything else None
    return Intent("cancel_alerts", "cancel_alert", {"alert_id": str(int(number))})


def _clear_drawings(words: Words) -> Intent | None:
    if not words.take(CLEAR):
        return None
    if not words.take_words(DRAWINGS_CORE):
        # «چارتەکە پاک بکەرەوە» / "clear the chart": only SAM's drawings go (never the user's).
        if not words.take_words(_n(["چارتەکە", "چارت", "chart"])):
            return None
    words.take_words(DRAWINGS)
    words.take_words(CLEAR_EXTRA)
    return Intent("clear_drawings", "clear_my_drawings", {})


def _analyze(words: Words) -> Intent | None:
    one = words.take(ANALYZE_ONE)
    if not one:
        if not words.take(ANALYZE):
            return None
        english = any(t.isascii() for t in words.tokens)
        if not words.take(ANALYZE_VERB) and not english:
            return None
    words.take_words(MARKET)
    symbol = words.take_window(instrument)
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
    symbol = words.take_window(instrument)
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


def _take_timeframe(words: Words) -> str | None:
    before = list(words.used)
    tf = words.take_window(timeframe, sizes=(2, 1))
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
    if price:
        words.take(WHAT_IS)
    words.take_words(PRICE)
    words.take_words(NOW)
    symbol = words.take_window(instrument)
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
    return (3 <= len(token) <= 10 and not token.isascii() and token.startswith("ن")
            and ("خ" in token or "ر" in token))


def _set_chart(words: Words) -> Intent | None:
    verb = words.take(SHOW)
    words.take_words(CHART)
    symbol = words.take_window(instrument)
    tf = _take_timeframe(words)
    if not (symbol or tf):
        return None
    if verb and verb[0] in WEAK_SHOW and not tf:
        return None       # «زێڕ بکە» / "set gold": only with a timeframe
    if not verb:
        # «گۆڵد لەسەر ١٥ خولەک» without a verb: only instrument + timeframe
        if not (symbol and tf):
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
