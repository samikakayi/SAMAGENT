"""Spoken app names -> apps: Sorani, colloquial and English aliases.

v1 found apps with ``shutil.which`` and missed Chrome, Edge, TradingView,
Telegram, MetaTrader 5, WhatsApp and Excel, although all 195 Start-menu apps
were installed (reports/computer-control.json). SAM 2 resolves a spoken name
in three steps (apps.py): this alias table (Sorani spellings as speech-to-text
writes them), then fuzzy matching against the Start-menu names, then a rough
Sorani->Latin transliteration for app names nobody put in the table
("سپۆتیفای" -> "spotifay" ~ "Spotify").

Every alias is compared after ``normalize_ckb`` (ي/ك/digit variants unified).
Speech-to-text writes the same Sorani word several ways (ترەیدینگ / تریدینگ /
ترێدینگ), so the common variants are listed explicitly and the rest is left to
rapidfuzz.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..textnorm import normalize_ckb


@dataclass(frozen=True)
class AppAlias:
    key: str                                   # stable id, e.g. "tradingview"
    display: str                               # English display name
    names: tuple[str, ...]                     # Start-menu names, best first
    aliases: tuple[str, ...]                   # spoken forms (Sorani + English)
    processes: tuple[str, ...] = ()            # exe basenames (lower case) of its windows
    url: str | None = None                     # web apps (YouTube)
    uri: str | None = None                     # shell URI fallback (ms-settings:)
    exe: str | None = None                     # executable fallback when not in the Start menu
    reuse_window: bool = True                  # focus a running window instead of launching again
    app_ids: tuple[str, ...] = field(default=())  # preferred AppsFolder ids (prefix match)


ALIASES: tuple[AppAlias, ...] = (
    AppAlias("tradingview", "TradingView", ("TradingView",),
             ("tradingview", "trading view", "trading-view", "ترەیدینگ ڤیو", "تریدینگ ڤیو", "ترێدینگ ڤیو",
              "ترەیدینگڤیو", "تریدینگڤیو", "ترێدینگڤیو", "ترەیدینگ ویو", "تریدینگ ویو", "ترێدینگ ویو",
              "تریدنگ ڤیو", "ترەیدنگ ڤیو", "ترەیدینگ", "تریدینگ", "ترێدینگ", "چارتی ترەیدینگ ڤیو"),
             processes=("tradingview.exe",), app_ids=("TradingView.Desktop_",)),
    AppAlias("mt5", "MetaTrader 5", ("MetaTrader 5", "MetaTrader"),
             ("metatrader 5", "metatrader", "meta trader", "mt5", "mt 5", "مێتاترەیدەر", "میتاترەیدەر",
              "مێتا ترەیدەر", "میتا ترەیدەر", "مێتاتریدەر", "میتاتریدەر", "میتا تریدەر", "متاتریدر",
              "مێتاترەیدەر ٥", "مێتاترەیدەر پێنج", "ئێم تی فایڤ", "ئێم تی ٥", "ام تی ٥", "ئێم تی پێنج"),
             processes=("terminal64.exe",)),
    AppAlias("chrome", "Google Chrome", ("Google Chrome", "Chrome"),
             ("chrome", "google chrome", "کرۆم", "کڕۆم", "کروم", "گووگڵ کرۆم", "گوگڵ کرۆم", "گوگل کروم",
              "گووگڵ کڕۆم", "کرۆمی گووگڵ"),
             processes=("chrome.exe",)),
    AppAlias("edge", "Microsoft Edge", ("Microsoft Edge", "Edge"),
             ("edge", "microsoft edge", "ms edge", "ئێج", "ئێدج", "ئەیج", "مایکرۆسۆفت ئێج", "مایکرۆسۆفت ئێدج"),
             processes=("msedge.exe",)),
    AppAlias("firefox", "Firefox", ("Firefox", "Mozilla Firefox"),
             ("firefox", "mozilla firefox", "فایەرفۆکس", "فایرفۆکس", "فایەرفاکس", "فایرفاکس"),
             processes=("firefox.exe",)),
    AppAlias("browser", "Web browser", ("Google Chrome", "Microsoft Edge", "Firefox"),
             ("browser", "web browser", "براوزەر", "براوسەر", "وێبگەڕ", "گەڕۆک"),
             processes=("chrome.exe", "msedge.exe", "firefox.exe")),
    AppAlias("vscode", "Visual Studio Code", ("Visual Studio Code",),
             ("vs code", "vscode", "visual studio code", "code", "ڤی ئێس کۆد", "ڤی ئێس کود", "ڤی ئیس کۆد",
              "وی ئێس کۆد", "ڤیژواڵ ستۆدیۆ کۆد", "ڤیژوال ستودیو کود", "ڤیشواڵ ستۆدیۆ کۆد", "کۆد"),
             processes=("code.exe",)),
    AppAlias("cursor", "Cursor", ("Cursor",),
             ("cursor", "cursor ai", "کێرسەر", "کەرسەر", "کورسەر", "کێرسۆر"),
             processes=("cursor.exe",)),
    AppAlias("notepad", "Notepad", ("Notepad",),
             ("notepad", "note pad", "نۆتپاد", "نۆت پاد", "نوتپاد", "نۆتپەد", "نۆتپاد ", "نۆتبووک"),
             processes=("notepad.exe",)),
    AppAlias("explorer", "File Explorer", ("File Explorer",),
             ("file explorer", "explorer", "files", "my computer", "this pc", "فایل ئێکسپلۆرەر", "ئێکسپلۆرەر",
              "فایل ئیکسپلۆرەر", "فایلەکان", "کۆمپیوتەرەکەم"),
             processes=("explorer.exe",), exe="explorer.exe", reuse_window=False),
    AppAlias("settings", "Settings", ("Settings",),
             ("settings", "windows settings", "ڕێکخستنەکان", "ڕێکخستنەکانی ویندۆز", "سێتینگ", "سێتینگز",
              "سێتینگەکان", "ڕێکخستن"),
             processes=("systemsettings.exe",), uri="ms-settings:"),
    AppAlias("calculator", "Calculator", ("Calculator",),
             ("calculator", "calc", "ژمێرەر", "ژمێرەرەکە", "حاسیبە", "حاسیبەکە", "کالکولەیتەر", "ئامێری ژمێرە"),
             processes=("calculatorapp.exe", "calc.exe")),
    AppAlias("telegram", "Telegram", ("Telegram", "Telegram Desktop"),
             ("telegram", "telegram desktop", "تێلێگرام", "تێلەگرام", "تلگرام", "تەلەگرام", "تیلیگرام", "تێلگرام"),
             processes=("telegram.exe",)),
    AppAlias("whatsapp", "WhatsApp", ("WhatsApp",),
             ("whatsapp", "whats app", "واتساپ", "وەتساپ", "واتس ئاپ", "وەتس ئەپ", "واتسئاپ", "وەتسئاپ", "واتس اپ"),
             processes=("whatsapp.exe", "whatsapp.root.exe")),
    AppAlias("youtube", "YouTube", ("YouTube",),
             ("youtube", "you tube", "یوتیوب", "یووتیوب", "یوتووب", "یووتووب", "یوتیووب"),
             url="https://www.youtube.com", reuse_window=False),
    AppAlias("word", "Word", ("Word", "Microsoft Word"),
             ("word", "microsoft word", "ms word", "وۆرد", "وۆڕد", "وۆرد ", "ڕەشنووسی وۆرد"),
             processes=("winword.exe",)),
    AppAlias("excel", "Excel", ("Excel", "Microsoft Excel"),
             ("excel", "microsoft excel", "ms excel", "ئێکسڵ", "ئێکسل", "ئیکسڵ", "ئەکسڵ", "ئێکسێل", "ئێکسڵی"),
             processes=("excel.exe",)),
    AppAlias("powerpoint", "PowerPoint", ("PowerPoint", "Microsoft PowerPoint"),
             ("powerpoint", "power point", "ms powerpoint", "پاوەرپۆینت", "پاوەر پۆینت", "پاوەرپوینت", "پاوەر پوینت"),
             processes=("powerpnt.exe",)),
    AppAlias("spotify", "Spotify", ("Spotify",),
             ("spotify", "سپۆتیفای", "سپوتیفای", "ئیسپۆتیفای"),
             processes=("spotify.exe",)),
    AppAlias("discord", "Discord", ("Discord",),
             ("discord", "دیسکۆرد", "دیسکۆڕد", "دیسکورد"),
             processes=("discord.exe",)),
    AppAlias("taskmgr", "Task Manager", ("Task Manager",),
             ("task manager", "taskmgr", "تاسک مانەجەر", "تاسک مەنەجەر", "تاسک مانیجەر", "بەڕێوەبەری ئەرکەکان"),
             processes=("taskmgr.exe",), exe="taskmgr.exe"),
    AppAlias("paint", "Paint", ("Paint",),
             ("paint", "mspaint", "ms paint", "پەینت", "پێنت", "پەینتی ویندۆز"),
             processes=("mspaint.exe",)),
    AppAlias("snipping", "Snipping Tool", ("Snipping Tool",),
             ("snipping tool", "snip", "screenshot tool", "سنیپینگ تووڵ", "سنیپینگ تول", "ئامرازی سکرینشۆت",
              "ئامرازی وێنەگرتنی شاشە"),
             processes=("snippingtool.exe",)),
    AppAlias("terminal", "Terminal", ("Terminal", "Windows Terminal"),
             ("terminal", "windows terminal", "powershell window", "تێرمیناڵ", "تێرمیناڵی ویندۆز", "تیرمیناڵ"),
             processes=("windowsterminal.exe",), reuse_window=False),
    AppAlias("cmd", "Command Prompt", ("Command Prompt",),
             ("cmd", "command prompt", "سی ئێم دی", "سی ئێم دی ", "کۆماند پرۆمپت", "کۆمەند پرۆمپت"),
             exe="cmd.exe", reuse_window=False),
    AppAlias("control", "Control Panel", ("Control Panel",),
             ("control panel", "کۆنترۆڵ پانێڵ", "کۆنتڕۆڵ پانێڵ", "کۆنترۆڵ پانڵ", "کۆنترۆل پانێل"),
             exe="control.exe", reuse_window=False),
)

# Words that surround an app name in a spoken command ("کرۆم بکەرەوە",
# "open the Chrome app"). Compared after normalize_ckb.
FILLER_WORDS: frozenset[str] = frozenset(normalize_ckb(w) for w in (
    "بکەرەوە", "بکەوە", "بکەرەوه", "بکرێتەوە", "بکەیتەوە", "کردنەوە", "کردنەوەی", "بۆم", "بۆ", "تکایە",
    "دەتوانیت", "دەتوانی", "ئەپی", "ئەپ", "ئەپەکە", "بەرنامەی", "بەرنامە", "بەرنامەکە", "پرۆگرامی", "پرۆگرام",
    "پرۆگرامەکە", "ئاپی", "ئاپ", "ئاپەکە", "ئەپلیکەیشنی", "ئەپلیکەیشن", "سام", "دەکەیتەوە", "هەڵبکە", "هەلبکە",
    "بێنە", "پیشان", "بدە", "بهێنە", "سەر", "شاشە", "و",
    "open", "launch", "start", "run", "please", "the", "app", "application", "program", "for", "me", "up",
))
_SUFFIXES = ("ەکانم", "ەکانت", "ەکان", "ەکەم", "ەکەت", "ەکەی", "یەکە", "ەکە", "یەک", "کە", "ی")
_LATIN_RE = re.compile(r"[^a-z0-9 ]+")


def strip_suffix(word: str) -> str:
    """Drop one Sorani definite/possessive suffix ("کرۆمەکە" -> "کرۆم")."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def clean_query(text: str) -> str:
    """Normalised app query without filler words ("کرۆم بکەرەوە" -> "کرۆم")."""
    tokens = normalize_ckb(text, strip_punct=True).split()
    kept = [t for t in tokens if t not in FILLER_WORDS]
    return " ".join(kept or tokens)


def query_variants(text: str) -> list[str]:
    """The cleaned query plus a suffix-stripped form (both normalised)."""
    base = clean_query(text)
    stripped = " ".join(strip_suffix(w) for w in base.split())
    return [base] if stripped == base else [base, stripped]


# Sorani (Arabic script) -> rough Latin, for fuzzy matching English app names
# that are not in the alias table. Vowel letters are approximations on purpose.
_TRANSLIT = {
    "ا": "a", "آ": "a", "أ": "a", "إ": "i", "ئ": "", "ء": "", "ب": "b", "پ": "p", "ت": "t", "ث": "s",
    "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ذ": "z", "ر": "r", "ڕ": "r", "ز": "z", "ژ": "zh",
    "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ع": "", "غ": "gh", "ف": "f", "ڤ": "v",
    "ق": "q", "ک": "k", "ك": "k", "گ": "g", "ل": "l", "ڵ": "l", "م": "m", "ن": "n", "ه": "h", "ھ": "h",
    "ە": "e", "ة": "e", "ۆ": "o", "ؤ": "o", "ێ": "e",
}
_VOWELS = set("aeiou")


def transliterate(text: str) -> str:
    """Very rough Sorani -> Latin ("تێلێگرام" -> "telegram", "واتساپ" -> "watsap")."""
    out: list[str] = []
    value = normalize_ckb(text).replace("وو", "u")
    for i, char in enumerate(value):
        if char == "و":
            prev = out[-1][-1:] if out and out[-1] else ""
            start = i == 0 or value[i - 1] == " "
            out.append("w" if start or prev in _VOWELS else "u")
        elif char == "ی":
            prev = out[-1][-1:] if out and out[-1] else ""
            start = i == 0 or value[i - 1] == " "
            out.append("y" if start or prev in _VOWELS else "i")
        elif char in _TRANSLIT:
            out.append(_TRANSLIT[char])
        else:
            out.append(char)
    return _LATIN_RE.sub("", "".join(out).lower()).strip()


def _score(query: str, alias: str) -> float:
    """Similarity 0..100 of a cleaned query and one alias (both normalised)."""
    from rapidfuzz import fuzz

    if not query or not alias:
        return 0.0
    if query == alias:
        return 100.0
    best = float(fuzz.ratio(query, alias))
    q_words, a_words = query.split(), alias.split()
    if len(q_words) > len(a_words):
        # The alias may be embedded in a longer sentence: compare windows of
        # the same word count ("زوو کرۆم بکەرەوە بۆم" -> "کرۆم").
        n = len(a_words)
        for i in range(len(q_words) - n + 1):
            window = " ".join(q_words[i:i + n])
            if window == alias:
                return 96.0
            best = max(best, float(fuzz.ratio(window, alias)) - 2.0)
    return best


def match_alias(text: str, *, threshold: float = 82.0,
                extra: Iterable[AppAlias] = ()) -> tuple[AppAlias, float] | None:
    """Best alias for a spoken app name, or None under ``threshold``.

    Short aliases (<= 4 letters, e.g. "ئێج", "mt5") must match exactly or as a
    whole word: one wrong letter in a 3-letter word is a different word.
    """
    best: tuple[AppAlias, float] | None = None
    variants = query_variants(text)
    for spec in (*extra, *ALIASES):
        for alias in spec.aliases:
            norm = normalize_ckb(alias, strip_punct=True)
            for query in variants:
                score = _score(query, norm)
                if len(norm.replace(" ", "")) <= 4 and score < 96.0:
                    continue
                if best is None or score > best[1]:
                    best = (spec, score)
    if best is None or best[1] < threshold:
        return None
    return best


def alias_by_key(key: str) -> AppAlias | None:
    return next((a for a in ALIASES if a.key == key), None)


def aliases_for_process(process: str) -> list[AppAlias]:
    """Alias specs whose windows belong to ``process`` (exe basename)."""
    name = process.lower()
    return [a for a in ALIASES if name in a.processes]


__all__ = ["ALIASES", "AppAlias", "FILLER_WORDS", "alias_by_key", "aliases_for_process", "clean_query",
           "match_alias", "query_variants", "strip_suffix", "transliterate"]
