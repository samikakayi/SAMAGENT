"""Spoken app names -> apps: Sorani/English aliases, spelling variants and
fuzzy matching against the Start-menu index (acceptance 6: Chrome, Edge,
TradingView, Telegram, MT5, VS Code, Notepad by English and Sorani names)."""

from __future__ import annotations

import pytest

from sam.hands.aliases import clean_query, match_alias, query_variants, strip_suffix, transliterate
from sam.hands.apps import AppIndex
from tests.hands_helpers import START_ROWS, fake_windows


@pytest.mark.parametrize("spoken, key", [
    ("ترەیدینگ ڤیو", "tradingview"), ("تریدینگ ڤیو", "tradingview"), ("ترێدینگ ڤیو", "tradingview"),
    ("ترێدینگ", "tradingview"), ("تریدینگ", "tradingview"), ("ترەیدینگ ڤیو بکەرەوە", "tradingview"),
    ("TradingView", "tradingview"), ("trading view", "tradingview"), ("تریدینگ ویو", "tradingview"),
    ("کرۆم", "chrome"), ("کڕۆم", "chrome"), ("کرۆم بکەرەوە", "chrome"), ("کرۆمەکە بکەرەوە", "chrome"),
    ("گووگڵ کرۆم", "chrome"), ("Google Chrome", "chrome"),
    ("ئێج", "edge"), ("مایکرۆسۆفت ئێج", "edge"), ("Edge", "edge"),
    ("تێلێگرام", "telegram"), ("تەلەگرام بکەرەوە", "telegram"), ("Telegram", "telegram"),
    ("مێتاترەیدەر", "mt5"), ("میتاترەیدەر", "mt5"), ("مێتاترەیدەر ٥", "mt5"), ("MT5", "mt5"),
    ("MetaTrader 5", "mt5"),
    ("واتساپ", "whatsapp"), ("WhatsApp", "whatsapp"), ("ئێکسڵ", "excel"), ("Excel", "excel"),
    ("ڤی ئێس کۆد", "vscode"), ("VS Code", "vscode"), ("visual studio code", "vscode"),
    ("نۆتپاد", "notepad"), ("نۆتپاد بکەرەوە", "notepad"), ("Notepad", "notepad"),
    ("ژمێرەر", "calculator"), ("یوتیوب", "youtube"), ("ڕێکخستنەکان", "settings"), ("تاسک مانەجەر", "taskmgr"),
    ("پاوەرپۆینت", "powerpoint"), ("وۆرد", "word"), ("کێرسەر", "cursor"), ("تێرمیناڵ", "terminal"),
    ("سنیپینگ تووڵ", "snipping"), ("پەینت", "paint"), ("کۆنترۆڵ پانێڵ", "control"), ("سپۆتیفای", "spotify"),
    ("دیسکۆرد", "discord"), ("فایل ئێکسپلۆرەر", "explorer"), ("سی ئێم دی", "cmd"),
])
def test_spoken_names_match_the_right_app(spoken: str, key: str) -> None:
    found = match_alias(spoken)
    assert found is not None, spoken
    assert found[0].key == key, (spoken, found[0].key, found[1])


@pytest.mark.parametrize("spoken", ["بەیانیت باش", "hello there", "ئەمڕۆ هەوا چۆنە", "xyz", "ئێ"])
def test_unrelated_words_match_no_app(spoken: str) -> None:
    assert match_alias(spoken) is None


def test_short_aliases_need_an_exact_word() -> None:
    # "ئێج" (Edge) is three letters: a one-letter slip is a different word.
    assert match_alias("ئێژ") is None
    assert match_alias("mt6") is None


def test_query_cleanup_and_suffixes() -> None:
    assert clean_query("سام، کرۆم بکەرەوە بۆم تکایە") == "کرۆم"
    assert clean_query("please open the Chrome app") == "chrome"
    assert strip_suffix("کرۆمەکە") == "کرۆم"
    assert strip_suffix("ژمێرەرەکە") == "ژمێرەر"
    assert strip_suffix("ئێج") == "ئێج"  # too short to strip
    assert query_variants("تێرمیناڵی") == ["تێرمیناڵی", "تێرمیناڵ"]
    # Arabic ي/ك spellings from speech-to-text are the same word
    assert clean_query("كرۆم") == clean_query("کرۆم")


def test_transliteration_reaches_english_names() -> None:
    assert transliterate("تێلێگرام") == "telegram"
    assert transliterate("واتساپ") == "watsap"
    assert transliterate("زووم") == "zum"


@pytest.fixture
async def index(make_app):
    app = make_app()
    app.load_packages(["sam.hands"])
    windows, _ = fake_windows([])
    idx = AppIndex(app, windows=windows, enumerate_fn=lambda: [dict(r) for r in START_ROWS])
    assert await idx.refresh() == len(START_ROWS)
    return idx


@pytest.mark.parametrize("spoken, name", [
    ("Chrome", "Google Chrome"), ("کرۆم", "Google Chrome"), ("Edge", "Microsoft Edge"), ("ئێج", "Microsoft Edge"),
    ("TradingView", "TradingView"), ("ترەیدینگ ڤیو", "TradingView"), ("Telegram", "Telegram"),
    ("تێلێگرام", "Telegram"), ("MetaTrader 5", "MetaTrader 5"), ("مێتاترەیدەر", "MetaTrader 5"),
    ("VS Code", "Visual Studio Code"), ("ڤی ئێس کۆد", "Visual Studio Code"), ("Notepad", "Notepad"),
    ("نۆتپاد", "Notepad"), ("WhatsApp", "WhatsApp"), ("واتساپ", "WhatsApp"), ("Excel", "Excel"), ("ئێکسڵ", "Excel"),
    ("obs studio", "OBS Studio"), ("obs", "OBS Studio"),
])
async def test_index_resolves_the_seven_apps_v1_missed_and_more(index, spoken: str, name: str) -> None:
    entry = await index.resolve(spoken)
    assert entry is not None, spoken
    assert entry.name == name


async def test_tradingview_prefers_the_configured_install(index) -> None:
    entry = await index.resolve("تریدینگ ڤیو")
    assert entry.aumid == "TradingView.Desktop_n534cwy3pjxzj!TradingView.Desktop"
    assert entry.kind == "packaged" and entry.alias_key == "tradingview"


async def test_entry_kinds_and_uninstallers(index) -> None:
    telegram = await index.resolve("telegram")
    assert telegram.kind == "desktop" and telegram.aumid == "Telegram.TelegramDesktop"
    assert "unins" not in (telegram.path or "")
    mt5 = await index.resolve("mt5")
    assert mt5.kind == "exe" and mt5.path.endswith("terminal64.exe")
    youtube = await index.resolve("یوتیوب")
    assert youtube.kind == "url" and youtube.path.startswith("https://")
    settings = await index.resolve("ڕێکخستنەکان")
    assert settings.name == "Settings" and settings.kind == "packaged"


async def test_firefox_is_not_confused_with_tor_browser(index) -> None:
    # Tor Browser ships a firefox.exe; asking for Firefox must not open it.
    entry, suggestions = await index.resolve_with_candidates("Firefox")
    assert entry is None or entry.name != "Tor Browser"


async def test_unknown_app_gives_suggestions(index) -> None:
    entry, suggestions = await index.resolve_with_candidates("Photoshop")
    assert entry is None
    assert len(suggestions) <= 3


async def test_user_aliases_from_settings(index) -> None:
    index.app.config.set("hands.app_aliases", {"بەرنامەی چارت": "TradingView", "loop": "loop2", "loop2": "loop"})
    entry = await index.resolve("بەرنامەی چارت")
    assert entry is not None and entry.name == "TradingView"
    assert await index.resolve("loop") is None  # cycles end


async def test_index_is_cached_in_the_database(make_app) -> None:
    app = make_app()
    app.load_packages(["sam.hands"])
    windows, _ = fake_windows([])
    first = AppIndex(app, windows=windows, enumerate_fn=lambda: [dict(r) for r in START_ROWS])
    await first.refresh()

    def never() -> list:
        raise AssertionError("the cached index must be used")
    second = AppIndex(app, windows=windows, enumerate_fn=never)
    entry = await second.resolve("کرۆم")
    assert entry is not None and entry.name == "Google Chrome"
