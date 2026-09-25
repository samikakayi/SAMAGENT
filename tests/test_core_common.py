from __future__ import annotations

import pytest

from sam.textnorm import is_arabic_script, normalize_ckb
from sam.timing import Timing
from sam.trading.common import canonical_symbol, from_tv_resolution, normalize_timeframe, to_tv_resolution


@pytest.mark.parametrize("value,expected", [
    ("15", "M15"), ("15m", "M15"), ("M15", "M15"), ("m15", "M15"), ("1h", "H1"), ("60", "H1"), ("240", "H4"),
    ("4H", "H4"), ("h4", "H4"), ("D", "D1"), ("1D", "D1"), ("daily", "D1"), ("W", "W1"), ("1M", "MN1"),
    ("1440", "D1"), ("١٥ خولەک", "M15"), ("۱۵ خولەک", "M15"), ("پازدە خولەک", "M15"), ("چوار سەعات", "H4"),
    ("یەک کاتژمێر", "H1"), ("ڕۆژانە", "D1"), ("هەفتانە", "W1"), ("15 min", "M15"), (5, "M5"),
    ("7m", None), ("banana", None), ("", None),
])
def test_normalize_timeframe(value, expected):
    assert normalize_timeframe(value) == expected


def test_tv_resolution_round_trip():
    for tf, res in {"M1": "1", "M15": "15", "H1": "60", "H4": "240", "D1": "1D", "W1": "1W", "MN1": "1M"}.items():
        assert to_tv_resolution(tf) == res
        assert from_tv_resolution(res) == tf
    assert from_tv_resolution("D") == "D1"


@pytest.mark.parametrize("value,expected", [
    ("زێڕ", "XAUUSD"), ("ئاڵتوون", "XAUUSD"), ("گۆڵد", "XAUUSD"), ("Gold", "XAUUSD"), ("OANDA:XAUUSD", "XAUUSD"),
    ("TVC:GOLD", "XAUUSD"), ("xauusd", "XAUUSD"), ("XAUUSD.m.e", "XAUUSD"), ("بیتکۆین", "BTCUSD"),
    ("BINANCE:BTCUSDT", "BTCUSD"), ("OANDA:NAS100USD", "NAS100"), ("dxy", "USDX"),   # same instrument (review 2026-09-24)
    # Latin spellings of زێڕ that models sent as a symbol (voice + integration live checks)
    ("ZAR", "XAUUSD"), ("ZEUR", "XAUUSD"), ("Zêr", "XAUUSD"), ("zhir", "XAUUSD"), ("EURUSD", "EURUSD"),
    ("USDZAR", "USDZAR"),
])
def test_canonical_symbol(value, expected):
    assert canonical_symbol(value) == expected


def test_normalize_ckb_variants():
    assert normalize_ckb("كوردي‌ ١٢٣") == "کوردی ۱۲۳".replace("۱۲۳", "123")
    assert normalize_ckb("  بەڵێ،  باشە!  ", strip_punct=True) == "بەڵێ باشە"
    assert is_arabic_script("سڵاو، چۆنی؟ SAM") and not is_arabic_script("hello there")


def test_timing_turn_persists_stages(tmp_path):
    from sam.db import Database

    db = Database(tmp_path / "t.sqlite3")
    timing = Timing(db)
    with timing.turn("cascade") as turn:
        turn.add("stt", 420.0)
        with turn.stage("llm_total", model="groq:x"):
            pass
        turn.mark("first_audio")
    rows = db.query("SELECT stage, kind, turn_id FROM timings ORDER BY id")
    assert [r["stage"] for r in rows] == ["stt", "llm_total", "first_audio", "total"]
    assert {r["turn_id"] for r in rows} == {turn.turn_id} and rows[0]["kind"] == "cascade"
    timing.record("tool:open_app", 12.5, kind="tool")
    assert timing.recent(1)[0]["stage"] == "tool:open_app"
