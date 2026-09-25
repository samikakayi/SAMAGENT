"""Chart bridge: spoken timeframes/symbols -> TradingView values, colours, draw-spec validation."""

from __future__ import annotations

import pytest

from sam.trading.common import COLORS
from sam.trading.tv_parse import (DEFAULT_SPAN_BARS, canonical_request, clean_tag, clean_text, instrument_key,
                                  kinds_summary_ckb, normalize_item, parse_color, parse_tv_resolution,
                                  resolution_label_ckb, same_resolution, semantic_role, tv_symbol_for)

NOW = 1_790_243_400.0


@pytest.mark.parametrize("text,expected", [
    # the phrases from the brief
    ("١ خولەک", "1"), ("1m", "1"), ("٥ خولەک", "5"), ("١٥ خولەک", "15"), ("کاتژمێرێک", "60"), ("1h", "60"),
    ("H1", "60"), ("٤ کاتژمێر", "240"), ("ڕۆژانە", "1D"), ("D", "1D"), ("هەفتانە", "1W"),
    # more spoken forms
    ("۱۵ خولەک", "15"), ("پازدە خولەک", "15"), ("خولەکێک", "1"), ("سەعاتێک", "60"), ("نیو کاتژمێر", "30"),
    ("چارەکێک", "15"), ("دوو کاتژمێر", "120"), ("سێ کاتژمێر", "180"), ("چوار سەعاتی", "240"), ("١٥ خولەکی", "15"),
    ("ڕۆژێک", "1D"), ("هەفتەیەک", "1W"), ("مانگانە", "1M"), ("four hours", "240"), ("one hour", "60"),
    ("fifteen minutes", "15"), ("daily", "1D"), ("weekly", "1W"), ("45", "45"), ("240", "240"), ("15", "15"),
    ("M15", "15"), ("4H", "240"), ("1D", "1D"), ("1W", "1W"), ("1M", "1M"), ("تایم فرەیمی ١٥", "15"),
    # unknown / not offered by TradingView without custom intervals
    ("7m", None), ("banana", None), ("", None), (None, None), ("دوو ڕۆژ", None), ("٩٠ خولەک", None),
])
def test_parse_tv_resolution(text, expected):
    assert parse_tv_resolution(text) == expected


def test_resolution_helpers():
    assert same_resolution("1D", "D") and same_resolution("60", "60") and not same_resolution("15", "60")
    assert not same_resolution(None, "15")
    assert resolution_label_ckb("15") == "پازدە خولەک"
    assert resolution_label_ckb("1D") == "ڕۆژانە" and resolution_label_ckb("D") == "ڕۆژانە"
    assert resolution_label_ckb("60") == "یەک کاتژمێر"
    assert resolution_label_ckb("7") == "7 خولەک"


@pytest.mark.parametrize("text,expected", [
    ("گۆڵد", "XAUUSD"), ("زێڕ", "XAUUSD"), ("ئاڵتوون", "XAUUSD"), ("gold", "XAUUSD"), ("XAUUSD", "XAUUSD"),
    ("زێڕی", "XAUUSD"), ("زێڕەکە", "XAUUSD"), ("xau/usd", "XAUUSD"), ("XAU/USD", "XAUUSD"), ("gold spot", "XAUUSD"),
    ("بیتکۆین", "BTCUSD"), ("ئیتریۆم", "ETHUSD"), ("یۆرۆ", "EURUSD"), ("نەوت", "USOIL"), ("dxy", "USDX"),
    ("AAPL", "AAPL"), ("نرخی زێڕ", "XAUUSD"), ("شتێکی نەناسراو", None), ("", None),
])
def test_canonical_request(text, expected):
    assert canonical_request(text) == expected


def test_instrument_key_equivalences():
    assert instrument_key("TVC:GOLD") == instrument_key("OANDA:XAUUSD") == instrument_key("XAUUSD.m.e") == "XAUUSD"
    assert instrument_key("BINANCE:BTCUSDT") == instrument_key("BTCUSD") == "BTCUSD"


@pytest.mark.parametrize("request_,current,expected", [
    ("گۆڵد", "TVC:GOLD", ("TVC:GOLD", "same")),           # keep the user's own gold feed
    ("زێڕ", "OANDA:XAUUSD", ("OANDA:XAUUSD", "same")),
    ("gold", "BINANCE:BTCUSDT", ("OANDA:XAUUSD", "mapped")),
    ("XAUUSD", None, ("OANDA:XAUUSD", "mapped")),
    ("بیتکۆین", "TVC:GOLD", ("BINANCE:BTCUSDT", "mapped")),
    ("btc", "BINANCE:BTCUSDT", ("BINANCE:BTCUSDT", "same")),
    ("dxy", "TVC:GOLD", ("TVC:DXY", "mapped")),
    ("oanda:xauusd", "TVC:GOLD", ("OANDA:XAUUSD", "explicit")),    # an explicit exchange is honoured
    ("TVC:GOLD", "TVC:GOLD", ("TVC:GOLD", "same")),
    ("AAPL", "TVC:GOLD", ("AAPL", "passthrough")),
    ("شتێکی نەناسراو", "TVC:GOLD", (None, "unknown")),
])
def test_tv_symbol_for(request_, current, expected):
    assert tv_symbol_for(request_, current=current) == expected


def test_tv_symbol_override_from_settings():
    overrides = {"XAUUSD": {"mt5": "XAUUSD.m.e", "tv": "PEPPERSTONE:XAUUSD"}}
    assert tv_symbol_for("زێڕ", current="BINANCE:BTCUSDT", overrides=overrides) == ("PEPPERSTONE:XAUUSD", "mapped")
    assert tv_symbol_for("زێڕ", current="TVC:GOLD", overrides=overrides) == ("TVC:GOLD", "same")


@pytest.mark.parametrize("value,expected", [
    ("#26A69A", "#26a69a"), ("26a69a", "#26a69a"), ("#fa0", "#ffaa00"), ("#11223344", "#112233"),
    ("support", COLORS["support"]), ("resistance", COLORS["resistance"]), ("red", "#f23645"), ("سەوز", "#089981"),
    ("شین", "#2962ff"), ("nonsense", None), ("", None), (None, None), (12, None),
])
def test_parse_color(value, expected):
    assert parse_color(value) == expected


@pytest.mark.parametrize("text,role", [
    ("هێڵی پشتگیری", "support"), ("بەرگری", "resistance"), ("Support 4250", "support"), ("SL", "stop"),
    ("ستۆپ", "stop"), ("TP1", None), ("tp", "target"), ("ئامانج", "target"), ("entry", "entry"), ("", None),
    ("slope", None),
])
def test_semantic_role(text, role):
    assert semantic_role(text) == role


def test_clean_text_and_tag():
    assert clean_text("  پشتگیری\n‮evil\x00 ") == "پشتگیری evil"
    assert len(clean_text("x" * 200)) == 60
    assert clean_tag("analysis:12") == "analysis:12" and clean_tag("drop table;--") == "droptable--"
    assert clean_tag("") == "user-request"


def test_normalize_horizontal_line_support_colour_and_label():
    spec, error = normalize_item({"kind": "horizontal_line", "points": [{"price": "4250.5"}], "text": "پشتگیری"},
                                 last_price=4255.0, now=NOW)
    assert error is None
    assert spec["kind"] == "horizontal_line" and spec["points"] == [{"price": 4250.5}]
    assert spec["color"] == COLORS["support"] and spec["role"] == "support"
    assert spec["overrides"]["linecolor"] == COLORS["support"] and spec["overrides"]["text"] == "پشتگیری"
    assert spec["lock"] is False


def test_normalize_aliases_numbers_and_explicit_colour():
    spec, error = normalize_item({"kind": "hline", "points": [4260], "color": "#abc", "style": {"width": 9,
                                  "line_style": "dashed", "lock": True}}, last_price=4255.0, now=NOW)
    assert error is None and spec["kind"] == "horizontal_line" and spec["color"] == "#aabbcc"
    assert spec["overrides"]["linewidth"] == 4 and spec["overrides"]["linestyle"] == 2 and spec["lock"] is True


def test_two_point_kinds_get_a_span_when_timeless():
    spec, error = normalize_item({"kind": "rectangle", "points": [{"price": 4240}, {"price": 4250}], "text": "zone"},
                                 last_price=4255.0, now=NOW)
    assert error is None
    assert spec["points"][0]["bars_ago"] == DEFAULT_SPAN_BARS and spec["points"][1]["bars_ago"] == 0
    assert spec["overrides"]["backgroundColor"].startswith("rgba(") and spec["overrides"]["fillBackground"] is True
    spec, _ = normalize_item({"kind": "trend_line", "points": [{"price": 4240, "time": NOW - 3600},
                                                              {"price": 4250}]}, last_price=4255.0, now=NOW)
    assert spec["points"][0] == {"price": 4240.0, "time": int(NOW - 3600)} and "bars_ago" not in spec["points"][1]


def test_time_in_milliseconds_is_converted():
    spec, error = normalize_item({"kind": "arrow_up", "points": [{"price": 4250, "time": int(NOW * 1000)}]},
                                 last_price=4255.0, now=NOW)
    assert error is None and spec["points"][0]["time"] == int(NOW) and spec["color"] == COLORS["support"]


@pytest.mark.parametrize("item,fragment", [
    ({"kind": "circle", "points": [{"price": 1}]}, "unknown kind"),
    ({"kind": "trend_line", "points": [{"price": 4250}]}, "exactly 2 points"),
    ({"kind": "long_position", "points": [{"price": 4250}]}, "entry, stop and target"),
    ({"kind": "horizontal_line", "points": [{"price": "abc"}]}, "numeric price"),
    ({"kind": "horizontal_line", "points": [{"price": 42}]}, "far from the chart price"),
    ({"kind": "horizontal_line", "points": [{"price": float("nan")}]}, "invalid price"),
    ({"kind": "text", "points": [{"price": 4250, "time": 5}]}, "time is out of range"),
    ({"kind": "text", "points": [{"price": 4250, "bars_ago": -2}]}, "bars_ago is out of range"),
    ({"kind": "horizontal_line"}, "needs points"),
    ("not a dict", "must be an object"),
    ({"kind": "long_position", "points": [{"price": 4250}, {"price": 4260}, {"price": 4270}]},
     "stop < entry < target"),
    ({"kind": "short_position", "points": [{"price": 4250}, {"price": 4240}, {"price": 4230}]},
     "target < entry < stop"),
])
def test_normalize_item_errors(item, fragment):
    spec, error = normalize_item(item, last_price=4255.0, now=NOW)
    assert spec is None and fragment in error


def test_position_spec_carries_stop_and_target():
    spec, error = normalize_item({"kind": "long_position", "text": "ignored",
                                  "points": [{"price": 4250, "bars_ago": 3}, {"price": 4240}, {"price": 4280}]},
                                 last_price=4255.0, now=NOW)
    assert error is None
    assert spec["points"] == [{"price": 4250.0, "bars_ago": 3}]
    assert spec["position"] == {"stop": 4240.0, "target": 4280.0} and spec["text"] == ""
    short, error = normalize_item({"kind": "short_position", "points": [{"price": 4250}, {"price": 4260},
                                                                        {"price": 4220}]}, last_price=4255.0, now=NOW)
    assert error is None and short["position"] == {"stop": 4260.0, "target": 4220.0}


def test_kinds_summary_is_sorani():
    text = kinds_summary_ckb(["horizontal_line", "horizontal_line", "trend_line", "fib_retracement"])
    assert text == "2 هێڵی ئاسۆیی، 1 هێڵی ترێند، 1 فیبۆناچی"
