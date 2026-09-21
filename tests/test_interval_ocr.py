"""Toolbar interval parsing. No live TradingView window is required."""

from __future__ import annotations

import pytest

from sam_backend.trading.types import parse_interval_token, pick_visible_interval


@pytest.mark.parametrize("token,expected", [
    ("15m", "M15"),
    ("15", "M15"),
    ("M15", "M15"),
    ("1h", "H1"),
    ("1H", "H1"),
    ("60", "H1"),
    ("4h", "H4"),
    ("D", "D1"),
    ("1D", "D1"),
    ("1", "M1"),
    ("1m", "M1"),
    ("1M", "MN1"),
    ("monthly", "MN1"),
    ("5", "M5"),
    ("W", "W1"),
])
def test_toolbar_tokens_map_onto_canonical_timeframes(token, expected):
    assert parse_interval_token(token) == expected


def test_noise_and_empty_tokens_are_ignored():
    assert parse_interval_token("") is None
    assert parse_interval_token("XAUUSD") is None
    assert parse_interval_token("Indicators") is None


def test_a_closed_toolbar_shows_one_interval_after_the_symbol():
    tokens = [("XAUUSD", 40.0), ("15m", 120.0), ("Indicators", 260.0)]
    assert pick_visible_interval(tokens, symbol="XAUUSD") == "M15"


def test_an_open_interval_menu_still_picks_the_button_next_to_the_symbol():
    tokens = [
        ("XAUUSD", 40.0),
        ("1m", 110.0),
        ("5m", 150.0),
        ("15m", 190.0),
        ("1h", 230.0),
        ("D", 270.0),
    ]
    assert pick_visible_interval(tokens, symbol="XAUUSD") == "M1"


def test_interval_tokens_to_the_left_of_the_symbol_are_not_the_chart_interval():
    tokens = [("15m", 10.0), ("XAUUSD", 80.0), ("1h", 140.0)]
    assert pick_visible_interval(tokens, symbol="XAUUSD") == "H1"


def test_a_toolbar_with_no_interval_returns_nothing():
    tokens = [("XAUUSD", 40.0), ("Indicators", 200.0)]
    assert pick_visible_interval(tokens, symbol="XAUUSD") is None
