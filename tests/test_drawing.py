from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from sam_backend.cancellation import CancellationManager
from sam_backend.config import Settings
from sam_backend.contracts import ExecutionStatus
from sam_backend.db import Database
from sam_backend.trading.calibration import (
    MAX_RESIDUAL_PIXELS,
    Calibration,
    ChartCalibrator,
    fit_linear,
    fit_with_outlier_rejection,
    geometry_hash,
)
from sam_backend.trading.drawing import SEMANTIC_TYPES, DrawingEngine, DrawRequest, Layer
from sam_backend.trading.ocr import parse_price
from sam_backend.trading.service import TradingService


@pytest.fixture()
def database(tmp_path: Path) -> Database:
    return Database(tmp_path / "sam.sqlite3")


@pytest.fixture()
def trading(tmp_path: Path) -> TradingService:
    settings = Settings(
        project_root=tmp_path,
        workspace_root=tmp_path / "workspace",
        data_dir=tmp_path / "data",
    )
    settings.prepare()
    return TradingService(settings, Database(settings.database_path), CancellationManager())


# --- Calibration math --------------------------------------------------------


def test_price_scale_fit_is_exact_for_a_linear_axis():
    points = [(100.0, 3450.0), (200.0, 3400.0), (300.0, 3350.0), (400.0, 3300.0)]
    slope, intercept = fit_linear(points)
    assert slope == pytest.approx(-0.5)
    assert intercept == pytest.approx(3500.0)
    assert slope * 250 + intercept == pytest.approx(3375.0)


def test_a_misread_price_label_is_rejected_instead_of_tilting_the_mapping():
    corrupted = [(100.0, 3450.0), (200.0, 3400.0), (300.0, 9350.0), (400.0, 3300.0), (500.0, 3250.0)]
    slope, intercept, inliers, worst = fit_with_outlier_rejection(corrupted)
    assert (300.0, 9350.0) not in inliers
    assert slope == pytest.approx(-0.5)
    assert slope * 300 + intercept == pytest.approx(3350.0)
    assert worst == pytest.approx(0.0)


def test_a_non_linear_axis_is_refused_rather_than_linearised():
    # A logarithmic scale has no linear consensus, so no mapping may be returned.
    logarithmic = [(float(y), 1000.0 * (1.5 ** (y / 100.0))) for y in (0, 100, 200, 300, 400)]
    assert fit_with_outlier_rejection(logarithmic) is None


def test_several_misread_labels_do_not_drag_the_fit():
    # Two corrupted readings among six; the majority must still win.
    points = [(100.0, 3450.0), (200.0, 3400.0), (300.0, 50.0), (400.0, 3300.0),
              (500.0, 3250.0), (600.0, 921.0)]
    slope, intercept, inliers, worst = fit_with_outlier_rejection(points)
    assert slope == pytest.approx(-0.5)
    assert len(inliers) == 4
    assert all(price > 1000 for _, price in inliers)


def test_clipped_edge_labels_are_dropped_by_magnitude():
    # OCR turns a half-drawn 3475.00 into 75.00 at the top of a real price scale.
    from sam_backend.trading.calibration import drop_magnitude_outliers

    kept = drop_magnitude_outliers([(10.0, 75.0), (100.0, 3450.0), (200.0, 3425.0),
                                    (300.0, 3400.0), (400.0, 3375.0)])
    assert (10.0, 75.0) not in kept
    assert len(kept) == 4


def test_moving_the_window_changes_the_viewport_identity():
    base = {"left": 0, "top": 0, "right": 1920, "bottom": 1080}
    assert geometry_hash(base) == geometry_hash(dict(base))
    assert geometry_hash(base) != geometry_hash({**base, "left": 1})
    assert geometry_hash(base) != geometry_hash({**base, "bottom": 1000})


def test_price_scale_region_stays_inside_the_window():
    geometry = {"left": 100, "top": 50, "right": 1300, "bottom": 850}
    scale = ChartCalibrator.price_scale_region(geometry)
    plot = ChartCalibrator.plot_region(geometry)
    assert scale["right"] == geometry["right"]
    assert scale["left"] > plot["left"]
    assert plot["right"] == scale["left"]
    assert plot["bottom"] < geometry["bottom"]


def test_calibration_round_trips_price_and_pixels():
    calibration = Calibration(slope=-0.5, intercept=3500.0, method="test", verified=True)
    assert calibration.price_at(200.0) == pytest.approx(3400.0)
    assert calibration.y_at(3400.0) == pytest.approx(200.0)
    assert calibration.price_at(calibration.y_at(3333.0)) == pytest.approx(3333.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("3412.50", 3412.5), ("3,412.50", 3412.5), ("1.1234", 1.1234), ("-45.5", -45.5),
     ("abc", None), ("12:30", None), ("", None), ("Vol", None)],
)
def test_only_numeric_axis_labels_become_price_anchors(text, expected):
    assert parse_price(text) == expected


# --- Pixel verification ------------------------------------------------------


def _chart(line_y: int | None = None) -> Image.Image:
    image = Image.new("RGB", (400, 300), "black")
    if line_y is not None:
        ImageDraw.Draw(image).line([(0, line_y), (399, line_y)], fill="white", width=2)
    return image


def test_a_drawn_line_is_detected_on_the_row_it_was_drawn():
    rows = DrawingEngine.changed_rows(_chart(), _chart(150))
    assert [row[0] for row in rows] == [150, 151]


def test_an_unchanged_chart_reports_no_drawing():
    assert DrawingEngine.changed_rows(_chart(), _chart()) == []


def test_a_line_at_the_wrong_price_does_not_satisfy_the_expected_row():
    rows = DrawingEngine.changed_rows(_chart(), _chart(40))
    assert all(abs(row[0] - 150) > 4 for row in rows)


# --- Ownership ---------------------------------------------------------------


def test_drawings_are_recorded_with_full_provenance(database: Database):
    record = database.record_drawing(
        symbol="xauusd", layer="SNR", drawing_type="resistance", label="R1",
        theory="snr", timeframe="H1", price=3412.5,
    )
    assert record["symbol"] == "XAUUSD"
    assert record["visible"] is True
    assert record["verified"] is False
    assert record["theory"] == "snr"
    assert record["price"] == pytest.approx(3412.5)


def test_owned_drawings_can_be_filtered_by_symbol_theory_and_layer(database: Database):
    database.record_drawing(symbol="XAUUSD", layer="SNR", drawing_type="support", theory="snr", price=1.0)
    database.record_drawing(symbol="XAUUSD", layer="ENTRY", drawing_type="entry", theory="snr", price=2.0)
    database.record_drawing(symbol="EURUSD", layer="SNR", drawing_type="support", theory="wyckoff", price=3.0)
    assert len(database.list_drawings()) == 3
    assert len(database.list_drawings(symbol="xauusd")) == 2
    assert len(database.list_drawings(theory="snr")) == 2
    assert len(database.list_drawings(layer="SNR")) == 2
    assert len(database.list_drawings(symbol="XAUUSD", layer="ENTRY")) == 1


def test_an_unfiltered_delete_is_refused_without_an_explicit_request(database: Database):
    database.record_drawing(symbol="XAUUSD", layer="SNR", drawing_type="support", price=1.0)
    with pytest.raises(ValueError, match="all_owned"):
        database.delete_drawings()
    assert len(database.list_drawings()) == 1


def test_deleting_one_theory_leaves_other_drawings_untouched(database: Database):
    database.record_drawing(symbol="XAUUSD", layer="SNR", drawing_type="support", theory="snr", price=1.0)
    database.record_drawing(symbol="XAUUSD", layer="THEORY", drawing_type="premium", theory="ict", price=2.0)
    removed = database.delete_drawings(theory="snr")
    assert len(removed) == 1
    remaining = database.list_drawings()
    assert len(remaining) == 1 and remaining[0]["theory"] == "ict"


def test_hiding_a_layer_only_affects_that_layer(database: Database):
    database.record_drawing(symbol="XAUUSD", layer="SNR", drawing_type="support", price=1.0)
    database.record_drawing(symbol="XAUUSD", layer="TARGETS", drawing_type="tp1", price=2.0)
    assert database.set_drawing_visibility(visible=False, layer="SNR") == 1
    visible = database.list_drawings(visible_only=True)
    assert len(visible) == 1 and visible[0]["layer"] == "TARGETS"


def test_verification_state_is_recorded_separately_from_creation(database: Database):
    record = database.record_drawing(symbol="XAUUSD", layer="SNR", drawing_type="support", price=1.0)
    assert record["verified"] is False and record["verified_at"] is None
    updated = database.mark_drawing_verified(record["id"], True, {"matched_row": 512})
    assert updated["verified"] is True
    assert updated["verified_at"] is not None
    assert updated["payload"]["matched_row"] == 512


# --- Calibration persistence -------------------------------------------------


def test_calibration_is_scoped_to_one_symbol_timeframe_and_viewport(database: Database):
    database.save_chart_calibration(
        window_handle=42, symbol="XAUUSD", timeframe="H1", geometry_hash="aaa",
        slope=-0.5, intercept=3500.0, method="windows_ocr_price_scale", verified=True,
    )
    assert database.get_chart_calibration(window_handle=42, symbol="XAUUSD", timeframe="H1", geometry_hash="aaa")
    # A different viewport, timeframe, or symbol must not reuse the mapping.
    assert database.get_chart_calibration(window_handle=42, symbol="XAUUSD", timeframe="H1", geometry_hash="bbb") is None
    assert database.get_chart_calibration(window_handle=42, symbol="XAUUSD", timeframe="M5", geometry_hash="aaa") is None
    assert database.get_chart_calibration(window_handle=42, symbol="EURUSD", timeframe="H1", geometry_hash="aaa") is None


def test_recalibrating_the_same_viewport_replaces_the_mapping(database: Database):
    for slope in (-0.5, -0.25):
        database.save_chart_calibration(
            window_handle=7, symbol="XAUUSD", timeframe="H1", geometry_hash="aaa",
            slope=slope, intercept=3500.0, method="windows_ocr_price_scale", verified=True,
        )
    stored = database.get_chart_calibration(window_handle=7, symbol="XAUUSD", timeframe="H1", geometry_hash="aaa")
    assert stored["slope"] == pytest.approx(-0.25)


def test_invalidating_a_window_clears_every_calibration_it_held(database: Database):
    for timeframe in ("H1", "M15"):
        database.save_chart_calibration(
            window_handle=9, symbol="XAUUSD", timeframe=timeframe, geometry_hash="aaa",
            slope=-0.5, intercept=3500.0, method="windows_ocr_price_scale", verified=True,
        )
    assert database.invalidate_chart_calibration(window_handle=9) == 2
    assert database.get_chart_calibration(window_handle=9, symbol="XAUUSD", timeframe="H1", geometry_hash="aaa") is None


# --- Service behaviour and permission gating ---------------------------------


def test_drawing_is_blocked_while_computer_control_is_off(trading: TradingService):
    result = trading.draw_annotation("support", 3400.0)
    assert result.status is ExecutionStatus.FAILED
    assert result.error_code == "COMPUTER_CONTROL_DISABLED"
    assert result.executed is False


def test_calibration_is_blocked_while_screen_access_is_off(trading: TradingService):
    result = trading.calibrate_chart()
    assert result.error_code == "SCREEN_ACCESS_DISABLED"
    assert result.verified is False


def test_drawing_without_an_analysis_does_not_invent_levels(trading: TradingService):
    result = trading.draw_analysis()
    assert result.status is ExecutionStatus.FAILED
    assert result.error_code == "NO_ANALYSIS"


def test_an_unknown_layer_is_rejected(trading: TradingService):
    assert trading.set_layer_visibility("NOT_A_LAYER", False).error_code == "UNKNOWN_LAYER"


def test_clearing_with_no_owned_drawings_succeeds_without_touching_the_chart(trading: TradingService):
    result = trading.clear_drawings(symbol="XAUUSD")
    assert result.status is ExecutionStatus.SUCCESS
    assert result.data["count"] == 0


def test_every_semantic_annotation_maps_to_a_known_layer():
    for annotation in SEMANTIC_TYPES:
        _, layer = DrawRequest(annotation=annotation, price=1.0).resolve()
        assert layer.value in Layer.__members__


def test_an_analysis_report_becomes_an_ordered_annotation_plan():
    report = {
        "setup": {
            "entry": 3400.0, "stop": 3390.0, "invalidation": 3388.0,
            "targets": [{"price": 3420.0}, {"price": 3440.0}, {"price": 3460.0}, {"price": 3999.0}],
        },
        "support": [{"price": 3380.0}, {"price": 3370.0}],
        "resistance": [{"price": 3430.0}],
    }
    plan = TradingService._drawing_plan(report)
    annotations = [item[0] for item in plan]
    assert annotations[:3] == ["entry", "stop", "invalidation"]
    assert "tp1" in annotations and "tp3" in annotations
    # Only the first three targets are annotated, so the chart stays readable.
    assert "tp4" not in annotations
    assert 3999.0 not in [item[1] for item in plan]
    assert annotations.count("support") == 2


def test_an_empty_report_produces_no_annotations():
    assert TradingService._drawing_plan({"setup": {}, "support": [], "resistance": []}) == []


# --- Regressions found by running against the live TradingView window --------


def test_screen_coordinates_and_captures_use_the_same_units():
    """A DPI-unaware process reports logical pixels but captures physical ones.

    At 175% scaling those differ by 1.75x, so every capture region derived from a
    window rectangle addressed the wrong part of the screen.
    """
    from sam_backend.dpi import ensure_dpi_awareness

    state = ensure_dpi_awareness()
    if state["mode"] == "not-windows":
        pytest.skip("DPI awareness is a Windows concern")
    assert state.get("coordinates_consistent") is True, (
        f"window coordinates and screen captures disagree: {state}"
    )


def test_the_plot_region_excludes_the_located_price_axis():
    geometry = {"left": 0, "top": 0, "right": 2880, "bottom": 1716}
    # A real chart put its axis at x=2652 with an icon rail to the right of it.
    bounded = ChartCalibrator.plot_region(geometry, axis_x=2652.5)
    assert bounded["right"] < 2652
    # Without a located axis the conservative fixed ratio still applies.
    fallback = ChartCalibrator.plot_region(geometry)
    assert fallback["right"] == ChartCalibrator.price_scale_region(geometry)["left"]


def test_an_implausible_axis_position_falls_back_instead_of_inverting_the_region():
    geometry = {"left": 0, "top": 0, "right": 2880, "bottom": 1716}
    for bogus in (10.0, -50.0, 9999.0):
        region = ChartCalibrator.plot_region(geometry, axis_x=bogus)
        assert region["right"] > region["left"]


def test_price_labels_are_grouped_into_columns_by_x_position():
    # A sidebar number and a chart watermark must not join the axis column.
    anchors = [
        {"x": 2650.0, "y": 500.0, "price": 4700.0},
        {"x": 2655.0, "y": 627.0, "price": 4660.0},
        {"x": 2652.0, "y": 756.0, "price": 4620.0},
        {"x": 400.0, "y": 300.0, "price": 174.0},
    ]
    columns = ChartCalibrator.cluster_columns(anchors)
    axis_column = max(columns, key=len)
    assert len(axis_column) == 3
    assert all(item["x"] > 2000 for item in axis_column)


def test_a_maximized_window_frame_is_not_used_as_the_capture_area():
    """A maximized frame rect overhangs the desktop by the invisible border."""
    from sam_backend.trading.drawing import DrawingEngine

    class State:
        window_geometry = {"left": -12, "top": -12, "right": 2892, "bottom": 1728}
        client_geometry = {"left": 0, "top": 0, "right": 2880, "bottom": 1716}

    assert DrawingEngine.chart_geometry(State()) == State.client_geometry

    class NoClient:
        window_geometry = {"left": 100, "top": 100, "right": 900, "bottom": 700}
        client_geometry = None

    assert DrawingEngine.chart_geometry(NoClient()) == NoClient.window_geometry


# --- Regressions from the live pointer-driving acceptance run ----------------


def test_the_hide_all_drawings_chord_is_on_the_forbidden_list():
    """Ctrl+Alt+H hides every annotation, including the user's own.

    Sending it during testing blanked the chart and made verification impossible,
    so it is recorded as forbidden rather than left as a latent hazard.
    """
    from sam_backend.trading.drawing import FORBIDDEN_CHORDS, SHORTCUTS

    assert ("ctrl", "alt", "h") in FORBIDDEN_CHORDS
    # Every chord SAM actually sends must be absent from the forbidden set.
    for primitive, chord in SHORTCUTS.items():
        assert tuple(chord) not in FORBIDDEN_CHORDS, primitive


def test_selection_retries_neighbouring_rows_before_giving_up():
    """A click computed from price can land a pixel beside a thin line."""
    from sam_backend.trading.drawing import DrawingEngine

    from sam_backend.trading.drawing import VERIFY_ROW_TOLERANCE

    offsets = DrawingEngine.SELECT_OFFSETS
    assert offsets[0] == 0, "the calibrated row must be tried first"
    assert set(offsets) >= {-1, 1, -2, 2}, "neighbouring rows must be attempted"
    assert len(offsets) == len(set(offsets)), "offsets must not repeat"
    # Never click further away than a line would still be judged correctly placed.
    assert max(abs(offset) for offset in offsets) <= VERIFY_ROW_TOLERANCE


def test_the_pointer_settles_before_a_shortcut_is_sent():
    """The crosshair only follows a move it actually receives."""
    from sam_backend.trading.drawing import (
        DRAW_SETTLE_SECONDS,
        MOUSE_SETTLE_OFFSET,
        MOUSE_SETTLE_SECONDS,
    )

    assert MOUSE_SETTLE_OFFSET > 0, "the pointer must approach from a different row"
    assert MOUSE_SETTLE_SECONDS >= 0.2, "too short a settle loses the crosshair update"
    assert DRAW_SETTLE_SECONDS >= 0.5, "the chart needs time to paint before verification"


# --- Two-anchor (drag) objects ------------------------------------------------


def test_two_anchor_types_map_to_drag_tools_with_layers():
    from sam_backend.trading.drawing import SHORTCUTS, TWO_ANCHOR_TYPES, TwoAnchorRequest

    for annotation in TWO_ANCHOR_TYPES:
        primitive, layer = TwoAnchorRequest(
            annotation=annotation, price_a=1.0, minutes_a=0.0, price_b=2.0, minutes_b=60.0
        ).resolve()
        assert primitive in SHORTCUTS, annotation
        assert layer.value in Layer.__members__


def test_the_time_axis_round_trips_minutes_and_pixels():
    calibration = Calibration(
        slope=-0.5, intercept=3500.0, method="test", verified=True,
        minutes_per_pixel=0.095238, time_intercept=1200.0,
    )
    assert calibration.time_calibrated is True
    assert calibration.minutes_at(calibration.x_at_minutes(1300.0)) == pytest.approx(1300.0)
    # A later time must sit further right.
    assert calibration.x_at_minutes(1300.0) > calibration.x_at_minutes(1200.0)


def test_an_uncalibrated_time_axis_refuses_to_place_an_anchor():
    calibration = Calibration(slope=-0.5, intercept=3500.0, method="test", verified=True)
    assert calibration.time_calibrated is False
    with pytest.raises(ValueError, match="time axis is not calibrated"):
        calibration.x_at_minutes(600.0)


def test_an_anchor_outside_the_plot_is_reported_off_screen():
    from sam_backend.trading.drawing import DrawingEngine

    calibration = Calibration(
        slope=-0.5, intercept=3500.0, method="test", verified=True,
        minutes_per_pixel=0.1, time_intercept=0.0,
    )
    engine = DrawingEngine.__new__(DrawingEngine)
    region = {"left": 0, "top": 0, "right": 1000, "bottom": 800}
    inside = DrawingEngine.anchor_to_screen(engine, calibration, 3300.0, 50.0, region)
    outside = DrawingEngine.anchor_to_screen(engine, calibration, 9999.0, 50.0, region)
    assert inside["on_screen"] is True
    assert outside["on_screen"] is False


def test_endpoint_change_detection_localises_to_a_box():
    from sam_backend.trading.drawing import DrawingEngine

    region = {"left": 0, "top": 0, "right": 400, "bottom": 300}
    before = Image.new("RGB", (400, 300), "black")
    after = before.copy()
    ImageDraw.Draw(after).line([(100, 100), (140, 140)], fill="white", width=3)

    at_line, ratio_here = DrawingEngine.changed_near(before, after, 120, 120, region)
    away, ratio_there = DrawingEngine.changed_near(before, after, 320, 60, region)
    assert at_line is True and ratio_here > 0
    assert away is False and ratio_there == 0.0


def test_a_clock_label_parses_only_when_it_is_a_real_time():
    from sam_backend.trading.calibration import parse_clock

    assert parse_clock("20:15") == 20 * 60 + 15
    assert parse_clock("0:00") == 0
    assert parse_clock("23:59") == 24 * 60 - 1
    for invalid in ("24:00", "20:75", "2015", "abc", ""):
        assert parse_clock(invalid) is None


# --- Regression: a watchlist column is not a price axis -----------------------


def test_unevenly_pitched_labels_are_not_accepted_as_an_axis():
    """A real axis prints labels at a regular pitch; a list of quotes does not."""
    from sam_backend.trading.calibration import spacing_is_regular

    axis_rows = [100.0, 200.0, 300.0, 400.0, 500.0]
    assert spacing_is_regular(axis_rows)[0] is True
    # One label the recognizer missed leaves a double gap, which is still an axis.
    with_gap = [100.0, 200.0, 400.0, 500.0]
    assert spacing_is_regular(with_gap)[0] is True
    # A crosshair or last-price tag between gridlines is still the same axis.
    with_tag = [100.0, 200.0, 247.0, 300.0, 400.0, 500.0]
    assert spacing_is_regular(with_tag)[0] is True
    # Watchlist rows land wherever their symbols happen to sit: no shared lattice.
    scattered = [100.0, 137.0, 141.0, 402.0, 623.0, 631.0, 904.0]
    assert spacing_is_regular(scattered)[0] is False


def test_an_axis_far_from_the_traded_price_is_rejected():
    """The live price is a cheap, decisive check against reading another panel.

    A watchlist's percent-change column fitted a descending line on a real chart
    and produced a 'price axis' spanning 0.2 to 0.3 while gold traded near 4454.
    """
    from sam_backend.trading.calibration import brackets_expected_price

    assert brackets_expected_price([4440.0, 4460.0, 4480.0], 4454.26) is True
    # Just off the visible range is still plausible.
    assert brackets_expected_price([4460.0, 4480.0], 4454.26) is True
    # A percent-change column is not.
    assert brackets_expected_price([-3.21, -0.43, 0.55, 1.48], 4454.26) is False
    # With no expectation available the check cannot veto anything.
    assert brackets_expected_price([-3.21, 0.55], None) is True


def test_an_axis_needs_more_labels_than_a_bare_minimum_fit():
    from sam_backend.trading.calibration import MIN_ANCHORS, MIN_AXIS_ANCHORS

    # Three collinear points are easy to find by chance among unrelated numbers.
    assert MIN_AXIS_ANCHORS > MIN_ANCHORS


# --- Regression: segment clipping ---------------------------------------------


def _clip_fixture():
    calibration = Calibration(
        slope=-0.1, intercept=500.0, method="test", verified=True,
        minutes_per_pixel=0.2, time_intercept=1000.0, minutes_span=[1000.0, 1400.0],
    )
    return calibration, {"left": 0, "top": 40, "right": 2000, "bottom": 1600}


def test_a_segment_fully_inside_the_chart_is_kept():
    """The clipper once rejected on-screen segments, so no construction drew."""
    from sam_backend.trading.drawing import DrawingEngine

    calibration, region = _clip_fixture()
    kept = DrawingEngine.clamp_segment(
        {"price": 460.0, "minutes": 1100.0}, {"price": 440.0, "minutes": 1300.0}, calibration, region
    )
    assert kept is not None
    first, second = kept
    assert first["price"] == pytest.approx(460.0, abs=0.5)
    assert second["price"] == pytest.approx(440.0, abs=0.5)


def test_a_segment_running_off_the_edge_is_trimmed_not_dropped():
    from sam_backend.trading.drawing import DrawingEngine

    calibration, region = _clip_fixture()
    trimmed = DrawingEngine.clamp_segment(
        {"price": 460.0, "minutes": 1100.0}, {"price": 300.0, "minutes": 3000.0}, calibration, region
    )
    assert trimmed is not None
    for point in trimmed:
        x = calibration.x_at_minutes(point["minutes"])
        y = calibration.y_at(point["price"])
        assert region["left"] - 1 <= x <= region["right"] + 1
        assert region["top"] - 1 <= y <= region["bottom"] + 1


def test_a_segment_entirely_off_screen_is_refused():
    from sam_backend.trading.drawing import DrawingEngine

    calibration, region = _clip_fixture()
    assert DrawingEngine.clamp_segment(
        {"price": 9000.0, "minutes": 1100.0}, {"price": 9500.0, "minutes": 1300.0}, calibration, region
    ) is None


def test_a_time_after_midnight_resolves_into_the_calibrated_span():
    """A chart spanning midnight runs past 1440 minutes; the wrap must be chosen."""
    calibration = Calibration(
        slope=-0.1, intercept=500.0, method="test", verified=True,
        minutes_per_pixel=0.2, time_intercept=1000.0, minutes_span=[1300.0, 1560.0],
    )
    # 00:30 belongs at 1470 on this axis, not at 30.
    assert calibration.resolve_minutes(30.0) == pytest.approx(1470.0)
    # A time already inside the span is left alone.
    assert calibration.resolve_minutes(1400.0) == pytest.approx(1400.0)


# --- Regression: a refusal must not claim it executed -------------------------


def test_an_off_screen_price_refusal_reports_nothing_was_executed(trading: TradingService):
    """A refusal that says executed=True would imply the chart was touched."""
    result = trading.draw_annotation("resistance", 999_999.0)
    assert result.verified is False
    assert result.executed is False, "a refused drawing must never report itself executed"
    assert result.error and "unchanged" in " ".join(result.observations).lower() or result.error


# --- Regression: levels must never land on the wrong instrument ---------------


class _ChartOn:
    """Stand-in for an observed TradingView window showing one symbol."""

    def __init__(self, symbol):
        self.symbol = symbol
        self.window_handle = 1
        self.client_geometry = {"left": 0, "top": 0, "right": 1200, "bottom": 800}
        self.window_geometry = self.client_geometry
        self.timeframe = "M15"
        self.current_price = 100.0
        self.active = True


def test_drawing_is_refused_when_the_chart_shows_another_instrument(trading: TradingService, monkeypatch):
    """Price alone is not a safe guard: two instruments can share a range.

    A GLD chart was open while XAUUSD had been analysed; only the price being
    off-screen prevented the levels from being drawn onto the wrong instrument.
    """
    monkeypatch.setattr(trading.tradingview, "observe", lambda: _ChartOn("GLD"))
    result = trading.draw_annotation("support", 100.0, symbol="XAUUSD")
    assert result.verified is False
    assert result.error_code == "SYMBOL_MISMATCH"
    assert result.executed is False
    assert "GLD" in result.error and "XAUUSD" in result.error


def test_drawing_proceeds_when_the_chart_shows_the_analysed_instrument(trading: TradingService, monkeypatch):
    monkeypatch.setattr(trading.tradingview, "observe", lambda: _ChartOn("XAUUSD"))
    result = trading.draw_annotation("support", 100.0, symbol="XAUUSD")
    # It gets past the symbol guard and stops at the permission gate instead.
    assert result.error_code != "SYMBOL_MISMATCH"


def test_provider_and_chart_spellings_of_one_instrument_agree(trading: TradingService, monkeypatch):
    for chart_symbol, analysis_symbol in (("XAUUSD", "XAUUSD.m"), ("OANDA:XAUUSD", "XAUUSD"), ("XAUUSD", "XAUUSD")):
        monkeypatch.setattr(trading.tradingview, "observe", lambda s=chart_symbol: _ChartOn(s))
        result = trading.draw_annotation("support", 100.0, symbol=analysis_symbol)
        assert result.error_code != "SYMBOL_MISMATCH", f"{chart_symbol} vs {analysis_symbol} wrongly rejected"


def test_an_analysis_report_cannot_be_drawn_onto_a_different_symbol(trading: TradingService, monkeypatch):
    monkeypatch.setattr(trading.tradingview, "observe", lambda: _ChartOn("GLD"))
    trading.analyst.latest = {"symbol": "XAUUSD", "setup": {"entry": 100.0, "stop": 99.0}, "support": [], "resistance": []}
    result = trading.draw_analysis()
    assert result.error_code == "SYMBOL_MISMATCH"
