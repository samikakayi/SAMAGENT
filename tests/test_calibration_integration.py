"""End-to-end calibration through the real Windows OCR engine.

These tests render a realistic price axis, feed it through the same capture ->
OCR -> fit -> verify pipeline the live chart uses, and assert that SAM converts
pixels into prices accurately enough to place a line. They skip themselves when
no OCR language pack is installed rather than reporting a false pass.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw, ImageFont

from sam_backend.trading.calibration import ChartCalibrator
from sam_backend.trading.ocr import WindowsOcrEngine

pytestmark = pytest.mark.skipif(
    WindowsOcrEngine().capability()["state"] != "AVAILABLE",
    reason="Windows OCR language pack is not installed on this machine",
)

GEOMETRY = {"left": 0, "top": 0, "right": 1200, "bottom": 800}


def _font(size: int = 18):
    for name in ("segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_price_axis(
    *, top_price: float, bottom_price: float, region: dict[str, int], dark: bool = True
) -> Image.Image:
    """Render a price scale whose labels sit at true linear positions."""
    width, height = region["width"], region["height"]
    background = "#131722" if dark else "white"
    foreground = "#d1d4dc" if dark else "black"
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    font = _font()
    steps = 8
    # Print enough decimals for the instrument: a chart that rounded 1.1875 to
    # 1.19 would not be showing a linear axis at all, and SAM rightly refuses it.
    decimals = 2 if abs(top_price) >= 100 else 4
    for index in range(steps + 1):
        ratio = index / steps
        y = ratio * (height - 1)
        price = top_price + (bottom_price - top_price) * ratio
        draw.text((8, y), f"{price:.{decimals}f}", fill=foreground, font=font, anchor="lm")
    return image


class StubCalibrator(ChartCalibrator):
    """Real OCR and real fitting; only the screen grab is substituted."""

    def __init__(self, image: Image.Image) -> None:
        super().__init__()
        self._image = image

    def _grab(self, region: dict[str, int]):  # noqa: D102
        return self._image


def test_a_rendered_price_axis_calibrates_and_predicts_prices_accurately():
    region = ChartCalibrator.price_scale_region(GEOMETRY)
    top_price, bottom_price = 3500.00, 3300.00
    calibrator = StubCalibrator(render_price_axis(top_price=top_price, bottom_price=bottom_price, region=region))

    result = calibrator.calibrate(geometry=GEOMETRY, symbol="XAUUSD", timeframe="H1", window_handle=1)

    assert result.verified, f"calibration failed: {result.error} / {result.observations}"
    data = result.data
    assert len(data["anchors"]) >= 3
    # The axis spans 200 price units over the scale height; check the mapping at
    # the midpoint against the value the renderer actually drew.
    slope, intercept = data["slope"], data["intercept"]
    mid_y = region["top"] + (region["height"] - 1) / 2
    predicted = slope * mid_y + intercept
    assert predicted == pytest.approx((top_price + bottom_price) / 2, abs=0.5)
    # Price decreases as Y increases on every standard chart.
    assert slope < 0


def test_calibration_survives_a_light_theme_axis():
    region = ChartCalibrator.price_scale_region(GEOMETRY)
    calibrator = StubCalibrator(
        render_price_axis(top_price=1.2000, bottom_price=1.1000, region=region, dark=False)
    )
    result = calibrator.calibrate(geometry=GEOMETRY, symbol="EURUSD", timeframe="M15", window_handle=2)
    assert result.verified, f"calibration failed: {result.error}"
    assert result.data["slope"] < 0


def test_a_blank_price_axis_refuses_to_calibrate_instead_of_guessing():
    region = ChartCalibrator.price_scale_region(GEOMETRY)
    blank = Image.new("RGB", (region["width"], region["height"]), "#131722")
    result = StubCalibrator(blank).calibrate(geometry=GEOMETRY, symbol="XAUUSD", timeframe="H1", window_handle=3)
    assert not result.verified
    assert result.error_code == "INSUFFICIENT_ANCHORS"


def test_a_verified_calibration_confirms_itself_against_a_fresh_read():
    from sam_backend.trading.calibration import calibration_from_row

    region = ChartCalibrator.price_scale_region(GEOMETRY)
    image = render_price_axis(top_price=3500.0, bottom_price=3300.0, region=region)
    calibrator = StubCalibrator(image)
    first = calibrator.calibrate(geometry=GEOMETRY, symbol="XAUUSD", timeframe="H1", window_handle=4)
    assert first.verified

    calibration = calibration_from_row({**first.data, "window_handle": 4})
    outcome = calibrator.verify(
        calibration, geometry=GEOMETRY, symbol="XAUUSD", timeframe="H1", window_handle=4
    )
    assert outcome.verified, f"re-verification failed: {outcome.error}"
    assert outcome.data["stable"] is True


def test_a_rescaled_chart_is_detected_as_drift_rather_than_silently_trusted():
    from sam_backend.trading.calibration import calibration_from_row

    region = ChartCalibrator.price_scale_region(GEOMETRY)
    original = StubCalibrator(render_price_axis(top_price=3500.0, bottom_price=3300.0, region=region))
    first = original.calibrate(geometry=GEOMETRY, symbol="XAUUSD", timeframe="H1", window_handle=5)
    assert first.verified
    calibration = calibration_from_row({**first.data, "window_handle": 5})

    # The user zoomed: the same pixels now show a different price range.
    zoomed = StubCalibrator(render_price_axis(top_price=3450.0, bottom_price=3400.0, region=region))
    outcome = zoomed.verify(calibration, geometry=GEOMETRY, symbol="XAUUSD", timeframe="H1", window_handle=5)

    assert not outcome.verified
    assert outcome.error_code == "CALIBRATION_DRIFTED"
    assert outcome.data["worst_drift"] > outcome.data["tolerance"]


def test_a_moved_window_invalidates_the_calibration_before_any_drawing():
    from sam_backend.trading.calibration import calibration_from_row

    region = ChartCalibrator.price_scale_region(GEOMETRY)
    calibrator = StubCalibrator(render_price_axis(top_price=3500.0, bottom_price=3300.0, region=region))
    first = calibrator.calibrate(geometry=GEOMETRY, symbol="XAUUSD", timeframe="H1", window_handle=6)
    calibration = calibration_from_row({**first.data, "window_handle": 6})

    moved = {"left": 40, "top": 0, "right": 1240, "bottom": 800}
    outcome = calibrator.verify(calibration, geometry=moved, symbol="XAUUSD", timeframe="H1", window_handle=6)

    assert not outcome.verified
    assert outcome.error_code == "CALIBRATION_STALE_GEOMETRY"
