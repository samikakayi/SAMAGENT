"""Chart calibration: the price/time <-> screen-coordinate mapping.

Every price-accurate drawing depends on this module. A calibration is only
returned as verified when several independent axis labels agree on one linear
fit, so SAM can refuse to draw rather than place a line at a guessed pixel.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..contracts import ExecutionStatus, StandardResult
from .ocr import OcrWord, WindowsOcrEngine, parse_price

# Fraction of the window width occupied by the right-hand price scale. TradingView
# keeps this strip narrow; sampling slightly wide is safe because non-numeric text
# is discarded during parsing.
PRICE_SCALE_WIDTH_RATIO = 0.085
# Share of the window width scanned when locating the price axis. A side
# watchlist or an open indicator panel pushes the chart's own axis well left
# of the window edge, so the band is wide; the spacing and live-price checks
# are what keep a wide scan from latching onto the wrong column.
PRICE_AXIS_SEARCH_RATIO = 0.48
PRICE_SCALE_MIN_WIDTH = 56
PRICE_SCALE_MAX_WIDTH = 190
TIME_SCALE_HEIGHT = 34
CHART_TOP_INSET = 40
CHART_BOTTOM_INSET = TIME_SCALE_HEIGHT
# Keep the axis labels themselves out of the verified plot area.
PLOT_AXIS_MARGIN = 8

MIN_ANCHORS = 3
# A real price axis prints labels at a regular pixel pitch. Requiring that the
# fitted inliers are close to evenly spaced rejects a spurious consensus found
# among unrelated numbers elsewhere on screen (a watchlist column, for example).
MIN_AXIS_ANCHORS = 4
MAX_SPACING_VARIATION = 0.35
# Share of labels that must sit on the same lattice for a column to be an axis.
MIN_LATTICE_SHARE = 0.6
MAX_LATTICE_OFFSET_PIXELS = 12.0
# The axis must bracket the price the instrument is actually trading at.
PRICE_SANITY_TOLERANCE = 0.35
# Calibration error originates in pixel space: the OCR bounding box for a label
# is centred within a few pixels of the gridline it belongs to. Expressing the
# tolerance in pixels therefore scales correctly across symbols and zoom levels,
# where a price-percentage tolerance would be far too tight on a zoomed-in gold
# chart and far too loose on a wide index chart.
MAX_RESIDUAL_PIXELS = 6.0


@dataclass(slots=True)
class Calibration:
    slope: float
    intercept: float
    method: str
    anchors: list[dict[str, Any]] = field(default_factory=list)
    max_error: float = 0.0
    verified: bool = False
    geometry_hash: str = ""
    symbol: str = ""
    timeframe: str = ""
    window_handle: int = 0
    created_at: float = field(default_factory=time.monotonic)
    time_origin: dict[str, Any] | None = None
    axis_x: float | None = None
    precision: float = 0.0
    minutes_per_pixel: float | None = None
    time_intercept: float | None = None
    time_axis_y: float | None = None
    minutes_span: list[float] | None = None

    def price_at(self, y: float) -> float:
        return self.slope * y + self.intercept

    @property
    def time_calibrated(self) -> bool:
        return self.minutes_per_pixel is not None and self.time_intercept is not None

    def minutes_at(self, x: float) -> float:
        if not self.time_calibrated:
            raise ValueError("The time axis is not calibrated")
        return self.minutes_per_pixel * x + self.time_intercept

    def resolve_minutes(self, minutes: float) -> float:
        """Pick the day-wrap of this clock time that lies in the calibrated span.

        Anchors arrive as minutes-of-day. On a chart that crosses midnight the
        axis runs past 1440, so the same clock time has two candidate positions
        and only one of them is on screen.
        """
        if not self.minutes_span:
            return minutes
        low, high = self.minutes_span
        best, best_distance = minutes, float("inf")
        for wrap in (-1440.0, 0.0, 1440.0, 2880.0):
            candidate = minutes + wrap
            distance = 0.0 if low <= candidate <= high else min(abs(candidate - low), abs(candidate - high))
            if distance < best_distance:
                best, best_distance = candidate, distance
        return best

    def x_at_minutes(self, minutes: float) -> float:
        if not self.time_calibrated or not self.minutes_per_pixel:
            raise ValueError("The time axis is not calibrated")
        return (self.resolve_minutes(minutes) - self.time_intercept) / self.minutes_per_pixel

    def y_at(self, price: float) -> float:
        if self.slope == 0:
            raise ZeroDivisionError("Calibration slope is zero")
        return (price - self.intercept) / self.slope

    def as_dict(self) -> dict[str, Any]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "method": self.method,
            "anchors": self.anchors,
            "max_error": self.max_error,
            "verified": self.verified,
            "geometry_hash": self.geometry_hash,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "window_handle": self.window_handle,
            "price_per_pixel": self.slope,
            "time_origin": self.time_origin,
            "axis_x": self.axis_x,
            "precision": self.precision,
            "minutes_per_pixel": self.minutes_per_pixel,
            "time_intercept": self.time_intercept,
            "time_axis_y": self.time_axis_y,
            "minutes_span": self.minutes_span,
        }


def geometry_hash(geometry: dict[str, int] | None) -> str:
    """Identify a chart viewport. Any move, resize, or monitor change changes this."""
    if not geometry:
        return "unknown"
    payload = "|".join(str(int(geometry.get(key, 0))) for key in ("left", "top", "right", "bottom"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def fit_linear(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Least-squares fit of price = slope * y + intercept."""
    count = len(points)
    if count < 2:
        return None
    sum_y = sum(point[0] for point in points)
    sum_price = sum(point[1] for point in points)
    mean_y = sum_y / count
    mean_price = sum_price / count
    numerator = sum((y - mean_y) * (price - mean_price) for y, price in points)
    denominator = sum((y - mean_y) ** 2 for y, _ in points)
    if denominator == 0:
        return None
    slope = numerator / denominator
    if not math.isfinite(slope) or slope == 0:
        return None
    return slope, mean_price - slope * mean_y


def _integer_digits(price: float) -> int:
    return len(str(abs(int(price))))


def drop_magnitude_outliers(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Discard labels whose digit count disagrees with the rest of the axis.

    OCR clips the partially drawn labels at the top and bottom of a price scale,
    turning `3475.00` into `75.00`. Every label on one axis shares a magnitude,
    so a lone short reading is a misread rather than a real price.
    """
    if len(points) < 3:
        return points
    digits = sorted(_integer_digits(price) for _, price in points)
    median_digits = digits[len(digits) // 2]
    kept = [point for point in points if abs(_integer_digits(point[1]) - median_digits) <= 1]
    return kept if len(kept) >= MIN_ANCHORS else points


def fit_with_outlier_rejection(
    points: list[tuple[float, float]],
) -> tuple[float, float, list[tuple[float, float]], float] | None:
    """Find the linear price/Y mapping the most axis labels agree on.

    A greedy "drop the worst residual" loop fails once two or more labels are
    misread, because the corrupted values can drag the fit far enough that good
    labels look like the outliers. Every candidate pair is tried instead and the
    model with the largest consensus wins, which tolerates several bad reads as
    long as a majority of labels remain correct.
    """
    candidates = drop_magnitude_outliers(sorted(set(points)))
    if len(candidates) < 2:
        return None

    best: tuple[int, float, float, float, list[tuple[float, float]]] | None = None
    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            y_a, price_a = candidates[first]
            y_b, price_b = candidates[second]
            if y_a == y_b:
                continue
            slope = (price_b - price_a) / (y_b - y_a)
            if not math.isfinite(slope) or slope == 0:
                continue
            intercept = price_a - slope * y_a
            # Tolerance scales with the price span this model implies across the
            # sampled axis height, so it adapts to gold, FX, and index charts.
            tolerance = max(abs(slope) * MAX_RESIDUAL_PIXELS, 1e-9)
            inliers = [
                point for point in candidates if abs(point[1] - (slope * point[0] + intercept)) <= tolerance
            ]
            if len(inliers) < 2:
                continue
            error = sum(abs(price - (slope * y + intercept)) for y, price in inliers)
            score = (len(inliers), -error)
            if best is None or score > (best[0], -best[1]):
                best = (len(inliers), error, slope, intercept, inliers)

    if best is None or best[0] < MIN_ANCHORS:
        return None

    inliers = best[4]
    refit = fit_linear(inliers)
    if refit is None:
        return None
    slope, intercept = refit
    worst = max(abs(price - (slope * y + intercept)) for y, price in inliers)
    return slope, intercept, sorted(inliers), worst


_TIME_PATTERN = re.compile(r"^(?:\d{1,2}:\d{2}|\d{1,2}\s?[A-Za-z]{3}|[A-Za-z]{3}\s?'?\d{2,4}|\d{4})$")
_CLOCK_PATTERN = re.compile(r"^(\d{1,2}):(\d{2})$")
# The time axis sits above the chart's bottom toolbar, not flush with the window.
TIME_AXIS_SEARCH_HEIGHT = 170
TIME_AXIS_ROW_TOLERANCE = 8
MIN_TIME_ANCHORS = 4


def parse_clock(text: str) -> int | None:
    """Minutes past midnight from an HH:MM axis label."""
    match = _CLOCK_PATTERN.match(text.strip())
    if not match:
        return None
    hours, minutes = int(match.group(1)), int(match.group(2))
    if hours > 23 or minutes > 59:
        return None
    return hours * 60 + minutes


def spacing_is_regular(y_values: list[float], tolerance: float = MAX_SPACING_VARIATION) -> tuple[bool, float]:
    """Do most of these labels sit on one evenly pitched lattice?

    A price axis prints gridline labels at a fixed pitch, but TradingView also
    draws highlighted tags for the crosshair and the last price between them.
    Those tags are real prices on the same axis, so demanding that *every* gap
    match would reject the genuine axis. The test is instead whether a clear
    majority of labels fall on a single lattice, which a scattered column of
    unrelated numbers cannot satisfy.
    """
    ordered = sorted(y_values)
    if len(ordered) < 3:
        return False, 1.0
    gaps = [later - earlier for earlier, later in zip(ordered, ordered[1:]) if later > earlier]
    if not gaps:
        return False, 1.0
    gaps.sort()
    pitch = gaps[len(gaps) // 2]
    if pitch <= 0:
        return False, 1.0
    # The allowance is a handful of pixels, not a share of the pitch: a label
    # sits within a few pixels of its gridline whatever the zoom, and a
    # proportional allowance would let a widely scattered column qualify.
    allowance = min(pitch * tolerance, MAX_LATTICE_OFFSET_PIXELS)
    best_share = 0.0
    # Try each label as the lattice origin; the true gridline set will dominate.
    for origin in ordered:
        on_lattice = 0
        for value in ordered:
            offset = abs(value - origin) / pitch
            if abs(offset - round(offset)) * pitch <= allowance:
                on_lattice += 1
        best_share = max(best_share, on_lattice / len(ordered))
    return best_share >= MIN_LATTICE_SHARE, 1.0 - best_share


def brackets_expected_price(prices: list[float], expected: float | None, tolerance: float = PRICE_SANITY_TOLERANCE) -> bool:
    """Does this candidate axis plausibly belong to the traded instrument?

    The window title and the market feed both know roughly where price is. An
    axis whose labels are nowhere near it is some other column of numbers.
    """
    if expected is None or not prices:
        return True
    low, high = min(prices), max(prices)
    if low <= expected <= high:
        return True
    span = max(high - low, abs(expected) * 1e-6)
    nearest = low if expected < low else high
    return abs(expected - nearest) <= max(span, abs(expected) * tolerance)


class ChartCalibrator:
    """Derives and verifies the price<->Y mapping for a TradingView chart window."""

    def __init__(self, ocr: WindowsOcrEngine | None = None) -> None:
        self.ocr = ocr or WindowsOcrEngine()

    def capability(self) -> dict[str, Any]:
        return self.ocr.capability()

    @staticmethod
    def price_scale_region(geometry: dict[str, int]) -> dict[str, int]:
        width = int(geometry["right"] - geometry["left"])
        height = int(geometry["bottom"] - geometry["top"])
        scale_width = int(min(max(width * PRICE_SCALE_WIDTH_RATIO, PRICE_SCALE_MIN_WIDTH), PRICE_SCALE_MAX_WIDTH))
        return {
            "left": int(geometry["right"]) - scale_width,
            "top": int(geometry["top"]) + CHART_TOP_INSET,
            "right": int(geometry["right"]),
            "bottom": int(geometry["bottom"]) - CHART_BOTTOM_INSET,
            "width": scale_width,
            "height": height - CHART_TOP_INSET - CHART_BOTTOM_INSET,
        }

    @staticmethod
    def plot_region(geometry: dict[str, int], axis_x: float | None = None) -> dict[str, int]:
        """Candle area only.

        When calibration located the price axis, everything from that column
        rightwards (the axis plus any icon rail) is excluded so drawing
        verification compares only pixels the chart itself paints.
        """
        if axis_x is not None and int(geometry["left"]) + 80 < axis_x <= int(geometry["right"]):
            right = int(axis_x) - PLOT_AXIS_MARGIN
        else:
            right = ChartCalibrator.price_scale_region(geometry)["left"]
        return {
            "left": int(geometry["left"]),
            "top": int(geometry["top"]) + CHART_TOP_INSET,
            "right": right,
            "bottom": int(geometry["bottom"]) - CHART_BOTTOM_INSET,
        }

    def _grab(self, region: dict[str, int]) -> Any:
        from PIL import ImageGrab

        from ..dpi import ensure_dpi_awareness

        ensure_dpi_awareness()

        return ImageGrab.grab(
            bbox=(region["left"], region["top"], region["right"], region["bottom"]),
            all_screens=True,
        )

    def anchors_from_words(self, words: list[OcrWord], region: dict[str, int]) -> list[dict[str, Any]]:
        """Convert OCR words into absolute-screen (y, price) anchors."""
        anchors: list[dict[str, Any]] = []
        for word in words:
            price = parse_price(word.text)
            if price is None:
                continue
            anchors.append(
                {
                    "text": word.text,
                    "price": price,
                    "y": region["top"] + word.center_y,
                    "local_y": word.center_y,
                    "x": region["left"] + word.center_x,
                    "local_x": word.center_x,
                }
            )
        anchors.sort(key=lambda item: item["y"])
        return anchors

    def search_region(self, geometry: dict[str, int]) -> dict[str, int]:
        """Band to scan for the price axis.

        The axis is not reliably flush with the window edge: TradingView keeps a
        right-hand icon rail, and an idea or multi-pane layout insets the chart
        further. Scanning a wide band and locating the axis by content is robust
        to all of those, where a fixed offset silently reads the wrong column.
        """
        width = int(geometry["right"] - geometry["left"])
        band = int(min(max(width * PRICE_AXIS_SEARCH_RATIO, 160), 1400))
        return {
            "left": int(geometry["right"]) - band,
            "top": int(geometry["top"]) + CHART_TOP_INSET,
            "right": int(geometry["right"]),
            "bottom": int(geometry["bottom"]) - CHART_BOTTOM_INSET,
            "width": band,
            "height": int(geometry["bottom"] - geometry["top"]) - CHART_TOP_INSET - CHART_BOTTOM_INSET,
        }

    @staticmethod
    def cluster_columns(anchors: list[dict[str, Any]], tolerance: float = 70.0) -> list[list[dict[str, Any]]]:
        """Group numeric labels into vertical columns by their X position.

        Comparing each candidate against the column's running mean -- not just
        the most recently added label -- stops a chain of labels, each a
        little further right than the last, from drifting a single "column"
        across two genuinely separate ones (the price axis and an adjacent
        watchlist or icon rail, for example). Letting that happen would dilute
        the real axis fit with unrelated numbers and could fail verification,
        or in the worst case pass a corrupted fit.
        """
        columns: list[list[dict[str, Any]]] = []
        sums: list[float] = []
        for anchor in sorted(anchors, key=lambda item: item["x"]):
            for index, column in enumerate(columns):
                mean_x = sums[index] / len(column)
                if abs(mean_x - anchor["x"]) <= tolerance:
                    column.append(anchor)
                    sums[index] += anchor["x"]
                    break
            else:
                columns.append([anchor])
                sums.append(anchor["x"])
        return columns

    def calibrate(
        self,
        *,
        geometry: dict[str, int],
        symbol: str,
        timeframe: str,
        window_handle: int,
        expected_price: float | None = None,
    ) -> StandardResult:
        """Locate the price axis, then derive a verified linear price<->Y mapping."""
        started = time.perf_counter()
        region = self.search_region(geometry)
        if region["width"] < 20 or region["height"] < 60:
            return StandardResult.failure(
                "The TradingView window is too small to expose a readable price scale.",
                error_code="CHART_TOO_SMALL",
                started_at=started,
            )
        try:
            image = self._grab(region)
        except Exception as exc:
            return StandardResult.failure(
                f"Could not capture the price scale: {exc}", error_code="SCREEN_CAPTURE_FAILED", started_at=started
            )
        try:
            words = self.ocr.recognize(image)
        except RuntimeError as exc:
            return StandardResult.failure(
                f"Automatic calibration is unavailable: {exc}", error_code="OCR_UNAVAILABLE", started_at=started
            )
        anchors = self.anchors_from_words(words, region)
        if len(anchors) < MIN_ANCHORS:
            return StandardResult.failure(
                f"Only {len(anchors)} readable price labels were found; {MIN_ANCHORS} are required.",
                error_code="INSUFFICIENT_ANCHORS",
                started_at=started,
                observations=[f"Recognized text in the search band: {[word.text for word in words][:16]}"],
            )
        # Try each candidate column and keep the one the most labels agree on.
        best_fit = None
        best_column: list[dict[str, Any]] = []
        rejected: list[str] = []
        for column in self.cluster_columns(anchors):
            if len(column) < MIN_AXIS_ANCHORS:
                continue
            candidate = fit_with_outlier_rejection([(item["y"], item["price"]) for item in column])
            if candidate is None or candidate[0] >= 0:
                continue
            inliers = candidate[2]
            if len(inliers) < MIN_AXIS_ANCHORS:
                continue
            regular, variation = spacing_is_regular([y for y, _ in inliers])
            if not regular:
                rejected.append(f"column at x~{column[0]['x']:.0f}: only {1 - variation:.0%} of labels share a pitch")
                continue
            if not brackets_expected_price([price for _, price in inliers], expected_price):
                rejected.append(
                    f"column at x~{column[0]['x']:.0f}: values "
                    f"{min(p for _, p in inliers):.4g}-{max(p for _, p in inliers):.4g} are not near {expected_price:.4g}"
                )
                continue
            if best_fit is None or len(inliers) > len(best_fit[2]):
                best_fit, best_column = candidate, column
        fit = best_fit
        if fit is None:
            return StandardResult.failure(
                "No column of labels looked like this chart's price axis: none was evenly pitched, "
                "descending, and priced near the instrument.",
                error_code="CALIBRATION_NOT_LINEAR",
                started_at=started,
                observations=[
                    f"Recognized text in the search band: {[word.text for word in words][:16]}",
                    *(f"Rejected {reason}" for reason in rejected[:4]),
                ],
            )
        anchors = best_column
        slope, intercept, inliers, worst = fit
        tolerance = max(abs(slope) * MAX_RESIDUAL_PIXELS, 1e-9)
        # A standard price scale runs high-to-low down the screen. A positive
        # slope means the axis is inverted or the labels were misread; either
        # way it must not silently authorize drawing.
        monotonic = slope < 0
        verified = len(inliers) >= MIN_ANCHORS and worst <= tolerance and monotonic
        axis_x = min(item["x"] for item in anchors)
        calibration = Calibration(
            slope=slope,
            intercept=intercept,
            method="windows_ocr_price_scale",
            axis_x=axis_x,
            precision=tolerance,
            anchors=[{"y": y, "price": price} for y, price in inliers],
            max_error=worst,
            verified=verified,
            geometry_hash=geometry_hash(geometry),
            symbol=symbol.upper(),
            timeframe=timeframe,
            window_handle=int(window_handle),
        )
        observations = [
            f"Fitted {len(inliers)} of {len(anchors)} price labels; worst residual {worst:.6g} against tolerance {tolerance:.6g}.",
            f"A drawn level is accurate to about {tolerance:.6g} price units at this zoom.",
        ]
        if len(inliers) < len(anchors):
            observations.append(f"Discarded {len(anchors) - len(inliers)} outlier label(s) before fitting.")
        if not monotonic:
            observations.append("The fitted scale increases downward, which a standard price axis never does.")
        if not verified:
            return StandardResult(
                ExecutionStatus.PARTIAL,
                True,
                False,
                data=calibration.as_dict(),
                error="The price-scale fit did not meet the residual tolerance; drawing stays blocked.",
                error_code="CALIBRATION_NOT_VERIFIED",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                observations=observations,
            )
        return StandardResult.success(calibration.as_dict(), verified=True, started_at=started, observations=observations)

    def verify(
        self,
        calibration: Calibration,
        *,
        geometry: dict[str, int],
        symbol: str,
        timeframe: str,
        window_handle: int,
        expected_price: float | None = None,
    ) -> StandardResult:
        """Re-read the axis and confirm the stored mapping still predicts it."""
        started = time.perf_counter()
        if calibration.geometry_hash != geometry_hash(geometry):
            return StandardResult.failure(
                "The chart viewport moved or resized since calibration.",
                error_code="CALIBRATION_STALE_GEOMETRY",
                started_at=started,
            )
        fresh = self.calibrate(geometry=geometry, symbol=symbol, timeframe=timeframe,
                               window_handle=window_handle, expected_price=expected_price)
        if not fresh.verified or not isinstance(fresh.data, dict):
            fresh.error_code = fresh.error_code or "CALIBRATION_REVERIFY_FAILED"
            return fresh
        anchors = fresh.data.get("anchors") or []
        drifts = [abs(calibration.price_at(anchor["y"]) - anchor["price"]) for anchor in anchors]
        worst_drift = max(drifts) if drifts else float("inf")
        tolerance = max(abs(calibration.slope) * MAX_RESIDUAL_PIXELS, 1e-9)
        stable = worst_drift <= tolerance
        payload = {
            "stable": stable,
            "worst_drift": worst_drift,
            "tolerance": tolerance,
            "stored": calibration.as_dict(),
            "observed": fresh.data,
        }
        if stable:
            return StandardResult.success(payload, verified=True, started_at=started)
        return StandardResult(
            ExecutionStatus.PARTIAL,
            True,
            False,
            data=payload,
            error=f"Calibration drifted by {worst_drift:.6g}; a recalibration is required before drawing.",
            error_code="CALIBRATION_DRIFTED",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def calibrate_time_axis(self, geometry: dict[str, int], axis_x: float | None = None) -> StandardResult:
        """Fit minutes-of-day against X from the chart's own time-axis labels.

        The axis row is found by content rather than a fixed offset, because the
        chart's bottom toolbar sits below it and its height varies by layout.
        Clipped labels (a half-drawn `20:00` read as `0:00`) are rejected by the
        same consensus fit the price axis uses.
        """
        started = time.perf_counter()
        plot = self.plot_region(geometry, axis_x)
        band = {
            "left": plot["left"],
            "top": max(int(geometry["top"]), int(geometry["bottom"]) - TIME_AXIS_SEARCH_HEIGHT),
            "right": plot["right"],
            "bottom": int(geometry["bottom"]),
        }
        if band["right"] - band["left"] < 120 or band["bottom"] - band["top"] < 20:
            return StandardResult.failure("The chart is too small to expose a time axis.",
                                          error_code="CHART_TOO_SMALL", started_at=started)
        try:
            words = self.ocr.recognize(self._grab(band))
        except RuntimeError as exc:
            return StandardResult.failure(f"Time-axis OCR unavailable: {exc}",
                                          error_code="OCR_UNAVAILABLE", started_at=started)
        except Exception as exc:
            return StandardResult.failure(f"Time-axis capture failed: {exc}",
                                          error_code="SCREEN_CAPTURE_FAILED", started_at=started)

        clocks = [
            {"x": band["left"] + word.center_x, "y": band["top"] + word.center_y,
             "minutes": parse_clock(word.text), "text": word.text}
            for word in words
        ]
        clocks = [item for item in clocks if item["minutes"] is not None]
        clocks.sort(key=lambda item: item["x"])
        # A chart that spans midnight prints 23:45 then 00:15. Minutes-of-day goes
        # backwards there, which no linear fit can represent, so the sequence is
        # unwrapped into a continuous timeline before fitting.
        offset = 0
        previous: float | None = None
        for item in clocks:
            if previous is not None and item["minutes"] + offset < previous - 600:
                offset += 1440
            item["minutes"] = item["minutes"] + offset
            previous = item["minutes"]
        if len(clocks) < MIN_TIME_ANCHORS:
            return StandardResult.failure(
                f"Only {len(clocks)} clock labels were readable; {MIN_TIME_ANCHORS} are required.",
                error_code="INSUFFICIENT_TIME_ANCHORS", started_at=started,
                observations=[f"Recognized text: {[word.text for word in words][:16]}"],
            )
        # Labels for one axis share a row; anything else is chart or toolbar text.
        rows: dict[int, list[dict[str, Any]]] = {}
        for item in clocks:
            key = next((existing for existing in rows if abs(existing - item["y"]) <= TIME_AXIS_ROW_TOLERANCE), None)
            rows.setdefault(key if key is not None else int(item["y"]), []).append(item)
        row_y, row = max(rows.items(), key=lambda entry: len(entry[1]))
        if len(row) < MIN_TIME_ANCHORS:
            return StandardResult.failure(
                f"The densest label row held only {len(row)} clock labels.",
                error_code="INSUFFICIENT_TIME_ANCHORS", started_at=started,
            )

        fit = fit_with_outlier_rejection([(item["x"], float(item["minutes"])) for item in row])
        if fit is None:
            return StandardResult.failure(
                "Time labels did not form a consistent linear scale.",
                error_code="TIME_CALIBRATION_NOT_LINEAR", started_at=started,
                observations=[f"Row labels: {[item['text'] for item in row]}"],
            )
        minutes_per_pixel, intercept, inliers, worst = fit
        # Time must increase to the right on every chart.
        if minutes_per_pixel <= 0:
            return StandardResult.failure(
                "The fitted time axis runs backwards, so the labels were misread.",
                error_code="TIME_CALIBRATION_NOT_LINEAR", started_at=started,
            )
        tolerance = max(minutes_per_pixel * MAX_RESIDUAL_PIXELS, 1e-9)
        verified = len(inliers) >= MIN_TIME_ANCHORS and worst <= tolerance
        payload = {
            "minutes_per_pixel": minutes_per_pixel,
            "time_intercept": intercept,
            "time_axis_y": float(row_y),
            "anchors": [{"x": x, "minutes": minutes} for x, minutes in inliers],
            "max_error_minutes": worst,
            "tolerance_minutes": tolerance,
            "seconds_per_pixel": minutes_per_pixel * 60,
            "minutes_span": [min(minutes for _, minutes in inliers), max(minutes for _, minutes in inliers)],
        }
        observations = [
            f"Fitted {len(inliers)} of {len(row)} clock labels; worst residual {worst:.4g} min "
            f"against tolerance {tolerance:.4g} min.",
        ]
        if verified:
            return StandardResult.success(payload, verified=True, started_at=started, observations=observations)
        return StandardResult(
            ExecutionStatus.PARTIAL, True, False, data=payload,
            error="The time-axis fit did not meet its residual tolerance.",
            error_code="TIME_CALIBRATION_NOT_VERIFIED",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            observations=observations,
        )

    def read_time_axis(self, geometry: dict[str, int]) -> StandardResult:
        """Read time-scale labels. Reported separately because formats vary by interval."""
        started = time.perf_counter()
        scale = self.price_scale_region(geometry)
        region = {
            "left": int(geometry["left"]),
            "top": int(geometry["bottom"]) - TIME_SCALE_HEIGHT,
            "right": scale["left"],
            "bottom": int(geometry["bottom"]),
        }
        if region["right"] - region["left"] < 80:
            return StandardResult.failure("The chart is too narrow to read a time scale.", error_code="CHART_TOO_SMALL", started_at=started)
        try:
            words = self.ocr.recognize(self._grab(region))
        except RuntimeError as exc:
            return StandardResult.failure(f"Time-axis OCR unavailable: {exc}", error_code="OCR_UNAVAILABLE", started_at=started)
        except Exception as exc:
            return StandardResult.failure(f"Time-axis capture failed: {exc}", error_code="SCREEN_CAPTURE_FAILED", started_at=started)
        labels = [
            {"text": word.text, "x": region["left"] + word.center_x, "local_x": word.center_x}
            for word in words
            if _TIME_PATTERN.match(word.text.strip())
        ]
        labels.sort(key=lambda item: item["x"])
        if len(labels) < 2:
            return StandardResult(
                ExecutionStatus.PARTIAL,
                True,
                False,
                data={"labels": labels, "region": region},
                error="Fewer than two time labels were recognized; time<->X mapping stays unverified.",
                error_code="INSUFFICIENT_TIME_LABELS",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        return StandardResult.success({"labels": labels, "region": region}, verified=True, started_at=started)


def calibration_from_row(row: dict[str, Any]) -> Calibration:
    return Calibration(
        axis_x=row.get("axis_x"),
        minutes_per_pixel=row.get("minutes_per_pixel"),
        time_intercept=row.get("time_intercept"),
        time_axis_y=row.get("time_axis_y"),
        minutes_span=row.get("minutes_span"),
        slope=float(row["slope"]),
        intercept=float(row["intercept"]),
        method=str(row.get("method") or "unknown"),
        anchors=list(row.get("anchors") or []),
        max_error=float(row.get("max_error") or 0.0),
        verified=bool(row.get("verified")),
        geometry_hash=str(row.get("geometry_hash") or ""),
        symbol=str(row.get("symbol") or ""),
        timeframe=str(row.get("timeframe") or ""),
        window_handle=int(row.get("window_handle") or 0),
    )


def utc_iso(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).astimezone(UTC).isoformat()
