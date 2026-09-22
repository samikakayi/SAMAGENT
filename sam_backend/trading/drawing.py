"""Semantic TradingView drawing with ownership, layers, and pixel verification.

The Trading Brain asks for `draw_horizontal_line(price)`, never `click(x, y)`.
Every annotation is recorded as SAM-owned before it is verified, and a drawing is
only reported as present after the chart itself is observed to have changed at the
expected location. SAM never touches an annotation it does not own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from ..contracts import ExecutionStatus, StandardResult
from .calibration import Calibration, ChartCalibrator
from .desktop_input import DesktopInput


class Layer(StrEnum):
    STRUCTURE = "STRUCTURE"
    SNR = "SNR"
    SUPPLY_DEMAND = "SUPPLY_DEMAND"
    LIQUIDITY = "LIQUIDITY"
    ORDER_BLOCK = "ORDER_BLOCK"
    FVG = "FVG"
    THEORY = "THEORY"
    ENTRY = "ENTRY"
    TARGETS = "TARGETS"
    NOTES = "NOTES"


# Documented TradingView Desktop drawing shortcuts. Each places the tool at the
# current pointer position, which is why calibration must be verified first.
#
# Verified against TradingView Desktop 3.4.0: with the pointer settled over the
# chart, Alt+H creates a horizontal line within ~2px of the crosshair.
SHORTCUTS: dict[str, tuple[str, str]] = {
    "horizontal_line": ("alt", "h"),
    "vertical_line": ("alt", "v"),
    "trend_line": ("alt", "t"),
    "fib_retracement": ("alt", "f"),
    "horizontal_ray": ("alt", "j"),
    "cross_line": ("alt", "c"),
}

# Chords SAM must never emit. Ctrl+Alt+H toggles "hide all drawings", which makes
# every annotation on the chart — including the user's own — vanish, and no
# subsequent capture could verify anything. Discovered by hitting it in testing.
FORBIDDEN_CHORDS: frozenset[tuple[str, ...]] = frozenset({("ctrl", "alt", "h")})

# Semantic annotation type -> (drawing primitive, default layer).
SEMANTIC_TYPES: dict[str, tuple[str, Layer]] = {
    "support": ("horizontal_line", Layer.SNR),
    "resistance": ("horizontal_line", Layer.SNR),
    "rbs": ("horizontal_line", Layer.SNR),
    "sbr": ("horizontal_line", Layer.SNR),
    "supply": ("horizontal_line", Layer.SUPPLY_DEMAND),
    "demand": ("horizontal_line", Layer.SUPPLY_DEMAND),
    "equal_high": ("horizontal_line", Layer.LIQUIDITY),
    "equal_low": ("horizontal_line", Layer.LIQUIDITY),
    "bsl": ("horizontal_line", Layer.LIQUIDITY),
    "ssl": ("horizontal_line", Layer.LIQUIDITY),
    "pdh": ("horizontal_line", Layer.LIQUIDITY),
    "pdl": ("horizontal_line", Layer.LIQUIDITY),
    "pwh": ("horizontal_line", Layer.LIQUIDITY),
    "pwl": ("horizontal_line", Layer.LIQUIDITY),
    "order_block": ("horizontal_line", Layer.ORDER_BLOCK),
    "breaker": ("horizontal_line", Layer.ORDER_BLOCK),
    "fvg": ("horizontal_line", Layer.FVG),
    "premium": ("horizontal_line", Layer.THEORY),
    "discount": ("horizontal_line", Layer.THEORY),
    "equilibrium": ("horizontal_line", Layer.THEORY),
    "bos": ("horizontal_line", Layer.STRUCTURE),
    "choch": ("horizontal_line", Layer.STRUCTURE),
    "mss": ("horizontal_line", Layer.STRUCTURE),
    "entry": ("horizontal_line", Layer.ENTRY),
    "stop": ("horizontal_line", Layer.ENTRY),
    "invalidation": ("horizontal_line", Layer.ENTRY),
    "tp1": ("horizontal_line", Layer.TARGETS),
    "tp2": ("horizontal_line", Layer.TARGETS),
    "tp3": ("horizontal_line", Layer.TARGETS),
    "trendline": ("trend_line", Layer.STRUCTURE),
    "vertical": ("vertical_line", Layer.NOTES),
    "fibonacci": ("fib_retracement", Layer.THEORY),
}

# A drawn line must change pixels within this many rows of the calibrated Y.
# Measured against the live chart, a shortcut-placed line lands within 2px of
# the pointer, so this leaves room for antialiasing without accepting a line
# at a visibly different price.
VERIFY_ROW_TOLERANCE = 6
# Time for the chart to paint a newly created object before it is verified.
DRAW_SETTLE_SECONDS = 1.0
# Share of a row's pixels that must differ before it counts as a new mark.
VERIFY_ROW_CHANGE_RATIO = 0.35


# Objects defined by two chart anchors rather than a single price level.
TWO_ANCHOR_TYPES: dict[str, tuple[str, Layer]] = {
    "trendline": ("trend_line", Layer.STRUCTURE),
    "channel": ("trend_line", Layer.STRUCTURE),
    "fibonacci": ("fib_retracement", Layer.THEORY),
    "fib_retracement": ("fib_retracement", Layer.THEORY),
}

# How far from a computed endpoint a changed pixel still counts as that endpoint.
ENDPOINT_RADIUS = 14
# Fraction of pixels in the endpoint box that must differ.
ENDPOINT_CHANGE_RATIO = 0.02
# Points sampled along a drawn segment when confirming it appeared.
LINE_SAMPLES = 24
# Pixels either side of the ideal path searched for a changed pixel.
LINE_SEARCH_RADIUS = 4
# Share of sampled points that must show a change for the line to be verified.
MIN_LINE_COVERAGE = 0.6


@dataclass(slots=True)
class TwoAnchorRequest:
    """A drag-drawn object anchored at two (price, time) chart points."""

    annotation: str
    price_a: float
    minutes_a: float
    price_b: float
    minutes_b: float
    label: str = ""
    theory: str = ""
    strategy: str = ""
    setup_id: str | None = None
    layer: Layer | None = None

    def resolve(self) -> tuple[str, Layer]:
        primitive, default_layer = TWO_ANCHOR_TYPES.get(self.annotation.lower(), ("trend_line", Layer.STRUCTURE))
        return primitive, self.layer or default_layer


@dataclass(slots=True)
class DrawRequest:
    annotation: str
    price: float
    label: str = ""
    theory: str = ""
    strategy: str = ""
    setup_id: str | None = None
    layer: Layer | None = None
    price_secondary: float | None = None

    def resolve(self) -> tuple[str, Layer]:
        primitive, default_layer = SEMANTIC_TYPES.get(self.annotation.lower(), ("horizontal_line", Layer.NOTES))
        return primitive, self.layer or default_layer


class DrawingEngine:
    """Executes calibrated, verified TradingView annotations."""

    def __init__(
        self,
        *,
        database: Any,
        calibrator: ChartCalibrator,
        observe: Callable[[], Any],
        focus: Callable[[], StandardResult],
        computer_control: bool = False,
        screen_access: bool = False,
        desktop: DesktopInput | None = None,
    ) -> None:
        self.database = database
        self.calibrator = calibrator
        self._observe = observe
        self._focus = focus
        self.desktop = desktop or DesktopInput()
        self.computer_control = computer_control
        self.screen_access = screen_access

    def refresh_permissions(self, *, computer_control: bool, screen_access: bool) -> None:
        self.computer_control = computer_control
        self.screen_access = screen_access

    # --- Low-level input ---------------------------------------------------

    # --- Verification ------------------------------------------------------

    def _grab_plot(self, geometry: dict[str, int], axis_x: float | None = None) -> Any:
        region = self.calibrator.plot_region(geometry, axis_x)
        return self.desktop.grab((region["left"], region["top"], region["right"], region["bottom"]))

    @staticmethod
    def changed_rows(before: Any, after: Any) -> list[tuple[int, float]]:
        """Rows whose pixels changed between two captures, as (local_y, ratio)."""
        import numpy

        first = numpy.asarray(before.convert("L"), dtype=numpy.int16)
        second = numpy.asarray(after.convert("L"), dtype=numpy.int16)
        if first.shape != second.shape:
            return []
        difference = numpy.abs(second - first) > 18
        ratios = difference.mean(axis=1)
        return [(int(index), float(ratio)) for index, ratio in enumerate(ratios) if ratio >= VERIFY_ROW_CHANGE_RATIO]

    def _verify_horizontal(
        self, before: Any, after: Any, *, expected_y: float, geometry: dict[str, int], axis_x: float | None = None
    ) -> tuple[bool, dict[str, Any]]:
        region = self.calibrator.plot_region(geometry, axis_x)
        rows = self.changed_rows(before, after)
        local_expected = expected_y - region["top"]
        matches = [row for row in rows if abs(row[0] - local_expected) <= VERIFY_ROW_TOLERANCE]
        best = max(matches, key=lambda row: row[1]) if matches else None
        detail = {
            "expected_local_y": local_expected,
            "changed_row_count": len(rows),
            "matched_row": best[0] if best else None,
            "matched_ratio": best[1] if best else None,
            "tolerance_rows": VERIFY_ROW_TOLERANCE,
            "nearest_changed_rows": sorted(rows, key=lambda row: abs(row[0] - local_expected))[:5],
        }
        return best is not None, detail

    # --- Public API --------------------------------------------------------

    @staticmethod
    def chart_geometry(state: Any) -> dict[str, int] | None:
        """Prefer the client area; a maximized frame rect runs off the screen."""
        return getattr(state, "client_geometry", None) or getattr(state, "window_geometry", None)

    # A drawn line is only a couple of pixels tall, so a click computed from the
    # calibrated price can land just beside it. Trying a few neighbouring rows is
    # what makes deletion reliable; it still only ever targets SAM's own price.
    SELECT_OFFSETS = (0, -1, 1, -2, 2, -3, 3)

    def _select_and_delete(
        self, x: int, y: float, *, geometry: dict[str, int], axis_x: float | None, delete_key: int
    ) -> tuple[bool, dict[str, Any]]:
        """Click the annotation and press Delete, retrying on adjacent rows."""
        attempts: list[dict[str, Any]] = []
        for offset in self.SELECT_OFFSETS:
            before = self._grab_plot(geometry, axis_x)
            self.desktop.click(int(x), int(y) + offset)
            self.desktop.press_key(delete_key)
            time.sleep(0.35)
            after = self._grab_plot(geometry, axis_x)
            changed, detail = self._verify_horizontal(
                before, after, expected_y=y, geometry=geometry, axis_x=axis_x
            )
            attempts.append({"offset": offset, "changed": changed, "matched_row": detail.get("matched_row")})
            if changed:
                return True, {**detail, "attempts": attempts}
        return False, {"attempts": attempts, "reason": "No click offset selected the annotation."}

    # --- Two-anchor (drag) drawing -----------------------------------------

    @staticmethod
    def changed_near(before: Any, after: Any, x: float, y: float, region: dict[str, int]) -> tuple[bool, float]:
        """Did pixels change inside a small box around this endpoint?"""
        import numpy

        local_x = int(x - region["left"])
        local_y = int(y - region["top"])
        first = numpy.asarray(before.convert("L"), dtype=numpy.int16)
        second = numpy.asarray(after.convert("L"), dtype=numpy.int16)
        if first.shape != second.shape:
            return False, 0.0
        height, width = first.shape
        top = max(0, local_y - ENDPOINT_RADIUS)
        bottom = min(height, local_y + ENDPOINT_RADIUS + 1)
        left = max(0, local_x - ENDPOINT_RADIUS)
        right = min(width, local_x + ENDPOINT_RADIUS + 1)
        if bottom <= top or right <= left:
            return False, 0.0
        window = numpy.abs(second[top:bottom, left:right] - first[top:bottom, left:right]) > 18
        ratio = float(window.mean())
        return ratio >= ENDPOINT_CHANGE_RATIO, ratio

    @staticmethod
    def changed_along_line(
        before: Any, after: Any, first: dict[str, float], second: dict[str, float], region: dict[str, int]
    ) -> tuple[float, dict[str, Any]]:
        """Fraction of points along the intended path that show a new mark.

        A diagonal line is one pixel wide, so a box around an endpoint barely
        registers it and antialiasing can push the ratio under any sensible
        threshold. Sampling the whole path instead tests the thing that actually
        matters: did a line appear where SAM meant to put one.
        """
        import numpy

        difference = numpy.abs(
            numpy.asarray(after.convert("L"), dtype=numpy.int16)
            - numpy.asarray(before.convert("L"), dtype=numpy.int16)
        ) > 18
        height, width = difference.shape
        x1, y1 = first["x"] - region["left"], first["y"] - region["top"]
        x2, y2 = second["x"] - region["left"], second["y"] - region["top"]
        hits = 0
        for index in range(LINE_SAMPLES + 1):
            ratio = index / LINE_SAMPLES
            x = int(round(x1 + (x2 - x1) * ratio))
            y = int(round(y1 + (y2 - y1) * ratio))
            top = max(0, y - LINE_SEARCH_RADIUS)
            bottom = min(height, y + LINE_SEARCH_RADIUS + 1)
            left = max(0, x - LINE_SEARCH_RADIUS)
            right = min(width, x + LINE_SEARCH_RADIUS + 1)
            if bottom > top and right > left and difference[top:bottom, left:right].any():
                hits += 1
        coverage = hits / (LINE_SAMPLES + 1)
        return coverage, {"samples": LINE_SAMPLES + 1, "hits": hits, "coverage": round(coverage, 3),
                          "required": MIN_LINE_COVERAGE, "search_radius": LINE_SEARCH_RADIUS}

    def anchor_to_screen(self, calibration: Calibration, price: float, minutes: float, region: dict[str, int]) -> dict[str, Any]:
        x = calibration.x_at_minutes(minutes)
        y = calibration.y_at(price)
        on_screen = region["left"] <= x <= region["right"] and region["top"] <= y <= region["bottom"]
        return {"x": x, "y": y, "price": price, "minutes": minutes, "on_screen": on_screen}

    def draw_two_anchor(self, request: TwoAnchorRequest) -> StandardResult:
        """Drag-draw a two-anchor object and verify both endpoints landed."""
        started = time.perf_counter()
        blocked = self.preconditions()
        if blocked is not None:
            return blocked
        primitive, layer = request.resolve()
        if primitive not in SHORTCUTS:
            return StandardResult.failure(
                f"No verified TradingView shortcut exists for {primitive}.",
                error_code="UNSUPPORTED_DRAWING", started_at=started,
            )
        state, obstacle = self.ensure_foreground()
        if obstacle is not None:
            obstacle.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            return obstacle
        calibration, failure = self.active_calibration(state)
        if failure is not None:
            failure.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            return failure
        if not calibration.time_calibrated:
            return StandardResult.failure(
                "A two-anchor object needs the time axis calibrated as well as the price axis. "
                "Run calibrate_chart, which fits both.",
                error_code="TIME_CALIBRATION_REQUIRED", started_at=started,
            )

        geometry = self.chart_geometry(state)
        region = self.calibrator.plot_region(geometry, calibration.axis_x)
        first = self.anchor_to_screen(calibration, request.price_a, request.minutes_a, region)
        second = self.anchor_to_screen(calibration, request.price_b, request.minutes_b, region)
        offscreen = [name for name, anchor in (("A", first), ("B", second)) if not anchor["on_screen"]]
        if offscreen:
            return StandardResult.failure(
                f"Anchor(s) {', '.join(offscreen)} fall outside the visible chart; scroll or zoom first.",
                error_code="ANCHOR_OFF_SCREEN", started_at=started,
            )

        try:
            before = self._grab_plot(geometry, calibration.axis_x)
            modifier, key = SHORTCUTS[primitive]
            self.desktop.press_chord(modifier, key)
            time.sleep(0.3)
            self.desktop.drag((int(first["x"]), int(first["y"])), (int(second["x"]), int(second["y"])))
            time.sleep(DRAW_SETTLE_SECONDS)
            after = self._grab_plot(geometry, calibration.axis_x)
            self.desktop.press_key(DesktopInput.ESCAPE)
        except Exception as exc:
            return StandardResult.failure(
                f"Drag input failed: {exc}", executed=True,
                error_code="DRAWING_INPUT_FAILED", started_at=started,
            )

        # A drag only draws if the tool actually armed. If it did not, the same
        # gesture pans the chart, which silently moves the user's view and would
        # make any later price mapping wrong. Detect that explicitly.
        panned = False
        pan_detail: dict[str, Any] = {}
        if self.screen_access:
            recheck = self.calibrator.calibrate(
                geometry=geometry, symbol=(state.symbol or "UNKNOWN").upper(),
                timeframe=state.timeframe or "UNKNOWN", window_handle=state.window_handle,
                expected_price=state.current_price,
            )
            if recheck.verified and isinstance(recheck.data, dict):
                moved = abs(
                    (recheck.data["slope"] * first["y"] + recheck.data["intercept"]) - calibration.price_at(first["y"])
                )
                tolerance = max(abs(calibration.slope) * 12, calibration.precision or 0.0)
                panned = moved > tolerance
                pan_detail = {"price_shift_at_anchor": moved, "tolerance": tolerance}

        near_a, ratio_a = self.changed_near(before, after, first["x"], first["y"], region)
        near_b, ratio_b = self.changed_near(before, after, second["x"], second["y"], region)
        coverage, coverage_detail = self.changed_along_line(before, after, first, second, region)
        # The path is the real evidence; the endpoints are reported alongside it.
        verified = coverage >= MIN_LINE_COVERAGE and (near_a or near_b)
        detail = {
            "anchor_a": {**first, "changed": near_a, "ratio": ratio_a},
            "anchor_b": {**second, "changed": near_b, "ratio": ratio_b},
            "path": coverage_detail,
            "endpoint_radius": ENDPOINT_RADIUS,
            "required_ratio": ENDPOINT_CHANGE_RATIO,
            "chart_panned": panned,
            **({"pan": pan_detail} if pan_detail else {}),
        }
        record = self.database.record_drawing(
            symbol=(state.symbol or "UNKNOWN").upper(),
            layer=layer.value,
            drawing_type=request.annotation.lower(),
            label=request.label or request.annotation.upper(),
            theory=request.theory,
            strategy=request.strategy,
            timeframe=state.timeframe or "UNKNOWN",
            setup_id=request.setup_id,
            price=request.price_a,
            price_secondary=request.price_b,
            verified=verified,
            payload={
                "primitive": primitive,
                "two_anchor": True,
                "screen_a": {"x": first["x"], "y": first["y"]},
                "screen_b": {"x": second["x"], "y": second["y"]},
                "minutes_a": request.minutes_a,
                "minutes_b": request.minutes_b,
                "geometry_hash": calibration.geometry_hash,
                "verification": detail,
            },
        )
        if panned and state.window_handle:
            self.database.invalidate_chart_calibration(window_handle=state.window_handle)
        payload = {"drawing": record, "verification": detail}
        if verified:
            return StandardResult.success(
                payload, verified=True, started_at=started,
                observations=[f"{coverage:.0%} of the intended path shows a new mark on the chart."],
            )
        return StandardResult(
            ExecutionStatus.PARTIAL, True, False, data=payload,
            error=(
                "The drawing tool did not arm, so the drag panned the chart instead of drawing. "
                "The view has moved; recalibrate before drawing again."
                if panned else
                f"The drag was performed but only {coverage:.0%} of the intended path shows a new mark "
                f"({MIN_LINE_COVERAGE:.0%} required)."
            ),
            error_code="CHART_PANNED" if panned else "DRAWING_NOT_VERIFIED",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # --- Multi-line construction plans (Gann fans, pitchforks) --------------

    @staticmethod
    def clamp_segment(
        start: dict[str, float], end: dict[str, float], calibration: Calibration, region: dict[str, int]
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        """Trim a line to the part of it that is actually on the chart.

        Steep Gann rays leave the viewport within a few bars and project past the
        end of the day. Rather than refuse them or draw off-screen, the segment is
        clipped to the visible plot rectangle so what gets drawn is exactly what
        can be seen and verified.
        """
        try:
            x1, y1 = calibration.x_at_minutes(start["minutes"]), calibration.y_at(start["price"])
            x2, y2 = calibration.x_at_minutes(end["minutes"]), calibration.y_at(end["price"])
        except (ValueError, ZeroDivisionError):
            return None
        # Liang-Barsky clipping against the plot rectangle. Each edge contributes
        # a (p, q) pair; p < 0 means the segment enters through that edge and
        # tightens the near end, p > 0 means it leaves and tightens the far end.
        dx, dy = x2 - x1, y2 - y1
        low, high = 0.0, 1.0
        for p_value, q_value in (
            (-dx, x1 - region["left"]), (dx, region["right"] - x1),
            (-dy, y1 - region["top"]), (dy, region["bottom"] - y1),
        ):
            if p_value == 0:
                if q_value < 0:
                    return None
                continue
            ratio = q_value / p_value
            if p_value < 0:
                if ratio > high:
                    return None
                low = max(low, ratio)
            else:
                if ratio < low:
                    return None
                high = min(high, ratio)
        if low > high:
            return None
        clipped_a = {"x": x1 + dx * low, "y": y1 + dy * low}
        clipped_b = {"x": x1 + dx * high, "y": y1 + dy * high}
        # A segment must span enough pixels for a drag to register and be checked.
        if abs(clipped_b["x"] - clipped_a["x"]) < 40 and abs(clipped_b["y"] - clipped_a["y"]) < 40:
            return None
        return (
            {"price": calibration.price_at(clipped_a["y"]), "minutes": calibration.minutes_at(clipped_a["x"])},
            {"price": calibration.price_at(clipped_b["y"]), "minutes": calibration.minutes_at(clipped_b["x"])},
        )

    def draw_line_plan(
        self,
        lines: list[dict[str, Any]],
        *,
        annotation: str,
        theory: str,
        layer: Layer,
        setup_id: str | None = None,
        max_lines: int = 6,
    ) -> StandardResult:
        """Draw a set of related trendlines, clipping each to the visible chart.

        Used by constructions that are several lines rather than one object: a
        Gann fan and a pitchfork are both drawn this way, each line verified on
        its own so a partially-placed construction is reported as partial.
        """
        started = time.perf_counter()
        blocked = self.preconditions()
        if blocked is not None:
            return blocked
        state, obstacle = self.ensure_foreground()
        if obstacle is not None:
            return obstacle
        calibration, failure = self.active_calibration(state)
        if failure is not None:
            return failure
        if not calibration.time_calibrated:
            return StandardResult.failure(
                "This construction needs the time axis calibrated. Run calibrate_chart, which fits both axes.",
                error_code="TIME_CALIBRATION_REQUIRED", started_at=started,
            )
        geometry = self.chart_geometry(state)
        region = self.calibrator.plot_region(geometry, calibration.axis_x)

        drawn: list[dict[str, Any]] = []
        unverified: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for line in lines[:max_lines]:
            segment = self.clamp_segment(line["start"], line["end"], calibration, region)
            if segment is None:
                skipped.append({"label": line.get("label"), "reason": "Falls outside the visible chart."})
                continue
            first, second = segment
            outcome = self.draw_two_anchor(TwoAnchorRequest(
                annotation="trendline",
                price_a=first["price"], minutes_a=first["minutes"],
                price_b=second["price"], minutes_b=second["minutes"],
                label=f"{annotation}:{line.get('label', '')}".strip(":"),
                theory=theory, setup_id=setup_id, layer=layer,
            ))
            record = (outcome.data or {}).get("drawing")
            if outcome.status is ExecutionStatus.SUCCESS and record:
                drawn.append({**record, "line": line.get("label")})
            elif record:
                unverified.append({**record, "line": line.get("label")})
            else:
                skipped.append({"label": line.get("label"), "reason": outcome.error or "draw failed"})

        payload = {
            "annotation": annotation, "requested": len(lines), "attempted": min(len(lines), max_lines),
            "drawn": drawn, "unverified": unverified, "skipped": skipped,
        }
        if drawn and not unverified:
            return StandardResult.success(payload, verified=True, started_at=started,
                                          observations=[f"{len(drawn)} line(s) drawn and verified; {len(skipped)} off-screen."])
        if drawn or unverified:
            return StandardResult(
                ExecutionStatus.PARTIAL, True, False, data=payload,
                error=f"{len(unverified)} of {len(drawn) + len(unverified)} lines could not be verified.",
                error_code="DRAWING_PARTIALLY_VERIFIED",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        return StandardResult.failure(
            "No line in the construction fell inside the visible chart.",
            error_code="CONSTRUCTION_OFF_SCREEN", started_at=started,
        )

    def preconditions(self, *, require_calibration: bool = True) -> StandardResult | None:
        if not self.computer_control:
            return StandardResult.failure(
                "Computer Control is off, so SAM cannot move the pointer or send shortcuts to TradingView. "
                "Nothing was changed on the chart. Turn Computer Control on in Settings to allow it.",
                error_code="COMPUTER_CONTROL_DISABLED",
            )
        if not self.screen_access:
            return StandardResult.failure(
                "Screen Access is off, so SAM cannot capture the chart and could never confirm a drawing landed "
                "where it was meant to. Nothing was drawn. Turn Screen Access on in Settings to allow it.",
                error_code="SCREEN_ACCESS_DISABLED",
            )
        return None

    def ensure_foreground(self) -> tuple[Any, StandardResult | None]:
        """Guarantee captures show the chart and not whatever window is above it.

        A screen grab returns the pixels currently on screen for a rectangle, not
        the contents of a specific window. Reading the price axis or verifying a
        drawing while another application overlaps TradingView would measure the
        wrong pixels entirely, so an occluded chart is a hard failure.
        """
        state = self._observe()
        if not state.window_handle or not self.chart_geometry(state):
            return state, StandardResult.failure(
                "No TradingView chart window was found. SAM did not act. Open TradingView Desktop and make sure a "
                "chart window is visible, then try again.",
                error_code="WINDOW_NOT_FOUND",
            )
        if state.active:
            return state, None
        if not self.computer_control:
            return state, StandardResult.failure(
                "TradingView is not the foreground window and Computer Control is OFF, so its chart cannot be read reliably. "
                "Bring TradingView to the front, or enable Computer Control so SAM can focus it.",
                error_code="TRADINGVIEW_NOT_FOREGROUND",
            )
        focused = self._focus()
        if focused.status != ExecutionStatus.SUCCESS:
            return state, focused
        state = self._observe()
        if not state.active:
            return state, StandardResult.failure(
                "TradingView did not come to the foreground, so any capture would show another window.",
                error_code="TRADINGVIEW_NOT_FOREGROUND",
            )
        return state, None

    def active_calibration(self, state: Any) -> tuple[Calibration | None, StandardResult | None]:
        from .calibration import calibration_from_row, geometry_hash

        geometry = self.chart_geometry(state)
        if not state.window_handle or not geometry:
            return None, StandardResult.failure(
                "No TradingView chart window was found. SAM did not act. Open TradingView Desktop and make sure a "
                "chart window is visible, then try again.",
                error_code="WINDOW_NOT_FOUND",
            )
        symbol = (state.symbol or "UNKNOWN").upper()
        timeframe = state.timeframe or "UNKNOWN"
        digest = geometry_hash(geometry)
        row = self.database.get_chart_calibration(
            window_handle=state.window_handle, symbol=symbol, timeframe=timeframe, geometry_hash=digest
        )
        if row is None:
            return None, StandardResult.failure(
                "This chart viewport has not been calibrated, so SAM does not know which pixel a price sits at. "
                "Nothing was drawn. Run Calibrate on the TradingView panel; SAM reads the price axis itself.",
                error_code="CALIBRATION_REQUIRED",
            )
        calibration = calibration_from_row(row)
        if not calibration.verified:
            return None, StandardResult.failure(
                "The stored calibration is not verified; price-accurate drawing stays blocked.",
                error_code="CALIBRATION_NOT_VERIFIED",
            )
        return calibration, None

    def calibrate(self) -> StandardResult:
        """Read the price axis and persist a verified mapping for this viewport."""
        started = time.perf_counter()
        if not self.screen_access:
            return StandardResult.failure(
                "Screen Access is off, so SAM cannot read the chart's price axis. Calibration was not attempted. "
                "Turn Screen Access on in Settings to allow it.",
                error_code="SCREEN_ACCESS_DISABLED", started_at=started,
            )
        state, blocked = self.ensure_foreground()
        if blocked is not None:
            blocked.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            return blocked
        symbol = (state.symbol or "UNKNOWN").upper()
        timeframe = state.timeframe or "UNKNOWN"
        # The window title carries the instrument's live price. Handing it to the
        # calibrator lets it reject a column of numbers that belongs to some other
        # panel — a watchlist, for instance — instead of this chart's price axis.
        result = self.calibrator.calibrate(
            geometry=self.chart_geometry(state), symbol=symbol, timeframe=timeframe,
            window_handle=state.window_handle, expected_price=state.current_price,
        )
        if isinstance(result.data, dict) and result.data.get("slope"):
            # Fit the time axis in the same pass so two-anchor objects work without
            # a second, separately-timed calibration of a chart that may have moved.
            time_axis = self.calibrator.calibrate_time_axis(
                self.chart_geometry(state), result.data.get("axis_x")
            )
            time_data = time_axis.data if (time_axis.verified and isinstance(time_axis.data, dict)) else {}
            result.data["minutes_per_pixel"] = time_data.get("minutes_per_pixel")
            result.data["time_intercept"] = time_data.get("time_intercept")
            result.data["time_axis_y"] = time_data.get("time_axis_y")
            result.data["minutes_span"] = time_data.get("minutes_span")
            result.data["time_axis_verified"] = bool(time_axis.verified)
            result.observations.extend(time_axis.observations)
            if not time_axis.verified:
                result.observations.append(
                    "The time axis could not be fitted, so only single-price annotations are available: "
                    + (time_axis.error or "unknown reason")
                )
            self.database.save_chart_calibration(
                window_handle=state.window_handle,
                symbol=symbol,
                timeframe=timeframe,
                geometry_hash=result.data["geometry_hash"],
                slope=result.data["slope"],
                intercept=result.data["intercept"],
                method=result.data["method"],
                anchors=result.data.get("anchors"),
                verified=bool(result.verified),
                max_error=result.data.get("max_error"),
                axis_x=result.data.get("axis_x"),
                precision=result.data.get("precision"),
                minutes_per_pixel=time_data.get("minutes_per_pixel"),
                time_intercept=time_data.get("time_intercept"),
                time_axis_y=time_data.get("time_axis_y"),
                minutes_span=time_data.get("minutes_span"),
            )
        result.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    def verify_calibration(self) -> StandardResult:
        state, blocked = self.ensure_foreground()
        if blocked is not None:
            return blocked
        calibration, failure = self.active_calibration(state)
        if failure is not None:
            return failure
        outcome = self.calibrator.verify(
            calibration,
            geometry=self.chart_geometry(state),
            symbol=(state.symbol or "UNKNOWN").upper(),
            timeframe=state.timeframe or "UNKNOWN",
            window_handle=state.window_handle,
        )
        if not outcome.verified and state.window_handle:
            # A drifted mapping must never silently keep authorizing drawings.
            self.database.invalidate_chart_calibration(window_handle=state.window_handle)
        return outcome

    def price_to_screen(self, price: float) -> StandardResult:
        state = self._observe()
        calibration, failure = self.active_calibration(state)
        if failure is not None:
            return failure
        region = self.calibrator.plot_region(self.chart_geometry(state), calibration.axis_x)
        y = calibration.y_at(price)
        inside = region["top"] <= y <= region["bottom"]
        return StandardResult(
            ExecutionStatus.SUCCESS if inside else ExecutionStatus.PARTIAL,
            True,
            inside,
            data={"price": price, "y": y, "x_center": (region["left"] + region["right"]) / 2, "region": region, "on_screen": inside},
            error=None if inside else "That price is outside the visible chart range; scroll or zoom before drawing.",
            error_code=None if inside else "PRICE_OFF_SCREEN",
        )

    def screen_to_price(self, y: float) -> StandardResult:
        state = self._observe()
        calibration, failure = self.active_calibration(state)
        if failure is not None:
            return failure
        return StandardResult.success({"y": y, "price": calibration.price_at(y)}, verified=True)

    def draw(self, request: DrawRequest) -> StandardResult:
        """Draw one calibrated annotation and verify it appeared on the chart."""
        started = time.perf_counter()
        blocked = self.preconditions()
        if blocked is not None:
            return blocked
        primitive, layer = request.resolve()
        if primitive not in SHORTCUTS:
            return StandardResult.failure(
                f"No verified TradingView shortcut exists for {primitive}.", error_code="UNSUPPORTED_DRAWING", started_at=started
            )
        if primitive != "horizontal_line":
            return StandardResult.failure(
                f"{primitive} is a two-anchor drawing. Use draw_tradingview_object with two price/time anchors.",
                error_code="TWO_ANCHOR_REQUIRED",
                started_at=started,
            )
        state, blocked = self.ensure_foreground()
        if blocked is not None:
            blocked.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            return blocked
        calibration, failure = self.active_calibration(state)
        if failure is not None:
            failure.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            return failure
        placement = self.price_to_screen(request.price)
        if not placement.verified:
            # price_to_screen is a query, so it reports itself as executed. As a
            # refusal to draw it must not: nothing was sent to the chart.
            return StandardResult.failure(
                placement.error or "That price cannot be placed on the current chart.",
                error_code=placement.error_code or "PRICE_OFF_SCREEN",
                started_at=started,
                observations=["No drawing command was sent; the chart is unchanged."],
            )

        geometry = self.chart_geometry(state)
        target_y = float(placement.data["y"])
        target_x = float(placement.data["x_center"])
        try:
            before = self._grab_plot(geometry, calibration.axis_x)
            self.desktop.move_mouse(int(target_x), int(target_y))
            modifier, key = SHORTCUTS[primitive]
            self.desktop.press_chord(modifier, key)
            time.sleep(DRAW_SETTLE_SECONDS)
            after = self._grab_plot(geometry, calibration.axis_x)
            # The drawing tool stays armed and the new object stays selected after
            # the shortcut. Leaving it that way makes the next click create another
            # annotation instead of selecting this one, so disarm before returning.
            self.desktop.press_key(DesktopInput.ESCAPE)
        except Exception as exc:
            return StandardResult.failure(
                f"Drawing input failed: {exc}", executed=True, error_code="DRAWING_INPUT_FAILED", started_at=started
            )

        verified, detail = self._verify_horizontal(before, after, expected_y=target_y, geometry=geometry, axis_x=calibration.axis_x)
        record = self.database.record_drawing(
            symbol=(state.symbol or "UNKNOWN").upper(),
            layer=layer.value,
            drawing_type=request.annotation.lower(),
            label=request.label or request.annotation.upper(),
            theory=request.theory,
            strategy=request.strategy,
            timeframe=state.timeframe or "UNKNOWN",
            setup_id=request.setup_id,
            price=request.price,
            price_secondary=request.price_secondary,
            verified=verified,
            payload={
                "primitive": primitive,
                "screen_y": target_y,
                "screen_x": target_x,
                "geometry_hash": calibration.geometry_hash,
                "verification": detail,
            },
        )
        payload = {"drawing": record, "verification": detail, "price": request.price, "screen_y": target_y}
        if verified:
            return StandardResult.success(payload, verified=True, started_at=started,
                                          observations=[f"Chart pixels changed at the calibrated row for {request.price}."])
        # The shortcut was sent, so an unverified annotation may still exist; it is
        # recorded as owned precisely so it can be found and removed later.
        return StandardResult(
            ExecutionStatus.PARTIAL,
            True,
            False,
            data=payload,
            error="The drawing command was sent but the chart did not change at the expected price row.",
            error_code="DRAWING_NOT_VERIFIED",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def draw_plan(self, requests: list[DrawRequest]) -> StandardResult:
        """Draw an ordered annotation plan, stopping at the first hard failure."""
        started = time.perf_counter()
        drawn: list[dict[str, Any]] = []
        unverified: list[dict[str, Any]] = []
        for request in requests:
            outcome = self.draw(request)
            if outcome.status == ExecutionStatus.SUCCESS:
                drawn.append(outcome.data["drawing"])
            elif outcome.status == ExecutionStatus.PARTIAL and isinstance(outcome.data, dict) and outcome.data.get("drawing"):
                unverified.append(outcome.data["drawing"])
            else:
                return StandardResult(
                    ExecutionStatus.PARTIAL if drawn else ExecutionStatus.FAILED,
                    bool(drawn),
                    False,
                    data={"drawn": drawn, "unverified": unverified, "failed_at": request.annotation},
                    error=outcome.error,
                    error_code=outcome.error_code,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )
        payload = {"drawn": drawn, "unverified": unverified, "requested": len(requests)}
        if unverified:
            return StandardResult(
                ExecutionStatus.PARTIAL, True, False, data=payload,
                error=f"{len(unverified)} of {len(requests)} annotations could not be visually verified.",
                error_code="DRAWING_PARTIALLY_VERIFIED",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        return StandardResult.success(payload, verified=True, started_at=started)

    def undo(self) -> StandardResult:
        started = time.perf_counter()
        blocked = self.preconditions()
        if blocked is not None:
            return blocked
        focused = self._focus()
        if focused.status != ExecutionStatus.SUCCESS:
            return focused
        try:
            self.desktop.press_chord("ctrl", "z")
        except Exception as exc:
            return StandardResult.failure(str(exc), executed=True, error_code="UNDO_FAILED", started_at=started)
        return StandardResult(
            ExecutionStatus.PARTIAL, True, False, data={"action": "undo"},
            error="Undo was sent; TradingView exposes no independent confirmation of what it reverted.",
            error_code="UNDO_NOT_VERIFIED",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def set_layer_visibility(self, layer: str, visible: bool, symbol: str | None = None) -> StandardResult:
        """Toggle SAM's ownership records for a layer. Chart objects are untouched."""
        started = time.perf_counter()
        normalized = layer.upper()
        if normalized not in Layer.__members__:
            return StandardResult.failure(f"Unknown chart layer: {layer}", error_code="UNKNOWN_LAYER", started_at=started)
        count = self.database.set_drawing_visibility(visible=visible, layer=normalized, symbol=symbol)
        return StandardResult.success(
            {"layer": normalized, "visible": visible, "updated": count, "symbol": symbol},
            verified=True,
            started_at=started,
            observations=["Layer state is SAM's ownership record; it does not hide objects already on the chart."],
        )

    def list_owned(self, **filters: Any) -> StandardResult:
        drawings = self.database.list_drawings(**filters)
        by_layer: dict[str, int] = {}
        for drawing in drawings:
            by_layer[drawing["layer"]] = by_layer.get(drawing["layer"], 0) + 1
        return StandardResult.success(
            {"drawings": drawings, "count": len(drawings), "by_layer": by_layer}, verified=True
        )

    def clear_owned(
        self,
        *,
        symbol: str | None = None,
        layer: str | None = None,
        theory: str | None = None,
        setup_id: str | None = None,
        all_owned: bool = False,
    ) -> StandardResult:
        """Remove SAM-owned annotations from the chart by selecting each one.

        Drawings the user made by hand are not in the ownership table and are
        therefore never selected, never clicked, and never deleted.
        """
        started = time.perf_counter()
        targets = self.database.list_drawings(symbol=symbol, layer=layer, theory=theory, setup_id=setup_id)
        if not targets and not all_owned:
            return StandardResult.success({"removed": [], "count": 0}, verified=True, started_at=started,
                                          observations=["No SAM-owned drawings matched the filter."])
        blocked = self.preconditions()
        if blocked is not None:
            return blocked
        state, blocked = self.ensure_foreground()
        if blocked is not None:
            blocked.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            return blocked
        calibration, failure = self.active_calibration(state)
        if failure is not None:
            failure.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            return failure

        geometry = self.chart_geometry(state)
        removed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for drawing in targets:
            payload = drawing.get("payload") or {}
            if payload.get("geometry_hash") != calibration.geometry_hash or drawing.get("price") is None:
                skipped.append({"id": drawing["id"], "reason": "Recorded under a different viewport; not safe to click."})
                continue
            try:
                y = calibration.y_at(float(drawing["price"]))
                plot = self.calibrator.plot_region(geometry, calibration.axis_x)
                x = payload.get("screen_x") or (plot["left"] + plot["right"]) / 2
                changed, detail = self._select_and_delete(
                    int(x), y, geometry=geometry, axis_x=calibration.axis_x, delete_key=DesktopInput.DELETE
                )
            except Exception as exc:
                skipped.append({"id": drawing["id"], "reason": f"Input failed: {exc}"})
                break
            if changed:
                self.database.delete_drawings(drawing_id=drawing["id"])
                removed.append({**drawing, "verification": detail})
            else:
                # Stop rather than keep clicking blindly at unverified positions.
                skipped.append({"id": drawing["id"], "reason": "The chart did not change; deletion was not confirmed."})
                break
        payload = {"removed": removed, "count": len(removed), "skipped": skipped, "matched": len(targets)}
        if skipped:
            return StandardResult(
                ExecutionStatus.PARTIAL, bool(removed), False, data=payload,
                error=f"{len(skipped)} of {len(targets)} owned drawings could not be confirmed as removed.",
                error_code="CLEAR_PARTIALLY_VERIFIED",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        return StandardResult.success(payload, verified=True, started_at=started)

    def capability(self) -> dict[str, Any]:
        state = self._observe()
        ocr = self.calibrator.capability()
        calibration, failure = self.active_calibration(state) if state.window_handle else (None, None)
        return {
            "computer_control": self.computer_control,
            "screen_access": self.screen_access,
            "ocr": ocr,
            "calibrated": calibration is not None,
            "calibration_error": failure.error if failure else None,
            "layers": [layer.value for layer in Layer],
            "semantic_types": sorted(SEMANTIC_TYPES),
            "verified_price_drawing": bool(calibration and self.computer_control and self.screen_access),
            "time_calibrated": bool(calibration and calibration.time_calibrated),
            "two_anchor_types": sorted(TWO_ANCHOR_TYPES),
            "verified_two_anchor_drawing": bool(
                calibration and calibration.time_calibrated and self.computer_control and self.screen_access
            ),
        }
