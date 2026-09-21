from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Iterable

from .types import Candle


def _values(candles: Iterable[Candle], field: str = "close") -> list[float]:
    return [float(getattr(candle, field)) for candle in candles]


def sma(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
    result: list[float | None] = [None] * len(values)
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index >= period - 1:
            result[index] = running / period
    return result


def ema(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
    if not values:
        return []
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    seed = fmean(values[:period])
    result[period - 1] = seed
    multiplier = 2 / (period + 1)
    previous = seed
    for index in range(period, len(values)):
        previous = (values[index] - previous) * multiplier + previous
        result[index] = previous
    return result


def wma(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
    denominator = period * (period + 1) / 2
    result: list[float | None] = [None] * len(values)
    for index in range(period - 1, len(values)):
        window = values[index - period + 1:index + 1]
        result[index] = sum(value * weight for value, weight in zip(window, range(1, period + 1))) / denominator
    return result


def true_ranges(candles: list[Candle]) -> list[float]:
    result: list[float] = []
    for index, candle in enumerate(candles):
        previous_close = candles[index - 1].close if index else candle.close
        result.append(max(candle.high - candle.low, abs(candle.high - previous_close), abs(candle.low - previous_close)))
    return result


def atr(candles: list[Candle], period: int = 14) -> list[float | None]:
    ranges = true_ranges(candles)
    result: list[float | None] = [None] * len(ranges)
    if len(ranges) < period:
        return result
    previous = fmean(ranges[:period])
    result[period - 1] = previous
    for index in range(period, len(ranges)):
        previous = ((previous * (period - 1)) + ranges[index]) / period
        result[index] = previous
    return result


def rsi(values: list[float], period: int = 14) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return result
    gains: list[float] = []
    losses: list[float] = []
    for left, right in zip(values, values[1:period + 1]):
        change = right - left
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = fmean(gains)
    avg_loss = fmean(losses)

    def score() -> float:
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        relative = avg_gain / avg_loss
        return 100 - 100 / (1 + relative)

    result[period] = score()
    for index in range(period + 1, len(values)):
        change = values[index] - values[index - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
        result[index] = score()
    return result


def macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, list[float | None]]:
    fast_values = ema(values, fast)
    slow_values = ema(values, slow)
    line: list[float | None] = [
        (fast_value - slow_value) if fast_value is not None and slow_value is not None else None
        for fast_value, slow_value in zip(fast_values, slow_values)
    ]
    compact = [value for value in line if value is not None]
    compact_signal = ema(compact, signal)
    signal_line: list[float | None] = [None] * len(values)
    start = next((index for index, value in enumerate(line) if value is not None), len(values))
    for offset, value in enumerate(compact_signal):
        if start + offset < len(signal_line):
            signal_line[start + offset] = value
    histogram = [
        (left - right) if left is not None and right is not None else None
        for left, right in zip(line, signal_line)
    ]
    return {"macd": line, "signal": signal_line, "histogram": histogram}


def stochastic(candles: list[Candle], period: int = 14, smooth: int = 3) -> dict[str, list[float | None]]:
    k_values: list[float | None] = [None] * len(candles)
    for index in range(period - 1, len(candles)):
        window = candles[index - period + 1:index + 1]
        high = max(item.high for item in window)
        low = min(item.low for item in window)
        k_values[index] = 50.0 if high == low else (candles[index].close - low) / (high - low) * 100
    compact = [value for value in k_values if value is not None]
    compact_d = sma(compact, smooth)
    d_values: list[float | None] = [None] * len(candles)
    start = period - 1
    for offset, value in enumerate(compact_d):
        if start + offset < len(d_values):
            d_values[start + offset] = value
    return {"k": k_values, "d": d_values}


def stochastic_rsi(values: list[float], rsi_period: int = 14, stochastic_period: int = 14) -> list[float | None]:
    rsi_values = rsi(values, rsi_period)
    result: list[float | None] = [None] * len(values)
    for index in range(len(values)):
        start = index - stochastic_period + 1
        if start < 0:
            continue
        window = rsi_values[start:index + 1]
        if any(value is None for value in window) or rsi_values[index] is None:
            continue
        numeric = [float(value) for value in window if value is not None]
        low, high = min(numeric), max(numeric)
        result[index] = 50.0 if high == low else (float(rsi_values[index]) - low) / (high - low) * 100
    return result


def bollinger(values: list[float], period: int = 20, deviations: float = 2.0) -> dict[str, list[float | None]]:
    middle = sma(values, period)
    upper: list[float | None] = [None] * len(values)
    lower: list[float | None] = [None] * len(values)
    for index in range(period - 1, len(values)):
        deviation = pstdev(values[index - period + 1:index + 1])
        assert middle[index] is not None
        upper[index] = middle[index] + deviations * deviation
        lower[index] = middle[index] - deviations * deviation
    return {"middle": middle, "upper": upper, "lower": lower}


def keltner(candles: list[Candle], period: int = 20, multiplier: float = 2.0) -> dict[str, list[float | None]]:
    middle = ema(_values(candles), period)
    atr_values = atr(candles, period)
    upper = [m + multiplier * a if m is not None and a is not None else None for m, a in zip(middle, atr_values)]
    lower = [m - multiplier * a if m is not None and a is not None else None for m, a in zip(middle, atr_values)]
    return {"middle": middle, "upper": upper, "lower": lower}


def adx(candles: list[Candle], period: int = 14) -> list[float | None]:
    if len(candles) < period * 2:
        return [None] * len(candles)
    tr = true_ranges(candles)
    plus_dm = [0.0]
    minus_dm = [0.0]
    for previous, current in zip(candles, candles[1:]):
        up = current.high - previous.high
        down = previous.low - current.low
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    smoothed_tr = sum(tr[1:period + 1])
    smoothed_plus = sum(plus_dm[1:period + 1])
    smoothed_minus = sum(minus_dm[1:period + 1])
    dx: list[float | None] = [None] * len(candles)
    for index in range(period, len(candles)):
        if index > period:
            smoothed_tr = smoothed_tr - smoothed_tr / period + tr[index]
            smoothed_plus = smoothed_plus - smoothed_plus / period + plus_dm[index]
            smoothed_minus = smoothed_minus - smoothed_minus / period + minus_dm[index]
        plus_di = 100 * smoothed_plus / smoothed_tr if smoothed_tr else 0
        minus_di = 100 * smoothed_minus / smoothed_tr if smoothed_tr else 0
        denominator = plus_di + minus_di
        dx[index] = 100 * abs(plus_di - minus_di) / denominator if denominator else 0
    result: list[float | None] = [None] * len(candles)
    first = [value for value in dx[period:period * 2] if value is not None]
    if len(first) == period:
        previous_adx = fmean(first)
        result[period * 2 - 1] = previous_adx
        for index in range(period * 2, len(candles)):
            assert dx[index] is not None
            previous_adx = (previous_adx * (period - 1) + dx[index]) / period
            result[index] = previous_adx
    return result


def parabolic_sar(candles: list[Candle], step: float = 0.02, maximum: float = 0.2) -> list[float | None]:
    if len(candles) < 2:
        return [None] * len(candles)
    result: list[float | None] = [None] * len(candles)
    bullish = candles[1].close >= candles[0].close
    extreme = candles[0].high if bullish else candles[0].low
    sar = candles[0].low if bullish else candles[0].high
    acceleration = step
    for index in range(1, len(candles)):
        sar = sar + acceleration * (extreme - sar)
        if bullish:
            sar = min(sar, candles[index - 1].low, candles[index - 2].low if index > 1 else candles[index - 1].low)
            if candles[index].low < sar:
                bullish = False
                sar = extreme
                extreme = candles[index].low
                acceleration = step
            elif candles[index].high > extreme:
                extreme = candles[index].high
                acceleration = min(maximum, acceleration + step)
        else:
            sar = max(sar, candles[index - 1].high, candles[index - 2].high if index > 1 else candles[index - 1].high)
            if candles[index].high > sar:
                bullish = True
                sar = extreme
                extreme = candles[index].high
                acceleration = step
            elif candles[index].low < extreme:
                extreme = candles[index].low
                acceleration = min(maximum, acceleration + step)
        result[index] = sar
    return result


def ichimoku(candles: list[Candle]) -> dict[str, list[float | None]]:
    def midpoint(period: int) -> list[float | None]:
        result: list[float | None] = [None] * len(candles)
        for index in range(period - 1, len(candles)):
            window = candles[index - period + 1:index + 1]
            result[index] = (max(item.high for item in window) + min(item.low for item in window)) / 2
        return result

    conversion = midpoint(9)
    base = midpoint(26)
    span_b_raw = midpoint(52)
    span_a: list[float | None] = [None] * len(candles)
    span_b: list[float | None] = [None] * len(candles)
    lagging: list[float | None] = [None] * len(candles)
    for index in range(len(candles)):
        if index + 26 < len(candles):
            if conversion[index] is not None and base[index] is not None:
                span_a[index + 26] = (conversion[index] + base[index]) / 2
            span_b[index + 26] = span_b_raw[index]
        if index >= 26:
            lagging[index - 26] = candles[index].close
    return {"conversion": conversion, "base": base, "span_a": span_a, "span_b": span_b, "lagging": lagging}


def vwap(candles: list[Candle]) -> list[float | None]:
    result: list[float | None] = []
    cumulative_value = 0.0
    cumulative_volume = 0.0
    for candle in candles:
        typical = (candle.high + candle.low + candle.close) / 3
        cumulative_value += typical * candle.volume
        cumulative_volume += candle.volume
        result.append(cumulative_value / cumulative_volume if cumulative_volume else None)
    return result


def volume_profile(candles: list[Candle], bins: int = 48, value_area: float = 0.70) -> dict[str, object]:
    if not candles:
        raise ValueError("candles cannot be empty")
    bins = max(8, min(200, bins))
    low = min(candle.low for candle in candles)
    high = max(candle.high for candle in candles)
    if high == low:
        return {"poc": low, "vah": high, "val": low, "hvn": [low], "lvn": [], "bins": []}
    width = (high - low) / bins
    volumes = [0.0] * bins
    for candle in candles:
        typical = (candle.high + candle.low + candle.close) / 3
        index = min(bins - 1, max(0, int((typical - low) / width)))
        volumes[index] += candle.volume
    total = sum(volumes)
    poc_index = max(range(bins), key=volumes.__getitem__)
    selected = {poc_index}
    selected_volume = volumes[poc_index]
    left, right = poc_index - 1, poc_index + 1
    while total and selected_volume / total < value_area and (left >= 0 or right < bins):
        left_volume = volumes[left] if left >= 0 else -1
        right_volume = volumes[right] if right < bins else -1
        chosen = left if left_volume >= right_volume else right
        selected.add(chosen)
        selected_volume += volumes[chosen]
        if chosen == left:
            left -= 1
        else:
            right += 1
    centers = [low + (index + 0.5) * width for index in range(bins)]
    ranked = sorted(range(bins), key=volumes.__getitem__, reverse=True)
    positive = [value for value in volumes if value > 0]
    threshold = fmean(positive) * 0.35 if positive else 0
    return {
        "poc": centers[poc_index],
        "vah": centers[max(selected)],
        "val": centers[min(selected)],
        "hvn": [centers[index] for index in ranked[: min(5, bins)] if volumes[index] > 0],
        "lvn": [centers[index] for index, value in enumerate(volumes) if 0 < value <= threshold][:5],
        "bins": [{"price": centers[index], "volume": volumes[index]} for index in range(bins)],
    }


def pivot_points(high: float, low: float, close: float, kind: str = "standard") -> dict[str, float]:
    kind = kind.lower()
    pivot = (high + low + close) / 3
    spread = high - low
    if kind in {"standard", "traditional"}:
        return {
            "P": pivot,
            "R1": 2 * pivot - low,
            "S1": 2 * pivot - high,
            "R2": pivot + spread,
            "S2": pivot - spread,
            "R3": high + 2 * (pivot - low),
            "S3": low - 2 * (high - pivot),
        }
    if kind == "fibonacci":
        return {
            "P": pivot,
            "R1": pivot + 0.382 * spread,
            "S1": pivot - 0.382 * spread,
            "R2": pivot + 0.618 * spread,
            "S2": pivot - 0.618 * spread,
            "R3": pivot + spread,
            "S3": pivot - spread,
        }
    if kind == "woodie":
        pivot = (high + low + 2 * close) / 4
        return {"P": pivot, "R1": 2 * pivot - low, "S1": 2 * pivot - high, "R2": pivot + spread, "S2": pivot - spread}
    if kind == "camarilla":
        return {
            "P": pivot,
            **{f"R{level}": close + spread * multiplier for level, multiplier in enumerate((0.0916, 0.183, 0.275, 0.55), 1)},
            **{f"S{level}": close - spread * multiplier for level, multiplier in enumerate((0.0916, 0.183, 0.275, 0.55), 1)},
        }
    raise ValueError(f"Unsupported pivot type: {kind}")


def latest(values: list[float | None]) -> float | None:
    return next((value for value in reversed(values) if value is not None and math.isfinite(value)), None)


def find_swings(candles: list[Candle], left: int = 2, right: int = 2) -> dict[str, list[dict[str, Any]]]:
    """Pivot highs and lows. Lives here so pattern engines and the analysis layer
    can both use it without importing each other."""
    if left < 1 or right < 1:
        raise ValueError("left and right swing windows must be positive")
    highs: list[dict[str, Any]] = []
    lows: list[dict[str, Any]] = []
    for index in range(left, len(candles) - right):
        candle = candles[index]
        left_window = candles[index - left:index]
        right_window = candles[index + 1:index + right + 1]
        if all(candle.high > item.high for item in left_window) and all(candle.high >= item.high for item in right_window):
            highs.append({"index": index, "price": candle.high, "time": candle.time.isoformat(), "kind": "SWING_HIGH"})
        if all(candle.low < item.low for item in left_window) and all(candle.low <= item.low for item in right_window):
            lows.append({"index": index, "price": candle.low, "time": candle.time.isoformat(), "kind": "SWING_LOW"})
    return {"highs": highs, "lows": lows}
