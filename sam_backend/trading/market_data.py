from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

from ..contracts import CapabilityState, ExecutionStatus, StandardResult
from .types import Candle, MarketDataBatch, TIMEFRAME_SECONDS, capability, normalize_timeframe


class MarketDataError(RuntimeError):
    pass


class MetaTrader5Provider:
    """Read-only OHLCV/tick adapter for an installed MetaTrader 5 terminal."""

    provider_id = "metatrader5"
    tick_freshness_seconds = 300

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @staticmethod
    def _module():
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise MarketDataError("The MetaTrader5 Python package is not installed") from exc
        except OSError as exc:
            # The import system reads MetaTrader5/__init__.py and its _core
            # extension; a file it cannot read arrives as a raw OSError, not an
            # ImportError. Same meaning for us: no broker. The path stays out of
            # the message, which is returned to API clients.
            raise MarketDataError("The MetaTrader5 Python package could not be loaded") from exc
        return mt5

    @staticmethod
    def _timeframe_value(mt5: Any, timeframe: str) -> int:
        normalized = normalize_timeframe(timeframe)
        attribute = f"TIMEFRAME_{normalized}"
        value = getattr(mt5, attribute, None)
        if value is None:
            raise MarketDataError(f"MetaTrader 5 does not expose timeframe {timeframe}")
        return int(value)

    @staticmethod
    def _candidate_score(name: str, requested: str, visible: bool) -> tuple[int, int, int, str]:
        upper = name.upper()
        base = requested.upper()
        return (
            0 if upper == base else 1,
            0 if upper.startswith(base) else 1,
            0 if visible else 1,
            upper,
        )

    def _resolve_symbol(self, mt5: Any, requested: str) -> tuple[str, Any]:
        requested = requested.strip()
        if not requested or len(requested) > 64:
            raise MarketDataError("Symbol must contain 1 to 64 characters")
        exact = mt5.symbol_info(requested)
        if exact is not None:
            return str(exact.name), exact
        candidates = list(mt5.symbols_get(f"*{requested}*") or [])
        if not candidates:
            condensed = requested.replace("/", "").replace("-", "")
            candidates = list(mt5.symbols_get(f"*{condensed}*") or [])
        if not candidates:
            raise MarketDataError(f"No MetaTrader symbol matches {requested}")
        candidates.sort(key=lambda item: self._candidate_score(str(item.name), requested, bool(item.visible)))
        chosen = candidates[0]
        return str(chosen.name), chosen

    def _initialize(self, mt5: Any) -> None:
        if mt5.initialize():
            return
        candidates: list[str] = []
        try:
            import psutil

            for process in psutil.process_iter(["name", "exe"]):
                try:
                    if (process.info.get("name") or "").lower() in {"terminal64.exe", "terminal.exe"} and process.info.get("exe"):
                        candidates.append(str(process.info["exe"]))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            pass
        candidates.extend([
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        ])
        mt5.shutdown()
        for candidate in dict.fromkeys(candidates):
            if not candidate or not __import__("os").path.isfile(candidate):
                continue
            if mt5.initialize(path=candidate):
                return
            mt5.shutdown()
        raise MarketDataError(f"MetaTrader 5 initialization failed: {mt5.last_error()}")

    def health(self) -> dict[str, Any]:
        try:
            mt5 = self._module()
        except MarketDataError as exc:
            return {"provider": self.provider_id, "state": CapabilityState.UNAVAILABLE.value, "error": str(exc)}
        with self._lock:
            try:
                self._initialize(mt5)
                terminal = mt5.terminal_info()
                account = mt5.account_info()
                return {
                    "provider": self.provider_id,
                    "state": CapabilityState.AVAILABLE.value,
                    "terminal_connected": bool(getattr(terminal, "connected", False)),
                    "trade_allowed": False,
                    "read_only_adapter": True,
                    "broker": str(getattr(account, "company", "")) or None,
                    "server": str(getattr(account, "server", "")) or None,
                }
            except Exception as exc:
                return {"provider": self.provider_id, "state": CapabilityState.UNAVAILABLE.value, "error": str(exc)}
            finally:
                mt5.shutdown()

    def fetch(self, symbol: str, timeframe: str, count: int = 500) -> MarketDataBatch:
        count = max(50, min(10_000, int(count)))
        normalized = normalize_timeframe(timeframe)
        mt5 = self._module()
        with self._lock:
            try:
                self._initialize(mt5)
                resolved_symbol, info = self._resolve_symbol(mt5, symbol)
                if not bool(getattr(info, "visible", False)):
                    mt5.symbol_select(resolved_symbol, True)
                    info = mt5.symbol_info(resolved_symbol) or info
                mt5_timeframe = self._timeframe_value(mt5, normalized)
                rates = mt5.copy_rates_from_pos(resolved_symbol, mt5_timeframe, 0, count)
                if rates is None or len(rates) == 0:
                    raise MarketDataError(f"No {normalized} bars returned for {resolved_symbol}: {mt5.last_error()}")
                tick = mt5.symbol_info_tick(resolved_symbol)
                terminal = mt5.terminal_info()
                account = mt5.account_info()
                bid_raw = getattr(tick, "bid", None) if tick is not None else None
                ask_raw = getattr(tick, "ask", None) if tick is not None else None
                bid = float(bid_raw) if bid_raw not in {None, 0} else None
                ask = float(ask_raw) if ask_raw not in {None, 0} else None
                tick_time: datetime | None = None
                tick_timestamp_error: str | None = None
                if tick is not None:
                    tick_milliseconds = int(getattr(tick, "time_msc", 0) or 0)
                    tick_seconds = int(getattr(tick, "time", 0) or 0)
                    try:
                        if tick_milliseconds > 0:
                            tick_time = datetime.fromtimestamp(tick_milliseconds / 1000, UTC)
                        elif tick_seconds > 0:
                            tick_time = datetime.fromtimestamp(tick_seconds, UTC)
                    except (OSError, OverflowError, ValueError) as exc:
                        tick_timestamp_error = f"Provider tick timestamp could not be decoded: {exc}"
                candles_by_time: dict[int, Candle] = {}
                duplicates = 0
                has_real_volume = False
                for row in rates:
                    timestamp = int(row["time"])
                    real_volume = float(row["real_volume"]) if "real_volume" in row.dtype.names else 0.0
                    has_real_volume = has_real_volume or real_volume > 0
                    candle = Candle(
                        time=datetime.fromtimestamp(timestamp, UTC),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(real_volume if real_volume > 0 else row["tick_volume"]),
                        spread=float(row["spread"]) if "spread" in row.dtype.names else None,
                        real_volume=real_volume if real_volume > 0 else None,
                    )
                    if timestamp in candles_by_time:
                        duplicates += 1
                    candles_by_time[timestamp] = candle
                candles = [candles_by_time[key] for key in sorted(candles_by_time)]
                interval = TIMEFRAME_SECONDS.get(normalized)
                missing = 0
                if interval:
                    for left, right in zip(candles, candles[1:]):
                        delta = (right.time - left.time).total_seconds()
                        # Market closures are not missing candles. Count only short in-session gaps.
                        if interval * 1.5 < delta <= interval * 6:
                            missing += max(0, round(delta / interval) - 1)
                now = datetime.now(UTC)
                latest_bar_age_seconds = (now - candles[-1].time).total_seconds()
                stale_candle = bool(interval and latest_bar_age_seconds > max(interval * 4, 300))
                future_bar_offsets = [
                    (candle.time - now).total_seconds()
                    for candle in candles
                    if candle.time > now
                ]
                future_bar_count = len(future_bar_offsets)
                tick_age_seconds = (now - tick_time).total_seconds() if tick_time is not None else None
                future_tick = bool(tick_age_seconds is not None and tick_age_seconds < 0)
                tick_stale = bool(
                    tick_age_seconds is not None
                    and tick_age_seconds > self.tick_freshness_seconds
                )
                quote_present = bid is not None or ask is not None
                quote_timestamp_verified = bool(
                    not quote_present
                    or (
                        tick_time is not None
                        and not future_tick
                        and not tick_stale
                        and tick_timestamp_error is None
                    )
                )
                tick_future_offset = -tick_age_seconds if future_tick and tick_age_seconds is not None else 0.0
                max_future_offset_seconds = max([0.0, tick_future_offset, *future_bar_offsets])
                stale = bool(
                    stale_candle
                    or future_bar_count
                    or future_tick
                    or (quote_present and not quote_timestamp_verified)
                )
                feed_parts = [str(getattr(account, "company", "") or ""), str(getattr(account, "server", "") or "")]
                feed = " / ".join(part for part in feed_parts if part) or str(getattr(terminal, "company", "") or "MetaTrader 5")
                observations: list[str] = []
                if resolved_symbol.upper() != symbol.upper():
                    observations.append(f"Requested symbol {symbol} was mapped to broker symbol {resolved_symbol}.")
                if missing:
                    observations.append(f"Detected {missing} short in-session candle gap(s).")
                if future_bar_count:
                    observations.append(
                        f"Provider returned {future_bar_count} candle timestamp(s) up to "
                        f"{max(future_bar_offsets):.3f}s in the future. Values and timestamps were preserved, "
                        "but the data is unverified until provider clock skew is resolved."
                    )
                if future_tick:
                    observations.append(
                        f"Provider tick timestamp is {tick_future_offset:.3f}s in the future. "
                        "The quote was preserved unchanged but is not verified."
                    )
                if tick_stale and tick_age_seconds is not None:
                    observations.append(
                        f"Provider tick is {tick_age_seconds:.3f}s old, exceeding the "
                        f"{self.tick_freshness_seconds}s quote-freshness threshold."
                    )
                if tick_timestamp_error:
                    observations.append(tick_timestamp_error)
                elif quote_present and tick_time is None:
                    observations.append("Provider returned bid/ask values without a verifiable tick timestamp.")
                if stale_candle:
                    observations.append("The newest candle is older than the configured freshness threshold.")
                return MarketDataBatch(
                    provider=self.provider_id,
                    requested_symbol=symbol,
                    resolved_symbol=resolved_symbol,
                    timeframe=normalized,
                    candles=candles,
                    precision=int(getattr(info, "digits", 5) or 5),
                    point=float(getattr(info, "point", 0.00001) or 0.00001),
                    fetched_at=now,
                    bid=bid,
                    ask=ask,
                    tick_time=tick_time,
                    tick_age_seconds=round(tick_age_seconds, 3) if tick_age_seconds is not None else None,
                    tick_stale=tick_stale,
                    future_tick=future_tick,
                    future_bar_count=future_bar_count,
                    latest_bar_age_seconds=round(latest_bar_age_seconds, 3),
                    max_future_offset_seconds=round(max_future_offset_seconds, 3),
                    quote_timestamp_verified=quote_timestamp_verified,
                    feed=feed,
                    stale=stale,
                    missing_bars=missing,
                    duplicate_bars=duplicates,
                    volume_kind="real" if has_real_volume else "tick",
                    observations=observations,
                )
            finally:
                mt5.shutdown()

    def capabilities(self, symbol: str = "XAUUSD") -> dict[str, Any]:
        started = datetime.now(UTC)
        try:
            batch = self.fetch(symbol, "M1", 100)
        except Exception as exc:
            return {
                "provider": self.provider_id,
                "state": CapabilityState.UNAVAILABLE.value,
                "error": str(exc),
                "capabilities": [
                    capability("OHLC", CapabilityState.UNAVAILABLE, self.provider_id, str(exc)),
                    capability("Volume", CapabilityState.UNAVAILABLE, self.provider_id, str(exc)),
                ],
            }
        has_tick = batch.bid is not None and batch.ask is not None
        volume_state = CapabilityState.AVAILABLE if batch.volume_kind == "real" else CapabilityState.PARTIALLY_AVAILABLE
        volume_reason = None if batch.volume_kind == "real" else "Broker supplies tick volume, not centralized traded volume."
        unavailable_advanced = [
            "Aggressor Side", "True Delta", "CVD", "Footprint", "Level II", "Order Book",
            "Economic Calendar", "Sentiment", "Open Interest", "Options Data",
        ]
        items = [
            capability("OHLC", CapabilityState.AVAILABLE, self.provider_id),
            capability("Volume", volume_state, self.provider_id, volume_reason),
            capability("Bid", CapabilityState.AVAILABLE if has_tick else CapabilityState.UNAVAILABLE, self.provider_id),
            capability("Ask", CapabilityState.AVAILABLE if has_tick else CapabilityState.UNAVAILABLE, self.provider_id),
            capability("Ticks", CapabilityState.PARTIALLY_AVAILABLE, self.provider_id, "Tick history availability depends on the connected broker."),
            capability("Trade Prints", CapabilityState.UNAVAILABLE, self.provider_id, "No verified time-and-sales source is configured."),
        ]
        items.extend(capability(name, CapabilityState.UNAVAILABLE, self.provider_id, "No verified compatible source is configured.") for name in unavailable_advanced)
        return {
            "provider": self.provider_id,
            "state": CapabilityState.AVAILABLE.value,
            "symbol": batch.resolved_symbol,
            "feed": batch.feed,
            "checked_at": started.isoformat(),
            "capabilities": items,
        }


class MarketDataService:
    def __init__(self) -> None:
        self.providers = {"metatrader5": MetaTrader5Provider()}

    def health(self) -> dict[str, Any]:
        return {name: provider.health() for name, provider in self.providers.items()}

    def get_ohlcv(self, symbol: str, timeframe: str, count: int = 500, provider: str = "metatrader5") -> StandardResult:
        import time

        started_at = time.perf_counter()
        selected = self.providers.get(provider)
        if selected is None:
            return StandardResult.failure(
                f"Unknown market-data provider: {provider}",
                error_code="PROVIDER_NOT_FOUND",
                started_at=started_at,
            )
        try:
            batch = selected.fetch(symbol, timeframe, count)
            return StandardResult.success(
                {"metadata": batch.metadata(), "candles": [candle.as_dict() for candle in batch.candles]},
                verified=batch.data_quality_verified,
                started_at=started_at,
                observations=batch.observations,
            )
        except MarketDataError as exc:
            return StandardResult.failure(str(exc), error_code="MARKET_DATA_UNAVAILABLE", started_at=started_at)
        except Exception as exc:
            return StandardResult.failure(
                f"Market-data request failed: {exc}",
                status=ExecutionStatus.FAILED,
                error_code="PROVIDER_ERROR",
                started_at=started_at,
            )
