"""Live: price alerts fire on a synthetic feed and on the live MT5 gold feed (read only).

    .venv\\Scripts\\python.exe acceptance\\engine_alerts_live.py [--seconds 90]

1. Synthetic feed (design acceptance 5): a price_cross alert at 2700 fires
   exactly once when the feed crosses it, and the text SAM would speak is
   Sorani (captured from the bus: no voice package is loaded, nothing is
   played).
2. Live: two alerts 0.05 above and below the current MT5 gold bid; the monitor
   polls the real feed until one fires (the other is cancelled). MT5 is read
   only; the alerts live in a throw-away SAM home under work/.

No keys, no model calls, no speakers.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

from _common import ROOT, Acceptance

from sam.app import App
from sam.events import Alert, SpeakRequest
from sam.textnorm import is_arabic_script
from sam.trading import tools as trading_tools
from sam.trading.monitor import Monitor


class SyntheticFeed:
    """MT5-feed stand-in: one price that the check moves."""

    def __init__(self, price: float, start: float) -> None:
        self.price = price
        self.now = start

    async def tick(self, symbol: str) -> dict:
        return {"symbol": "XAUUSD", "bid": self.price, "ask": self.price + 0.2, "last": self.price,
                "spread": 0.2, "time": self.now}

    async def bars(self, symbol: str, timeframe: str, count: int) -> list[dict]:
        step = 60
        end = self.now - self.now % step
        return [{"time": end - step * i, "open": self.price, "high": self.price + 0.1, "low": self.price - 0.1,
                 "close": self.price, "volume": 100} for i in range(min(count, 5))][::-1]

    async def connect(self) -> bool:
        return True

    async def status(self) -> dict:
        return {"connected": True}


def new_app(home: Path) -> tuple[App, list]:
    (home / "data").mkdir(parents=True, exist_ok=True)
    app = App(home, environ={})
    app.loop = asyncio.get_running_loop()
    app.bus.bind_loop(app.loop)
    trading_tools.register(app)
    events: list = []
    app.bus.subscribe((Alert, SpeakRequest), events.append)
    return app, events


async def run(seconds: float) -> int:
    acc = Acceptance("engine_alerts_live")
    base = ROOT / "work" / "acceptance-alerts-home"
    shutil.rmtree(base, ignore_errors=True)
    with acc.check("synthetic feed: a price cross fires once, spoken in Sorani") as c:
        app, events = new_app(base / "synthetic")
        try:
            t0 = time.time()
            feed = SyntheticFeed(2690.0, t0)
            app.trading.mt5 = feed
            monitor = Monitor(app, clock=lambda: feed.now)
            app.trading.monitor = monitor
            alert = monitor.add({"kind": "price_cross", "symbol": "gold", "level": 2700, "direction": "up"},
                                price=2690.0)
            assert await monitor.check_once(now=t0 + 2) == []
            feed.price, feed.now = 2701.0, t0 + 4
            fired = await monitor.check_once(now=t0 + 4)
            again = await monitor.check_once(now=t0 + 6)
            await asyncio.sleep(0.05)
            spoken = [e.text_ckb for e in events if isinstance(e, SpeakRequest)]
            c.data = {"fired": len(fired), "fired_again": len(again), "spoken": spoken,
                      "status": monitor.get(alert["id"])["status"]}
            assert len(fired) == 1 and not again and spoken and is_arabic_script(spoken[0]), c.data
            c.detail = spoken[0]
        finally:
            app.close()

    with acc.check(f"live MT5 gold: an alert 0.05 from the bid fires within {seconds:.0f} s") as c:
        app, events = new_app(base / "live")
        try:
            feed = app.trading.mt5
            if not await feed.connect():
                c.skip("MetaTrader 5 is not connected")
            tick = await feed.tick("XAUUSD")
            bid = float(tick["bid"])
            if time.time() - float(tick["time"]) > 300:
                c.skip(f"the gold feed is {time.time() - float(tick['time']):.0f} s old (market closed?)")
            monitor = Monitor(app)
            app.trading.monitor = monitor
            up = monitor.add({"kind": "price_cross", "symbol": "XAUUSD", "level": round(bid + 0.05, 2),
                              "direction": "up"}, price=bid)
            down = monitor.add({"kind": "price_cross", "symbol": "XAUUSD", "level": round(bid - 0.05, 2),
                                "direction": "down"}, price=bid)
            began = time.monotonic()
            fired: list = []
            while time.monotonic() - began < seconds and not fired:
                fired = await monitor.check_once()
                if not fired:
                    await asyncio.sleep(1.0)
            await asyncio.sleep(0.05)
            spoken = [e.text_ckb for e in events if isinstance(e, SpeakRequest)]
            for item in (up, down):
                if monitor.get(item["id"])["status"] == "active":
                    monitor.cancel(item["id"])
            c.data = {"bid": bid, "fired": [f.get("kind") for f in fired], "seconds": round(time.monotonic() - began, 1),
                      "spoken": spoken}
            assert fired and spoken and is_arabic_script(spoken[0]), c.data
            c.detail = f"fired after {c.data['seconds']} s: {spoken[0]}"
        finally:
            await trading_tools.stop(app)
            app.close()
    shutil.rmtree(base, ignore_errors=True)
    return acc.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=90.0)
    sys.exit(asyncio.run(run(parser.parse_args().seconds)))
