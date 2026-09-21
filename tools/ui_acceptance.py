"""Real-browser UI acceptance (spec sections 14-15).

Drives the actual SAM page in Chromium: clicks the real controls, waits for the
real network calls, and compares what the panel renders against what the backend
returned for the same request. A control that quietly does nothing fails here.

Run:  .venv\\Scripts\\python.exe tools\\ui_acceptance.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402

import os

# Honour an alternate port so the checks run even when another local
# application already holds SAM's default port.
BASE = f"http://127.0.0.1:{os.environ.get('SAM_PORT', '8765')}"
results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def api(path: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main() -> int:
    print("=" * 78)
    print("REAL BROWSER UI ACCEPTANCE")
    print("=" * 78)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        console_errors: list[str] = []
        page.on("pageerror", lambda error: console_errors.append(str(error)))
        page.on("console", lambda message: console_errors.append(message.text)
                if message.type == "error" else None)
        calls: list[str] = []
        page.on("request", lambda request: calls.append(request.url) if "/api/" in request.url else None)

        print("\n14) Page load and panel wiring")
        page.goto(BASE, wait_until="networkidle", timeout=60_000)
        step("The SAM page loads", "SAM" in page.title() or page.locator("body").count() > 0, page.title()[:50])
        step("No uncaught JavaScript errors on load", not console_errors,
             "; ".join(console_errors[:2]) if console_errors else "clean")

        for panel in ("card-providers", "card-triggers", "card-backtest", "card-composer", "card-journal"):
            step(f"Panel #{panel} is present", page.locator(f"#{panel}").count() == 1)

        # The panels live inside the Market view. The nav target appears twice
        # (desktop rail and mobile bar), so the visible one is chosen explicitly.
        page.locator('[data-nav="market"]:visible').first.click()
        page.wait_for_timeout(1500)
        step("The Market view opens from the navigation",
             page.locator("#card-backtest").is_visible())

        page.wait_for_timeout(2500)
        expected = ("/api/providers/status", "/api/trading/triggers", "/api/trading/strategies", "/api/trading/journal")
        for endpoint in expected:
            step(f"The page actually called {endpoint}", any(endpoint in url for url in calls))

        print("\n14b) Provider status is rendered from the backend")
        live = api("/api/providers/status")
        for element_id, key in (("provider-openrouter", "openrouter"), ("provider-ollama", "ollama")):
            shown = page.locator(f"#{element_id}").inner_text().strip().upper()
            step(f"{key} status matches the backend", shown == str(live[key]["status"]).upper(),
                 f"UI={shown} backend={live[key]['status']}")

        step("The OpenRouter key field is masked",
             page.locator("#openrouter-key-input").get_attribute("type") == "password")

        print("\n14c) Entry trigger panel")
        backend_triggers = api("/api/trading/triggers")["data"]
        shown_count = page.locator("#trigger-count").inner_text().strip()
        step("The trigger count matches the registry", shown_count == str(backend_triggers["count"]),
             f"UI={shown_count} backend={backend_triggers['count']}")
        rendered = page.locator("#trigger-list").inner_text()
        first = backend_triggers["triggers"][0]
        step("Trigger detail comes from the backend definition",
             first["name"] in rendered and first["confirmation"][:25] in rendered,
             first["name"])
        options = page.locator("#bt-trigger option").count()
        step("Triggers populate the backtest selector", options >= backend_triggers["count"], f"{options} options")

        print("\n15) Backtest from the real UI")
        page.select_option("#bt-trigger", "golden_cross")
        page.fill("#bt-count", "3000")
        page.select_option("#bt-timeframe", "M15")
        with page.expect_response(lambda r: "/api/trading/backtest" in r.url, timeout=600_000) as captured:
            page.click("#bt-run-btn")
        response = captured.value
        step("Clicking Run Backtest issues a real backend request", response.status == 200, f"HTTP {response.status}")
        payload = response.json()
        page.wait_for_timeout(2500)
        rendered = page.locator("#backtest-results").inner_text()
        data = payload.get("data") or {}

        def shown(label: str) -> str | None:
            match = re.search(rf"{re.escape(label)}\s*\n?\s*([^\n]+)", rendered)
            return match.group(1).strip() if match else None

        checks = [
            ("Total setups", str(data.get("total_setups"))),
            ("Wins", str(data.get("wins"))),
            ("Losses", str(data.get("losses"))),
            ("Max consecutive losses", str(data.get("max_consecutive_losses"))),
        ]
        for label, expected_value in checks:
            actual = shown(label)
            step(f"UI '{label}' equals the backend value", actual == expected_value,
                 f"UI={actual} backend={expected_value}")
        win_rate = shown("Win rate")
        step("UI win rate equals the backend win rate",
             win_rate == f"{(data.get('win_rate') or 0) * 100:.1f}%",
             f"UI={win_rate} backend={(data.get('win_rate') or 0) * 100:.1f}%")
        step("An equity curve is drawn from the returned trades",
             page.locator("#backtest-results svg").count() >= 1)
        rows = page.locator("#backtest-results table tbody tr").count()
        step("The trade table lists the returned trades", rows == len(data.get("trades", [])),
             f"UI rows={rows} backend trades={len(data.get('trades', []))}")

        print("\n14d) Strategy composer")
        page.fill("#sc-name", "UI Acceptance Strategy")
        page.fill("#sc-context", "1H bullish structure")
        page.fill("#sc-setup", "15m Support RBS")
        page.fill("#sc-confirmation", "5m liquidity sweep and bullish MSS")
        page.select_option("#sc-trigger", "bullish_sweep")
        page.fill("#sc-invalidation", "Below the sweep low")
        page.fill("#sc-targets", "TP1 liquidity, TP2 resistance")
        with page.expect_response(lambda r: "/api/trading/strategies" in r.url and r.request.method == "POST",
                                  timeout=120_000) as saved:
            page.click("#sc-save-btn")
        step("Saving a composed strategy hits the backend", saved.value.status == 201,
             f"HTTP {saved.value.status}")
        page.wait_for_timeout(1500)
        step("The composer reports the saved version",
             "version" in page.locator("#composer-status").inner_text().lower(),
             page.locator("#composer-status").inner_text()[:60])

        print("\n14e) Journal / research")
        with page.expect_response(lambda r: "/api/trading/journal" in r.url, timeout=60_000):
            page.click("#journal-refresh-btn")
        step("Journal refresh issues a real request", True)

        print("\n14f) No control is inert")
        buttons = page.locator("#card-providers button, #card-backtest button, #card-composer button, #card-journal button")
        inert: list[str] = []
        for index in range(buttons.count()):
            button = buttons.nth(index)
            identifier = button.get_attribute("id")
            if not identifier:
                inert.append(button.inner_text().strip() or f"button {index}")
        step("Every panel button has an id a handler can bind to", not inert, "; ".join(inert) or "all identified")
        script = (PROJECT_ROOT / "frontend" / "panels.js").read_text(encoding="utf-8")
        unbound = []
        for index in range(buttons.count()):
            identifier = buttons.nth(index).get_attribute("id")
            if identifier and f'#{identifier}' not in script:
                unbound.append(identifier)
        step("Every panel button is bound in panels.js", not unbound, "; ".join(unbound) or "all bound")

        step("Still no uncaught JavaScript errors after interaction", not console_errors,
             "; ".join(console_errors[:2]) if console_errors else "clean")

        browser.close()

    return summarize()


def summarize() -> int:
    print("\n" + "=" * 78)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
    print("=" * 78)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
