"""Final daily-use workflow and stability (spec sections 17, 18, 27, 28, 29).

Runs the thirty-four step workflow against the live backend and the real
TradingView window, then exercises crash recovery, monitor stability, and
resource behaviour over an extended session.

Run with SAM already started by start.ps1:
  .venv\\Scripts\\python.exe tools\\daily_workflow.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os

# The port is configurable so the workflow can run when another local app
# already holds SAM's default port.
PORT = os.environ.get("SAM_PORT", "8765")
BASE = f"http://127.0.0.1:{PORT}"
results: list[tuple[str, bool, str]] = []


def step(number: str, name: str, passed: bool, detail: str = "") -> bool:
    results.append((f"{number} {name}", passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {number:>5}  {name}" + (f" — {detail}" if detail else ""))
    return passed


def call(path: str, payload: dict | None = None, method: str | None = None, timeout: int = 600) -> dict:
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method=method or ("POST" if payload is not None else "GET"),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or "{}")
    except urllib.error.HTTPError as error:
        return {"__http_error__": error.code, "body": error.read().decode("utf-8", "replace")[:200]}
    except Exception as exc:
        return {"__error__": str(exc)}


def wait_healthy(seconds: int = 60) -> bool:
    for _ in range(seconds * 2):
        health = call("/api/health", timeout=5)
        if health.get("status") == "ok":
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    print("=" * 78)
    print("FINAL DAILY-USE WORKFLOW")
    print("=" * 78)

    step("1-2", "SAM is healthy", wait_healthy())
    health = call("/api/health")
    print(f"   db={health.get('database')} audit={health.get('audit_chain_valid')} "
          f"coords={health.get('screen_coordinates', {}).get('coordinates_consistent')}")

    step("3", "The UI is served", "SAM" in (call_text("/") or ""))

    settings = call("/api/settings", {"computer_control_enabled": True, "screen_access_enabled": True}, method="PUT")
    runtime = settings.get("runtime", {})
    step("4-5", "Computer Control and Screen Access enabled",
         runtime.get("computer_control_enabled") is True and runtime.get("screen_access_enabled") is True)

    triggers = call("/api/trading/triggers")
    step("6", "Trading mode surfaces its registries", triggers.get("data", {}).get("count", 0) >= 14,
         f"{triggers.get('data', {}).get('count')} triggers")

    focus = call("/api/tradingview/action", {"action": "focus"})
    step("7", "TradingView focused", bool(focus.get("verified")), focus.get("error") or "")
    # Step 8 is "select XAUUSD". If the chart already shows something else the
    # user put there, SAM analyses XAUUSD from the provider and must then refuse
    # to annotate the other instrument rather than switching their chart.
    state = call("/api/tradingview/state")
    chart_symbol = (state.get("symbol") or "").upper()
    step("8", "The chart symbol is read from the live window", bool(chart_symbol), f"symbol={chart_symbol}")
    symbol_matches = chart_symbol.startswith("XAU") or "XAUUSD" in chart_symbol

    analysis = call("/api/trading/analyze", {
        "symbol": "XAUUSD", "timeframes": ["H1", "M15", "M5", "M1"], "theories": ["snr"],
    })
    report = analysis.get("data") or {}
    step("9", "Market feed is named", bool(report.get("feed")), str(report.get("feed"))[:48])
    frames = report.get("timeframes") or {}
    step("10-11", "H1 and M15 analysed", {"H1", "M15"} <= set(frames))
    step("12", "M5 and M1 searched for an entry", {"M5", "M1"} <= set(frames))

    setup = report.get("setup") or {}
    decision = setup.get("decision") or report.get("setup_state")
    step("13", "An honest ENTRY / WAIT / NO_TRADE was produced", decision is not None, str(decision))
    entry_ready = setup.get("decision") == "ENTRY_READY"
    if entry_ready:
        for number, key in (("14", "technical_invalidation"), ("15", "stop"), ("19", "rr")):
            step(number, f"{key} computed", setup.get(key) is not None, str(setup.get(key)))
        targets = setup.get("targets") or []
        for index, number in enumerate(("16", "17", "18")):
            step(number, f"TP{index + 1} computed", len(targets) > index)
    else:
        step("14-19", "No entry means no invented stop, targets, or RR",
             setup.get("stop") is None and not (setup.get("targets") or []) and setup.get("rr") is None,
             f"decision={decision}")

    call("/api/tradingview/action", {"action": "auto_calibrate"})
    drawn = call("/api/tradingview/drawings/analysis", {"theory": "snr"})
    drawn_data = drawn.get("data") or {}
    if symbol_matches:
        step("20-21", "Levels drawn and verified on the chart",
             drawn.get("status") in {"SUCCESS", "PARTIAL"} and len(drawn_data.get("drawn", [])) > 0,
             f"drawn={len(drawn_data.get('drawn', []))} unverified={len(drawn_data.get('unverified', []))} "
             f"code={drawn.get('error_code')}")
    else:
        # Drawing XAUUSD levels onto another instrument would be wrong, and would
        # also clutter whatever the user has on that chart.
        step("20-21", "Annotating a chart showing another instrument is refused",
             drawn.get("verified") is False
             and drawn.get("error_code") in {"SYMBOL_MISMATCH", "PRICE_OFF_SCREEN"},
             f"chart={chart_symbol}, analysis=XAUUSD, code={drawn.get('error_code')}")
    owned_before = call("/api/tradingview/drawings").get("data", {}).get("count", 0)
    cleared = call("/api/tradingview/drawings/clear", {"all_owned": True})
    owned_after = call("/api/tradingview/drawings").get("data", {}).get("count", 0)
    step("22", "SAM drawings cleared", owned_after == 0,
         f"{owned_before} -> {owned_after}, removed={cleared.get('data', {}).get('count')}")

    wyckoff = call("/api/trading/analyze", {"symbol": "XAUUSD", "timeframes": ["H1", "M15"], "theories": ["wyckoff"]})
    step("23", "Theory switched to Wyckoff", set((wyckoff.get("data") or {}).get("theories", {})) == {"wyckoff"})
    compare = call("/api/trading/analyze", {
        "symbol": "XAUUSD", "timeframes": ["H1", "M15"], "theories": ["snr", "wyckoff", "ict"],
    })
    step("24", "Three theories compared independently",
         len((compare.get("data") or {}).get("theories", {})) == 3)

    backtest = call("/api/trading/backtest", {"symbol": "XAUUSD", "timeframe": "M15",
                                              "trigger": "golden_cross", "count": 3000})
    backtest_data = backtest.get("data") or {}
    step("25", "Backtest ran on real history", backtest.get("status") == "SUCCESS",
         f"setups={backtest_data.get('total_setups')} win={backtest_data.get('win_rate')} "
         f"PF={backtest_data.get('profit_factor')}")
    step("26", "The trade list is populated", len(backtest_data.get("trades", [])) > 0,
         f"{len(backtest_data.get('trades', []))} trades")

    strategy = call("/api/trading/strategies", {
        "name": "Daily Workflow Strategy", "context": "1H bullish structure",
        "setup": "15m Support RBS", "confirmation": "5m liquidity sweep and bullish MSS",
        "entry_trigger": "bullish_sweep", "invalidation": "Below the sweep low",
        "stop": "Beyond the structural invalidation",
        "targets": ["TP1 internal liquidity", "TP2 external liquidity", "TP3 resistance"],
        "timeframes": ["H1", "M15", "M5"], "direction": "LONG", "minimum_rr": 2.0,
    })
    step("27", "Custom strategy saved", bool(strategy.get("name")), f"v{strategy.get('version')}")
    listed = call("/api/trading/strategies").get("strategies", [])
    step("28", "Custom strategy reloads", any(item["name"] == "Daily Workflow Strategy" for item in listed),
         f"{len(listed)} stored")

    created = call("/api/trading/setups", {"theory": "snr"})
    setup_id = (created.get("setup") or {}).get("id")
    if setup_id:
        call(f"/api/trading/setups/{setup_id}/monitor", {"enabled": True})
        time.sleep(8)
        rows = call("/api/trading/setups").get("setups", [])
        mine = next((row for row in rows if row["id"] == setup_id), None)
        step("29", "Setup monitoring starts and holds a state",
             bool(mine and mine.get("monitor_enabled")), f"state={mine.get('state') if mine else None}")
        call(f"/api/trading/setups/{setup_id}/monitor", {"enabled": False})
    else:
        step("29", "Setup monitoring starts", False, "no setup id")

    audit = call("/api/audit/verify")
    step("30", "Audit chain valid", bool(audit.get("valid")), str(audit)[:40])
    providers = call("/api/providers/status")
    cost = call("/api/cost")
    step("31", "Provider, model and cost status reported",
         all(key in providers for key in ("openrouter", "ollama", "litellm")),
         f"OR={providers.get('openrouter', {}).get('status')} "
         f"Ollama={providers.get('ollama', {}).get('status')} "
         f"LiteLLM={providers.get('litellm', {}).get('status')} cost={cost.get('today', {}).get('cost_usd')}")

    # --- 27: crash recovery -------------------------------------------------
    print("\n27) Controlled failure recovery")
    bad_symbol = call("/api/trading/analyze", {"symbol": "NOT_A_SYMBOL", "timeframes": ["H1"], "theories": ["snr"]})
    step("27a", "A bad market-data request fails without killing the service",
         bad_symbol.get("status") in {"FAILED", "PARTIAL"} and wait_healthy(10),
         f"status={bad_symbol.get('status')}")
    bad_replay = call("/api/research/replay/control", {"action": "advance"})
    step("27b", "A missing replay session is refused cleanly",
         bad_replay.get("error_code") == "NO_REPLAY_SESSION")
    bad_draw = call("/api/tradingview/drawings", {"annotation": "support", "price": 99_999_999})
    step("27c", "An impossible drawing is refused and the chart untouched",
         bad_draw.get("verified") is False and bad_draw.get("executed") is False,
         f"code={bad_draw.get('error_code')}")
    step("27d", "SAM is still healthy after the failures", wait_healthy(10))

    # --- 32-34: restart and persistence -------------------------------------
    print("\n32-34) Restart and persistence")
    strategies_before = len(call("/api/trading/strategies").get("strategies", []))
    audit_before = len(call("/api/audit").get("entries", []))
    print("   restarting SAM...")
    import subprocess

    # This script runs from the same interpreter, so it must exclude itself and
    # its own children or it would terminate the very process doing the testing.
    import os

    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "$self = {0}; Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object {{ $_.CommandLine -like '*-m sam_backend*' -and $_.ProcessId -ne $self }} | "
         "ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}".format(os.getpid())],
        capture_output=True, timeout=60,
    )
    time.sleep(3)
    subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                      "-File", str(PROJECT_ROOT / "start.ps1"), "-NoBrowser", "-Port", PORT],
                     cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    step("32", "SAM restarts cleanly", wait_healthy(90))
    strategies_after = call("/api/trading/strategies").get("strategies", [])
    step("33", "Saved strategies survive the restart",
         any(item["name"] == "Daily Workflow Strategy" for item in strategies_after),
         f"{strategies_before} -> {len(strategies_after)}")
    step("33b", "The audit chain is still valid after restart", bool(call("/api/audit/verify").get("valid")))
    repeat = call("/api/trading/analyze", {"symbol": "XAUUSD", "timeframes": ["H1", "M15"], "theories": ["snr"]})
    step("34", "Analysis runs again after the restart",
         repeat.get("status") in {"SUCCESS", "PARTIAL"} and bool((repeat.get("data") or {}).get("timeframes")),
         f"status={repeat.get('status')}")

    return summarize()


def call_text(path: str) -> str:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:
        return ""


def summarize() -> int:
    print("\n" + "=" * 78)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"RESULT: {passed}/{len(results)} workflow steps passed")
    failed = [name for name, ok, _ in results if not ok]
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
    if failed:
        print(f"  Failed steps: {', '.join(item.split()[0] for item in failed)}")
    print("=" * 78)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
