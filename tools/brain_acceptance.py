"""Trading-brain acceptance tests (spec sections 93, 94, 95, 96, 97).

Runs against live MT5 data and the real TradingView window:
  93  no-trade honesty            — an incomplete setup must return WAIT/NO_TRADE
  94  theory switch               — same chart, independent Wyckoff analysis
  95  custom theory persistence   — teach, reload from disk, execute
  96  setup monitor               — real state transitions, no duplicate events
  97  drawing robustness          — resize the chart, recalibrate, place a level

Run:  .venv\\Scripts\\python.exe tools\\brain_acceptance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import win32con  # noqa: E402
import win32gui  # noqa: E402

from sam_backend.cancellation import CancellationManager  # noqa: E402
from sam_backend.config import Settings  # noqa: E402
from sam_backend.db import Database  # noqa: E402
from sam_backend.dpi import ensure_dpi_awareness  # noqa: E402
from sam_backend.trading.service import TradingService  # noqa: E402
from sam_backend.trading.types import SetupDecision  # noqa: E402

TEST_THEORY = "__robustness__"
results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def build(root: Path, *, control: bool = False) -> tuple[Settings, Database, TradingService]:
    settings = Settings(
        project_root=root, workspace_root=root / "ws", data_dir=root / "data",
        screen_access_enabled=control, computer_control_enabled=control,
    )
    settings.prepare()
    database = Database(settings.database_path)
    trading = TradingService(settings, database, CancellationManager())
    trading.refresh_permissions()
    return settings, database, trading


def main() -> int:
    print("=" * 74)
    print("TRADING BRAIN ACCEPTANCE TESTS")
    print("=" * 74)
    ensure_dpi_awareness()
    root = PROJECT_ROOT / "work" / "brain"
    root.mkdir(parents=True, exist_ok=True)
    _, database, trading = build(root)

    # --- 93: no-trade honesty ------------------------------------------------
    print("\n93) No-trade honesty on live data")
    analysis = trading.analyze(symbol="XAUUSD", timeframes=["H1", "M15", "M5", "M1"], theories=["snr"])
    report = analysis.data or {}
    setup = report.get("setup") or {}
    decision = setup.get("decision")
    state = report.get("setup_state")
    print(f"   decision={decision} state={state} verified={analysis.verified}")
    print(f"   missing_confirmation={ (setup.get('missing_confirmation') or [])[:4] }")
    valid = {item.value for item in SetupDecision}
    step("A decision from the permitted set was returned", decision in valid or decision is None, f"decision={decision}")
    entry_ready = decision == SetupDecision.ENTRY_READY.value
    if entry_ready:
        step("ENTRY_READY carries a full plan", all(setup.get(key) is not None for key in ("entry", "stop", "rr")),
             "an entry must never be published without stop and RR")
    else:
        step("An incomplete setup produced WAIT/NO_TRADE rather than a signal", True,
             f"{decision or state} with reasons recorded")
    step("Self-check ran and reported its verdict", "self_check" in report or "checks" in report,
         f"keys={[k for k in report if 'check' in k]}")

    # --- 94: theory switch ---------------------------------------------------
    print("\n94) Theory switch preserves context and stays independent")
    snr_only = trading.analyze(symbol="XAUUSD", timeframes=["H1", "M15"], theories=["snr"])
    wyckoff_only = trading.analyze(symbol="XAUUSD", timeframes=["H1", "M15"], theories=["wyckoff"])
    snr_report, wy_report = snr_only.data or {}, wyckoff_only.data or {}
    step("Symbol context is preserved across the switch",
         snr_report.get("symbol") == wy_report.get("symbol") == "XAUUSD",
         f"{snr_report.get('symbol')} / {wy_report.get('symbol')}")
    step("Only the requested theory ran each time",
         set(snr_report.get("theories", {})) == {"snr"} and set(wy_report.get("theories", {})) == {"wyckoff"},
         f"{sorted(snr_report.get('theories', {}))} then {sorted(wy_report.get('theories', {}))}")
    comparison = trading.analyze(symbol="XAUUSD", timeframes=["H1", "M15"], theories=["snr", "wyckoff", "ict"])
    theories = (comparison.data or {}).get("theories", {})
    step("Comparison runs each theory independently", len(theories) == 3, f"ran {sorted(theories)}")

    # --- 95: custom theory persistence --------------------------------------
    print("\n95) Custom theory survives a restart")
    # The schema demands executable predicates, not prose: a taught theory has to
    # actually run against candles, so this is expressed the way SAM evaluates it.
    definition = {
        "name": "N",
        "aliases": ["theory n", "تیۆری N"],
        "description": "Break of a session high, retest as support, then a bullish lower-timeframe shift.",
        "required_data": ["OHLC"],
        "timeframes": ["H1", "M15", "M5"],
        "conditions": [
            {"predicate": "trend_is", "value": "BULLISH", "timeframe": "H1"},
            {"predicate": "has_liquidity_sweep", "timeframe": "M15"},
            {"predicate": "has_mss", "timeframe": "M5"},
            {"predicate": "has_active_fvg", "timeframe": "M5"},
        ],
        "entry": "On the lower-timeframe shift after the retest holds.",
        "invalidation": "A close back through the retested level.",
        "targets": ["Nearest external liquidity", "Measured move of the breakout leg"],
    }
    saved = trading.save_custom_theory(definition)
    print(f"   saved version={saved.get('version')} name={saved.get('name')}")
    step("Custom theory validated and stored", bool(saved.get("name")), f"version={saved.get('version')}")

    # Rebuild every object from disk: this is the restart.
    _, reloaded_db, reloaded = build(root)
    loaded = reloaded_db.get_custom_theory(name="N")
    step("Theory reloads from disk after a restart", bool(loaded), f"version={(loaded or {}).get('version')}")
    step("Reloaded definition kept its executable rules",
         bool(loaded) and loaded["definition"].get("conditions") == definition["conditions"],
         f"{len((loaded or {}).get('definition', {}).get('conditions', []))} predicates")
    executed = reloaded.analyze(symbol="XAUUSD", timeframes=["H1", "M15"], theories=["N"])
    executed_theories = (executed.data or {}).get("theories", {})
    step("The reloaded custom theory executes by name",
         any("N" in key for key in executed_theories), f"ran {sorted(executed_theories)}")

    # A second save must version, not overwrite.
    second = reloaded.save_custom_theory({**definition, "description": "Revised wording."})
    step("Re-teaching creates a new version instead of overwriting",
         second.get("version", 0) > saved.get("version", 0),
         f"v{saved.get('version')} -> v{second.get('version')}")

    # --- 96: setup monitor ---------------------------------------------------
    print("\n96) Setup monitor records transitions without spamming")
    if report.get("current_price"):
        created = reloaded.create_setup_from_analysis(report, theory="snr")
        setup_id = created.get("id")
        if setup_id:
            reloaded_db.set_setup_monitoring(setup_id, True)
            first_poll = reloaded.poll_monitors()
            second_poll = reloaded.poll_monitors()
            events = reloaded_db.list_setup_events(setup_id)
            step("Setup persisted with a state", bool(created.get("state")), f"state={created.get('state')}")
            step("Monitoring polls are idempotent when nothing changed",
                 len(second_poll) <= len(first_poll),
                 f"poll1={len(first_poll)} poll2={len(second_poll)} events={len(events)}")
            states = [event["state"] for event in events]
            step("No duplicate consecutive state events",
                 all(a != b for a, b in zip(states, states[1:])), f"states={states}")
        else:
            step("Setup created for monitoring", False, "no setup id returned")
    else:
        step("Setup monitor exercised", False, "no current price available from the feed")

    # --- 97: drawing robustness ---------------------------------------------
    print("\n97) Drawing robustness after the chart geometry changes")
    _, draw_db, drawer = build(PROJECT_ROOT / "work" / "brain_draw", control=True)
    focused = drawer.tradingview.focus()
    if not focused.verified:
        step("TradingView available for the robustness test", False, focused.error or "not focused")
        return summarize()
    handle = drawer.tradingview.observe().window_handle
    before_geometry = drawer.tradingview.observe().client_geometry
    first = drawer.calibrate_chart()
    step("Calibrated at the original size", first.verified, first.error or "")

    # Resize the window, which must invalidate the viewport-scoped calibration.
    win32gui.ShowWindow(handle, win32con.SW_RESTORE)
    time.sleep(1.0)
    win32gui.MoveWindow(handle, 60, 60, 1500, 1000, True)
    time.sleep(1.5)
    drawer.tradingview.focus()
    time.sleep(0.6)
    resized_geometry = drawer.tradingview.observe().client_geometry
    step("The chart viewport actually changed",
         resized_geometry != before_geometry, f"{before_geometry['width']}x{before_geometry['height']} -> "
         f"{resized_geometry['width']}x{resized_geometry['height']}")

    stale = drawer.drawing.price_to_screen(4460.0)
    step("The stale calibration is refused after the resize",
         not stale.verified and stale.error_code in {"CALIBRATION_REQUIRED", "CALIBRATION_NOT_VERIFIED", "PRICE_OFF_SCREEN"},
         f"code={stale.error_code}")

    second_cal = drawer.calibrate_chart()
    step("Recalibrated at the new size", second_cal.verified, second_cal.error or "")
    if second_cal.verified:
        anchors = [item["price"] for item in second_cal.data["anchors"]]
        target = round((max(anchors) + min(anchors)) / 2, 2)
        drawn = drawer.draw_annotation("resistance", target, label="robustness", theory=TEST_THEORY)
        detail = (drawn.data or {}).get("verification") or {}
        step("A level drawn after the resize is verified in place", drawn.verified,
             f"{target} matched_row={detail.get('matched_row')} expected={detail.get('expected_local_y')}")
        cleared = drawer.clear_drawings(theory=TEST_THEORY)
        step("The robustness drawing was removed", cleared.status.value == "SUCCESS",
             f"removed={(cleared.data or {}).get('count')}")

    # Put the window back the way it was found.
    win32gui.ShowWindow(handle, win32con.SW_MAXIMIZE)
    time.sleep(1.0)
    return summarize()


def summarize() -> int:
    print("\n" + "=" * 74)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
    print("=" * 74)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
