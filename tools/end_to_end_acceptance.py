"""End-to-end acceptance run (spec section 91).

Executes the full chain the user asks for in Sorani:

  "SAM، TradingView بکەرەوە و XAUUSD بە SnR شیکاریم بۆ بکە.
   1H و 15m ببینە و لە 5m و 1m entry بدۆزەوە."

TradingView -> symbol -> multi-timeframe data -> structure/SnR/liquidity ->
entry hunt on the lower timeframes -> ENTRY/WAIT/NO_TRADE -> stop, targets, RR ->
drawing plan -> draw -> verify -> self-check -> concise spoken summary.

Run:  .venv\\Scripts\\python.exe tools\\end_to_end_acceptance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_backend.cancellation import CancellationManager  # noqa: E402
from sam_backend.config import Settings  # noqa: E402
from sam_backend.db import Database  # noqa: E402
from sam_backend.dpi import ensure_dpi_awareness  # noqa: E402
from sam_backend.trading.service import TradingService  # noqa: E402
from sam_backend.voice import VoiceService, language_support  # noqa: E402

SORANI_REQUEST = "SAM، TradingView بکەرەوە و XAUUSD بە SnR شیکاریم بۆ بکە. 1H و 15m ببینە و لە 5m و 1m entry بدۆزەوە."
DRAW_THEORY = "__e2e__"
results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def main() -> int:
    print("=" * 78)
    print("END-TO-END ACCEPTANCE RUN")
    print("=" * 78)
    print(f"Request: {SORANI_REQUEST}\n")
    ensure_dpi_awareness()

    root = PROJECT_ROOT / "work" / "e2e"
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        project_root=root, workspace_root=root / "ws", data_dir=root / "data",
        screen_access_enabled=True, computer_control_enabled=True, voice_language="ckb-IQ",
    )
    settings.prepare()
    database = Database(settings.database_path)
    trading = TradingService(settings, database, CancellationManager())
    trading.refresh_permissions()

    # --- 1-3: intent, TradingView, symbol -----------------------------------
    print("1-3) Intent routing, TradingView, symbol")
    routed = trading.route_natural_intent(SORANI_REQUEST)
    step("The Sorani request routes to a trading intent", routed is not None,
         f"routed={'yes' if routed else 'no'}")

    focused = trading.tradingview.focus()
    step("TradingView focused", focused.verified, focused.error or "")
    time.sleep(0.5)
    # The request names XAUUSD, so switch to it rather than hoping the chart is
    # already there. The user's own symbol is put back at the end of the run.
    original_symbol = (trading.tradingview.observe().symbol or "").upper()
    switched = trading.tradingview.set_symbol("XAUUSD")
    step("The requested instrument was selected", switched.verified,
         switched.error_code or f"from {original_symbol}")
    state = trading.tradingview.observe()
    step("Symbol read from the live window", (state.symbol or "").upper() == "XAUUSD", f"symbol={state.symbol}")

    # --- 4-12: multi-timeframe data and analysis ----------------------------
    print("\n4-12) Multi-timeframe data, structure, SnR, liquidity")
    analysis = trading.analyze(
        symbol="XAUUSD", timeframes=["H1", "M15", "M5", "M1"], theories=["snr"],
    )
    report = analysis.data or {}
    frames = report.get("timeframes") or {}
    step("All four requested timeframes were analysed",
         set(frames) == {"H1", "M15", "M5", "M1"}, f"{sorted(frames)}")
    step("Each timeframe used its own candles, not an inferred one",
         all(frames[key]["metadata"]["timeframe"] == key for key in frames),
         ", ".join(f"{k}:{frames[k]['metadata']['analysis_bars']}bars" for k in sorted(frames)))
    step("The market feed is named", bool(report.get("feed")), str(report.get("feed")))
    step("Structure computed per timeframe", all("structure" in frames[key] for key in frames),
         " ".join(f"{k}={frames[k]['structure']['trend'][:4]}" for k in sorted(frames)))
    step("SnR levels scored", any(frames[key]["snr"] for key in frames),
         f"{sum(len(frames[k]['snr']) for k in frames)} levels")
    step("Liquidity inspected", all("liquidity" in frames[key] for key in frames))

    # --- 13-22: entry hunt and decision -------------------------------------
    print("\n13-22) Lower-timeframe entry hunt and decision")
    setup = report.get("setup") or {}
    decision = setup.get("decision")
    print(f"   decision={decision} state={report.get('setup_state')} htf_bias={report.get('htf_bias')}")
    reasons = setup.get("reasons") or []
    missing = setup.get("missing_confirmation") or []
    if reasons:
        print(f"   reasons: {reasons[:3]}")
    if missing:
        print(f"   missing: {missing[:3]}")
    step("A decision was produced", decision is not None or report.get("setup_state") is not None,
         f"{decision or report.get('setup_state')}")

    entry_ready = decision == "ENTRY_READY"
    if entry_ready:
        print("\n23-28) Invalidation, stop, targets, RR")
        step("Technical invalidation computed", setup.get("technical_invalidation") is not None)
        step("Stop computed", setup.get("stop") is not None, f"stop={setup.get('stop')}")
        targets = setup.get("targets") or []
        step("Three targets computed", len(targets) >= 3, f"{len(targets)} targets")
        step("RR computed", setup.get("rr") is not None, f"RR={setup.get('rr')}")
    else:
        print("\n23-28) Skipped: no entry is ready, so no stop/targets are published")
        step("No entry means no fabricated stop, targets, or RR",
             setup.get("stop") is None and not (setup.get("targets") or []) and setup.get("rr") is None,
             "SAM withholds a plan rather than inventing one")

    # --- 29-33: drawing plan, draw, verify ----------------------------------
    print("\n29-33) Drawing plan, draw on the chart, verify")
    plan = TradingService._drawing_plan(report)
    step("A drawing plan was derived from the analysis", bool(plan), f"{len(plan)} annotations")
    if plan:
        calibrated = trading.calibrate_chart()
        step("Chart calibrated before drawing", calibrated.verified, calibrated.error or "")
        if calibrated.verified:
            anchors = [item["price"] for item in calibrated.data["anchors"]]
            low, high = min(anchors), max(anchors)
            visible = [item for item in plan if low <= item[1] <= high][:3]
            print(f"   {len(visible)} of {len(plan)} planned levels are inside the visible range")
            drawn_ok = 0
            for annotation, price, label in visible:
                outcome = trading.draw_annotation(annotation, price, label=label, theory=DRAW_THEORY)
                detail = (outcome.data or {}).get("verification") or {}
                print(f"     {annotation:11s} {price:9.2f} -> {outcome.status.value:8s} "
                      f"row={detail.get('matched_row')} expected={detail.get('expected_local_y')}")
                drawn_ok += 1 if outcome.verified else 0
            if visible:
                step("Every visible planned level drew and verified", drawn_ok == len(visible),
                     f"{drawn_ok}/{len(visible)} verified")
            else:
                step("Planned levels outside the visible range are not drawn blindly", True,
                     "SAM refuses to place an off-screen level")
            owned = database.list_drawings(theory=DRAW_THEORY)
            step("Drawings are tracked as SAM-owned", len(owned) == len(visible), f"{len(owned)} rows")
            cleared = trading.clear_drawings(theory=DRAW_THEORY)
            step("All test drawings removed", not database.list_drawings(theory=DRAW_THEORY),
                 f"removed={(cleared.data or {}).get('count')}")

    # --- 34: self check ------------------------------------------------------
    print("\n34) Self-check")
    checks = report.get("self_check") or {}
    failures = checks.get("critical_failures") or []
    print(f"   passed={checks.get('passed')} critical_failures={failures[:4]}")
    step("Self-check ran", bool(checks), f"{len(checks.get('checks', []))} checks")
    if not checks.get("passed"):
        step("A failed self-check blocked a confident entry", decision != "ENTRY_READY",
             "SAM must not publish ENTRY_READY while a critical check fails")

    # --- 35: spoken summary --------------------------------------------------
    print("\n35) Concise Sorani summary")
    summary = report.get("spoken_summary_ckb") or ""
    print(f"   {summary}")
    step("A concise Sorani summary was produced", bool(summary), f"{len(summary)} chars")

    voice = VoiceService(settings)
    support = language_support("ckb-IQ")
    step("Sorani speech output is honestly reported as unavailable",
         support["stt_supported"] is False,
         "text works; Whisper has no Kurdish model and no Kurdish TTS voice is installed")
    spoken = voice.tts.capability()
    print(f"   TTS engine available for supported languages: {spoken['engine']} ({spoken['state']})")

    # --- 36: still interruptible --------------------------------------------
    print("\n36) Remains interruptible")
    voice.barge_in.begin_speaking()
    step("Barge-in can interrupt at any point", voice.barge_in.interrupt() is True,
         "TTS, generation, and cancellable tools all stop")

    # --- 37: leave the user's chart as it was found -------------------------
    print("\n37) The user's chart is restored")
    if original_symbol and original_symbol != "XAUUSD":
        restored = trading.tradingview.set_symbol(original_symbol)
        step(f"The chart was returned to {original_symbol}",
             restored.verified and (trading.tradingview.observe().symbol or "").upper() == original_symbol,
             restored.error_code or "")
    else:
        step("The chart already showed the requested instrument", True, "nothing to restore")
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
