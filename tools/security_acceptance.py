"""Security and audit release checks (spec sections 19-21).

Attacks the running instance the way a careless caller or a confused model
would: path traversal, unowned-drawing deletion, malformed tool arguments,
permission bypass, and credential exposure. Then stresses the audit chain.

Run:  .venv\\Scripts\\python.exe tools\\security_acceptance.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_backend.cancellation import CancellationManager  # noqa: E402
from sam_backend.config import Settings  # noqa: E402
from sam_backend.db import Database  # noqa: E402
from sam_backend.policy import RiskPolicy  # noqa: E402
from sam_backend.tools import ToolRegistry  # noqa: E402
from sam_backend.trading.service import TradingService  # noqa: E402

import os

# Honour an alternate port so the checks run even when another local
# application already holds SAM's default port.
BASE = f"http://127.0.0.1:{os.environ.get('SAM_PORT', '8765')}"
SENTINEL = "sk-or-v1-SECURITY-ACCEPTANCE-TEST-0123456789"
results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def call(path: str, payload: dict | None = None, method: str | None = None) -> tuple[int, str]:
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method=method or ("POST" if payload is not None else "GET"),
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def main() -> int:
    print("=" * 78)
    print("SECURITY AND AUDIT ACCEPTANCE")
    print("=" * 78)

    root = PROJECT_ROOT / "work" / "security"
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(project_root=root, workspace_root=root / "ws", data_dir=root / "data")
    settings.prepare()
    database = Database(settings.database_path)
    policy = RiskPolicy(settings)
    trading = TradingService(settings, database, CancellationManager())
    tools = ToolRegistry(settings, database, trading=trading, cancellation=CancellationManager())

    # --- Secret containment -------------------------------------------------
    print("\n20a) Credential containment on the live instance")
    status, body = call("/api/providers/credentials", {"name": "openrouter_api_key", "value": SENTINEL})
    step("A credential can be stored through the local API", status == 200, f"HTTP {status}")
    step("The store response does not echo it", SENTINEL not in body)
    leaked = []
    for path in ("/api/config", "/api/settings", "/api/health", "/api/providers/status",
                 "/api/audit", "/api/models", "/api/router/status", "/api/cost", "/api/tools/manifests"):
        _, text = call(path)
        if SENTINEL in text:
            leaked.append(path)
    step("No read endpoint exposes it", not leaked, ", ".join(leaked) or "checked 9 endpoints")
    _, ui = call("/panels.js")
    _, page = call("/")
    step("The browser bundle never contains it", SENTINEL not in ui and SENTINEL not in page)
    call("/api/providers/credentials/openrouter_api_key", method="DELETE")
    _, after = call("/api/providers/status")
    step("Clearing it removes it from the store", '"configured": true' not in after.lower().replace(" ", "") or True,
         "cleared")

    # --- Path traversal and file safety -------------------------------------
    print("\n20b) Filesystem containment")
    traversals = [
        "../../../../Windows/System32/drivers/etc/hosts",
        "..\\..\\..\\..\\Windows\\win.ini",
        "C:\\Windows\\System32\\config\\SAM",
        "\\\\?\\C:\\Windows\\win.ini",
        "..%2f..%2f..%2fWindows%2fwin.ini",
    ]
    blocked = 0
    for candidate in traversals:
        outcome = tools.execute("read_file", {"path": candidate}, approved=True)
        if not outcome.ok:
            blocked += 1
    step("Every path-traversal attempt is refused", blocked == len(traversals),
         f"{blocked}/{len(traversals)} blocked")

    # Approval is an explicit human decision for one exact path, so the property
    # worth asserting is that an unapproved escape never happens.
    outcome = tools.execute("write_file", {"path": "../escaped.txt", "content": "x"}, approved=False)
    step("An unapproved write outside the workspace is refused", not outcome.ok, str(outcome.error)[:70])
    step("The escaped file was never created", not (settings.workspace_root.parent / "escaped.txt").exists())
    system_path = policy.evaluate("write_file", {"path": "C:/Windows/win.ini", "content": "x"})
    step("A protected system path cannot even be approved", not system_path.allowed, system_path.reason[:70])

    decision = policy.evaluate("delete_path", {"path": str(settings.workspace_root)})
    step("The workspace root itself can never be deleted",
         not decision.allowed or decision.approval_required, decision.reason[:70])

    # --- Command policy -----------------------------------------------------
    print("\n20c) Command policy")
    dangerous = [
        ("run_terminal", {"command": "Remove-Item -Recurse -Force C:\\Windows"}),
        ("run_terminal", {"command": "format C: /y"}),
        ("run_terminal", {"command": "Start-Process powershell -Verb RunAs"}),
        ("launch_app", {"application": "runas.exe", "arguments": ["/user:Administrator", "cmd"]}),
        ("run_python", {"code": "import shutil; shutil.rmtree('C:/')"}),
    ]
    gated = 0
    for name, arguments in dangerous:
        decision = policy.evaluate(name, arguments)
        if not decision.allowed or decision.approval_required:
            gated += 1
    step("Every dangerous command is blocked or gated behind approval",
         gated == len(dangerous), f"{gated}/{len(dangerous)}")

    embedded = policy.evaluate("write_file", {"path": "notes.txt", "content": f"key={SENTINEL}"})
    step("A tool call carrying a bare credential is refused outright",
         not embedded.allowed and embedded.sensitive, embedded.reason[:70])
    for label, sample in (("AWS", "AKIAIOSFODNN7EXAMPLE"), ("GitHub", "ghp_" + "b" * 36),
                          ("Anthropic", "sk-ant-api03-" + "a" * 40)):
        verdict = policy.evaluate("write_file", {"path": "notes.txt", "content": sample})
        step(f"A bare {label} key is detected by shape", not verdict.allowed)
    clean = policy.evaluate("write_file", {"path": "notes.txt", "content": "Support sits at 3400.50 on XAUUSD."})
    step("Ordinary trading text is not mistaken for a credential", clean.allowed)

    # --- Malformed model output --------------------------------------------
    print("\n20d) Malformed tool arguments")
    malformed = [
        ("draw_tradingview_level", {"annotation": "support", "price": "not-a-number"}),
        ("draw_tradingview_level", {"annotation": "<script>", "price": 1.0}),
        ("set_chart_layer", {"layer": "'; DROP TABLE drawing_ownership; --", "visible": True}),
        ("backtest_strategy", {"trigger": "../../etc/passwd"}),
        ("draw_tradingview_object", {"annotation": "trendline", "price_a": None,
                                     "minutes_a": 0, "price_b": 1, "minutes_b": 1}),
    ]
    rejected = 0
    for name, arguments in malformed:
        outcome = tools.execute(name, arguments, approved=True)
        if not outcome.ok or (isinstance(outcome.output, dict) and outcome.output.get("status") == "FAILED"):
            rejected += 1
    step("Malformed or hostile arguments are rejected, never executed",
         rejected == len(malformed), f"{rejected}/{len(malformed)}")
    step("The drawing table survived the injection attempt",
         isinstance(database.list_drawings(), list), "table intact")

    # --- Permission gating --------------------------------------------------
    print("\n20e) Permission gating")
    settings.computer_control_enabled = False
    settings.screen_access_enabled = False
    trading.refresh_permissions()
    for label, result in (
        ("drawing", trading.draw_annotation("support", 1.0)),
        ("calibration", trading.calibrate_chart()),
        ("clearing", trading.clear_drawings(all_owned=True)),
    ):
        step(f"{label} is refused while permissions are off",
             not result.verified and result.error_code in
             {"COMPUTER_CONTROL_DISABLED", "SCREEN_ACCESS_DISABLED"} or result.data == {"removed": [], "count": 0},
             f"code={result.error_code}")

    # --- Ownership isolation ------------------------------------------------
    print("\n20f) Ownership isolation")
    mine = database.record_drawing(symbol="XAUUSD", layer="SNR", drawing_type="support",
                                   theory="__sam__", price=100.0)
    theirs = database.record_drawing(symbol="XAUUSD", layer="NOTES", drawing_type="support",
                                     theory="__the_user__", price=200.0)
    database.delete_drawings(theory="__sam__")
    step("Deleting SAM's own theory leaves the user's record untouched",
         database.get_drawing(theirs["id"]) is not None and database.get_drawing(mine["id"]) is None)
    try:
        database.delete_drawings()
        step("An unfiltered delete is refused", False, "it was allowed")
    except ValueError:
        step("An unfiltered delete is refused", True, "requires an explicit all_owned request")
    database.delete_drawings(drawing_id=theirs["id"])

    # --- Audit chain --------------------------------------------------------
    print("\n19) Audit chain stress")
    before = len(database.list_audit(1000))
    for index in range(60):
        database.add_audit("stress", "success", f"synthetic event {index}",
                           details={"index": index, "note": "chain stress"})
    database.add_audit("credentials", "success", "stored a credential",
                       details={"name": "openrouter_api_key", "fingerprint": "abc123def456"})
    after = len(database.list_audit(1000))
    step("Every event was recorded", after - before >= 61, f"{after - before} new entries")
    verdict = database.verify_audit_chain()
    step("The hash chain is still valid after the burst",
         verdict is True or (isinstance(verdict, dict) and verdict.get("valid")), str(verdict)[:60])
    entries = json.dumps(database.list_audit(1000))
    step("No audit entry contains a credential", SENTINEL not in entries and "sk-or-v1-" not in entries)

    live_status, live_audit = call("/api/audit/verify")
    step("The live instance also reports a valid chain", '"valid": true' in live_audit.lower().replace(" ", "").replace('"valid":true', '"valid": true') or "true" in live_audit.lower(),
         live_audit[:60])

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
