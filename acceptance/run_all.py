"""Run SAM 2's live acceptance scripts on this PC and print one table.

    .venv\\Scripts\\python.exe acceptance\\run_all.py [--home PATH] [--only TEXT ...] [--list]
                                               [--timeout S] [--json PATH]

Discovery: every ``acceptance/<module>_*.py`` (``run_all.py`` and files that
start with ``_`` are not scripts). They run ONE AT A TIME -- they share live
apps (TradingView drawings, MT5, the microphone) -- each with:

- ``SAM_HOME`` = ``--home`` (default: env ``SAM_HOME``, else the repository root),
- ``SAM_ACCEPTANCE_OUT`` = a temp file where the script may write its JSON result,
- ``PYTHONPATH`` = the repository root, ``PYTHONIOENCODING=utf-8``, cwd = the repo root.

Result protocol (``acceptance/_common.py`` implements it):
- exit code 0 = pass, 77 = skipped (a precondition is missing), else fail;
- optional JSON object ``{"ok", "skipped", "summary", "checks": [...]}`` in
  ``SAM_ACCEPTANCE_OUT`` or as the last stdout line; it refines the exit code
  (``"ok": false`` fails a script that exited 0).

Everything printed or saved is redacted (``sam.secrets.redact``). The full
report goes to ``work/acceptance/<timestamp>.json`` (gitignored) unless
``--json`` names another file. Exit code: 0 when nothing failed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
EXIT_SKIP = 77
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _redact(text: str) -> str:
    try:
        from sam.secrets import redact
        return redact(text)
    except Exception:  # noqa: BLE001
        return text


@dataclass
class ScriptResult:
    name: str
    status: str                 # pass | fail | skip | timeout | error
    seconds: float
    exit_code: int | None
    summary: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    output_tail: str = ""


def discover(directory: Path = HERE, only: list[str] | None = None) -> list[Path]:
    scripts = sorted(p for p in directory.glob("*_*.py")
                     if p.name != "run_all.py" and not p.name.startswith("_"))
    if only:
        wanted = [o.lower() for o in only]
        scripts = [p for p in scripts if any(o in p.stem.lower() for o in wanted)]
    return scripts


def _last_json_line(text: str) -> dict[str, Any] | None:
    """The JSON object a script printed last: one line, or pretty-printed
    (a block from a line starting with "{" to the end of the output)."""
    lines = text.strip().splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].lstrip().startswith("{"):
            continue
        # The block to the end first: an object nested in pretty-printed JSON
        # never parses that way. A lone line only counts when not indented.
        candidates = ["\n".join(lines[index:])]
        if lines[index].startswith("{"):
            candidates.append(lines[index])
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def run_script(script: Path, *, home: Path, timeout_s: float, python: str = sys.executable) -> ScriptResult:
    with tempfile.TemporaryDirectory(prefix="sam-acc-") as tmp:
        out_file = Path(tmp) / "result.json"
        env = {**os.environ, "SAM_HOME": str(home), "SAM_ACCEPTANCE_OUT": str(out_file),
               "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
               "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH", "")]))}
        began = time.perf_counter()
        try:
            proc = subprocess.run([python, str(script)], cwd=str(ROOT), env=env, capture_output=True,
                                  timeout=timeout_s, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            tail = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
            return ScriptResult(script.stem, "timeout", round(time.perf_counter() - began, 2), None,
                                f"no result within {timeout_s:.0f} s", output_tail=_redact(tail[-1500:]))
        except OSError as exc:
            return ScriptResult(script.stem, "error", round(time.perf_counter() - began, 2), None,
                                _redact(f"could not start: {exc}"))
        seconds = round(time.perf_counter() - began, 2)
        payload: dict[str, Any] | None = None
        if out_file.is_file():
            try:
                payload = json.loads(out_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
        if payload is None:
            payload = _last_json_line(proc.stdout or "")
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode == EXIT_SKIP or (payload or {}).get("skipped"):
        status = "skip"
    elif proc.returncode == 0 and (payload is None or payload.get("ok", True)):
        status = "pass"
    else:
        status = "fail"
    lines = [line for line in output.strip().splitlines() if line.strip()]
    summary = str((payload or {}).get("summary") or (lines[-1] if lines and status != "pass" else "") or "")
    if status == "fail" and not summary:
        summary = f"exit code {proc.returncode}"
    return ScriptResult(script.stem, status, seconds, proc.returncode, _redact(summary)[:300],
                        [c for c in (payload or {}).get("checks", []) if isinstance(c, dict)],
                        _redact(output[-3000:]) if status != "pass" else "")


def format_table(results: list[ScriptResult]) -> str:
    rows = [("script", "status", "sec", "summary")]
    rows += [(r.name, r.status.upper(), f"{r.seconds:.1f}", r.summary.replace("\n", " ")[:90]) for r in results]
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(3)) + "  " + row[3])
        if index == 0:
            lines.append("  ".join("-" * w for w in widths) + "  " + "-" * 7)
    counts = {s: sum(1 for r in results if r.status == s) for s in ("pass", "fail", "skip", "timeout", "error")}
    lines.append("")
    lines.append(", ".join(f"{n} {s}" for s, n in counts.items() if n) or "no scripts")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SAM 2 acceptance scripts (live, on this PC).")
    parser.add_argument("--home", default=os.environ.get("SAM_HOME") or str(ROOT))
    parser.add_argument("--only", nargs="*", default=None, help="substring(s) of script names")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per script")
    parser.add_argument("--list", action="store_true", help="list the scripts and exit")
    parser.add_argument("--json", default=None, help="report path (default work/acceptance/<stamp>.json)")
    parser.add_argument("--dir", default=str(HERE), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:  # Sorani summaries on a cp1252 console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

    scripts = discover(Path(args.dir), args.only)
    if args.list:
        print("\n".join(p.name for p in scripts) or "no acceptance scripts found")
        return 0
    home = Path(args.home)
    print(f"SAM_HOME = {home}   ({len(scripts)} scripts, one at a time)", flush=True)
    results: list[ScriptResult] = []
    for script in scripts:
        print(f"-> {script.name} ...", flush=True)
        result = run_script(script, home=home, timeout_s=args.timeout)
        print(f"   {result.status.upper()} in {result.seconds:.1f} s  {result.summary}", flush=True)
        results.append(result)
    print()
    print(format_table(results))
    report_path = Path(args.json) if args.json else ROOT / "work" / "acceptance" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {"home": str(home), "at": time.time(), "results": [asdict(r) for r in results]}
        report_path.write_text(_redact(json.dumps(report, ensure_ascii=False, indent=1)), encoding="utf-8")
        print(f"\nreport: {report_path}")
    except OSError as exc:
        print(f"\ncould not write the report: {exc}")
    return 1 if any(r.status in ("fail", "timeout", "error") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
