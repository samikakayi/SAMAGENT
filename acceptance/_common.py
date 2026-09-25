"""Tiny helper for live acceptance scripts (``acceptance/<module>_*.py``).

Scripts do not have to use it: ``run_all.py`` also understands a plain exit
code (0 pass, 77 skip, anything else fail) and a JSON object printed as the
last stdout line. With the helper::

    from _common import Acceptance

    acc = Acceptance("hands_open_app")
    with acc.check("open Chrome by Sorani name") as c:
        ...                      # raise/assert to fail; c.detail = "..." ; c.data["ms"] = 812
    acc.skip("TradingView is not installed")    # optional, whole script
    sys.exit(acc.finish())       # writes the JSON result, returns the exit code

Output is redacted with ``sam.secrets.redact``. Rules for every live script:
clean up after yourself (drawings, symbol/timeframe, temp files), never place
orders, never play audio on the speakers, never print a key.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 77


def _redact(text: str) -> str:
    try:
        from sam.secrets import redact
        return redact(text)
    except Exception:  # noqa: BLE001
        return text


def sam_home() -> Path:
    """The SAM_HOME the runner passed (default: the repository root)."""
    return Path(os.environ.get("SAM_HOME") or ROOT)


class _Check:
    def __init__(self, name: str) -> None:
        self.name, self.ok, self.detail, self.ms = name, True, "", 0.0
        self.data: dict[str, Any] = {}
        self.skipped = False

    def skip(self, reason: str) -> None:
        self.skipped, self.detail = True, reason
        raise _SkipCheck(reason)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "skipped": self.skipped, "detail": _redact(self.detail)[:2000],
                "ms": round(self.ms, 1), "data": json.loads(_redact(json.dumps(self.data, ensure_ascii=False, default=str)))}


class _SkipCheck(Exception):
    pass


class Acceptance:
    def __init__(self, name: str) -> None:
        self.name = name
        self.checks: list[_Check] = []
        self.skipped_reason = ""
        self.started = time.perf_counter()

    @contextmanager
    def check(self, name: str) -> Iterator[_Check]:
        item = _Check(name)
        began = time.perf_counter()
        try:
            yield item
        except _SkipCheck:
            pass
        except Exception as exc:  # noqa: BLE001 - a failed check is a result, not a crash
            item.ok = False
            item.detail = (item.detail + "\n" if item.detail else "") + "".join(
                traceback.format_exception_only(type(exc), exc)).strip()
        finally:
            item.ms = (time.perf_counter() - began) * 1000.0
            self.checks.append(item)
            status = "SKIP" if item.skipped else ("ok" if item.ok else "FAIL")
            print(_redact(f"[{status}] {name} ({item.ms:.0f} ms) {item.detail}".rstrip()), flush=True)

    def skip(self, reason: str) -> None:
        self.skipped_reason = reason

    def result(self) -> dict[str, Any]:
        failed = [c for c in self.checks if not c.ok]
        ok = not failed and not self.skipped_reason
        passed = sum(1 for c in self.checks if c.ok and not c.skipped)
        summary = (f"skipped: {self.skipped_reason}" if self.skipped_reason else
                   f"{passed}/{len(self.checks)} checks passed" + (f"; failed: {', '.join(c.name for c in failed)}" if failed else ""))
        return {"name": self.name, "ok": ok, "skipped": bool(self.skipped_reason), "summary": _redact(summary),
                "checks": [c.as_dict() for c in self.checks],
                "seconds": round(time.perf_counter() - self.started, 2)}

    def finish(self) -> int:
        """Write the JSON result (file from the runner + last stdout line); return the exit code."""
        result = self.result()
        text = json.dumps(result, ensure_ascii=False)
        out = os.environ.get("SAM_ACCEPTANCE_OUT")
        if out:
            Path(out).write_text(text, encoding="utf-8")
        print(text, flush=True)
        if result["skipped"]:
            return EXIT_SKIP
        return EXIT_PASS if result["ok"] else EXIT_FAIL


__all__ = ["Acceptance", "sam_home", "ROOT", "EXIT_PASS", "EXIT_FAIL", "EXIT_SKIP"]
