"""Live check of build_project (the "Hamawmin demo") with the real model:
one streamed request on the ``strong`` ladder (free tier) through the brain's
Worker.build_project and the ``files`` tool, real VS Code opened on the new
folder, the entry file shown in it.

    set SAM_HOME=C:\\Users\\samit\\Desktop\\SAM-Agent
    .venv\\Scripts\\python.exe acceptance\\hands_build_live.py [--budget 540]

Clean-up and safety: the project goes to a throw-away folder under work/
(``hands.projects_dir`` and ``hands.build_timeout_s`` are overridden IN
MEMORY, so no setting is written to SAM_HOME's database); the browser preview
is recorded instead of opened (a tab in the user's browser could not be
closed safely); every VS Code window titled with a folder of that throw-away
directory is closed again -- also when the tool failed or timed out -- and
the folder is deleted. Skips (exit 77) when no provider of the strong ladder
has a key. A progress timeline (seconds since the call) is recorded so a slow
model can be told apart from a hang.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from _common import ROOT, Acceptance, sam_home

from sam.app import App
from sam.events import WorkerProgress

DESCRIPTION = ("ماڵپەڕێکی یەک لاپەڕەیی بچووک بۆ دوکانێکی قاوە بە ناوی «قاوەی هەولێر»: سەردێڕ، سێ جۆر قاوە "
               "بە نرخەوە، و دوگمەیەک بۆ پەیوەندی.")


def strong_configured(app: App) -> list[str]:
    providers = {ref.split(":", 1)[0] for ref in app.llm.ladder("strong")}
    backends = app.llm.backends
    return sorted(p for p in providers if p in backends and backends[p].configured())


def our_vscode_windows(windows: list[Any], projects: Path) -> list[Any]:
    """VS Code windows whose title names a folder SAM created under ``projects``."""
    names = [p.name.lower() for p in projects.iterdir()] if projects.is_dir() else []
    return [w for w in windows if w.process.lower() == "code.exe" and any(n in w.title.lower() for n in names)]


async def close_our_vscode(app: App, projects: Path) -> bool:
    """Close only VS Code windows titled with OUR throw-away folders."""
    for _ in range(4):
        mine = our_vscode_windows(await app.hands.windows.list(), projects)
        if not mine:
            return True
        for window in mine:
            await app.hands.windows.act("close", window)
        await asyncio.sleep(1.0)
    return not our_vscode_windows(await app.hands.windows.list(), projects)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=None, help="hands.build_timeout_s for this run")
    args = parser.parse_args()
    acc = Acceptance("hands_build_live")
    app = App(str(sam_home()) if os.environ.get("SAM_HOME") else None)
    projects = Path(tempfile.mkdtemp(prefix="sam2-build-live-", dir=str(ROOT / "work"))) / "SAM Projects"
    real_get = app.config.get
    overrides: dict[str, Any] = {"hands.projects_dir": str(projects)}
    if args.budget:
        overrides["hands.build_timeout_s"] = args.budget

    def config_get(key: str, default: Any = None) -> Any:
        return overrides[key] if key in overrides else real_get(key, default)
    app.config.get = config_get  # type: ignore[method-assign] - in memory only, never persisted
    app.load_packages(["sam.hands", "sam.brain.worker"])
    await app.start()
    timeline: list[tuple[float, str]] = []
    started = time.perf_counter()

    def on_progress(event: WorkerProgress) -> None:
        timeline.append((round(time.perf_counter() - started, 1), f"{event.step}/{event.max_steps} {event.text_ckb}"))
    app.bus.subscribe(WorkerProgress, on_progress)
    try:
        if not strong_configured(app):
            acc.skip("no provider of the strong ladder has a key")
            return acc.finish()
        previews: list[str] = []
        app.hands.code._startfile = previews.append
        with acc.check("build_project writes a Sorani website through the worker and opens it in VS Code") as c:
            started = time.perf_counter()
            result = await app.tools.dispatch("build_project", {"description": DESCRIPTION, "name": "qawa test"},
                                              source="text")
            data = result.get("data") or {}
            folder = Path(data["path"]) if data.get("path") else None
            html = (folder / "index.html").read_text(encoding="utf-8") if folder and (folder / "index.html").is_file() \
                else ""
            writes = app.db.query("SELECT ok, duration_ms FROM activity WHERE name = 'files' AND at > ?",
                                  (time.time() - 900,))
            c.data.update(ok=result["ok"], summary=result["summary"], ms=round((time.perf_counter() - started) * 1000),
                          budget_s=app.hands.code.time_budget(), files=data.get("files"), model=data.get("model"),
                          timed_out=data.get("timed_out"), vscode=data.get("vscode"), preview_recorded=previews,
                          files_tool_writes=len(writes), html_rtl='dir="rtl"' in html or "dir='rtl'" in html,
                          html_has_sorani="قاوە" in html, html_chars=len(html), timeline=timeline)
            assert result["ok"], result["summary"]
            assert (data.get("vscode") or {}).get("verified"), "VS Code window was not seen"
            assert c.data["html_has_sorani"] and writes, "no Sorani page / not written through the files tool"
    finally:
        closed = await close_our_vscode(app, projects)
        with acc.check("clean-up: our VS Code window closed and the test folder deleted") as c:
            await asyncio.sleep(0.5)  # VS Code releases the folder after its window closes
            shutil.rmtree(projects.parent, ignore_errors=True)
            c.data.update(vscode_closed=closed, folder_removed=not projects.parent.exists())
            assert closed and not projects.parent.exists(), c.data
        await app.stop()
        app.close()
    return acc.finish()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
