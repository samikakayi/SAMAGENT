"""build_project with a fake model: safe paths only, files written under
the projects folder, real VS Code (not Cursor's `code` shim) and a browser
preview, progress reported."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from sam.hands.code import CodeBuilder, slugify, unique_folder, validate_files
from tests.conftest import FakeBackend
from tests.hands_helpers import fake_windows, win


def test_slugify_handles_sorani_and_english() -> None:
    assert slugify("Portfolio site for Sami!") == "portfolio-site-for-sami"
    sorani = slugify("ماڵپەڕی فرۆشتنی جل")
    assert sorani and sorani.isascii() and " " not in sorani
    assert slugify("!!!").startswith("project-")


def test_unique_folder(tmp_path: Path) -> None:
    (tmp_path / "site").mkdir()
    (tmp_path / "site-2").mkdir()
    assert unique_folder(tmp_path, "site") == tmp_path / "site-3"


def test_validate_files_rejects_unsafe_paths_and_types() -> None:
    kept, problems = validate_files([
        {"path": "index.html", "content": "<h1>x</h1>"},
        {"path": "./css/style.css", "content": "body{}"},
        {"path": "../escape.html", "content": "x"},
        {"path": "C:/Windows/evil.html", "content": "x"},
        {"path": "tool.exe", "content": "MZ"},
        {"path": ".gitignore", "content": "node_modules"},
        {"path": ".env", "content": "KEY=1"},
        "not a dict",
    ])
    assert [p for p, _ in kept] == ["index.html", "css/style.css", ".gitignore"]
    assert len(problems) == 4


def fake_vscode(app, tmp_path: Path) -> Path:
    """A fake VS Code install (Code.exe next to bin/code.cmd, as on this PC)."""
    code_dir = tmp_path / "VS Code"
    (code_dir / "bin").mkdir(parents=True)
    (code_dir / "Code.exe").write_bytes(b"MZ")
    app.config.set("hands.vscode_path", str(code_dir / "bin" / "code.cmd"))
    return code_dir / "Code.exe"


def recording_popen(api: Any, launched: list[list[str]]) -> Any:
    def popen(argv: list[str], **kwargs: Any) -> Any:
        launched.append(argv)
        # VS Code gets an environment without Electron host variables: a SAM
        # started from a VS Code terminal would otherwise make Code.exe exit.
        env = kwargs.get("env")
        assert env is not None and "ELECTRON_RUN_AS_NODE" not in env
        assert not any(name.upper().startswith("VSCODE_") for name in env)
        if argv[1] != "--reuse-window":  # opening a folder makes a VS Code window titled with it
            api.add(win(77, f"{Path(argv[1]).name} - Visual Studio Code", "Code.exe", pid=77))
    return popen


async def test_build_without_the_brain_uses_one_structured_call(make_app, tmp_path: Path) -> None:
    project = {"files": [{"path": "index.html", "content": "<!doctype html><html lang='ckb' dir='rtl'>سڵاو</html>"},
                         {"path": "css/style.css", "content": "body{font-family:Vazirmatn}"}],
               "entry": "index.html", "summary": "A one-page Sorani site."}
    backend = FakeBackend("omniroute", {"sam-strong": [json.dumps(project, ensure_ascii=False)]})
    app = make_app(backends={"omniroute": backend})
    assert app.worker is None
    app.config.set("hands.projects_dir", str(tmp_path / "SAM Projects"))
    code_exe = fake_vscode(app, tmp_path)
    windows, api = fake_windows([])
    launched: list[list[str]] = []
    previewed: list[str] = []
    builder = CodeBuilder(app, windows=windows, popen_fn=recording_popen(api, launched), startfile_fn=previewed.append)
    steps: list[tuple[Any, ...]] = []
    result = await builder.build("ماڵپەڕێکی سادە بە کوردی", name="my site",
                                 progress=lambda *a, **k: steps.append((*a, *sorted(k.items()))))
    assert result["ok"], result
    folder = tmp_path / "SAM Projects" / "my-site"
    assert Path(result["path"]) == folder
    assert (folder / "index.html").read_text(encoding="utf-8").endswith("سڵاو</html>")
    assert (folder / "css" / "style.css").exists()
    # real VS Code (not Cursor's `code` shim): the folder first, then the entry file in that window
    assert launched == [[str(code_exe), str(folder)], [str(code_exe), "--reuse-window", str(folder / "index.html")]]
    assert result["vscode"]["verified"] and previewed == [str(folder / "index.html")]
    assert steps[0][:2] == (1, 8) and steps[-1][:2] == (8, 8) and ("done", True) in steps[-1]
    request = backend.calls[0][1]
    assert request.json_schema is not None and request.max_tokens >= 16000


class FakeWorker:
    """Stands in for app.worker.build_project (the brain's streamed builder)."""

    def __init__(self, files: dict[str, str], *, ok: bool = True) -> None:
        self.files, self.ok = files, ok
        self.calls: list[dict[str, Any]] = []

    async def build_project(self, description: str, *, project_dir: Any, kind: str = "website", name: str = "",
                            progress: Any = None, cancel: Any = None, source: str = "worker") -> dict[str, Any]:
        self.calls.append({"description": description, "project_dir": Path(project_dir), "kind": kind,
                           "source": source, "folder_existed": Path(project_dir).is_dir()})
        for step, (rel, content) in enumerate(self.files.items(), start=1):
            target = Path(project_dir) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            if progress is not None:
                progress(step, 6, f"فایلی {rel} نووسرا")
        return {"ok": self.ok, "project_dir": str(project_dir), "files": sorted(self.files),
                "entry": "index.html" if "index.html" in self.files else None,
                "summary": "Wrote files", "model": "omniroute:sam-strong", "errors": [] if self.ok else ["x: cut"]}


async def test_build_delegates_generation_to_the_worker_after_opening_vscode(make_app, tmp_path: Path) -> None:
    app = make_app()
    app.worker = FakeWorker({"index.html": "<html lang='ckb' dir='rtl'>سڵاو</html>", "style.css": "body{}"})
    app.config.set("hands.projects_dir", str(tmp_path / "SAM Projects"))
    code_exe = fake_vscode(app, tmp_path)
    windows, api = fake_windows([])
    launched: list[list[str]] = []
    previewed: list[str] = []
    steps: list[tuple[Any, ...]] = []
    builder = CodeBuilder(app, windows=windows, popen_fn=recording_popen(api, launched), startfile_fn=previewed.append)
    result = await builder.build("ماڵپەڕێک بۆ دوکانەکەم", name="Shop", source="live",
                                 progress=lambda *a, **k: steps.append(a))
    folder = tmp_path / "SAM Projects" / "shop"
    assert result["ok"] and result["files"] == ["index.html", "style.css"], result
    call = app.worker.calls[0]
    assert call["project_dir"] == folder and call["folder_existed"] and call["source"] == "live"
    assert launched[0] == [str(code_exe), str(folder)]            # VS Code shows the folder before files exist
    assert launched[1] == [str(code_exe), "--reuse-window", str(folder / "index.html")]
    assert previewed == [str(folder / "index.html")]
    # the worker's 6 steps are relayed inside SAM's 8-step progress line
    assert [s[:2] for s in steps] == [(1, 8), (2, 8), (3, 8), (8, 8)]


async def test_build_through_the_real_brain_worker_and_files_tool(make_app, tmp_path: Path) -> None:
    site = ("<<<FILE: index.html>>>\n<html lang=\"ckb\" dir=\"rtl\"><link rel=\"stylesheet\" href=\"style.css\">"
            "<h1>سڵاو</h1></html>\n<<<END FILE>>>\n<<<FILE: style.css>>>\nh1{color:teal}\n<<<END FILE>>>\n")
    app = make_app(backends={"omniroute": FakeBackend("omniroute", {"sam-strong": [site]})})
    assert app.load_packages(["sam.hands", "sam.brain.worker"]) == {"sam.hands": "ok", "sam.brain.worker": "ok"}
    app.config.set("hands.projects_dir", str(tmp_path / "SAM Projects"))
    fake_vscode(app, tmp_path)
    windows, api = fake_windows([])
    launched: list[list[str]] = []
    builder = CodeBuilder(app, windows=windows, popen_fn=recording_popen(api, launched), startfile_fn=lambda p: None)
    result = await builder.build("ماڵپەڕێکی سادە", name="hello")
    folder = tmp_path / "SAM Projects" / "hello"
    assert result["ok"], result
    assert sorted(result["files"]) == ["index.html", "style.css"]
    assert "سڵاو" in (folder / "index.html").read_text(encoding="utf-8")
    writes = app.db.query("SELECT name, ok FROM activity WHERE name = 'files'")
    assert len(writes) == 2 and all(row["ok"] for row in writes)   # written through the files tool


async def test_build_reports_a_model_failure_honestly(make_app, tmp_path: Path) -> None:
    backend = FakeBackend("omniroute", {"sam-strong": ["not json at all"]})
    app = make_app(backends={"omniroute": backend})
    app.config.set("hands.projects_dir", str(tmp_path / "P"))
    fake_vscode(app, tmp_path)
    windows, api = fake_windows([])
    launched: list[list[str]] = []
    builder = CodeBuilder(app, windows=windows, popen_fn=recording_popen(api, launched), startfile_fn=lambda p: None)
    result = await builder.build("a site")
    assert not result["ok"] and "Could not generate" in result["summary"]
    assert not any((tmp_path / "P").iterdir())                  # the empty project folder was removed again
    # the empty VS Code window SAM opened for it is closed again
    assert launched and result["vscode_closed"] is True and ("close", 77) in api.calls
    assert not api.is_window(77)


class SlowWorker:
    """Writes one complete file, then keeps 'streaming' forever (a slow free model)."""

    async def build_project(self, description: str, *, project_dir: Any, **kwargs: Any) -> dict[str, Any]:
        (Path(project_dir) / "index.html").write_text("<html dir='rtl'>سڵاو</html>", encoding="utf-8")
        await asyncio.sleep(3600)
        raise AssertionError("not reached")


async def test_a_slow_model_gives_an_honest_partial_result_before_the_tool_timeout(make_app, tmp_path: Path) -> None:
    app = make_app()
    app.worker = SlowWorker()
    app.config.set("hands.projects_dir", str(tmp_path / "P"))
    app.config.set("hands.build_timeout_s", 30)
    fake_vscode(app, tmp_path)
    windows, api = fake_windows([])
    launched: list[list[str]] = []
    builder = CodeBuilder(app, windows=windows, popen_fn=recording_popen(api, launched), startfile_fn=lambda p: None)
    assert builder.time_budget() == 30.0
    builder.time_budget = lambda: 0.3  # type: ignore[method-assign] - keep the test fast
    result = await builder.build("ماڵپەڕێک", name="slow")
    assert not result["ok"] and result["timed_out"]
    assert result["files"] == ["index.html"] and result["path"] == str(tmp_path / "P" / "slow")
    assert "did not finish within" in result["summary"]
    assert api.is_window(77)                                     # the partial project stays open in VS Code


def test_build_tool_allows_more_time_than_the_generation_budget(make_app) -> None:
    from sam.hands.code import BUILD_TIMEOUT_S

    app = make_app()
    app.load_packages(["sam.hands"])
    spec = app.tools.get("build_project")
    assert spec.timeout_s >= BUILD_TIMEOUT_S + 30
    assert app.config.get("hands.build_timeout_s") == BUILD_TIMEOUT_S


async def test_a_partial_worker_result_is_not_reported_as_success(make_app, tmp_path: Path) -> None:
    app = make_app()
    app.worker = FakeWorker({"style.css": "body{}"}, ok=False)    # no index.html for a website
    app.config.set("hands.projects_dir", str(tmp_path / "P"))
    app.config.set("hands.vscode_path", str(tmp_path / "missing" / "bin" / "code.cmd"))  # -> Explorer
    windows, _ = fake_windows([])
    builder = CodeBuilder(app, windows=windows, popen_fn=lambda *a, **k: None, startfile_fn=lambda p: None)
    result = await builder.build("a site")
    assert not result["ok"] and "Problems" in result["summary"]
