"""build_project: the "Hamawmin demo" -- say what you want, SAM writes a
small project, opens it in VS Code and previews it in the browser.

Order of work (the reference demo shows VS Code first, then the files
appearing): create ``hands.projects_dir/<slug>`` -> open that folder in VS
Code -> generate the files -> open the entry file in the same VS Code
window -> preview ``index.html`` in the default browser.

Files come from the brain's ``app.worker.build_project`` (one STREAMED call
on the ``strong`` ladder in a delimited multi-file format; each file is
written through the ``files`` tool the moment it is complete, cut-off or
referenced-but-missing files are regenerated). Without the brain package,
ONE structured call returning ``{files: [{path, content}], entry, summary}``
is the fallback. Either way paths are validated (relative, no ``..``,
allowed types) and stay under ``hands.projects_dir`` (~/SAM Projects).

VS Code: ``code`` on PATH is Cursor's shim on this PC (measured), so SAM
starts real VS Code from ``%LOCALAPPDATA%\\Programs\\Microsoft VS Code``
(setting ``hands.vscode_path``) and verifies its window appeared.

Time budget: the brain's stream has only a per-chunk timeout, so a slow
free model can write for many minutes (the first live run on this PC hit the
old 300 s tool timeout with one file written and the result -- folder, files
-- was lost). Generation therefore gets its own deadline
(``hands.build_timeout_s``, below the tool's timeout); when it passes, the
files already on disk are reported honestly as a partial project.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from ._win import launch_environment
from .aliases import transliterate

MAX_FILES = 20
MAX_FILE_BYTES = 200_000
BUILD_TIMEOUT_S = 540.0  # default for hands.build_timeout_s; the build_project tool allows 600 s
ALLOWED_SUFFIXES = {".html", ".css", ".js", ".mjs", ".json", ".md", ".txt", ".svg", ".py", ".toml", ".cfg",
                    ".ini", ".yaml", ".yml", ".csv", ".ts", ".tsx", ".jsx", ".xml", ".webmanifest", ".gitignore"}

PROJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "files": {"type": "array", "items": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        "entry": {"type": "string", "description": "file to open first, e.g. index.html"},
        "summary": {"type": "string", "description": "one sentence: what was built"},
    },
    "required": ["files", "entry"],
}

SYSTEM = {
    "website": (
        "You are a senior front-end developer. Build a complete, polished, modern, responsive static website with "
        "plain HTML, CSS and JavaScript (no build step, no frameworks, no external JS; Google Fonts allowed). "
        "Use semantic HTML, a pleasing colour palette, good spacing, hover states and a mobile layout. If the "
        "request is in Kurdish (Sorani) write the page text in Sorani with dir=\"rtl\" and lang=\"ckb\" and a "
        "font that supports Arabic script (e.g. Vazirmatn or Noto Naskh Arabic). Put the entry page in index.html. "
        "Return JSON only."),
    "python": (
        "You are a senior Python developer. Write a small, complete, runnable Python 3.13 project for the request "
        "with a main.py entry point, standard library only unless the request needs a package (then add "
        "requirements.txt), clear comments and a README.md with how to run it. Return JSON only."),
    "other": (
        "You are a senior developer. Create a small, complete project for the request with sensible files and a "
        "README.md. Return JSON only."),
}


def slugify(text: str) -> str:
    """ASCII folder name ("ماڵپەڕی فرۆشتنی جل" -> "malperi-froshtni-jl")."""
    value = text if text.isascii() else transliterate(text)
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    words = [w for w in value.split("-") if w][:5]
    slug = "-".join(words)[:40].strip("-")
    return slug or "project-" + _dt.datetime.now().strftime("%Y%m%d-%H%M")


def unique_folder(base: Path, slug: str) -> Path:
    candidate = base / slug
    counter = 2
    while candidate.exists():
        candidate = base / f"{slug}-{counter}"
        counter += 1
    return candidate


def validate_files(files: Any) -> tuple[list[tuple[str, str]], list[str]]:
    """Keep safe relative paths only. Returns (files, problems)."""
    kept: list[tuple[str, str]] = []
    problems: list[str] = []
    if not isinstance(files, list):
        return [], ["the model returned no file list"]
    for item in files[:MAX_FILES]:
        if not isinstance(item, dict):
            continue
        raw = re.sub(r"^(\./)+", "", str(item.get("path", "")).strip().replace("\\", "/")).lstrip("/")
        content = str(item.get("content", ""))
        parts = [p for p in raw.split("/") if p]
        if not parts or any(p in ("..", "") or p.startswith(".") and p != ".gitignore" or ":" in p for p in parts):
            problems.append(f"skipped unsafe path {raw!r}")
            continue
        suffix = Path(parts[-1]).suffix.lower() or parts[-1].lower()
        if suffix not in ALLOWED_SUFFIXES:
            problems.append(f"skipped {raw!r} (type not allowed)")
            continue
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            problems.append(f"skipped {raw!r} (too large)")
            continue
        kept.append(("/".join(parts), content))
    if len(files) > MAX_FILES:
        problems.append(f"only the first {MAX_FILES} files were kept")
    return kept, problems


class CodeBuilder:
    def __init__(self, app: Any, *, windows: Any, popen_fn: Any = None, startfile_fn: Any = None) -> None:
        self.app = app
        self.windows = windows
        self._popen = popen_fn or subprocess.Popen
        self._startfile = startfile_fn or (lambda p: os.startfile(p))  # type: ignore[attr-defined]
        self.vscode_wait_s = 15.0  # a cold VS Code start shows its window in 2-5 s on this PC

    def projects_dir(self) -> Path:
        return Path(str(self.app.config.get("hands.projects_dir") or Path.home() / "SAM Projects"))

    def vscode_exe(self) -> Path | None:
        """Code.exe of the configured install (the setting defaults to
        %LOCALAPPDATA%/Programs/Microsoft VS Code/bin/code.cmd). Code.exe is
        preferred over code.cmd: no console window and no batch-file quoting."""
        configured = Path(str(self.app.config.get("hands.vscode_path") or ""))
        if not str(configured) or str(configured) == ".":
            return None
        for candidate in (configured.parent.parent / "Code.exe", configured):
            if candidate.name and candidate.is_file():
                return candidate
        return None

    async def generate(self, description: str, kind: str) -> dict[str, Any]:
        messages = [{"role": "system", "content": SYSTEM.get(kind, SYSTEM["other"])},
                    {"role": "user", "content": f"Build this ({kind}): {description}\n"
                                                f"Return JSON: {{\"files\": [{{\"path\", \"content\"}}], "
                                                f"\"entry\", \"summary\"}}."}]
        response = await self.app.llm.chat(messages, ladder="strong", json_schema=PROJECT_SCHEMA, max_tokens=16000,
                                           reasoning="low", timeout_s=150)
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("the model did not return a JSON object")
        data["_model"] = response.model_ref
        return data

    async def _run_vscode(self, argv_tail: list[str]) -> Path | None:
        exe = self.vscode_exe()
        if exe is None:
            return None
        # launch_environment: without it, Code.exe started from a SAM that
        # itself runs under VS Code exits at once (see _win.py).
        await asyncio.to_thread(self._popen, [str(exe), *argv_tail], env=launch_environment(), stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=0x08000000 if exe.suffix == ".cmd" else 0)
        return exe

    async def open_in_vscode(self, folder: Path, entry: Path | None = None, *,
                             wait_s: float | None = None) -> dict[str, Any]:
        """Open ``folder`` (and optionally a file in it) in real VS Code and
        wait until a Code.exe window titled with the folder name appears."""
        tail = [str(folder)] + ([str(entry)] if entry is not None else [])
        if await self._run_vscode(tail) is None:
            await asyncio.to_thread(self._startfile, str(folder))
            return {"opened": "explorer", "verified": True}
        deadline = time.monotonic() + (self.vscode_wait_s if wait_s is None else wait_s)
        while time.monotonic() < deadline:
            for window in await self.windows.list():
                if window.process.lower() == "code.exe" and folder.name.lower() in window.title.lower():
                    return {"opened": "vscode", "verified": True, "window": window.title, "hwnd": window.hwnd}
            await asyncio.sleep(0.4)
        return {"opened": "vscode", "verified": False}

    async def _generate_with_worker(self, worker: Any, description: str, folder: Path, kind: str, name: str,
                                    say: Any, cancel: Any, source: str) -> tuple[list[str], str | None, dict[str, Any]]:
        def relay(step: int, total: int, text_ckb: str, **_: Any) -> None:
            say(1 + min(int(step), int(total)) * 6 // max(1, int(total)), 8, text_ckb)

        result = await worker.build_project(description, project_dir=folder, kind=kind, name=name,
                                            progress=relay, cancel=cancel, source=source)
        written = [str(p) for p in result.get("files") or [] if (folder / str(p)).is_file()]
        return written, result.get("entry"), result

    async def _generate_single_call(self, description: str, folder: Path,
                                    kind: str) -> tuple[list[str], str | None, dict[str, Any]]:
        data = await self.generate(description, kind)
        files, problems = validate_files(data.get("files"))

        def write_all() -> None:
            for rel, content in files:
                target = folder / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8", newline="\n")
        await asyncio.to_thread(write_all)
        written = [rel for rel, _ in files if (folder / rel).is_file()]
        return written, data.get("entry"), {"ok": bool(files) and len(written) == len(files), "errors": problems,
                                            "model": data.get("_model"), "summary": data.get("summary") or ""}

    def time_budget(self) -> float:
        try:
            value = float(self.app.config.get("hands.build_timeout_s", BUILD_TIMEOUT_S) or BUILD_TIMEOUT_S)
        except (TypeError, ValueError):
            value = BUILD_TIMEOUT_S
        return max(30.0, value)

    async def build(self, description: str, *, name: str | None = None, kind: str = "website",
                    progress: Any = None, cancel: Any = None, source: str = "worker") -> dict[str, Any]:
        kind = kind if kind in SYSTEM else "other"
        started = time.perf_counter()
        say = progress or (lambda *a, **k: None)
        folder = unique_folder(self.projects_dir(), slugify(name or description))
        await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
        say(1, 8, "ڤی ئێس کۆد دەکەمەوە")
        # VS Code starts while the model writes; its window is checked afterwards.
        vscode = asyncio.ensure_future(self.open_in_vscode(folder))
        worker = getattr(self.app, "worker", None)
        use_worker = worker is not None and hasattr(worker, "build_project")
        budget = self.time_budget()
        timed_out = False
        try:
            if use_worker:
                generation = self._generate_with_worker(worker, description, folder, kind, name or "", say, cancel,
                                                        source)
            else:
                say(2, 8, "کۆدەکە دەنووسم")
                generation = self._generate_single_call(description, folder, kind)
            written, entry_name, gen = await asyncio.wait_for(generation, budget)
        except TimeoutError:
            # Keep what the model finished: each file is written once complete.
            timed_out = True
            written = await asyncio.to_thread(_files_on_disk, folder)
            entry_name = None
            gen = {"ok": False, "errors": [f"the model did not finish within {budget:.0f} s"]}
        except Exception as exc:  # noqa: BLE001 - honest failure result
            written, entry_name, gen = [], None, {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"[:300]]}
        except BaseException:  # stop_all cancelled the tool: stop waiting for VS Code too
            vscode.cancel()
            raise
        problems = list(gen.get("errors") or [])
        if not written:
            # Nothing to show: close the empty VS Code window SAM opened and
            # drop the empty folder, so a failed build leaves nothing behind.
            closed = await self._close_opened_vscode(vscode)
            await asyncio.to_thread(_remove_if_empty, folder)
            say(8, 8, "دروست نەکرا", done=True, ok_=False)
            return {"ok": False, "path": str(folder), "problems": problems[:10], "timed_out": timed_out,
                    "vscode_closed": closed,
                    "summary": "Could not generate the project: no file was written"
                               + (f" ({problems[0]})." if problems else ".")}
        default = "index.html" if kind == "website" else "main.py"
        entry_rel = str(entry_name or default)
        entry = folder / entry_rel if (folder / entry_rel).is_file() else folder / written[0]
        opened = await vscode
        if opened.get("opened") == "vscode":
            await self._run_vscode(["--reuse-window", str(entry)])  # show the entry file in that window
        preview = None
        if entry.suffix.lower() == ".html":
            await asyncio.to_thread(self._startfile, str(entry))
            preview = str(entry)
        good = bool(gen.get("ok", True)) and (kind != "website" or any(p.endswith(".html") for p in written))
        say(8, 8, "تەواو بوو" if good else "بەشێکی دروست کرا", done=True, ok_=good)
        extra = "" if use_worker else str(gen.get("summary") or "")
        return {"ok": good, "path": str(folder), "files": written, "entry": str(entry), "preview": preview,
                "vscode": opened, "problems": problems[:10], "model": gen.get("model"), "timed_out": timed_out,
                "ms": round((time.perf_counter() - started) * 1000),
                "summary": f"Built {len(written)} files in {folder}"
                           + (" and opened it in VS Code" if opened.get("verified") else "")
                           + (" and in the browser." if preview else ".")
                           + (f" {extra}" if extra else "")
                           + ("" if good else f" Problems: {'; '.join(problems[:3]) or 'incomplete project'}.")}

    async def _close_opened_vscode(self, vscode: "asyncio.Future[dict[str, Any]]") -> bool | None:
        """Close the VS Code window this build opened (its workspace is the
        new, still empty folder, so nothing can be lost). None = no window."""
        try:
            opened = await vscode
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - nothing to close
            return None
        hwnd = opened.get("hwnd")
        if opened.get("opened") != "vscode" or not hwnd:
            return None
        window = await self.windows.find(int(hwnd))
        if window is None:
            return True
        result = await self.windows.act("close", window)
        return bool(result.get("ok"))


def _files_on_disk(folder: Path) -> list[str]:
    """Relative paths of the files already written (after a timeout)."""
    try:
        return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    except OSError:
        return []


def _remove_if_empty(folder: Path) -> None:
    """Drop the project folder again when nothing was written into it."""
    try:
        folder.rmdir()  # only succeeds for an empty folder
    except OSError:
        pass


__all__ = ["CodeBuilder", "PROJECT_SCHEMA", "slugify", "unique_folder", "validate_files"]
