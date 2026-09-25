"""Generate a small multi-file project (website, Python, other) with the text
model and write it through the ``files`` tool.

This is the "Hamawmin demo" path (reports/reference-agents.json: VS Code opens
and a Kurdish website appears). ``hands.build_project`` creates/opens the
folder and calls ``app.worker.build_project(...)``; this module only writes
files.

Why a delimited format instead of JSON: whole HTML/CSS/JS files inside JSON
strings need escaping that free models get wrong on long outputs, while

    <<<FILE: index.html>>>
    ...content...
    <<<END FILE>>>

survives streaming and can be parsed incrementally, so each file is written
(and shows up in VS Code) the moment it is complete. One streamed request
builds the whole project (free quotas are small); only files lost to a
length cut-off or referenced but missing are generated one by one (max 3).
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .llm import LLMError

log = logging.getLogger("sam.project_builder")

FILE_OPEN = re.compile(r"^\s*<<<\s*FILE:\s*(?P<path>.+?)\s*>>>\s*$")
FILE_END = re.compile(r"^\s*<<<\s*END(?: FILE)?\s*>>>\s*$")
_SAFE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\- ]{0,80}$")
_REF = re.compile(r"""(?:href|src)\s*=\s*["']([^"':#?]+?)["']""", re.I)
MAX_FILES = 12
MAX_FILE_CHARS = 200_000
MAX_REPAIRS = 3
_DIGITS_CKB = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

KIND_RULES = {
    "website": (
        "A static website: index.html at the root plus the CSS/JS files it links with relative paths "
        "(e.g. style.css, script.js). Polished, modern and responsive (CSS variables, flex/grid, generous "
        "spacing, hover states, a subtle animation), works by opening index.html directly: no build step, no "
        "JS frameworks (Google Fonts are fine). If the description is in Sorani, every visible text is "
        "natural Sorani, <html lang=\"ckb\" dir=\"rtl\">, and the font renders Arabic script well (Vazirmatn "
        "or Noto Sans Arabic from Google Fonts). Use realistic content, not lorem ipsum. At most 6 files."),
    "python": (
        "A Python 3.13 project: main.py as the entry point, standard library preferred, requirements.txt only "
        "if third-party packages are needed, README.md with how to run it. At most 8 files."),
    "other": (
        "Choose a sensible minimal structure for what is described; include README.md explaining how to "
        "use it. At most 8 files."),
}

SYSTEM_PROMPT = (
    "You are an expert software developer building a small, complete, working project for a Kurdish user. "
    "Output ONLY the project's files, each in exactly this format:\n"
    "<<<FILE: relative/path.ext>>>\n(the complete file content)\n<<<END FILE>>>\n"
    "No explanations before, between or after the files, and no Markdown code fences. Paths are relative, "
    "use only letters, digits, '.', '-', '_' and '/'. Every file must be complete: no placeholders, no "
    "'rest of the code here'.\n")


def safe_relpath(raw: str) -> str | None:
    """A normalised relative POSIX path, or None when the path could escape
    the project folder or has odd characters."""
    value = raw.strip().strip("`'\"").replace("\\", "/")
    if not value or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        return None
    parts = [p for p in PurePosixPath(value).parts if p not in ("", ".")]
    if not parts or len(parts) > 5:
        return None
    for part in parts:
        if part == ".." or not _SAFE_PART.match(part):
            return None
    return "/".join(parts)


def strip_fences(content: str) -> str:
    """Remove a Markdown fence wrapped around a whole file (models add them
    even when told not to)."""
    lines = content.strip("\n").splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]
    return "\n".join(lines) + "\n"


class FileBlockParser:
    """Incremental parser for the delimited file format."""

    def __init__(self) -> None:
        self._line = ""
        self._path: str | None = None
        self._raw_path = ""
        self._lines: list[str] = []
        self.rejected: list[str] = []

    def feed(self, delta: str) -> list[tuple[str, str]]:
        done: list[tuple[str, str]] = []
        self._line += delta
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            done.extend(self._consume(line.rstrip("\r")))
        return done

    def _consume(self, line: str) -> list[tuple[str, str]]:
        opened = FILE_OPEN.match(line)
        if self._path is None and not self._raw_path:
            if opened:
                self._start(opened.group("path"))
            return []
        if FILE_END.match(line):
            return self._close()
        if opened:  # a new file began without END: close the current one first
            finished = self._close()
            self._start(opened.group("path"))
            return finished
        self._lines.append(line)
        return []

    def _start(self, raw: str) -> None:
        self._raw_path = raw
        self._path = safe_relpath(raw)
        if self._path is None:
            self.rejected.append(raw[:120])
        self._lines = []

    def _close(self) -> list[tuple[str, str]]:
        path, lines = self._path, self._lines
        self._path, self._raw_path, self._lines = None, "", []
        if path is None:
            return []
        return [(path, strip_fences("\n".join(lines)))]

    def pending(self) -> tuple[str, str] | None:
        """The unterminated file at the end of the stream, if any."""
        if self._path is None:
            return None
        lines = self._lines + ([self._line] if self._line else [])
        return self._path, strip_fences("\n".join(lines))


def missing_references(files: dict[str, str]) -> list[str]:
    """Relative CSS/JS files that HTML files link but nobody wrote (images
    are left alone: a missing image does not break the page)."""
    missing: list[str] = []
    for path, content in files.items():
        if not path.endswith((".html", ".htm")):
            continue
        base = PurePosixPath(path).parent
        for ref in _REF.findall(content):
            if ref.startswith(("http", "//", "data:", "mailto:", "tel:", "javascript:")):
                continue
            target = safe_relpath(str(base / ref))
            if target and target not in files and target not in missing and target.endswith((".css", ".js")):
                missing.append(target)
    return missing


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


class _Writer:
    """Writes through the ``files`` tool when hands registered it (policy,
    activity log, events); otherwise directly, inside the project folder."""

    def __init__(self, app: Any, project_dir: Path, source: str) -> None:
        self.app = app
        self.root = project_dir
        self.source = source
        roots = [Path(str(app.config.get("hands.projects_dir", "") or "")), Path(app.config.workspace_dir)]
        self.trusted_root = any(str(r) not in ("", ".") and _inside(project_dir, r) for r in roots)

    async def write(self, relpath: str, content: str) -> tuple[bool, str]:
        target = self.root / Path(*relpath.split("/"))
        if not _inside(target, self.root):
            return False, "path escapes the project folder"
        if self.trusted_root:
            await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        tools = getattr(self.app, "tools", None)
        if tools is not None and tools.get("files") is not None:
            result = await tools.dispatch("files", {"action": "write", "path": str(target), "content": content},
                                          source=self.source)
            if not result.get("ok"):
                return False, str(result.get("summary", "write failed"))[:200]
        elif self.trusted_root:
            await asyncio.to_thread(target.write_text, content, encoding="utf-8")
        else:
            return False, "no files tool and the folder is outside the projects folder"
        exists = await asyncio.to_thread(lambda: target.is_file() and target.stat().st_size > 0)
        return (True, "") if exists else (False, "file not found after writing")


async def build_project(app: Any, description: str, *, project_dir: str | Path, kind: str = "website",
                        name: str = "", progress: Callable[..., Any] | None = None,
                        cancel: asyncio.Event | None = None, source: str = "worker",
                        ladder: str | None = None) -> dict[str, Any]:
    """Generate and write the project. Returns {"ok", "project_dir", "files",
    "entry", "summary", "summary_ckb", "model", "errors"}; ``ok`` only when
    every written file was found on disk afterwards."""
    root = Path(project_dir).expanduser()
    kind = kind if kind in KIND_RULES else "other"
    ladder = ladder or str(app.config.get("worker.build_ladder", "strong") or "strong")
    writer = _Writer(app, root, source)
    written: dict[str, str] = {}
    errors: list[str] = []
    model = ""
    steps = 6

    def report(step: int, text_ckb: str) -> None:
        if progress is not None:
            try:
                progress(min(step, steps), steps, text_ckb)
            except Exception:  # noqa: BLE001
                log.exception("progress callback failed")

    async def save(path: str, content: str) -> None:
        if not content.strip():
            errors.append(f"{path}: empty")
            return
        if len(written) >= MAX_FILES:
            errors.append(f"{path}: too many files")
            return
        if len(content) > MAX_FILE_CHARS:
            errors.append(f"{path}: too large")
            return
        ok_write, why = await writer.write(path, content)
        if ok_write:
            written[path] = content
            report(1 + len(written), f"فایلی {path} نووسرا")
        else:
            errors.append(f"{path}: {why}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT + KIND_RULES[kind]},
                {"role": "user", "content": f"Project name: {name or root.name}\nWhat the user wants:\n{description}"}]
    report(1, "کۆدەکە دەنووسم")
    parser = FileBlockParser()
    finish_reason = None
    raw_text: list[str] = []
    try:
        async for chunk in app.llm.stream(messages, ladder=ladder, max_tokens=16000, reasoning="low", timeout_s=120):
            if cancel is not None and cancel.is_set():
                raise asyncio.CancelledError
            if chunk.kind == "text":
                raw_text.append(chunk.text)
                for path, content in parser.feed(chunk.text):
                    await save(path, content)
            elif chunk.kind == "done" and chunk.response is not None:
                finish_reason = chunk.response.finish_reason
                model = chunk.response.model_ref
    except LLMError as err:
        errors.append(f"model: {err.kind}")
    for path, content in parser.feed("\n"):
        await save(path, content)
    pending = parser.pending()
    todo: list[str] = []
    if pending is not None:
        if finish_reason in ("length", "max_tokens", "MAX_TOKENS"):
            todo.append(pending[0])
        else:
            await save(*pending)  # the model just forgot the last END marker
    if not written and not todo and kind == "website":
        fenced = re.search(r"```html\s*(.*?)```", "".join(raw_text), re.S | re.I)
        if fenced:
            await save("index.html", fenced.group(1).strip() + "\n")
    todo += [p for p in missing_references(written) if p not in todo]
    for path in todo[:MAX_REPAIRS]:
        if cancel is not None and cancel.is_set():
            break
        report(len(written) + 1, f"فایلی {path} تەواو دەکەم")
        content = await _single_file(app, messages, written, path, ladder)
        if content:
            await save(path, content)
        else:
            errors.append(f"{path}: could not be generated")
    entry = next((p for p in ("index.html", "main.py", "README.md") if p in written),
                 next(iter(written), None))
    ok_all = bool(written) and (kind != "website" or any(p.endswith(".html") for p in written))
    count = str(len(written)).translate(_DIGITS_CKB)
    summary_ckb = (f"پرۆژەکە بە {count} فایل دروست کرا." if ok_all
                   else "ببورە، نەمتوانی پرۆژەکە بە تەواوی دروست بکەم.")
    report(steps, summary_ckb)
    return {"ok": ok_all, "project_dir": str(root), "files": sorted(written), "entry": entry,
            "summary": (f"Wrote {len(written)} files: {', '.join(sorted(written))}" if written else "No files written")
                       + (f"; problems: {'; '.join(errors[:5])}" if errors else ""),
            "summary_ckb": summary_ckb, "model": model, "errors": errors[:10]}


async def _single_file(app: Any, messages: list[dict[str, Any]], written: dict[str, str], path: str,
                       ladder: str) -> str:
    """Generate one file that was cut off or is referenced but missing."""
    listing = "\n".join(f"- {p} ({len(c)} chars)" for p, c in written.items()) or "(none yet)"
    html = next((c for p, c in written.items() if p.endswith(".html")), "")
    ask = (f"Files already written:\n{listing}\n\n" + (f"index.html for reference:\n{html[:6000]}\n\n" if html else "")
           + f"Now write ONLY the complete file {path} in the same format (<<<FILE: {path}>>> ... <<<END FILE>>>).")
    try:
        response = await app.llm.chat(messages + [{"role": "user", "content": ask}], ladder=ladder,
                                      max_tokens=12000, reasoning="low", timeout_s=120)
    except LLMError:
        return ""
    parser = FileBlockParser()
    files = parser.feed(response.text + "\n")
    pending = parser.pending()
    for candidate, content in files + ([pending] if pending else []):
        if candidate == path and content.strip():
            return content
    text = strip_fences(response.text or "")
    return text if text.strip() and "<<<" not in text else ""


__all__ = ["build_project", "FileBlockParser", "safe_relpath", "strip_fences", "missing_references", "KIND_RULES"]
