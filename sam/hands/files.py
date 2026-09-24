"""Files in the user's folders: list, read, write, append, copy, move,
rename, delete (to the Recycle Bin), open, search, reveal.

Paths may start with a known-folder word in English or Sorani ("Desktop/…",
"دێسکتۆپ/…", "Downloads", "پرۆژەکان") -- see ``policy.FOLDER_ALIASES``; the
risk of each call is decided by ``Policy.classify_path`` before this code
runs. Writes are atomic (temp file + replace) and an overwritten file is
backed up first (v1 kept the same safety net), outside SAM_HOME because the
dev SAM_HOME is v1's folder.
"""

from __future__ import annotations

import ctypes
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

from ..textnorm import normalize_ckb

READ_MAX_BYTES = 2_000_000
READ_RETURN_CHARS = 5000
WRITE_MAX_BYTES = 2_000_000
LIST_MAX = 200
SEARCH_MAX = 60
SEARCH_BUDGET_S = 6.0
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "$recycle.bin", "appdata"}
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".html", ".css", ".xml", ".log", ".ini",
                 ".yaml", ".yml", ".toml", ".bat", ".ps1", ".sql", ".mq5", ".mqh", ".pine", ".tsx", ".jsx"}


def _human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024  # type: ignore[assignment]
    return f"{size} B"


def recycle(path: Path) -> None:
    """Delete to the Recycle Bin (recoverable) with SHFileOperationW."""
    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]
    FO_DELETE, FOF_ALLOWUNDO, FOF_NOCONFIRMATION, FOF_SILENT, FOF_NOERRORUI = 3, 0x40, 0x10, 0x4, 0x400
    op = SHFILEOPSTRUCTW(None, FO_DELETE, str(path) + "\0", None,
                         FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI, False, None, None)
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if result != 0 or op.fAnyOperationsAborted:
        raise OSError(f"the Recycle Bin refused the file (code {result})")


class Files:
    def __init__(self, policy: Any, *, backup_dir: Path | None = None, recycle_fn: Any = None,
                 startfile_fn: Any = None, popen_fn: Any = None) -> None:
        self.policy = policy
        self.backup_dir = backup_dir
        self._recycle = recycle_fn or recycle
        self._startfile = startfile_fn or (lambda p: os.startfile(p))  # type: ignore[attr-defined]
        self._popen = popen_fn or subprocess.Popen

    # -- helpers -----------------------------------------------------------------
    def _backup(self, path: Path) -> str | None:
        if not path.is_file() or self.backup_dir is None:
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        safe = re.sub(r"[^\w.-]+", "_", path.name)[:80]
        target = self.backup_dir / f"{int(time.time() * 1000)}-{digest}-{safe}.bak"
        shutil.copy2(path, target)
        return str(target)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    # -- actions (sync; the tool runs them in a thread) --------------------------------
    def run(self, action: str, path: str, *, content: str | None = None, dest: str | None = None,
            pattern: str | None = None) -> dict[str, Any]:
        handler = getattr(self, f"_{action}", None)
        if handler is None:
            return {"ok": False, "summary": f"Unknown file action '{action}'."}
        try:
            return handler(path, content=content, dest=dest, pattern=pattern)
        except FileNotFoundError as exc:
            return {"ok": False, "summary": f"Not found: {exc.filename or exc}"}
        except PermissionError as exc:
            return {"ok": False, "summary": f"Windows denied access: {exc.filename or exc}"}
        except (OSError, ValueError, UnicodeError) as exc:
            return {"ok": False, "summary": f"{action} failed: {type(exc).__name__}: {exc}"}

    def _list(self, raw: str, **_: Any) -> dict[str, Any]:
        root = self.policy.resolve(raw)
        if root.is_file():
            stat = root.stat()
            return {"ok": True, "summary": f"{root.name} is a file ({_human(stat.st_size)}).", "path": str(root),
                    "entries": [{"name": root.name, "type": "file", "size": stat.st_size}]}
        entries = []
        with os.scandir(root) as items:
            for item in sorted(items, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower())):
                if len(entries) >= LIST_MAX:
                    break
                try:
                    is_dir = item.is_dir(follow_symlinks=False)
                    entries.append({"name": item.name, "type": "folder" if is_dir else "file",
                                    **({} if is_dir else {"size": item.stat().st_size})})
                except OSError:
                    continue
        folders = sum(1 for e in entries if e["type"] == "folder")
        return {"ok": True, "path": str(root), "entries": entries, "truncated": len(entries) >= LIST_MAX,
                "summary": f"{root.name or root}: {folders} folders, {len(entries) - folders} files."}

    def _read(self, raw: str, **_: Any) -> dict[str, Any]:
        path = self.policy.resolve(raw)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        size = path.stat().st_size
        if size > READ_MAX_BYTES:
            return {"ok": False, "summary": f"{path.name} is {_human(size)}; SAM reads files up to {_human(READ_MAX_BYTES)}."}
        data = path.read_bytes()
        if b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
            return {"ok": False, "summary": f"{path.name} is a binary file, not text."}
        text = data.decode("utf-16") if data.startswith((b"\xff\xfe", b"\xfe\xff")) else data.decode(
            "utf-8-sig", errors="replace")
        shown = text[:READ_RETURN_CHARS]
        return {"ok": True, "path": str(path), "size": size, "chars": len(text), "truncated": len(text) > len(shown),
                "untrusted": shown, "summary": f"Read {path.name} ({len(text)} characters)."}

    def _write(self, raw: str, *, content: str | None = None, append: bool = False, **_: Any) -> dict[str, Any]:
        path = self.policy.resolve(raw)
        text = content or ""
        data = text.encode("utf-8")
        if len(data) > WRITE_MAX_BYTES:
            return {"ok": False, "summary": f"The content is larger than {_human(WRITE_MAX_BYTES)}."}
        if path.is_dir():
            return {"ok": False, "summary": f"{path} is a folder."}
        existed = path.exists()
        backup = self._backup(path) if existed else None
        if append and existed:
            with path.open("ab") as handle:
                handle.write(data)
        else:
            self._atomic_write(path, data)
        verified = path.exists() and (path.read_bytes().endswith(data) if data else True)
        verb = "Appended to" if append and existed else "Overwrote" if existed else "Created"
        return {"ok": verified, "path": str(path), "bytes": len(data), "backup": backup,
                "summary": f"{verb} {path.name}." if verified else f"Writing {path.name} could not be verified."}

    def _append(self, raw: str, **kwargs: Any) -> dict[str, Any]:
        return self._write(raw, append=True, **kwargs)

    def _transfer(self, raw: str, dest: str | None, action: str) -> dict[str, Any]:
        source = self.policy.resolve(raw)
        if not source.exists():
            raise FileNotFoundError(str(source))
        target = self.policy.dest_path(raw, dest or "", action)
        if target.is_dir() and action != "rename":
            target = target / source.name
        if target.exists():
            return {"ok": False, "summary": f"{target} already exists; choose another name."}
        target.parent.mkdir(parents=True, exist_ok=True)
        if action == "copy":
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        else:
            shutil.move(str(source), str(target))
        verified = target.exists() and (action == "copy" or not source.exists())
        verbs = {"copy": "Copied", "move": "Moved", "rename": "Renamed"}
        return {"ok": verified, "path": str(target), "summary": f"{verbs[action]} {source.name} to {target}."
                if verified else f"{verbs[action]} {source.name}: result could not be verified."}

    def _copy(self, raw: str, *, dest: str | None = None, **_: Any) -> dict[str, Any]:
        return self._transfer(raw, dest, "copy")

    def _move(self, raw: str, *, dest: str | None = None, **_: Any) -> dict[str, Any]:
        return self._transfer(raw, dest, "move")

    def _rename(self, raw: str, *, dest: str | None = None, **_: Any) -> dict[str, Any]:
        return self._transfer(raw, dest, "rename")

    def _delete(self, raw: str, **_: Any) -> dict[str, Any]:
        path = self.policy.resolve(raw)
        if not path.exists():
            raise FileNotFoundError(str(path))
        self._recycle(path)
        gone = not path.exists()
        return {"ok": gone, "path": str(path), "recoverable": True,
                "summary": f"Moved {path.name} to the Recycle Bin." if gone else f"{path.name} is still there."}

    def _open(self, raw: str, **_: Any) -> dict[str, Any]:
        path = self.policy.resolve(raw)
        if not path.exists():
            raise FileNotFoundError(str(path))
        self._startfile(str(path))
        return {"ok": True, "path": str(path), "summary": f"Opened {path.name}."}

    def _reveal(self, raw: str, **_: Any) -> dict[str, Any]:
        path = self.policy.resolve(raw)
        if not path.exists():
            raise FileNotFoundError(str(path))
        self._popen(["explorer.exe", f"/select,{path}"])
        return {"ok": True, "path": str(path), "summary": f"Showed {path.name} in File Explorer."}

    def _search(self, raw: str, *, pattern: str | None = None, content: str | None = None, **_: Any) -> dict[str, Any]:
        """Find names matching ``pattern`` (glob or words; Sorani ok) under a
        folder; with ``content``, also look inside small text files."""
        root = self.policy.resolve(raw)
        if not root.is_dir():
            raise FileNotFoundError(str(root))
        wanted = (pattern or "").strip()
        glob = wanted if any(ch in wanted for ch in "*?[") else None
        words = normalize_ckb(wanted, strip_punct=True).split() if wanted and not glob else []
        needle = (content or "").strip().lower()
        deadline = time.monotonic() + SEARCH_BUDGET_S
        hits: list[dict[str, Any]] = []
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith(".")]
            depth = len(Path(current).relative_to(root).parts)
            if depth >= 6:
                dirs[:] = []
            for name in [*dirs, *files]:
                full = Path(current) / name
                name_ok = (glob and fnmatch.fnmatch(name.lower(), glob.lower())) or (
                    words and all(w in normalize_ckb(name) for w in words)) or (not glob and not words)
                if not name_ok:
                    continue
                if needle:
                    if name in dirs or full.suffix.lower() not in TEXT_SUFFIXES:
                        continue
                    try:
                        if full.stat().st_size > 1_000_000 or needle not in full.read_text(
                                encoding="utf-8", errors="ignore").lower():
                            continue
                    except OSError:
                        continue
                hits.append({"path": str(full), "type": "folder" if name in dirs else "file"})
                if len(hits) >= SEARCH_MAX:
                    break
            if len(hits) >= SEARCH_MAX or time.monotonic() > deadline:
                break
        timed_out = time.monotonic() > deadline
        return {"ok": True, "root": str(root), "matches": hits, "truncated": len(hits) >= SEARCH_MAX or timed_out,
                "summary": f"Found {len(hits)} match(es) under {root.name or root}"
                           + (" (search stopped early)." if timed_out else ".")}


__all__ = ["Files", "recycle"]
