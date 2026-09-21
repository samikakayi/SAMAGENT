"""Workspace snapshots: the mechanics of undoing what a run changed.

Pure filesystem work, deliberately free of orchestration. It knows how to
snapshot a file before it is written, how to describe the difference between a
snapshot and what is on disk now, and how to put the snapshot back. It does
not know about task state, events, audit or approvals -- the orchestrator owns
those and calls in here for the mechanics.

The snapshot is taken from the working tree, not from git HEAD, because what a
rollback must restore is what the user actually had -- including their own
uncommitted edits -- and because a workspace need not be a repository at all.
"""

from __future__ import annotations

import difflib
import os
import shutil
from pathlib import Path
from typing import Any

# Only these tools change a file, so only these need a snapshot first.
MODIFYING_TOOLS = {"write_file", "replace_text", "delete_path"}


class WorkspaceCheckpoints:
    """Snapshot, diff and restore files inside one workspace."""

    def __init__(self, workspace: Path, snapshot_root: Path) -> None:
        self.workspace = Path(workspace)
        self.snapshot_root = Path(snapshot_root)

    # -- locating ----------------------------------------------------------
    def target_path(self, arguments: dict[str, Any]) -> Path | None:
        """Where a modifying tool call is about to write."""
        raw = str(arguments.get("path") or "").strip()
        if not raw:
            return None
        path = Path(os.path.expandvars(raw)).expanduser()
        return (path if path.is_absolute() else self.workspace / path).resolve(strict=False)

    def relative(self, path: str | Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.workspace.resolve()).as_posix()
        except ValueError:
            return str(path)

    # -- snapshotting ------------------------------------------------------
    def snapshot(self, existing: list[dict[str, Any]], tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Record a file's current contents, once, before the first change.

        Returns the new entry, or None when there is nothing to record --
        a non-modifying tool, no path, or a file already snapshotted by an
        earlier step of the same run.
        """
        if tool_name not in MODIFYING_TOOLS:
            return None
        target = self.target_path(arguments)
        if target is None or any(item["path"] == str(target) for item in existing):
            return None
        entry: dict[str, Any] = {"path": str(target), "existed": target.is_file(), "snapshot": None}
        if entry["existed"]:
            self.snapshot_root.mkdir(parents=True, exist_ok=True)
            destination = self.snapshot_root / f"{len(existing)}.bin"
            shutil.copyfile(target, destination)
            entry["snapshot"] = str(destination)
        return entry

    # -- reporting ---------------------------------------------------------
    def diff(self, checkpoints: list[dict[str, Any]], preexisting: list[str]) -> list[dict[str, Any]]:
        """Per-file unified diffs of this run's own changes.

        Computed against the snapshots rather than `git diff`, which would
        fold in whatever the user had already changed in the same files.
        """
        already_dirty = {item.replace("\\", "/") for item in preexisting}
        files: list[dict[str, Any]] = []
        for entry in checkpoints:
            target = Path(entry["path"])
            before = Path(entry["snapshot"]).read_bytes() if entry.get("snapshot") else b""
            after = target.read_bytes() if target.is_file() else b""
            relative = self.relative(entry["path"])
            unified = "".join(difflib.unified_diff(
                before.decode("utf-8", "replace").splitlines(keepends=True),
                after.decode("utf-8", "replace").splitlines(keepends=True),
                fromfile=f"a/{relative}" if entry["existed"] else "/dev/null",
                tofile=f"b/{relative}" if target.is_file() else "/dev/null",
            ))
            if entry["existed"] and not target.is_file():
                status = "deleted"
            elif not entry["existed"]:
                status = "created"
            else:
                status = "modified" if before != after else "unchanged"
            files.append({
                "path": relative,
                "status": status,
                "diff": unified,
                "had_user_changes": relative in already_dirty,
            })
        return files

    # -- undoing -----------------------------------------------------------
    def restore(self, checkpoints: list[dict[str, Any]]) -> list[str]:
        """Put every snapshotted file back, and remove files the run created."""
        restored: list[str] = []
        for entry in checkpoints:
            target = Path(entry["path"])
            if entry.get("snapshot"):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(entry["snapshot"], target)
            elif target.is_file():
                target.unlink()
            restored.append(self.relative(entry["path"]))
        return restored
