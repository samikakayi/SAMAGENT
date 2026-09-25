"""File actions in temp folders: atomic writes with backups, Recycle Bin
deletes (faked), copy/move/rename verified, search, size caps."""

from __future__ import annotations

from pathlib import Path

import pytest

from sam.hands.files import Files
from sam.hands.policy import Policy


@pytest.fixture
def env(tmp_path: Path):
    home = tmp_path / "home"
    folders = {name: home / name.capitalize() for name in ("desktop", "documents", "downloads", "pictures")}
    for folder in folders.values():
        folder.mkdir(parents=True)
    projects = home / "SAM Projects"
    projects.mkdir()
    folders.update({"projects": projects, "home": home})
    policy = Policy(home=home, projects_dir=projects, folders=folders)
    recycled: list[Path] = []

    def recycle(path: Path) -> None:
        recycled.append(path)
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
        else:
            path.unlink()
    opened: list[str] = []
    files = Files(policy, backup_dir=tmp_path / "backups", recycle_fn=recycle, startfile_fn=opened.append,
                  popen_fn=lambda argv: opened.append(" ".join(argv)))
    return files, folders, recycled, opened, tmp_path


def test_write_read_append_with_sorani_text(env) -> None:
    files, folders, *_ = env
    result = files.run("write", "Desktop/تێبینی.txt", content="سڵاو\nhello")
    assert result["ok"] and "Created" in result["summary"]
    assert (folders["desktop"] / "تێبینی.txt").read_text(encoding="utf-8") == "سڵاو\nhello"
    result = files.run("append", "دێسکتۆپ/تێبینی.txt", content="\nزیاتر")
    assert result["ok"] and "Appended" in result["summary"]
    read = files.run("read", "Desktop/تێبینی.txt")
    assert read["ok"] and read["untrusted"] == "سڵاو\nhello\nزیاتر"


def test_overwrite_keeps_a_backup_outside_sam_home(env) -> None:
    files, folders, _, _, tmp = env
    target = folders["documents"] / "plan.md"
    target.write_text("old plan", encoding="utf-8")
    result = files.run("write", "Documents/plan.md", content="new plan")
    assert result["ok"] and "Overwrote" in result["summary"]
    backup = Path(result["backup"])
    assert backup.parent == tmp / "backups" and backup.read_text(encoding="utf-8") == "old plan"


def test_list_search_and_reveal(env) -> None:
    files, folders, _, opened, _ = env
    (folders["downloads"] / "report 2026.pdf").write_bytes(b"%PDF")
    (folders["downloads"] / "notes.txt").write_text("gold at 2700", encoding="utf-8")
    (folders["downloads"] / "sub").mkdir()
    listing = files.run("list", "Downloads")
    assert listing["ok"] and [e["name"] for e in listing["entries"]][0] == "sub"
    assert "1 folders, 2 files" in listing["summary"]
    assert [m["path"].endswith("report 2026.pdf") for m in files.run("search", "Downloads", pattern="*.pdf")["matches"]] == [True]
    assert len(files.run("search", "Downloads", pattern="report")["matches"]) == 1
    hits = files.run("search", "Downloads", content="gold")
    assert hits["ok"] and len(hits["matches"]) == 1 and hits["matches"][0]["path"].endswith("notes.txt")
    assert files.run("reveal", "Downloads/notes.txt")["ok"]
    assert opened[-1].startswith("explorer.exe /select,")


def test_copy_move_rename_are_verified(env) -> None:
    files, folders, *_ = env
    (folders["desktop"] / "a.txt").write_text("A", encoding="utf-8")
    assert files.run("copy", "Desktop/a.txt", dest="Projects")["ok"]
    assert (folders["projects"] / "a.txt").exists() and (folders["desktop"] / "a.txt").exists()
    assert files.run("rename", "Projects/a.txt", dest="b.txt")["ok"]
    assert (folders["projects"] / "b.txt").exists()
    assert files.run("move", "Projects/b.txt", dest="Documents/b.txt")["ok"]
    assert (folders["documents"] / "b.txt").exists() and not (folders["projects"] / "b.txt").exists()
    clash = files.run("copy", "Desktop/a.txt", dest="Documents/b.txt")
    assert not clash["ok"] and "already exists" in clash["summary"]


def test_delete_goes_to_the_recycle_bin(env) -> None:
    files, folders, recycled, *_ = env
    (folders["desktop"] / "old.txt").write_text("x", encoding="utf-8")
    result = files.run("delete", "Desktop/old.txt")
    assert result["ok"] and result["recoverable"] and recycled == [folders["desktop"] / "old.txt"]
    missing = files.run("delete", "Desktop/nothing.txt")
    assert not missing["ok"] and "Not found" in missing["summary"]


def test_size_caps_and_binary_files(env, monkeypatch) -> None:
    files, folders, *_ = env
    (folders["pictures"] / "photo.bin").write_bytes(b"\x89PNG\x00\x00binary")
    assert "binary" in files.run("read", "Pictures/photo.bin")["summary"]
    big = folders["documents"] / "big.txt"
    big.write_text("x" * 12_000, encoding="utf-8")
    read = files.run("read", "Documents/big.txt")
    assert read["ok"] and read["truncated"] and len(read["untrusted"]) == 5000
    import sam.hands.files as module
    monkeypatch.setattr(module, "WRITE_MAX_BYTES", 10)
    assert not files.run("write", "Projects/x.txt", content="0123456789ABC")["ok"]


def test_open_uses_the_default_app(env) -> None:
    files, folders, _, opened, _ = env
    (folders["pictures"] / "a.jpg").write_bytes(b"\xff\xd8")
    assert files.run("open", "Pictures/a.jpg")["ok"] and opened[-1].endswith("a.jpg")
    assert not files.run("frobnicate", "x")["ok"]
