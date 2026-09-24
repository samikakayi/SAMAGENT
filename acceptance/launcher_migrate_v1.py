"""Live: import the REAL SAM v1 database (read-only) into a TEMP SAM 2 database.

Never writes ``<SAM_HOME>\\data\\sam2.sqlite3`` and never writes v1's file:
the target is a throw-away home under %TEMP%, and the check proves the v1
connection is read-only and leaves no side files behind. Prints the counts.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

from _common import Acceptance, sam_home


def main() -> int:
    acc = Acceptance("launcher_migrate_v1")
    v1 = sam_home() / "data" / "sam.sqlite3"
    if not v1.is_file():
        acc.skip(f"no v1 database at {v1}")
        return acc.finish()

    from sam import migrate_v1
    from sam.app import App

    temp = Path(tempfile.mkdtemp(prefix="sam2-migrate-"))
    app = None
    try:
        (temp / "data").mkdir()
        app = App(temp, environ={})      # no keys: this check needs none
        app.load_packages(["sam.migrate_v1"])
        sides_before = {p.name for p in v1.parent.glob(v1.name + "-*")}
        main_mtime = v1.stat().st_mtime

        with acc.check("first import into a temp SAM 2 database") as c:
            began = time.perf_counter()
            report = migrate_v1.run_migration(app, v1)
            c.data = {"ms": round((time.perf_counter() - began) * 1000), "read": report["read"],
                      "strategy_cards": report["strategy_cards"], "card_versions": report["card_versions"],
                      "archived": report["archived"], "skipped": report["skipped"], "settings": report["settings"]}
            assert report["ok"], report
            c.detail = (f"{report['strategy_cards']} cards ({report['card_versions']} versions), "
                        f"{sum(report['archived'].values())} archive rows, "
                        f"{report['skipped']['messages_key_shaped']} key-shaped messages skipped")

        with acc.check("every imported card is archived with the v1 note") as c:
            cards = app.db.query("SELECT id, status, note, version FROM strategy_cards")
            c.data["cards"] = cards
            assert cards and all(r["status"] == "archived" for r in cards)
            assert all(r["note"] == migrate_v1.CARD_NOTE for r in cards)

        with acc.check("second run adds nothing") as c:
            again = migrate_v1.run_migration(app, v1, force=True)
            c.data = {"cards": again["strategy_cards"], "versions": again["card_versions"],
                      "archive": sum(again["archived"].values())}
            assert again["strategy_cards"] == 0 and again["card_versions"] == 0 and sum(again["archived"].values()) == 0

        with acc.check("v1 connection is read-only") as c:
            conn = migrate_v1.open_v1_readonly(v1)
            try:
                try:
                    conn.execute("CREATE TABLE sam2_probe (x)")
                    raise AssertionError("a write through the v1 connection succeeded")
                except sqlite3.OperationalError as exc:
                    c.detail = str(exc)
            finally:
                conn.close()

        with acc.check("v1 file state before/after (informational)") as c:
            # v1 may be running and legitimately checkpoint its file or create/remove its
            # own -wal/-shm meanwhile, so this is reported, not asserted. That SAM 2 itself
            # creates no side files is proven in tests/test_launcher_migrate_v1.py.
            sides_after = {p.name for p in v1.parent.glob(v1.name + "-*")}
            c.data = {"side_files_before": sorted(sides_before), "side_files_after": sorted(sides_after),
                      "main_file_mtime_unchanged": v1.stat().st_mtime == main_mtime}
            c.detail = "unchanged" if sides_after == sides_before and c.data["main_file_mtime_unchanged"] \
                else "changed (v1 is probably running and writing its own file)"
    finally:
        if app is not None:
            app.close()
        shutil.rmtree(temp, ignore_errors=True)
    return acc.finish()


if __name__ == "__main__":
    sys.exit(main())
