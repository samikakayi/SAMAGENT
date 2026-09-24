"""Start-menu app index + launching (open_app).

Measured on this PC (2026-09-24): enumerating ``shell:AppsFolder`` through
the Shell COM object lists all 195 Start-menu apps in ~0.9 s (plus ~0.2 s for
the pywin32 import); reading each shortcut's target path adds ~0.6 s. That is
too slow for a spoken command, so the index is built in the background at
start-up and cached in the DB (table ``hands_apps``) -- never as a file,
because the dev SAM_HOME is v1's folder where SAM 2 may only write its DB.

AppIDs come in three shapes, each launched differently:
- packaged (MSIX/Store) ``Family!App`` -> ``IApplicationActivationManager``
  (``sam.winapp.activate_aumid``) -- the only way that passes arguments
  (TradingView needs ``--remote-debugging-port``) and returns the pid;
- registered desktop AUMIDs (``Chrome``, ``MSEdge``, ``Telegram.TelegramDesktop``)
  -> ``shell:AppsFolder\\<id>`` (exactly what the Start menu does); with
  arguments, the shortcut's target exe is started directly;
- known-folder paths (``{6D809377-...}\\MetaTrader 5\\terminal64.exe``) -> the
  expanded exe path.

Every launch is verified by waiting for a window of the app (or its process)
before reporting success.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..textnorm import normalize_ckb
from . import APPS_SCHEMA, _win
from .aliases import ALIASES, AppAlias, clean_query, match_alias, query_variants, transliterate

log = logging.getLogger("sam.hands.apps")

SCHEMA = APPS_SCHEMA  # table hands_apps (defined in sam/hands/__init__.py)


@dataclass
class AppEntry:
    """One launchable app. ``aumid`` is the AppsFolder id (packaged or
    registered); ``path`` the exe/URI/URL used when there is no id."""

    name: str
    aumid: str | None = None
    path: str | None = None
    aliases: tuple[str, ...] = ()
    kind: str = "desktop"            # packaged | desktop | exe | uri | url
    processes: tuple[str, ...] = ()  # exe basenames of its windows (lower case)
    reuse_window: bool = True
    alias_key: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "aumid": self.aumid, "path": self.path, "kind": self.kind}


def _kind_of(app_id: str) -> str:
    if "!" in app_id:
        return "packaged"
    if app_id.startswith("{") or ":\\" in app_id:
        return "exe"
    return "desktop"


def enumerate_start_apps(with_targets: bool = True) -> list[dict[str, Any]]:
    """Read ``shell:AppsFolder`` (Get-StartApps equivalent). Runs on a COM
    (STA) worker thread; returns plain dicts."""
    import win32com.client  # pywin32; imported on the worker only

    shell = win32com.client.Dispatch("Shell.Application")
    folder = shell.NameSpace("shell:AppsFolder")
    rows: list[dict[str, Any]] = []
    for item in folder.Items():
        name, app_id = str(item.Name or "").strip(), str(item.Path or "").strip()
        if not name or not app_id:
            continue
        target = args = None
        if with_targets and "!" not in app_id:
            try:
                target = item.ExtendedProperty("System.Link.TargetParsingPath") or None
                args = item.ExtendedProperty("System.Link.Arguments") or None
            except Exception:  # noqa: BLE001 - some shell items refuse property reads
                target = args = None
        if target is None and app_id.startswith("{"):
            target = _win.expand_known_folder_path(app_id)
        rows.append({"app_id": app_id, "name": name, "target": target, "target_args": args})
    return rows


class AppIndex:
    """Cached Start-menu index + alias resolution + verified launching."""

    def __init__(self, app: Any, *, windows: Any, enumerate_fn: Callable[[], list[dict[str, Any]]] | None = None,
                 activate_fn: Callable[[str, str], int] | None = None,
                 startfile_fn: Callable[[str], None] | None = None,
                 popen_fn: Callable[..., Any] | None = None) -> None:
        self.app = app
        self.windows = windows
        self._enumerate = enumerate_fn or enumerate_start_apps
        self._activate = activate_fn or self._default_activate
        self._startfile = startfile_fn or (lambda target: os.startfile(target))  # type: ignore[attr-defined]
        self._popen = popen_fn or subprocess.Popen
        self._rows: list[dict[str, Any]] | None = None
        self._refresh_task: asyncio.Task[Any] | None = None
        self._worker = _win.Worker("sam-apps", _win.com_sta_initializer)
        self.last_refresh: dict[str, Any] = {}

    # -- index -----------------------------------------------------------------
    @staticmethod
    def _default_activate(aumid: str, arguments: str) -> int:
        from ..winapp import activate_aumid

        return activate_aumid(aumid, arguments)

    def _load_cached(self) -> list[dict[str, Any]]:
        if self._rows is None:
            try:
                self._rows = self.app.db.query("SELECT app_id, name, target, target_args FROM hands_apps")
            except Exception:  # noqa: BLE001
                self._rows = []
        return self._rows

    def _store(self, rows: list[dict[str, Any]]) -> None:
        now = time.time()
        with self.app.db.transaction() as conn:
            conn.execute("DELETE FROM hands_apps")
            conn.executemany(
                "INSERT OR REPLACE INTO hands_apps(app_id, name, target, target_args, updated_at) VALUES (?,?,?,?,?)",
                [(r["app_id"], r["name"], r.get("target"), r.get("target_args"), now) for r in rows])

    async def refresh(self) -> int:
        """Rebuild the index (~1.5 s on a COM thread) and cache it in the DB."""
        started = time.perf_counter()
        rows = await self._worker.run(self._enumerate)
        if rows:
            self._rows = rows
            await asyncio.to_thread(self._store, rows)
        took = (time.perf_counter() - started) * 1000.0
        self.last_refresh = {"at": time.time(), "count": len(rows), "ms": round(took)}
        try:
            self.app.timing.record("hands:app_index", took, kind="startup", count=len(rows))
        except Exception:  # noqa: BLE001
            pass
        return len(rows)

    def refresh_in_background(self) -> None:
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = self.app.spawn(self.refresh(), "hands-app-index")

    async def _ready(self) -> list[dict[str, Any]]:
        rows = self._load_cached()
        if rows:
            return rows
        task = self._refresh_task
        if task is None or task.done():
            await self.refresh()
        else:
            try:
                await asyncio.wait_for(asyncio.shield(task), 8.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
        return self._rows or []

    def entries(self) -> list[AppEntry]:
        return [self._entry(r) for r in self._load_cached()]

    def _entry(self, row: dict[str, Any], spec: AppAlias | None = None) -> AppEntry:
        app_id = row["app_id"]
        kind = _kind_of(app_id)
        target = row.get("target")
        processes = spec.processes if spec else ()
        if not processes and target and str(target).lower().endswith(".exe"):
            processes = (os.path.basename(str(target)).lower(),)
        return AppEntry(name=row["name"], aumid=app_id if kind != "exe" else None,
                        path=target if target and not str(target).startswith("::") else None,
                        aliases=spec.aliases if spec else (), kind=kind, processes=processes,
                        reuse_window=spec.reuse_window if spec else True, alias_key=spec.key if spec else "")

    # -- resolution --------------------------------------------------------------
    def _rows_named(self, rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
        wanted = name.lower()
        return [r for r in rows if r["name"].lower() == wanted]

    def _pick_for_alias(self, spec: AppAlias, rows: list[dict[str, Any]]) -> AppEntry | None:
        preferred: list[str] = []
        if spec.key == "tradingview":
            tv = self.app.config.get("trading.tv_aumid")
            if tv:
                preferred.append(str(tv))
        preferred.extend(spec.app_ids)
        for name in spec.names:
            matches = self._rows_named(rows, name)
            if not matches:
                continue
            for prefix in preferred:
                for row in matches:
                    if row["app_id"] == prefix or row["app_id"].startswith(prefix):
                        return self._entry(row, spec)
            return self._entry(matches[0], spec)
        # Broker/OEM builds carry a prefix ("Exness MetaTrader 5"). Matching by
        # exe name instead is wrong: Tor Browser also ships a firefox.exe.
        for name in spec.names:
            wanted = name.lower()
            for row in rows:
                if len(wanted) >= 5 and wanted in row["name"].lower():
                    return self._entry(row, spec)
        if spec.url:
            return AppEntry(spec.display, path=spec.url, kind="url", aliases=spec.aliases, reuse_window=False,
                            alias_key=spec.key)
        if spec.uri:
            return AppEntry(spec.display, path=spec.uri, kind="uri", aliases=spec.aliases,
                            processes=spec.processes, alias_key=spec.key)
        if spec.exe:
            return AppEntry(spec.display, path=spec.exe, kind="exe", aliases=spec.aliases,
                            processes=spec.processes, reuse_window=spec.reuse_window, alias_key=spec.key)
        return None

    def _user_aliases(self) -> dict[str, str]:
        value = self.app.config.get("hands.app_aliases", {}) or {}
        if not isinstance(value, dict):
            return {}
        table: dict[str, str] = {}
        for key, target in value.items():
            table[normalize_ckb(str(key), strip_punct=True)] = str(target)
            table.setdefault(clean_query(str(key)), str(target))
        return table

    async def resolve(self, name: str) -> AppEntry | None:
        """Best app for a spoken/typed name (Sorani or English), or None."""
        entry, _ = await self.resolve_with_candidates(name)
        return entry

    async def resolve_with_candidates(self, name: str, _depth: int = 0) -> tuple[AppEntry | None, list[str]]:
        from rapidfuzz import fuzz, process

        rows = [r for r in await self._ready() if "uninstall" not in r["name"].lower()]
        variants = query_variants(name)
        user = self._user_aliases() if _depth < 3 else {}
        for query in [normalize_ckb(name, strip_punct=True), *variants]:
            if query in user and normalize_ckb(user[query]) != query:
                return await self.resolve_with_candidates(user[query], _depth + 1)
        found = match_alias(name)
        if found is not None:
            entry = self._pick_for_alias(found[0], rows)
            if entry is not None:
                return entry, []
        names = [r["name"] for r in rows]
        if not names:
            return None, []
        best: tuple[str, float] | None = None
        for query in variants:
            latin = query if query.isascii() else transliterate(query)
            threshold = 86.0 if query.isascii() else 78.0
            if len(latin) < 3:
                continue
            hit = process.extractOne(latin, names, scorer=fuzz.WRatio, processor=lambda s: s.lower())
            if hit is not None and hit[1] >= threshold and (best is None or hit[1] > best[1]):
                # WRatio over-rewards tiny substrings ("pad" in many names): also
                # require a decent plain ratio for short queries.
                if len(latin) >= 6 or fuzz.ratio(latin, hit[0].lower()) >= 70 or latin in hit[0].lower().split():
                    best = (hit[0], float(hit[1]))
        if best is not None:
            row = next(r for r in rows if r["name"] == best[0])
            return self._entry(row), []
        query = variants[-1]
        latin = query if query.isascii() else transliterate(query)
        suggestions = [h[0] for h in process.extract(latin, names, scorer=fuzz.WRatio, limit=3)] if latin else []
        return None, suggestions

    # -- launching -----------------------------------------------------------------
    async def launch(self, name_or_entry: str | AppEntry, args: str = "", *, new_window: bool = False,
                     wait_s: float | None = None, direct: bool = False,
                     confirm: Callable[..., Any] | None = None) -> dict[str, Any]:
        """Open (or focus) an app and verify a window appeared.

        ``confirm`` (the tool's ``ctx.confirm``) is only used for TradingView:
        a TradingView running without SAM's DevTools port must be restarted,
        and the chart bridge asks the user first.

        Returns ``{"ok", "summary", "state": focused|started|delegated|failed, "window", "entry", ...}``.
        """
        if isinstance(name_or_entry, AppEntry):
            entry: AppEntry | None = name_or_entry
            suggestions: list[str] = []
        else:
            entry, suggestions = await self.resolve_with_candidates(name_or_entry)
        if entry is None:
            hint = f" Closest names: {', '.join(suggestions)}." if suggestions else ""
            return {"ok": False, "state": "not_found", "summary": f"No installed app matches '{name_or_entry}'.{hint}",
                    "suggestions": suggestions}
        wait_s = float(wait_s if wait_s is not None else self.app.config.get("hands.launch_wait_s", 12))
        blocked = self._args_blocked(entry, args)
        if blocked:
            return {"ok": False, "state": "blocked", "blocked": True, "summary": blocked, "entry": entry.as_dict()}

        tv_aumid = str(self.app.config.get("trading.tv_aumid") or "")
        if not direct and (entry.alias_key == "tradingview" or (entry.aumid and tv_aumid and entry.aumid == tv_aumid)):
            return await self._launch_tradingview(entry, args, wait_s, confirm)

        if entry.kind == "url":
            await asyncio.to_thread(self._startfile, entry.path or "")
            return {"ok": True, "state": "started", "summary": f"Opened {entry.name} in the browser.",
                    "entry": entry.as_dict()}

        if entry.reuse_window and not new_window and not args and entry.processes:
            existing = await self.windows.find_by_process(entry.processes)
            if existing is not None:
                focused = await self.windows.focus(existing.hwnd)
                if focused:
                    return {"ok": True, "state": "focused", "summary": f"{entry.name} was already open; brought it to the front.",
                            "window": existing.title, "entry": entry.as_dict()}

        before = {w.hwnd for w in await self.windows.list()}
        started = time.perf_counter()
        try:
            pid = await self._start(entry, args)
        except Exception as exc:  # noqa: BLE001 - honest failure result
            return {"ok": False, "state": "failed", "summary": f"Could not start {entry.name}: {type(exc).__name__}: {exc}",
                    "entry": entry.as_dict()}
        window = await self._wait_for_window(entry, pid, before, wait_s)
        took = round((time.perf_counter() - started) * 1000)
        if window is not None:
            await self.windows.focus(window.hwnd)
            return {"ok": True, "state": "started", "summary": f"{entry.name} is open.", "window": window.title,
                    "hwnd": window.hwnd, "pid": window.pid, "ms": took, "entry": entry.as_dict()}
        running = bool(pid and _win.pid_alive(pid)) or await self._process_running(entry)
        if running:
            return {"ok": True, "state": "started", "verified": False, "ms": took, "entry": entry.as_dict(),
                    "summary": f"{entry.name} started but no window appeared within {wait_s:.0f} s "
                               "(it may be starting slowly or sit in the tray)."}
        return {"ok": False, "state": "failed", "ms": took, "entry": entry.as_dict(),
                "summary": f"Asked Windows to start {entry.name}, but no window or process appeared."}

    @staticmethod
    def _args_blocked(entry: AppEntry, args: str) -> str | None:
        """Second check on the RESOLVED program (the tool's classifier only saw
        the spoken name; fuzzy matching can turn 'terminal' into a shell):
        script hosts never get arguments, shells get run_powershell's rules."""
        if not (args or "").strip():
            return None
        from .guards import SCRIPT_HOSTS, SHELL_NAMES
        from .policy import classify_powershell

        exe = Path(entry.path or "").name.lower()
        if exe in SCRIPT_HOSTS:
            return f"{entry.name} is a script host; SAM never starts it with arguments."
        if exe in SHELL_NAMES or exe.removesuffix(".exe") in SHELL_NAMES:
            risk, reason = classify_powershell(args)
            if risk == "blocked":
                return reason
        return None

    async def _start(self, entry: AppEntry, args: str) -> int:
        """Start the app; returns a pid when Windows gives one (else 0)."""
        if entry.kind == "packaged" and entry.aumid:
            return int(await asyncio.to_thread(self._activate, entry.aumid, args) or 0)
        if entry.kind == "uri" and entry.path:
            await asyncio.to_thread(self._startfile, entry.path)
            return 0
        exe = entry.path if entry.path and entry.path.lower().endswith(".exe") else None
        if args and exe:
            argv = [exe, *shlex.split(args, posix=False)]
            proc = await asyncio.to_thread(self._popen, argv, cwd=str(Path(exe).parent) if os.path.isabs(exe) else None,
                                           env=_win.launch_environment(), stdin=subprocess.DEVNULL,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return int(getattr(proc, "pid", 0) or 0)
        if entry.kind == "desktop" and entry.aumid:
            await asyncio.to_thread(self._startfile, f"shell:AppsFolder\\{entry.aumid}")
            return 0
        if exe and os.path.isabs(exe):
            proc = await asyncio.to_thread(self._popen, [exe], cwd=str(Path(exe).parent), env=_win.launch_environment(),
                                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return int(getattr(proc, "pid", 0) or 0)
        if entry.path:
            await asyncio.to_thread(self._startfile, entry.path)
            return 0
        raise OSError("nothing to launch")

    async def _wait_for_window(self, entry: AppEntry, pid: int, before: set[int], wait_s: float) -> Any:
        deadline = time.monotonic() + wait_s
        processes = set(entry.processes)
        while time.monotonic() < deadline:
            windows = await self.windows.list()
            for window in windows:
                if pid and window.pid == pid and (window.hwnd not in before or entry.kind == "packaged"):
                    return window
            for window in windows:
                if window.hwnd not in before and processes and window.process.lower() in processes:
                    return window
            if pid:
                # Packaged apps often hand off to an existing process: accept a
                # foreground window of the app even if it is not new.
                fg = next((w for w in windows if w.foreground), None)
                if fg is not None and (fg.pid == pid or fg.process.lower() in processes):
                    return fg
            await asyncio.sleep(0.2)
        return None

    async def _process_running(self, entry: AppEntry) -> bool:
        if not entry.processes:
            return False
        wanted = set(entry.processes)
        running = await asyncio.to_thread(_win.running_processes)
        return any(name.lower() in wanted for name in running.values())

    async def _launch_tradingview(self, entry: AppEntry, args: str, wait_s: float,
                                  confirm: Callable[..., Any] | None = None) -> dict[str, Any]:
        """TradingView must run with its local DevTools port so the chart tools
        work: delegate to the chart bridge (contract 3.3). A TradingView that
        runs WITHOUT the port is restarted only after the user agrees: the
        bridge asks through ``confirm``; without a confirm callback it is never
        restarted (``needs_restart`` is reported instead)."""
        tv = getattr(self.app.trading, "tv", None)
        if tv is not None:
            try:
                result = await tv.ensure_running(allow_restart=confirm is not None, confirm=confirm, focus=True)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "state": "failed", "summary": f"TradingView bridge failed: {type(exc).__name__}: {exc}"}
            state = str(result.get("state", ""))
            window = await self.windows.find_by_process(("tradingview.exe",))
            data = {"state": "delegated", "tv_state": state, "ms": result.get("ms"),
                    "window": getattr(window, "title", None)}
            if result.get("ok"):
                what = {"connected": "TradingView is open and SAM's chart connection works",
                        "started": "Started TradingView with SAM's chart connection",
                        "restarted": "Restarted TradingView with SAM's chart connection"}.get(state, f"TradingView: {state}")
                return {"ok": True, "summary": what + ".", **data}
            if state == "needs_restart":
                if window is not None and window.minimized:
                    await self.windows.focus(window.hwnd)  # at least show it to the user
                why = ("the user did not approve the restart" if result.get("declined")
                       else "a restart is needed and nobody could be asked")
                return {"ok": False, "declined": bool(result.get("declined")), **data,
                        "summary": f"TradingView is open but without SAM's chart connection ({why}); "
                                   "the chart tools will not work until it is restarted with tv_open."}
            return {"ok": False, **data,
                    "summary": f"TradingView could not be opened: {result.get('detail') or state}."}
        port = int(self.app.config.get("trading.tv_port", 9222) or 9222)
        flag = f"--remote-debugging-port={port}"
        combined = f"{flag} {args}".strip() if flag not in args else args
        forced = AppEntry(entry.name, aumid=str(self.app.config.get("trading.tv_aumid") or entry.aumid),
                          kind="packaged", processes=("tradingview.exe",), alias_key="")
        existing = await self.windows.find_by_process(("tradingview.exe",))
        if existing is not None:
            await self.windows.focus(existing.hwnd)
            return {"ok": True, "state": "focused", "window": existing.title,
                    "summary": "TradingView was already open; brought it to the front."}
        return await self.launch(forced, combined, wait_s=wait_s, direct=True)


__all__ = ["AppEntry", "AppIndex", "SCHEMA", "enumerate_start_apps", "ALIASES"]
