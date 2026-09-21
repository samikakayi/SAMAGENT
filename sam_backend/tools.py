from __future__ import annotations

import json
import hashlib
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .browser_automation import BrowserAutomationError, run_browser_workflow
from .cancellation import CancellationManager
from .capabilities import CapabilityRegistry
from .config import Settings
from .contracts import ExecutionStatus, ToolManifest
from .db import Database
from .policy import RiskPolicy, redact_secrets, windows_path_violation
from .project_map import ProjectScanner, run_git
from .verification import VerificationEngine
from .trading.drawing import Layer, SEMANTIC_TYPES, TWO_ANCHOR_TYPES
from .trading.service import TradingService
from .windows_control import WindowsController


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    sensitive: bool = False
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_text(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, default=str)


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


class ToolRegistry:
    """Deterministic broker for SAM's local capabilities."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        *,
        trading: TradingService | None = None,
        windows: WindowsController | None = None,
        cancellation: CancellationManager | None = None,
        scanner: ProjectScanner | None = None,
        verifier: VerificationEngine | None = None,
        capabilities: CapabilityRegistry | None = None,
    ):
        self.settings = settings
        self.database = database
        self.trading = trading
        self.windows = windows
        self.cancellation = cancellation
        self.workspace = settings.workspace_root.resolve()
        # Engineering collaborators are shared with the orchestrator so the
        # project map is scanned once per change, not once per tool call.
        self.scanner = scanner or ProjectScanner()
        self.verifier = verifier or VerificationEngine()
        self.capabilities = capabilities or CapabilityRegistry()
        self._handlers: dict[str, Callable[[dict[str, Any], bool], ToolResult]] = {
            "list_files": self._list_files,
            "search_files": self._search_files,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "replace_text": self._replace_text,
            "delete_path": self._delete_path,
            "run_terminal": self._run_terminal,
            "run_python": self._run_python,
            "open_url": self._open_url,
            "browser_automate": self._browser_automate,
            "launch_app": self._launch_app,
            "remember": self._remember,
            "recall": self._recall,
            "create_plan": self._create_plan,
            "project_map": self._project_map,
            "git_status": self._git_status,
            "git_diff": self._git_diff,
            "git_log": self._git_log,
            "run_tests": self._run_tests,
            "list_capabilities": self._list_capabilities,
        }
        if trading is not None:
            self._handlers.update({
                "market_snapshot": self._market_snapshot,
                "analyze_market": self._analyze_market,
                "get_tradingview_state": self._get_tradingview_state,
                "focus_tradingview": self._focus_tradingview,
                "set_tradingview_symbol": self._set_tradingview_symbol,
                "set_tradingview_timeframe": self._set_tradingview_timeframe,
                "save_custom_theory": self._save_custom_theory,
                "list_theories": self._list_theories,
                "calibrate_chart": self._calibrate_chart,
                "verify_chart_calibration": self._verify_chart_calibration,
                "draw_tradingview_level": self._draw_tradingview_level,
                "draw_tradingview_object": self._draw_tradingview_object,
                "draw_analysis_on_chart": self._draw_analysis_on_chart,
                "list_sam_drawings": self._list_sam_drawings,
                "clear_sam_drawings": self._clear_sam_drawings,
                "set_chart_layer": self._set_chart_layer,
                "list_entry_triggers": self._list_entry_triggers,
                "backtest_strategy": self._backtest_strategy,
            })
        if windows is not None:
            self._handlers.update({
                "list_processes": self._list_processes,
                "process_info": self._process_info,
                "stop_process": self._stop_process,
                "list_windows": self._list_windows,
                "window_action": self._window_action,
                "list_installed_apps": self._list_installed_apps,
                "read_clipboard": self._read_clipboard,
                "write_clipboard": self._write_clipboard,
                "capture_screen": self._capture_screen,
                "inspect_ui_tree": self._inspect_ui_tree,
            })
        if cancellation is not None:
            self._handlers["emergency_stop"] = self._emergency_stop

    @property
    def specs(self) -> list[dict[str, Any]]:
        definitions = [
            ("list_files", "List a workspace directory without reading file contents.", _schema({
                "path": {"type": "string", "default": "."},
                "max_depth": {"type": "integer", "minimum": 0, "maximum": 8, "default": 2},
            })),
            ("search_files", "Search text across workspace files.", _schema({
                "query": {"type": "string"}, "path": {"type": "string", "default": "."},
                "glob": {"type": "string", "default": "*"}, "case_sensitive": {"type": "boolean", "default": False},
            }, ["query"])),
            ("read_file", "Read a UTF-8 text file with optional line range.", _schema({
                "path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            }, ["path"])),
            ("write_file", "Atomically create or overwrite a UTF-8 text file.", _schema({
                "path": {"type": "string"}, "content": {"type": "string"},
                "create_parents": {"type": "boolean", "default": True},
            }, ["path", "content"])),
            ("replace_text", "Replace an exact text fragment in one file.", _schema({
                "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
                "expected_replacements": {"type": "integer", "minimum": 1, "default": 1},
            }, ["path", "old_text", "new_text"])),
            ("delete_path", "Permanently delete one file or directory. Always requires approval.", _schema({
                "path": {"type": "string"}, "recursive": {"type": "boolean", "default": False},
            }, ["path"])),
            ("run_terminal", "Run a PowerShell command on Windows (or a POSIX shell elsewhere).", _schema({
                "command": {"type": "string"}, "cwd": {"type": "string", "default": "."},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
            }, ["command"])),
            ("run_python", "Run Python code in an isolated interpreter process. Always requires approval.", _schema({
                "code": {"type": "string"}, "cwd": {"type": "string", "default": "."},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
            }, ["code"])),
            ("open_url", "Open an http(s) URL in the user's default browser.", _schema({"url": {"type": "string"}}, ["url"])),
            ("browser_automate", "Run a bounded browser workflow in a fresh profile with no saved cookies or passwords. Interactive actions require approval.", _schema({
                "url": {"type": "string"},
                "actions": {
                    "type": "array", "minItems": 1, "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["goto", "click", "fill", "press", "wait_for", "extract_text", "screenshot"]},
                            "selector": {"type": "string"}, "url": {"type": "string"},
                            "value": {"type": "string"}, "key": {"type": "string"},
                            "path": {"type": "string"}, "full_page": {"type": "boolean"},
                        },
                        "required": ["type"], "additionalProperties": False,
                    },
                },
                "headless": {"type": "boolean", "default": False},
                "timeout_ms": {"type": "integer", "minimum": 1000, "maximum": 120000, "default": 30000},
            }, ["url", "actions"])),
            ("launch_app", "Launch a desktop application. Always requires approval.", _schema({
                "application": {"type": "string"}, "arguments": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string", "default": "."},
            }, ["application"])),
            ("remember", "Save durable, non-secret user context in SAM's local memory.", _schema({
                "content": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}, "default": []},
                "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "domain": {"type": "string", "enum": ["session", "user", "project", "trading", "task", "episodic"], "default": "user"},
            }, ["content"])),
            ("recall", "Search SAM's durable local memory.", _schema({
                "query": {"type": "string", "default": ""}, "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            })),
            ("create_plan", "Create a structured task plan without executing it.", _schema({
                "goal": {"type": "string"}, "steps": {"type": "array", "items": {"type": "string"}},
            }, ["goal", "steps"])),
            ("project_map", "Understand a project: structure, languages, dependencies, commands, API routes, test suites and git state. Cached; prefer this over listing files one by one.", _schema({
                "path": {"type": "string", "default": "."},
                "refresh": {"type": "boolean", "default": False},
                "detail": {"type": "string", "enum": ["summary", "full"], "default": "summary"},
            })),
            ("git_status", "Show the working tree status: branch, staged, modified and untracked files.", _schema({
                "path": {"type": "string", "default": "."},
            })),
            ("git_diff", "Show a unified diff of uncommitted changes, optionally for one path.", _schema({
                "path": {"type": "string", "default": "."},
                "target": {"type": "string", "default": ""},
                "staged": {"type": "boolean", "default": False},
            })),
            ("git_log", "Show recent commits for orientation.", _schema({
                "path": {"type": "string", "default": "."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            })),
            ("run_tests", "Run the project's own declared checks (test/build/lint/typecheck) and return a parsed verdict.", _schema({
                "path": {"type": "string", "default": "."},
                "kinds": {"type": "array", "items": {"type": "string", "enum": ["test", "build", "lint", "typecheck"]}, "maxItems": 4},
                "command": {"type": "string", "default": ""},
                "timeout_seconds": {"type": "integer", "minimum": 5, "maximum": 1800, "default": 600},
            })),
            ("list_capabilities", "List which developer tools and runtimes are actually installed on this machine.", _schema({
                "category": {"type": "string", "default": ""},
            })),
            ("market_snapshot", "Read an exact current MetaTrader 5 snapshot with feed metadata.", _schema({
                "symbol": {"type": "string", "default": "XAUUSD"},
                "timeframes": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            })),
            ("analyze_market", "Run deterministic multi-timeframe trading analysis. It can return WAIT or NO_TRADE and never submits orders.", _schema({
                "symbol": {"type": "string", "default": "XAUUSD"},
                "timeframes": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                "theories": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                "count": {"type": "integer", "minimum": 100, "maximum": 5000, "default": 600},
                "minimum_rr": {"type": "number", "minimum": 0.1, "maximum": 20},
            })),
            ("get_tradingview_state", "Observe TradingView process, active window, native title, symbol, price, and geometry without changing it.", _schema({})),
            ("focus_tradingview", "Launch or focus TradingView Desktop. Requires Computer Control.", _schema({})),
            ("set_tradingview_symbol", "Change the TradingView symbol and verify it from the native window title.", _schema({"symbol": {"type": "string"}}, ["symbol"])),
            ("set_tradingview_timeframe", "Send a semantic TradingView timeframe command. With Screen Access on, toolbar OCR confirms the interval; otherwise returns PARTIAL.", _schema({"timeframe": {"type": "string"}}, ["timeframe"])),
            ("save_custom_theory", "Validate and save a new version of a structured custom trading theory.", _schema({
                "definition": {"type": "object", "additionalProperties": True},
            }, ["definition"])),
            ("list_theories", "List built-in and user-defined trading theories with health and version metadata.", _schema({})),
            ("calibrate_chart", "Read the TradingView price axis and derive a verified price-to-pixel mapping. Required before any price-accurate drawing.", _schema({})),
            ("verify_chart_calibration", "Re-read the price axis and confirm the stored calibration still holds. Invalidates it on drift.", _schema({})),
            ("draw_tradingview_level", "Draw one verified price annotation on the TradingView chart and confirm it appeared. Never claims success without visual verification.", _schema({
                "annotation": {
                    "type": "string",
                    "enum": sorted(SEMANTIC_TYPES),
                    "description": "Semantic annotation type; SAM chooses the chart layer from it.",
                },
                "price": {"type": "number"},
                "label": {"type": "string", "default": ""},
                "theory": {"type": "string", "default": ""},
                "setup_id": {"type": "string"},
                "layer": {"type": "string", "enum": [layer.value for layer in Layer]},
                "symbol": {"type": "string", "description": "Instrument the price belongs to; the chart must be showing it."},
            }, ["annotation", "price"])),
            ("draw_tradingview_object", "Drag-draw a two-anchor chart object (trendline, channel, Fibonacci) between two price/time anchors and verify both endpoints.", _schema({
                "annotation": {"type": "string", "enum": sorted(TWO_ANCHOR_TYPES)},
                "price_a": {"type": "number"}, "minutes_a": {"type": "number", "minimum": 0, "maximum": 1440},
                "price_b": {"type": "number"}, "minutes_b": {"type": "number", "minimum": 0, "maximum": 1440},
                "label": {"type": "string", "default": ""}, "theory": {"type": "string", "default": ""},
                "setup_id": {"type": "string"},
                "layer": {"type": "string", "enum": [layer.value for layer in Layer]},
            }, ["annotation", "price_a", "minutes_a", "price_b", "minutes_b"])),
            ("draw_analysis_on_chart", "Draw the entry, stop, targets, and nearest levels from the most recent completed analysis, verifying each annotation.", _schema({
                "theory": {"type": "string", "default": ""},
                "setup_id": {"type": "string"},
            })),
            ("list_sam_drawings", "List the annotations SAM owns, with layer, price, and verification state.", _schema({
                "symbol": {"type": "string"}, "layer": {"type": "string", "enum": [layer.value for layer in Layer]},
                "theory": {"type": "string"}, "setup_id": {"type": "string"},
                "visible_only": {"type": "boolean", "default": False},
            })),
            ("clear_sam_drawings", "Remove annotations SAM owns from the chart, verifying each removal. Drawings made by the user are never touched.", _schema({
                "symbol": {"type": "string"}, "layer": {"type": "string", "enum": [layer.value for layer in Layer]},
                "theory": {"type": "string"}, "setup_id": {"type": "string"},
                "all_owned": {"type": "boolean", "default": False},
            })),
            ("set_chart_layer", "Show or hide one of SAM's chart annotation layers in its ownership records.", _schema({
                "layer": {"type": "string", "enum": [layer.value for layer in Layer]},
                "visible": {"type": "boolean"},
                "symbol": {"type": "string"},
            }, ["layer", "visible"])),
            ("list_entry_triggers", "List SAM's reusable entry triggers with their requirements, confirmation, and invalidation rules.", _schema({})),
            ("backtest_strategy", "Backtest an entry trigger (or compare all of them) on real historical candles, with no lookahead. Returns win rate, expectancy, profit factor, and drawdown.", _schema({
                "symbol": {"type": "string", "default": "XAUUSD"},
                "timeframe": {"type": "string", "default": "M15"},
                "trigger": {"type": "string", "description": "Omit to compare every registered trigger."},
                "count": {"type": "integer", "minimum": 300, "maximum": 20000, "default": 3000},
                "stop_atr_multiple": {"type": "number", "minimum": 0.2, "maximum": 10, "default": 1.5},
                "reward_multiple": {"type": "number", "minimum": 0.2, "maximum": 20, "default": 2.0},
                "max_bars": {"type": "integer", "minimum": 5, "maximum": 500, "default": 60},
            })),
            ("list_processes", "List Windows processes without changing them.", _schema({"query": {"type": "string", "default": ""}})),
            ("process_info", "Inspect one Windows process.", _schema({"pid": {"type": "integer", "minimum": 1}}, ["pid"])),
            ("stop_process", "Terminate one explicit process tree. Always requires approval.", _schema({"pid": {"type": "integer", "minimum": 1}}, ["pid"])),
            ("list_windows", "List visible Windows application windows and geometry.", _schema({})),
            ("window_action", "Focus, minimize, maximize, restore, move, or close one explicit window.", _schema({
                "hwnd": {"type": "integer", "minimum": 1},
                "action": {"type": "string", "enum": ["focus", "minimize", "maximize", "restore", "move", "close"]},
                "x": {"type": "integer"}, "y": {"type": "integer"}, "width": {"type": "integer"}, "height": {"type": "integer"},
            }, ["hwnd", "action"])),
            ("list_installed_apps", "List installed Windows applications from read-only registry keys.", _schema({"query": {"type": "string", "default": ""}})),
            ("read_clipboard", "Read text from the Windows clipboard. Requires approval because it may contain secrets.", _schema({})),
            ("write_clipboard", "Replace Windows clipboard text. Requires approval.", _schema({"text": {"type": "string", "maxLength": 100000}}, ["text"])),
            ("capture_screen", "Capture the Windows desktop when Screen Access is enabled. Requires approval.", _schema({})),
            ("inspect_ui_tree", "Inspect a window through Windows UI Automation when pywinauto is installed.", _schema({"hwnd": {"type": "integer", "minimum": 1}}, ["hwnd"])),
            ("emergency_stop", "Immediately cancel active cancellable SAM tasks and monitoring work.", _schema({"reason": {"type": "string", "default": "emergency_stop"}})),
        ]
        return [
            {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}
            for name, description, parameters in definitions if name in self._handlers
        ]

    @property
    def manifests(self) -> list[dict[str, Any]]:
        destructive = {"delete_path", "stop_process", "window_action", "clear_sam_drawings"}
        cancellable = {"run_terminal", "run_python", "browser_automate", "analyze_market", "draw_analysis_on_chart", "clear_sam_drawings", "backtest_strategy", "run_tests"}
        permission_classes = {
            "list_files": "filesystem", "search_files": "filesystem", "read_file": "filesystem", "write_file": "filesystem",
            "replace_text": "filesystem", "delete_path": "destructive_actions", "run_terminal": "shell", "run_python": "shell",
            "open_url": "browser", "browser_automate": "browser", "launch_app": "applications",
            "market_snapshot": "network", "analyze_market": "network", "get_tradingview_state": "desktop",
            "focus_tradingview": "desktop", "set_tradingview_symbol": "desktop", "set_tradingview_timeframe": "desktop",
            "list_processes": "applications", "process_info": "applications", "stop_process": "destructive_actions",
            "list_windows": "desktop", "window_action": "desktop", "list_installed_apps": "applications",
            "read_clipboard": "credentials", "write_clipboard": "desktop", "capture_screen": "screen", "inspect_ui_tree": "desktop",
            "save_custom_theory": "memory", "list_theories": "memory", "remember": "memory", "recall": "memory",
            "calibrate_chart": "screen", "verify_chart_calibration": "screen",
            "list_entry_triggers": "memory", "backtest_strategy": "network",
            "draw_tradingview_level": "desktop", "draw_tradingview_object": "desktop", "draw_analysis_on_chart": "desktop",
            "list_sam_drawings": "memory", "clear_sam_drawings": "destructive_actions", "set_chart_layer": "memory",
            "create_plan": "planning", "emergency_stop": "safety",
            "project_map": "filesystem", "git_status": "filesystem", "git_diff": "filesystem",
            "git_log": "filesystem", "run_tests": "shell", "list_capabilities": "agent",
        }
        result = []
        for spec in self.specs:
            function = spec["function"]
            name = function["name"]
            manifest = ToolManifest(
                name=name,
                description=function["description"],
                input_schema=function["parameters"],
                output_schema={
                    "type": "object",
                    "required": ["status", "executed", "verified", "data", "error", "duration_ms", "observations"],
                },
                permission_class=permission_classes.get(name, "agent"),
                timeout_seconds=self.settings.command_timeout_seconds if name in cancellable else 30,
                cancellable=name in cancellable or name == "emergency_stop",
                max_retries=0 if name in destructive else 2 if name in {"market_snapshot", "get_tradingview_state"} else 1,
                verification="Verify the postcondition from an independent observation; never equate launch/request with success.",
                audit_behavior="Arguments are normalized and secret fields redacted; result metadata is hash-linked in the audit log.",
                secret_policy="Never persist or return credential material to a model.",
                error_codes=["INVALID_ARGUMENTS", "DENIED", "APPROVAL_REQUIRED", "TIMEOUT", "CANCELLED", "VERIFICATION_FAILED"],
            )
            result.append(manifest.as_dict())
        return result

    def has(self, name: str) -> bool:
        return name in self._handlers

    def execute(self, name: str, arguments: dict[str, Any], *, approved: bool = False) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(False, error=f"Unknown tool: {name}")
        decision = RiskPolicy(self.settings).evaluate(name, arguments)
        if not decision.allowed:
            return ToolResult(False, error=decision.reason, sensitive=decision.sensitive)
        if decision.approval_required and not approved:
            return ToolResult(False, error=f"Approval required: {decision.reason}", sensitive=decision.sensitive)
        try:
            return handler(arguments, approved)
        except (OSError, ValueError, TypeError, UnicodeError, RuntimeError, KeyError) as exc:
            return ToolResult(False, error=f"{type(exc).__name__}: {exc}")

    def _resolve(self, raw_path: str | None, *, approved: bool, must_exist: bool = False) -> Path:
        raw_path = os.path.expandvars(raw_path or ".")
        violation = windows_path_violation(raw_path)
        if violation:
            raise PermissionError(violation)
        path = Path(raw_path).expanduser()
        candidate = path.resolve(strict=False) if path.is_absolute() else (self.workspace / path).resolve(strict=False)
        if candidate == self.settings.data_dir or candidate.is_relative_to(self.settings.data_dir):
            raise PermissionError("SAM's private database and audit directory is not exposed to tools.")
        if not candidate.is_relative_to(self.workspace) and not approved:
            raise PermissionError("Path is outside SAM's workspace and has not been approved.")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    @staticmethod
    def _safe_environment() -> dict[str, str]:
        allowed = {
            "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "LOCALAPPDATA",
            "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA", "USERPROFILE", "USERNAME", "USERDOMAIN",
            "LANG", "LC_ALL", "TERM", "SHELL", "HOME",
        }
        return {key: value for key, value in os.environ.items() if key.upper() in allowed}

    def _truncate(self, value: str) -> tuple[str, bool]:
        if len(value) <= self.settings.max_output_chars:
            return value, False
        return value[: self.settings.max_output_chars] + "\n…[output truncated]", True

    def _list_files(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        root = self._resolve(str(arguments.get("path", ".")), approved=approved, must_exist=True)
        max_depth = max(0, min(8, int(arguments.get("max_depth", 2))))
        if root.is_file():
            return ToolResult(True, {"root": str(root), "entries": [{"path": root.name, "type": "file", "size": root.stat().st_size}]})
        ignored = {".git", "node_modules", "__pycache__", ".venv", "venv"}
        entries: list[dict[str, Any]] = []
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            depth = len(current_path.relative_to(root).parts)
            dirs[:] = sorted(item for item in dirs if item not in ignored and depth < max_depth)
            for directory in dirs:
                entries.append({"path": str((current_path / directory).relative_to(root)), "type": "directory"})
            for filename in sorted(files):
                path = current_path / filename
                try:
                    entries.append({"path": str(path.relative_to(root)), "type": "file", "size": path.stat().st_size})
                except OSError:
                    continue
                if len(entries) >= 2000:
                    return ToolResult(True, {"root": str(root), "entries": entries, "truncated": True}, truncated=True)
        return ToolResult(True, {"root": str(root), "entries": entries, "truncated": False})

    def _search_files(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        root = self._resolve(str(arguments.get("path", ".")), approved=approved, must_exist=True)
        query = str(arguments.get("query", ""))
        if not query:
            raise ValueError("query cannot be empty")
        pattern = re.compile(re.escape(query), 0 if arguments.get("case_sensitive") else re.IGNORECASE)
        glob = str(arguments.get("glob", "*"))
        paths = [root] if root.is_file() else root.rglob(glob)
        matches: list[dict[str, Any]] = []
        for path in paths:
            if not path.is_file() or any(part in {".git", "node_modules", ".venv", "data"} for part in path.parts):
                continue
            try:
                if path.stat().st_size > self.settings.max_file_bytes:
                    continue
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if pattern.search(line):
                            matches.append({"path": str(path), "line": line_number, "text": line.rstrip()[:500]})
                            if len(matches) >= 500:
                                return ToolResult(True, {"matches": matches, "truncated": True}, truncated=True)
            except OSError:
                continue
        return ToolResult(True, {"matches": matches, "truncated": False})

    def _read_file(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        path = self._resolve(str(arguments.get("path", "")), approved=approved, must_exist=True)
        if not path.is_file():
            raise ValueError("Path is not a file")
        if path.stat().st_size > self.settings.max_file_bytes:
            raise ValueError(f"File exceeds {self.settings.max_file_bytes} byte limit")
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        start = max(1, int(arguments.get("start_line", 1)))
        end = min(len(lines), int(arguments.get("end_line", len(lines)))) if lines else 0
        selected = "".join(lines[start - 1:end])
        selected, truncated = self._truncate(selected)
        return ToolResult(True, {"path": str(path), "start_line": start, "end_line": end, "content": selected}, truncated=truncated)

    def _write_file(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        path = self._resolve(str(arguments.get("path", "")), approved=approved)
        content = str(arguments.get("content", ""))
        encoded = content.encode("utf-8")
        if len(encoded) > self.settings.max_file_bytes:
            raise ValueError(f"Content exceeds {self.settings.max_file_bytes} byte limit")
        existed = path.exists()
        if existed and not approved:
            raise PermissionError("Overwriting an existing file requires approval.")
        before_hash = None
        backup_path = None
        if existed and path.is_file():
            before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            is_ordinary_workspace_file = path.is_relative_to(self.workspace) and not any(
                part.lower() in {".ssh", ".aws", ".azure", ".gnupg", "credentials", "secrets"}
                for part in path.parts
            ) and path.name.lower() not in {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519", "credentials.json"}
            if is_ordinary_workspace_file:
                backup_dir = self.settings.data_dir / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name)[:100]
                backup_path = backup_dir / f"{int(time.time() * 1000)}-{before_hash[:12]}-{safe_name}.bak"
                shutil.copy2(path, backup_path)
        if arguments.get("create_parents", True):
            path.parent.mkdir(parents=True, exist_ok=True)
        if not path.parent.exists():
            raise FileNotFoundError(path.parent)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return ToolResult(True, {
            "path": str(path), "bytes": len(encoded), "created": not existed, "overwritten": existed,
            "before_sha256": before_hash, "after_sha256": hashlib.sha256(encoded).hexdigest(),
            "backup": str(backup_path) if backup_path else None,
        })

    def _replace_text(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        if not approved:
            raise PermissionError("Replacing text overwrites a file and requires approval.")
        path = self._resolve(str(arguments.get("path", "")), approved=True, must_exist=True)
        if path.stat().st_size > self.settings.max_file_bytes:
            raise ValueError("File is too large")
        old = str(arguments.get("old_text", ""))
        new = str(arguments.get("new_text", ""))
        expected = int(arguments.get("expected_replacements", 1))
        if not old:
            raise ValueError("old_text cannot be empty")
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != expected:
            raise ValueError(f"Expected {expected} exact occurrence(s), found {count}; no change was made")
        return self._write_file({"path": str(path), "content": text.replace(old, new), "create_parents": False}, True)

    def _delete_path(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        if not approved:
            raise PermissionError("Deletion requires approval")
        path = self._resolve(str(arguments.get("path", "")), approved=True, must_exist=True)
        protected_exact = {self.workspace, self.settings.project_root, Path(path.anchor)}
        protected_system_roots: set[Path] = set()
        for variable in ("SYSTEMROOT", "WINDIR", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA"):
            value = os.getenv(variable)
            if value:
                protected_system_roots.add(Path(value).resolve(strict=False))
        if path.parent == path or path in protected_exact or any(path == root or path.is_relative_to(root) for root in protected_system_roots):
            raise PermissionError("Deleting a workspace, project, filesystem, or protected Windows path is blocked")
        if path.is_dir():
            if not arguments.get("recursive", False):
                path.rmdir()
            else:
                shutil.rmtree(path)
        else:
            path.unlink()
        return ToolResult(True, {"path": str(path), "deleted": True, "recoverable": False})

    def _run_process(self, command: list[str], cwd: Path, timeout: int) -> ToolResult:
        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(
            command, cwd=cwd, env=self._safe_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
            creationflags=creation_flags, start_new_session=os.name != "nt",
        )
        timed_out = False
        try:
            stdout_raw, stderr_raw = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    check=False,
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            # Always terminate the direct process too; taskkill can be unavailable
            # or restricted even when SAM is allowed to stop its own child.
            try:
                process.kill()
            except OSError:
                pass
            stdout_raw, stderr_raw = process.communicate(timeout=5)

        stdout, out_truncated = self._truncate(stdout_raw or "")
        stderr, err_truncated = self._truncate(stderr_raw or "")
        if timed_out:
            return ToolResult(
                False,
                {"exit_code": process.returncode, "stdout": stdout, "stderr": stderr, "cwd": str(cwd)},
                f"Process timed out after {timeout}s and its process tree was stopped.",
                truncated=out_truncated or err_truncated,
            )
        return ToolResult(
            process.returncode == 0,
            {"exit_code": process.returncode, "stdout": stdout, "stderr": stderr, "cwd": str(cwd)},
            None if process.returncode == 0 else f"Process exited with code {process.returncode}",
            truncated=out_truncated or err_truncated,
        )

    def _run_terminal(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        command_text = str(arguments.get("command", ""))
        cwd = self._resolve(str(arguments.get("cwd", ".")), approved=approved, must_exist=True)
        timeout = max(1, min(300, int(arguments.get("timeout_seconds", self.settings.command_timeout_seconds))))
        if os.name == "nt":
            executable = shutil.which("pwsh") or shutil.which("powershell")
            if not executable:
                raise FileNotFoundError("PowerShell was not found")
            command = [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command_text]
        else:
            command = ["/bin/sh", "-lc", command_text]
        return self._run_process(command, cwd, timeout)

    def _run_python(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        if not approved:
            raise PermissionError("Python execution requires approval")
        code = str(arguments.get("code", ""))
        cwd = self._resolve(str(arguments.get("cwd", ".")), approved=True, must_exist=True)
        timeout = max(1, min(300, int(arguments.get("timeout_seconds", self.settings.command_timeout_seconds))))
        return self._run_process([sys.executable, "-I", "-c", code], cwd, timeout)

    def _open_url(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        url = str(arguments.get("url", ""))
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Only absolute http(s) URLs can be opened")
        opened = webbrowser.open(url, new=2, autoraise=True)
        return ToolResult(bool(opened), {"url": url, "opened": bool(opened)}, None if opened else "The operating system did not confirm a browser launch")

    def _browser_automate(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        try:
            output = run_browser_workflow(
                arguments,
                workspace=self.workspace,
                max_output_chars=self.settings.max_output_chars,
            )
            return ToolResult(True, output)
        except BrowserAutomationError as exc:
            return ToolResult(False, error=str(exc))

    def _launch_app(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        if not approved:
            raise PermissionError("Application launch requires approval")
        application = str(arguments.get("application", "")).strip()
        if not application:
            raise ValueError("application cannot be empty")
        app_args = [str(item) for item in arguments.get("arguments", [])]
        cwd = self._resolve(str(arguments.get("cwd", ".")), approved=True, must_exist=True)
        executable = shutil.which(application) or application
        process = subprocess.Popen(
            [executable, *app_args], cwd=cwd, env=self._safe_environment(), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return ToolResult(True, {"application": application, "pid": process.pid, "arguments": app_args})

    def _remember(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        content = str(arguments.get("content", "")).strip()
        if not content:
            raise ValueError("content cannot be empty")
        if re.search(r"(?i)(api[_ -]?key|password|secret|bearer\s+[a-z0-9._-]+|-----BEGIN .*PRIVATE KEY-----)", content):
            raise PermissionError("SAM will not store likely credentials in durable memory")
        memory = self.database.add_memory(
            content,
            list(arguments.get("tags", [])),
            float(arguments.get("importance", 0.5)),
            domain=str(arguments.get("domain", "user")),
        )
        return ToolResult(True, memory)

    def _recall(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        return ToolResult(True, self.database.list_memories(str(arguments.get("query", "")), int(arguments.get("limit", 10))))

    def _create_plan(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        goal = str(arguments.get("goal", "")).strip()
        steps = [str(step).strip() for step in arguments.get("steps", []) if str(step).strip()]
        if not goal or not steps:
            raise ValueError("goal and at least one step are required")
        return ToolResult(True, {"goal": goal, "steps": [{"index": index, "status": "pending", "text": step} for index, step in enumerate(steps, 1)]})

    # -- project intelligence and engineering ------------------------------

    def _project_map(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        root = self._resolve(str(arguments.get("path", ".")), approved=approved, must_exist=True)
        project_map = self.scanner.scan(root, force=bool(arguments.get("refresh", False)))
        if str(arguments.get("detail", "summary")) == "full":
            return ToolResult(True, project_map.as_dict())
        # The summary view is what a model should normally read: the full map
        # of a large repository would dominate the context window.
        payload = project_map.as_dict()
        for heavy in ("tree", "dependencies", "scripts"):
            payload.pop(heavy, None)
        payload["api_routes"] = payload["api_routes"][:40]
        payload["summary"] = project_map.summary_text()
        return ToolResult(True, payload)

    def _git_root(self, arguments: dict[str, Any], approved: bool) -> Path:
        root = self._resolve(str(arguments.get("path", ".")), approved=approved, must_exist=True)
        if not (root / ".git").exists():
            # Walk up: the agent is often pointed at a subdirectory of a repo.
            for parent in root.parents:
                if (parent / ".git").exists():
                    return parent
            raise FileNotFoundError(f"{root} is not inside a git repository")
        return root

    def _git_status(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        root = self._git_root(arguments, approved)
        porcelain = run_git(root, "status", "--porcelain=v1", "--branch")
        entries: list[dict[str, str]] = []
        branch = ""
        for line in porcelain.splitlines():
            if line.startswith("##"):
                branch = line[2:].strip()
                continue
            if len(line) > 3:
                entries.append({"code": line[:2].strip() or "??", "path": line[3:].strip()})
        return ToolResult(True, {
            "root": str(root), "branch": branch, "changed": entries, "clean": not entries,
            "head": run_git(root, "rev-parse", "--short", "HEAD").strip(),
        })

    def _git_diff(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        root = self._git_root(arguments, approved)
        command = ["diff"]
        if bool(arguments.get("staged", False)):
            command.append("--cached")
        target = str(arguments.get("target", "")).strip()
        if target:
            command.extend(["--", target])
        raw_diff = run_git(root, *command, timeout=30)
        # A tracked .env or a pasted key would otherwise travel straight into
        # model context and the UI through the diff.
        safe_diff, redactions = redact_secrets(raw_diff)
        diff, truncated = self._truncate(safe_diff)
        return ToolResult(True, {
            "root": str(root), "diff": diff, "empty": not diff.strip(), "redactions": redactions,
        }, truncated=truncated)

    def _git_log(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        root = self._git_root(arguments, approved)
        limit = max(1, min(50, int(arguments.get("limit", 10))))
        raw = run_git(root, "log", f"-{limit}", "--pretty=format:%h%x1f%an%x1f%ar%x1f%s")
        commits = []
        for line in raw.splitlines():
            parts = line.split("")
            if len(parts) == 4:
                commits.append({"hash": parts[0], "author": parts[1], "when": parts[2], "subject": parts[3]})
        return ToolResult(True, {"root": str(root), "commits": commits})

    def _run_tests(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        root = self._resolve(str(arguments.get("path", ".")), approved=approved, must_exist=True)
        timeout = max(5, min(1800, int(arguments.get("timeout_seconds", 600))))
        verifier = VerificationEngine(timeout=timeout, python_executable=self.verifier.python_executable)
        explicit = str(arguments.get("command", "")).strip()
        if explicit:
            check = verifier.run_check("test", explicit, root, timeout=timeout)
            return ToolResult(check.ok, check.as_dict(), None if check.ok else (check.reason or check.summary))
        project_map = self.scanner.scan(root)
        kinds = [str(kind) for kind in arguments.get("kinds", []) if str(kind)] or None
        report = verifier.verify(project_map, root, kinds=kinds)
        payload = report.as_dict()
        payload["summary"] = report.summary_text()
        # A report whose checks were all skipped is not a pass; the caller
        # must be able to tell "nothing failed" from "nothing ran".
        return ToolResult(
            report.ok, payload,
            None if report.ok else f"Verification failed: {report.summary_text()[:500]}",
        )

    def _list_capabilities(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        category = str(arguments.get("category", "")).strip() or None
        payload = self.capabilities.as_dict()
        if category:
            payload["capabilities"] = {
                name: item for name, item in payload["capabilities"].items() if item.get("category") == category
            }
        payload["summary"] = self.capabilities.summary_text()
        return ToolResult(True, payload)

    @staticmethod
    def _standard_result(result: Any) -> ToolResult:
        payload = result.as_dict()
        return ToolResult(
            result.status in {ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL},
            payload,
            result.error,
            truncated=False,
        )

    def _market_snapshot(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return self._standard_result(self.trading.market_snapshot(str(arguments.get("symbol", "XAUUSD")), arguments.get("timeframes")))

    def _analyze_market(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return self._standard_result(self.trading.analyze(
            str(arguments.get("symbol", "XAUUSD")),
            arguments.get("timeframes"),
            arguments.get("theories"),
            count=int(arguments.get("count", 600)),
            minimum_rr=float(arguments["minimum_rr"]) if arguments.get("minimum_rr") is not None else None,
        ))

    def _get_tradingview_state(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        state = self.trading.tradingview.observe()
        return ToolResult(state.running, state.as_dict(), None if state.running else "TradingView is not running or has no visible chart window")

    def _focus_tradingview(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        self.trading.refresh_permissions()
        return self._standard_result(self.trading.tradingview.launch())

    def _set_tradingview_symbol(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        self.trading.refresh_permissions()
        return self._standard_result(self.trading.tradingview.set_symbol(str(arguments.get("symbol", ""))))

    def _set_tradingview_timeframe(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        self.trading.refresh_permissions()
        return self._standard_result(self.trading.tradingview.set_timeframe(str(arguments.get("timeframe", ""))))

    def _save_custom_theory(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return ToolResult(True, self.trading.save_custom_theory(dict(arguments.get("definition") or {})))

    def _list_theories(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return ToolResult(True, {"built_in": self.trading.knowledge.list(), "custom": self.database.list_custom_theories()})

    def _calibrate_chart(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return self._standard_result(self.trading.calibrate_chart())

    def _verify_chart_calibration(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return self._standard_result(self.trading.verify_calibration())

    def _draw_tradingview_level(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        annotation = str(arguments.get("annotation", "")).lower()
        if annotation not in SEMANTIC_TYPES:
            return ToolResult(False, {"error": f"Unsupported annotation type: {annotation}"})
        try:
            price = float(arguments.get("price"))
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "price must be a finite number"})
        return self._standard_result(self.trading.draw_annotation(
            annotation,
            price,
            label=str(arguments.get("label", "")),
            theory=str(arguments.get("theory", "")),
            setup_id=arguments.get("setup_id"),
            layer=arguments.get("layer"),
            symbol=arguments.get("symbol"),
        ))

    def _draw_tradingview_object(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        annotation = str(arguments.get("annotation", "")).lower()
        if annotation not in TWO_ANCHOR_TYPES:
            return ToolResult(False, {"error": f"Unsupported two-anchor object: {annotation}"})
        try:
            values = [float(arguments[key]) for key in ("price_a", "minutes_a", "price_b", "minutes_b")]
        except (TypeError, ValueError, KeyError):
            return ToolResult(False, {"error": "price_a, minutes_a, price_b and minutes_b must all be numbers"})
        return self._standard_result(self.trading.draw_two_anchor(
            annotation, *values,
            label=str(arguments.get("label", "")), theory=str(arguments.get("theory", "")),
            setup_id=arguments.get("setup_id"), layer=arguments.get("layer"),
        ))

    def _draw_analysis_on_chart(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return self._standard_result(self.trading.draw_analysis(
            theory=str(arguments.get("theory", "")), setup_id=arguments.get("setup_id")
        ))

    def _list_sam_drawings(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return self._standard_result(self.trading.list_drawings(
            symbol=arguments.get("symbol"), layer=arguments.get("layer"),
            theory=arguments.get("theory"), setup_id=arguments.get("setup_id"),
            visible_only=bool(arguments.get("visible_only", False)),
        ))

    def _clear_sam_drawings(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return self._standard_result(self.trading.clear_drawings(
            symbol=arguments.get("symbol"), layer=arguments.get("layer"),
            theory=arguments.get("theory"), setup_id=arguments.get("setup_id"),
            all_owned=bool(arguments.get("all_owned", False)),
        ))

    def _set_chart_layer(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return self._standard_result(self.trading.set_layer_visibility(
            str(arguments.get("layer", "")), bool(arguments.get("visible", True)), arguments.get("symbol")
        ))

    def _list_entry_triggers(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return self._standard_result(self.trading.list_entry_triggers())

    def _backtest_strategy(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.trading is not None
        return self._standard_result(self.trading.backtest(
            str(arguments.get("symbol", "XAUUSD")),
            str(arguments.get("timeframe", "M15")),
            trigger=arguments.get("trigger"),
            count=int(arguments.get("count", 3000)),
            stop_atr_multiple=float(arguments.get("stop_atr_multiple", 1.5)),
            reward_multiple=float(arguments.get("reward_multiple", 2.0)),
            max_bars=int(arguments.get("max_bars", 60)),
        ))

    def _list_processes(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.windows is not None
        return self._standard_result(self.windows.list_processes(str(arguments.get("query", ""))))

    def _process_info(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.windows is not None
        return self._standard_result(self.windows.process_info(int(arguments.get("pid", 0))))

    def _stop_process(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.windows is not None
        return self._standard_result(self.windows.stop_process(int(arguments.get("pid", 0))))

    def _list_windows(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.windows is not None
        return self._standard_result(self.windows.enumerate_windows())

    def _window_action(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.windows is not None
        return self._standard_result(self.windows.window_action(
            int(arguments.get("hwnd", 0)),
            str(arguments.get("action", "")),
            x=arguments.get("x"), y=arguments.get("y"), width=arguments.get("width"), height=arguments.get("height"),
        ))

    def _list_installed_apps(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.windows is not None
        return self._standard_result(self.windows.list_installed_apps(str(arguments.get("query", ""))))

    def _read_clipboard(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.windows is not None
        result = self.windows.read_clipboard()
        result.data = "[Clipboard content hidden from model context]" if result.data else result.data
        return self._standard_result(result)

    def _write_clipboard(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.windows is not None
        return self._standard_result(self.windows.write_clipboard(str(arguments.get("text", ""))))

    def _capture_screen(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.windows is not None
        return self._standard_result(self.windows.capture_screen())

    def _inspect_ui_tree(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.windows is not None
        return self._standard_result(self.windows.inspect_ui_tree(int(arguments.get("hwnd", 0))))

    def _emergency_stop(self, arguments: dict[str, Any], approved: bool) -> ToolResult:
        assert self.cancellation is not None
        return ToolResult(True, self.cancellation.emergency_stop(str(arguments.get("reason", "emergency_stop"))))
