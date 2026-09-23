"""The tool catalogue: what SAM advertises to a model, and how each tool is
classified for permissions and manifests.

Declarative data, deliberately apart from the engine that executes tools. A
maintainer adding a tool's schema edits this file; one changing how tools run
edits registry.py. Both halves are keyed by the same tool name.

Every function takes the registry so a catalogue entry can depend on what is
actually wired up -- trading and desktop tools only exist when their
subsystems were constructed.
"""

from __future__ import annotations

from typing import Any

from ..contracts import ToolManifest
from ..trading.drawing import Layer, SEMANTIC_TYPES, TWO_ANCHOR_TYPES


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


def tool_specs(registry: Any) -> list[dict[str, Any]]:
    """OpenAI-style function specifications for every wired-up tool."""
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
        ("web_search", "Search the public web and return titles, URLs and snippets. Read-only: it fetches results, it does not open pages or run anything.", _schema({
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        }, ["query"])),
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
        for name, description, parameters in definitions if name in registry._handlers
    ]



def tool_manifests(registry: Any) -> list[dict[str, Any]]:
    """Per-tool permission class, timeout, retry and audit metadata."""
    destructive = {"delete_path", "stop_process", "window_action", "clear_sam_drawings"}
    cancellable = {"run_terminal", "run_python", "browser_automate", "analyze_market", "draw_analysis_on_chart", "clear_sam_drawings", "backtest_strategy", "run_tests"}
    permission_classes = {
        "list_files": "filesystem", "search_files": "filesystem", "read_file": "filesystem", "write_file": "filesystem",
        "replace_text": "filesystem", "delete_path": "destructive_actions", "run_terminal": "shell", "run_python": "shell",
        "open_url": "browser", "browser_automate": "browser", "launch_app": "applications",
        # Reading search results is a network read, not browser control: it
        # opens nothing and runs nothing on the machine.
        "web_search": "network",
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
    for spec in registry.specs:
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
            timeout_seconds=registry.settings.command_timeout_seconds if name in cancellable else 30,
            cancellable=name in cancellable or name == "emergency_stop",
            max_retries=0 if name in destructive else 2 if name in {"market_snapshot", "get_tradingview_state"} else 1,
            verification="Verify the postcondition from an independent observation; never equate launch/request with success.",
            audit_behavior="Arguments are normalized and secret fields redacted; result metadata is hash-linked in the audit log.",
            secret_policy="Never persist or return credential material to a model.",
            error_codes=["INVALID_ARGUMENTS", "DENIED", "APPROVAL_REQUIRED", "TIMEOUT", "CANCELLED", "VERIFICATION_FAILED"],
        )
        result.append(manifest.as_dict())
    return result

