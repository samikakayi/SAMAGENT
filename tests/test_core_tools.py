from __future__ import annotations

import asyncio
import json

import pytest
from conftest import FAKE_GROQ

from sam.brain.confirm import ConfirmBroker
from sam.brain.tools import ToolContext, ToolRegistry, fail, ok, tool, validate_args
from sam.events import ConfirmRequest, EventBus, ToolFinished, ToolStarted, WorkerProgress
from sam.secrets import redact_obj

PARAMS = {"type": "object", "properties": {
    "name": {"type": "string", "description": "App name"},
    "count": {"type": "integer"},
    "mode": {"type": "string", "enum": ["fast", "slow"]},
}, "required": ["name"]}


@tool("open_app", description="Open or focus an app. Works with Sorani names.", description_ckb="کردنەوەی بەرنامە",
      params=PARAMS, blocking=True, examples_ckb=("کرۆم بکەرەوە",))
async def open_app(ctx: ToolContext, name: str, count: int = 1, mode: str = "fast") -> dict:
    return ok(f"opened {name}", count=count, mode=mode, source=ctx.source)


@tool("analyze_later", description="Slow analysis.", blocking=False)
async def analyze_later(ctx: ToolContext) -> dict:
    ctx.progress(1, 3, "شیکردنەوە")
    return ok("done")


@tool("delete_file", description="Delete a file.", risk="confirm",
      params={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
      confirm_text_ckb="فایلی {path} بسڕمەوە؟")
async def delete_file(ctx: ToolContext, path: str) -> dict:
    return ok(f"deleted {path}")


def _powershell_risk(args):
    command = args.get("command", "").lower()
    if "mimikatz" in command:
        return "blocked", "ئەم فەرمانە قەدەغەیە."
    return ("safe", None) if command.startswith("get-") else ("confirm", "ئەم فەرمانە جێبەجێ بکەم؟")


@tool("run_powershell", description="Run PowerShell.", classify=_powershell_risk,
      params={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]})
async def run_powershell(ctx: ToolContext, command: str) -> dict:
    return ok("ran", output=f"out of {command}")


@tool("leaky", description="Returns a key by mistake.")
async def leaky(ctx: ToolContext) -> dict:
    return ok(f"here {FAKE_GROQ}", api_key=FAKE_GROQ, text="x" * 20000)


@tool("crashes", description="Raises.")
async def crashes(ctx: ToolContext) -> dict:
    raise RuntimeError(f"connection failed with key {FAKE_GROQ}")


@tool("sleepy", description="Sleeps.", timeout_s=0.2)
async def sleepy(ctx: ToolContext) -> dict:
    await asyncio.sleep(5)
    return ok("never")


def make_registry(timeout_s: float = 0.3):
    bus = EventBus()
    events: list = []
    bus.subscribe(None, events.append)
    broker = ConfirmBroker(bus, timeout_s=timeout_s)
    registry = ToolRegistry(bus=bus, confirm=broker, redact_obj=lambda o: redact_obj(o))
    for fn in (open_app, analyze_later, delete_file, run_powershell, leaky, crashes, sleepy):
        registry.add(fn, owner="test")
    return registry, broker, events


def test_openai_schema_generation():
    registry, _, _ = make_registry()
    tools = {t["function"]["name"]: t for t in registry.openai_tools()}
    assert tools["open_app"]["type"] == "function"
    assert tools["open_app"]["function"]["parameters"] == PARAMS
    assert "Sorani examples: کرۆم بکەرەوە" in tools["open_app"]["function"]["description"]
    assert tools["analyze_later"]["function"]["parameters"] == {"type": "object", "properties": {}}
    json.dumps(tools)  # JSON-able


def test_gemini_declarations_live_and_text():
    from google.genai import types

    registry, _, _ = make_registry()
    live = {d.name: d for d in registry.gemini_declarations()}
    assert isinstance(live["open_app"], types.FunctionDeclaration)
    assert live["open_app"].behavior == types.Behavior.BLOCKING
    assert live["analyze_later"].behavior == types.Behavior.NON_BLOCKING
    assert live["open_app"].parameters_json_schema == PARAMS
    assert live["analyze_later"].parameters_json_schema is None
    text = {d.name: d for d in registry.gemini_declarations(live=False)}
    assert text["open_app"].behavior is None  # behavior is Live-only (BidiGenerateContent)
    tool_list = registry.gemini_tools(["open_app"])
    assert len(tool_list) == 1 and tool_list[0].function_declarations[0].name == "open_app"


def test_duplicate_names_rejected_and_add_from_module():
    registry = ToolRegistry()
    registry.add(open_app)
    with pytest.raises(ValueError):
        registry.add(open_app)
    import types as pytypes
    module = pytypes.ModuleType("fake_pkg")
    module.a, module.b, module.c = analyze_later, delete_file, 42
    assert sorted(registry.add_from(module, owner="fake")) == ["analyze_later", "delete_file"]
    assert registry.get("delete_file").owner == "fake"


def test_validate_args_coerces_and_checks():
    args, error = validate_args(PARAMS, {"name": "Chrome", "count": "3", "extra": 1})
    assert error is None and args == {"name": "Chrome", "count": 3}      # undeclared keys dropped
    open_schema = {**PARAMS, "additionalProperties": True}
    assert validate_args(open_schema, {"name": "Chrome", "extra": 1})[0] == {"name": "Chrome", "extra": 1}
    # A parameterless tool called with a stray key (Gemini did this to tv_open) gets {}.
    assert validate_args({"type": "object", "properties": {}}, {"reason": "user asked"}) == ({}, None)
    assert validate_args(PARAMS, {"count": 1})[1] == "missing required argument 'name'"
    assert "must be one of" in validate_args(PARAMS, {"name": "x", "mode": "turbo"})[1]
    assert "must be integer" in validate_args(PARAMS, {"name": "x", "count": "many"})[1]


async def test_dispatch_success_events_and_json_string_args():
    registry, _, events = make_registry()
    result = await registry.dispatch("open_app", '{"name": "Chrome", "count": "2"}', source="live", call_id="c1")
    assert result == {"ok": True, "summary": "opened Chrome", "data": {"count": 2, "mode": "fast", "source": "live"}}
    started = [e for e in events if isinstance(e, ToolStarted)]
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert started[0].call_id == "c1" and started[0].args == {"name": "Chrome", "count": 2}
    assert finished[0].ok and finished[0].name == "open_app"


async def test_dispatch_errors_are_results_not_exceptions():
    registry, _, _ = make_registry()
    assert (await registry.dispatch("nope", {}))["ok"] is False
    assert "missing required" in (await registry.dispatch("open_app", {}))["summary"]
    assert (await registry.dispatch("open_app", "{not json"))["ok"] is False
    crashed = await registry.dispatch("crashes", {})
    assert crashed["ok"] is False and "RuntimeError" in crashed["summary"] and FAKE_GROQ not in crashed["summary"]
    slow = await registry.dispatch("sleepy", {})
    assert slow["ok"] is False and "timed out" in slow["summary"]


async def test_results_are_redacted_and_capped():
    registry, _, _ = make_registry()
    result = await registry.dispatch("leaky", {})
    dumped = json.dumps(result)
    assert FAKE_GROQ not in dumped
    assert result["data"]["truncated"] is True and len(result["data"]["preview"]) <= 6000


async def test_confirm_yes_by_voice_runs_the_tool():
    registry, broker, events = make_registry(timeout_s=2)

    async def answer():
        await asyncio.sleep(0.05)
        request = [e for e in events if isinstance(e, ConfirmRequest)][0]
        assert request.question_ckb == "فایلی a.txt بسڕمەوە؟" and request.tool_name == "delete_file"
        assert broker.offer_transcript("بەڵێ") is True

    result, _ = await asyncio.gather(registry.dispatch("delete_file", {"path": "a.txt"}), answer())
    assert result["ok"] is True and result["summary"] == "deleted a.txt"


async def test_confirm_no_and_timeout_do_not_run():
    registry, broker, _ = make_registry(timeout_s=2)

    async def say_no():
        await asyncio.sleep(0.05)
        broker.offer_transcript("نەخێر مەیکە")

    result, _ = await asyncio.gather(registry.dispatch("delete_file", {"path": "a.txt"}), say_no())
    assert result["ok"] is False and result["data"] == {"declined": True}
    registry2, _, _ = make_registry(timeout_s=0.1)
    timed_out = await registry2.dispatch("delete_file", {"path": "b.txt"})
    assert timed_out["ok"] is False and timed_out["data"] == {"declined": True}


async def test_classifier_decides_risk_from_arguments():
    registry, _, _ = make_registry(timeout_s=0.1)
    assert (await registry.dispatch("run_powershell", {"command": "Get-Process"}))["ok"] is True
    blocked = await registry.dispatch("run_powershell", {"command": "mimikatz"})
    assert blocked["ok"] is False and blocked["data"] == {"blocked": True}
    assert registry.risk_of("run_powershell", {"command": "rm x"}) == ("confirm", "ئەم فەرمانە جێبەجێ بکەم؟")
    unconfirmed = await registry.dispatch("run_powershell", {"command": "Remove-Item x"})
    assert unconfirmed["data"] == {"declined": True}


async def test_no_broker_means_no():
    registry = ToolRegistry()
    registry.add(delete_file)
    assert (await registry.dispatch("delete_file", {"path": "x"}))["ok"] is False


async def test_cancel_all_stops_running_tools_and_progress_events():
    registry, _, events = make_registry()

    @tool("long_job", description="Long.", timeout_s=10)
    async def long_job(ctx: ToolContext) -> dict:
        await asyncio.sleep(10)
        return ok("never")

    registry.add(long_job)
    task = asyncio.create_task(registry.dispatch("long_job", {}))
    await asyncio.sleep(0.05)
    assert [r["name"] for r in registry.running()] == ["long_job"]
    assert registry.cancel_all() == 1
    result = await asyncio.wait_for(task, 2)
    assert result["ok"] is False and result["data"] == {"cancelled": True}
    await registry.dispatch("analyze_later", {})
    assert any(isinstance(e, WorkerProgress) and e.text_ckb == "شیکردنەوە" for e in events)


async def test_sync_handlers_and_plain_returns_are_normalised():
    registry = ToolRegistry()

    @tool("plain", description="Plain.")
    def plain(ctx: ToolContext) -> str:
        return "just text"

    @tool("failing", description="Fails politely.")
    async def failing(ctx: ToolContext) -> dict:
        return fail("TradingView is not running.", hint="open it")

    registry.add(plain)
    registry.add(failing)
    assert await registry.dispatch("plain", {}) == {"ok": True, "summary": "just text", "data": None}
    assert await registry.dispatch("failing", {}) == {"ok": False, "summary": "TradingView is not running.",
                                                      "data": {"hint": "open it"}}


def test_describe_for_prompt():
    registry, _, _ = make_registry()
    text = registry.describe_for_prompt(["open_app", "delete_file"])
    assert text.splitlines() == ["- open_app: Open or focus an app", "- delete_file: Delete a file."]
