from __future__ import annotations

import asyncio
import json
import sys
import threading
import types as pytypes

from conftest import FAKE_GROQ, FakeBackend

from sam.bridge import CoreThread, UiAdapter
from sam.brain.tools import ToolContext, ok, tool
from sam.events import (Caption, ComponentStatus, EventBus, LevelMeter, SettingsChanged, ToolFinished, VoiceState)


# --- EventBus ---------------------------------------------------------------------

async def test_bus_sync_async_and_failing_subscribers():
    bus = EventBus(asyncio.get_running_loop())
    got_sync, got_async = [], []

    def broken(event):
        raise RuntimeError("subscriber bug")

    async def slow(event):
        await asyncio.sleep(0)
        got_async.append(event)

    bus.subscribe(Caption, broken)
    bus.subscribe(Caption, got_sync.append)
    unsubscribe = bus.subscribe((Caption, VoiceState), slow)
    bus.publish(Caption(text="سڵاو"))
    bus.publish(VoiceState(state="listening"))
    await asyncio.sleep(0.01)
    assert [e.text for e in got_sync] == ["سڵاو"] and len(got_async) == 2
    unsubscribe()
    bus.publish(Caption(text="again"))
    await asyncio.sleep(0.01)
    assert len(got_async) == 2


async def test_bus_publish_threadsafe_and_wait_for():
    bus = EventBus(asyncio.get_running_loop())
    waiter = asyncio.create_task(bus.wait_for(ComponentStatus, lambda e: e.component == "mt5", timeout=2))
    await asyncio.sleep(0)
    thread = threading.Thread(target=bus.publish_threadsafe,
                              args=(ComponentStatus(component="mt5", state="ok"),))
    thread.start()
    thread.join()
    event = await waiter
    assert event.state == "ok"


# --- CoreThread / UiAdapter -----------------------------------------------------------

def test_core_thread_submit_run_sync_call_soon():
    core = CoreThread()
    core.start()
    try:
        async def work(x):
            await asyncio.sleep(0.01)
            return x * 2, threading.current_thread().name

        value, name = core.run_sync(work(21), timeout=5)
        assert value == 42 and name == "sam-core"
        done = threading.Event()
        core.call_soon(done.set)
        assert done.wait(2)
    finally:
        core.stop()


def test_ui_adapter_forwards_from_core_thread_and_throttles_meters():
    core = CoreThread()
    core.start()
    received = []
    bus = EventBus(core.loop)
    adapter = UiAdapter(bus, received.append, max_meter_hz=10).attach()
    try:
        async def emit():
            bus.publish(Caption(text="x"))
            for i in range(5):
                bus.publish(LevelMeter(source="mic", level=0.5, at=100.0 + i * 0.01))
            bus.publish(LevelMeter(source="mic", level=0.5, at=101.0))

        core.run_sync(emit(), timeout=5)
        kinds = [type(e).__name__ for e in received]
        assert kinds.count("Caption") == 1 and kinds.count("LevelMeter") == 2
        adapter.detach()
        core.run_sync(emit(), timeout=5)
        assert len(received) == 3
    finally:
        core.stop()


# --- App ---------------------------------------------------------------------------------

def _fake_package(name, events):
    module = pytypes.ModuleType(name)

    @tool("hello", description="Say hello.")
    async def hello(ctx: ToolContext) -> dict:
        return ok("hello", key=FAKE_GROQ)

    def register(app):
        events.append("register")
        app.tools.add(hello, owner=name)
        app.memory = "memory-object"

    async def start(app):
        events.append("start")

    async def stop(app):
        events.append("stop")

    module.register, module.start, module.stop = register, start, stop
    return module


def _broken_package(name):
    module = pytypes.ModuleType(name)

    def register(app):
        raise RuntimeError(f"broken with {FAKE_GROQ}")

    module.register = register
    return module


async def test_app_loads_packages_defensively(make_app, monkeypatch):
    events: list[str] = []
    monkeypatch.setitem(sys.modules, "fakepkg_ok", _fake_package("fakepkg_ok", events))
    monkeypatch.setitem(sys.modules, "fakepkg_broken", _broken_package("fakepkg_broken"))
    app = make_app()
    status = app.load_packages(("fakepkg_ok", "fakepkg_broken", "sam.does_not_exist_yet"))
    assert status == {"fakepkg_ok": "ok", "fakepkg_broken": "failed: RuntimeError", "sam.does_not_exist_yet": "missing"}
    assert FAKE_GROQ not in app.failed["fakepkg_broken"]
    assert app.memory == "memory-object" and "hello" in app.tools.names()
    await app.start()
    result = await app.tools.dispatch("hello", {})
    assert result["ok"] and FAKE_GROQ not in json.dumps(result)
    await app.stop()
    assert events == ["register", "start", "stop"]
    assert app.db.scalar("SELECT count(*) FROM timings WHERE stage LIKE 'startup:%'") >= 2


async def test_stop_all_tool_cancels_running_work(make_app):
    app = make_app()
    await app.start()

    @tool("forever", description="Runs forever.", timeout_s=30)
    async def forever(ctx: ToolContext) -> dict:
        await asyncio.sleep(30)
        return ok("no")

    app.tools.add(forever)
    running = asyncio.create_task(app.tools.dispatch("forever", {}))
    await asyncio.sleep(0.05)
    stopped = await app.tools.dispatch("stop_all", {})
    assert stopped["ok"] and stopped["data"]["tools"] == 1
    assert (await asyncio.wait_for(running, 2))["data"] == {"cancelled": True}
    await app.stop()


async def test_settings_change_event_and_llm_through_app(make_app):
    app = make_app(backends={"groq": FakeBackend("groq"), "omniroute": FakeBackend("omniroute")},
                   env_text="LITELLM_FAST_MODEL=sam-fast\n")
    await app.start()
    seen = []
    app.bus.subscribe(SettingsChanged, seen.append)
    app.config.set("voice.voice_name", "Puck")
    await asyncio.sleep(0.01)
    assert seen and seen[0].key == "voice.voice_name" and seen[0].value == "Puck"
    response = await app.llm.chat([{"role": "user", "content": "hi"}])
    assert response.model_ref == "groq:openai/gpt-oss-20b"
    assert app.db.usage_for("groq", "openai/gpt-oss-20b")["requests"] == 1
    await app.stop()


def test_status_has_no_key_values(make_app):
    app = make_app(env_text="LITELLM_API_KEY=sk-" + "Q1w" * 10 + "\n")
    app.secrets.set("groq_api_key", FAKE_GROQ)
    dumped = json.dumps(app.status())
    assert FAKE_GROQ not in dumped and "Q1w" * 10 not in dumped
    assert app.status()["keys"]["groq_api_key"]["configured"] is True
    assert app.redact(f"x {FAKE_GROQ}") == "x [REDACTED]"


async def test_tool_events_reach_a_ui_sink(make_app):
    app = make_app()
    await app.start()
    got = []
    UiAdapter(app.bus, got.append, types=(ToolFinished,)).attach()
    await app.tools.dispatch("stop_all", {})
    assert [e.name for e in got] == ["stop_all"]
    await app.stop()
