"""The local brain (Ollama) as the last rung of every ladder: message
conversion, the native /api/chat backend over httpx.MockTransport, the ladder
fallback, the server manager (fake Popen), the UI hints. No network, no
process is started."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from brain_helpers import Reply, brain_app
from conftest import FakeBackend, rate_limited

from sam.brain.llm import LLMChunk, LLMError, LLMRequest, LLMResponse, ToolCall
from sam.brain.llm_local import CLOUD_BACK_CKB, LOCAL_NOTICE_CKB
from sam.brain.llm_ollama import OllamaBackend, split_context, to_ollama_messages
from sam.brain.local_server import OllamaServer, split_host
from sam.brain.persona import CONTEXT_HEADING
from sam.brain.responder import LOCAL_ACK_CKB, LOCAL_NOT_DONE_CKB
from sam.brain.tools import ok, tool
from sam.events import ComponentStatus
from sam.voice.notices import VoiceNotice


class FakeServer:
    """Stands in for OllamaServer: never starts anything."""

    def __init__(self, *, up: bool = True, loaded: list[str] | None = None) -> None:
        self.up = up
        self.ensured = 0
        self.stopped = 0
        self.base_url = "http://127.0.0.1:11434"
        self.last_error = "" if up else "ollama.exe not found"
        self._loaded = loaded or []

    def available(self) -> bool:
        return self.up

    def listening(self) -> bool:
        return self.up

    async def ensure(self) -> bool:
        self.ensured += 1
        self.last_error = "" if self.up else "ollama.exe not found"
        return self.up

    async def stop(self) -> bool:
        self.stopped += 1
        return True


class FakeLocal:
    """A scripted 'ollama' backend for ladder tests."""

    provider = "ollama"

    def __init__(self, replies: list[Any] | None = None, *, delay: float = 0.0, loaded: list[str] | None = None) -> None:
        self.replies = list(replies or ["سڵاو، فەرموو."])
        self.calls: list[tuple[str, LLMRequest]] = []
        self.delay = delay
        self.warmed: list[tuple[str, Any, Any]] = []
        self._loaded = loaded or []
        self.server = FakeServer()

    def configured(self) -> bool:
        return True

    def _next(self) -> Any:
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]

    async def complete(self, model: str, req: LLMRequest) -> LLMResponse:
        self.calls.append((model, req))
        if self.delay:
            await asyncio.sleep(self.delay)
        item = self._next()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Reply):
            calls = [ToolCall(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(item.calls)]
            return LLMResponse(text=item.text, tool_calls=calls, provider="ollama", model=model, total_ms=1.0)
        return LLMResponse(text=str(item), provider="ollama", model=model, total_ms=1.0)

    async def stream(self, model: str, req: LLMRequest):
        response = await self.complete(model, req)
        if response.text:
            yield LLMChunk(kind="text", text=response.text)
        yield LLMChunk(kind="done", response=response)

    async def list_models(self) -> list[str]:
        return ["qwen3:8b", "qwen3.5:4b"]

    async def loaded(self) -> list[str]:
        return list(self._loaded)

    async def warm(self, model: str, messages: Any = None, tools: Any = None) -> bool:
        self.warmed.append((model, messages, tools))
        return True

    async def aclose(self) -> None:
        pass


# --- message conversion ---------------------------------------------------------------------------------

def test_the_per_turn_context_moves_next_to_the_users_words():
    system = f"You are SAM.\n\nTools: open_app.\n\n{CONTEXT_HEADING}\nNow: Thursday 23:10.\n\nFacts: likes gold."
    stable, context = split_context(system)
    assert stable == "You are SAM.\n\nTools: open_app." and context.startswith(CONTEXT_HEADING)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "سڵاو"}, {"role": "assistant", "content": "سڵاو، فەرموو."},
                {"role": "user", "content": "نرخی زێڕ چەندە"},
                {"role": "assistant", "content": "", "_gemini_content": {"x": 1},
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "get_price", "arguments": "{\"symbol\": \"زێڕ\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "name": "get_price", "content": "{\"ok\": true}"}]
    out = to_ollama_messages(messages)
    # The stable prefix (instruction + tools) is byte-identical every turn: Ollama keeps it cached.
    assert out[0] == {"role": "system", "content": "You are SAM.\n\nTools: open_app."}
    assert out[1]["content"] == "سڵاو"                                  # earlier turns unchanged
    assert out[3]["content"].startswith(CONTEXT_HEADING) and out[3]["content"].endswith("نرخی زێڕ چەندە")
    assert out[4]["tool_calls"] == [{"function": {"name": "get_price", "arguments": {"symbol": "زێڕ"}}}]
    assert "_gemini_content" not in out[4]
    assert out[5] == {"role": "tool", "content": "{\"ok\": true}", "tool_name": "get_price"}


def test_the_persona_puts_its_changing_parts_after_the_heading(make_app):
    app, _ = brain_app(make_app)
    text = app.persona.system_instruction("voice")
    stable, context = split_context(text)
    assert "Now:" not in stable and "Now:" in context and "RESPOND IN CENTRAL KURDISH" in stable
    assert stable == split_context(app.persona.system_instruction("voice"))[0]


# --- the native backend ----------------------------------------------------------------------------------

def backend_with(handler, make_app, **settings) -> tuple[OllamaBackend, FakeServer, Any]:
    app = make_app(backends={})
    for key, value in settings.items():
        app.config.set(key, value)
    server = FakeServer()
    return OllamaBackend(app.config, server, transport=httpx.MockTransport(handler)), server, app


async def test_complete_sends_no_thinking_keep_alive_and_capped_reply(make_app):
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": "tv_open", "arguments": {}}}]},
            "done": True, "done_reason": "stop", "prompt_eval_count": 4146, "eval_count": 12})

    backend, server, _ = backend_with(handler, make_app)
    req = LLMRequest(messages=[{"role": "user", "content": "ترەیدینگ ڤیو بکەرەوە"}],
                     tools=[{"type": "function", "function": {"name": "tv_open", "parameters": {}}}], max_tokens=4096)
    response = await backend.complete("qwen3:8b", req)
    body = seen["body"]
    assert seen["url"] == "http://127.0.0.1:11434/api/chat" and server.ensured == 1
    assert body["think"] is False and body["stream"] is False and body["keep_alive"] == "5m"
    assert body["options"]["num_ctx"] == 8192 and body["options"]["num_predict"] == 1024   # CPU: 8 tok/s
    assert body["tools"][0]["function"]["name"] == "tv_open"
    assert response.provider == "ollama" and response.model_ref == "ollama:qwen3:8b"
    assert [c.name for c in response.tool_calls] == ["tv_open"] and response.tool_calls[0].id.startswith("call_")
    assert response.usage == {"tokens_in": 4146, "tokens_out": 12}
    assert response.assistant_message()["tool_calls"][0]["function"]["name"] == "tv_open"


async def test_errors_are_classified_and_a_missing_server_is_a_network_error(make_app):
    backend, server, _ = backend_with(lambda r: httpx.Response(404, json={"error": "model 'x' not found"}), make_app)
    with pytest.raises(LLMError) as info:
        await backend.complete("x", LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    assert info.value.kind == "not_found"
    server.up = False
    with pytest.raises(LLMError) as info:
        await backend.complete("qwen3:8b", LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    assert info.value.kind == "network" and "not found" in info.value.message


async def test_a_model_without_a_thinking_switch_is_asked_again_without_it(make_app):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": "\"x\" does not support thinking"})
        return httpx.Response(200, json={"message": {"content": "باشە."}, "done": True})

    backend, _, _ = backend_with(handler, make_app)
    response = await backend.complete("x", LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    assert response.text == "باشە." and len(bodies) == 2 and "think" not in bodies[1]


async def test_stream_yields_text_then_tool_calls_then_done(make_app):
    lines = [{"message": {"content": "باشە"}, "done": False},
             {"message": {"content": "، ئێستا."}, "done": False},
             {"message": {"content": "", "tool_calls": [{"function": {"name": "get_price",
                                                                      "arguments": {"symbol": "زێڕ"}}}]},
              "done": False},
             {"message": {"content": ""}, "done": True, "done_reason": "stop", "prompt_eval_count": 10,
              "eval_count": 5}]
    body = "\n".join(json.dumps(item, ensure_ascii=False) for item in lines).encode()
    backend, _, _ = backend_with(lambda r: httpx.Response(200, content=body), make_app)
    chunks = [c async for c in backend.stream("qwen3:8b", LLMRequest(messages=[{"role": "user", "content": "x"}]))]
    assert [c.kind for c in chunks] == ["text", "text", "tool_call", "done"]
    assert chunks[-1].response.text == "باشە، ئێستا." and chunks[2].tool_call.arguments == {"symbol": "زێڕ"}


async def test_aclose_stops_only_through_the_server_manager(make_app):
    backend, server, app = backend_with(lambda r: httpx.Response(200, json={}), make_app)
    await backend.aclose()
    assert server.stopped == 1
    app.config.set("llm.local.stop_on_quit", False)
    await backend.aclose()
    assert server.stopped == 1


# --- the last rung of every ladder ------------------------------------------------------------------------

def ladder_app(make_app, cloud: dict[str, Any], local: FakeLocal) -> Any:
    app = make_app(backends={**cloud, "ollama": local})
    events: list[Any] = []
    app.bus.subscribe((ComponentStatus, VoiceNotice), events.append)
    app.test_events = events
    return app


async def test_the_local_brain_answers_only_after_every_cloud_rung_failed(make_app):
    groq = FakeBackend("groq", {"m": [rate_limited("groq", "m")]})
    local = FakeLocal(["سڵاو، فەرموو."])
    app = ladder_app(make_app, {"groq": groq}, local)
    response = await app.llm.chat([{"role": "user", "content": "سڵاو"}], ladder=["groq:m"])
    assert response.provider == "ollama" and response.text == "سڵاو، فەرموو." and len(groq.calls) == 1
    assert local.calls[0][0] == "qwen3:8b" and local.calls[0][1].timeout_s == 150   # its own timeout
    assert app.llm.brain_mode == "local"
    notices = [e for e in app.test_events if isinstance(e, VoiceNotice)]
    assert [n.kind for n in notices] == ["local"] and notices[0].text_ckb == LOCAL_NOTICE_CKB
    assert any(isinstance(e, ComponentStatus) and e.component == "brain" and e.state == "degraded"
               for e in app.test_events)
    # the cloud is back: said once
    groq.script["m"] = ["باشە."]
    app.llm._cooldown.clear()  # noqa: SLF001
    response = await app.llm.chat([{"role": "user", "content": "سڵاو"}], ladder=["groq:m"])
    assert response.provider == "groq" and app.llm.brain_mode == "cloud"
    assert [n.kind for n in app.test_events if isinstance(n, VoiceNotice)] == ["local", "cloud"]
    assert [n.text_ckb for n in app.test_events if isinstance(n, VoiceNotice)][-1] == CLOUD_BACK_CKB
    assert app.db.usage_for("ollama", "qwen3:8b")["requests"] == 1


async def test_a_healthy_cloud_rung_never_reaches_the_local_brain(make_app):
    local = FakeLocal()
    app = ladder_app(make_app, {"groq": FakeBackend("groq", {"m": ["باشە."]})}, local)
    response = await app.llm.chat([{"role": "user", "content": "x"}], ladder=["groq:m"])
    assert response.provider == "groq" and local.calls == [] and not app.test_events


async def test_callers_can_refuse_the_local_brain_and_images_never_go_to_it(make_app):
    local = FakeLocal()
    app = ladder_app(make_app, {"groq": FakeBackend("groq", {"m": [rate_limited("groq", "m")]})}, local)
    with pytest.raises(LLMError) as info:
        await app.llm.chat([{"role": "user", "content": "x"}], ladder=["groq:m"], local=False)
    assert info.value.kind == "exhausted"
    image = [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                                          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]
    with pytest.raises(LLMError):
        await app.llm.chat(image, ladder=["groq:m"])
    assert local.calls == []


async def test_the_local_rung_is_not_cut_by_the_rounds_deadline(make_app):
    async def slow(*_a, **_k):
        await asyncio.sleep(1.0)

    groq = FakeBackend("groq", {"m": ["late"]})
    groq.complete = lambda model, req: slow()          # type: ignore[method-assign]
    local = FakeLocal(["سڵاو."], delay=0.3)
    app = ladder_app(make_app, {"groq": groq}, local)
    response = await app.llm.chat([{"role": "user", "content": "x"}], ladder=["groq:m"], deadline_s=0.6)
    assert response.provider == "ollama" and response.text == "سڵاو."


async def test_a_local_failure_rests_it_and_the_error_lists_every_attempt(make_app):
    local = FakeLocal([LLMError("network", "ollama.exe not found", provider="ollama", model="qwen3:8b")])
    app = ladder_app(make_app, {"groq": FakeBackend("groq", {"m": [rate_limited("groq", "m")]})}, local)
    with pytest.raises(LLMError) as info:
        await app.llm.chat([{"role": "user", "content": "x"}], ladder=["groq:m"])
    assert info.value.kind == "exhausted" and "ollama:qwen3:8b: network" in str(info.value)
    assert app.llm.cooling("ollama:*") and not app.llm.local_ready()
    with pytest.raises(LLMError):                           # rests: not asked again at once
        await app.llm.chat([{"role": "user", "content": "x"}], ladder=["groq:m"])
    assert len(local.calls) == 1


async def test_streaming_falls_back_to_the_local_brain_too(make_app):
    local = FakeLocal(["سڵاو، فەرموو."])
    app = ladder_app(make_app, {"groq": FakeBackend("groq", {"m": [rate_limited("groq", "m")]})}, local)
    chunks = [c async for c in app.llm.stream([{"role": "user", "content": "x"}], ladder=["groq:m"])]
    assert chunks[-1].kind == "done" and chunks[-1].response.provider == "ollama"


async def test_a_missing_main_model_falls_back_to_an_installed_one(make_app):
    local = FakeLocal(["باشە."])

    async def listing() -> list[str]:
        return ["qwen3.5:4b"]

    local.list_models = listing                              # type: ignore[method-assign]
    app = ladder_app(make_app, {}, local)
    response = await app.llm.chat([{"role": "user", "content": "x"}], ladder=["groq:m"])
    assert response.model == "qwen3.5:4b"


# --- the conversation on the local brain ------------------------------------------------------------------

@tool("get_price", description="Current price.", params={"type": "object", "properties": {"symbol": {"type": "string"}}},
      risk="safe", blocking=True)
async def fake_price(ctx, symbol: str = "") -> dict[str, Any]:
    return ok("زێڕ ئێستا لەسەر ٤٣١٢ مامەڵە دەکرێت.", symbol="XAUUSD", bid=4312.4)


async def test_a_local_tool_call_is_answered_with_the_tools_own_sentence(make_app):
    app, groq = brain_app(make_app, [rate_limited("groq", "openai/gpt-oss-20b")] * 4, tools=(fake_price,))
    local = FakeLocal([Reply(calls=[("get_price", {"symbol": "زێڕ"})])])
    app.llm.backends["ollama"] = local
    chunks = [c async for c in app.conversation.respond_stream("زێڕ ئێستا لە چ ئاستێکە؟", source="text")]
    assert "".join(chunks).strip() == "زێڕ ئێستا لەسەر ٤٣١٢ مامەڵە دەکرێت."
    assert len(local.calls) == 1                     # no second (3-10 s) local wording round


async def test_a_cold_local_brain_is_announced_in_voice(make_app):
    app, groq = brain_app(make_app, [rate_limited("groq", "openai/gpt-oss-20b")] * 6)
    local = FakeLocal(["سڵاو، فەرموو."])
    app.llm.backends["ollama"] = local
    for ref in ("groq:openai/gpt-oss-20b", "groq:openai/gpt-oss-120b"):
        app.llm._cool(ref, 600)                      # noqa: SLF001 - every cloud rung rests
    chunks = [c async for c in app.conversation.respond_stream("سڵاو", source="cascade")]
    assert chunks[0] == LOCAL_ACK_CKB and chunks[-1].strip() == "سڵاو، فەرموو."
    local._loaded = ["qwen3:8b"]                      # noqa: SLF001 - warm now: no announcement
    chunks = [c async for c in app.conversation.respond_stream("سوپاس", source="cascade")]
    assert LOCAL_ACK_CKB not in chunks


async def test_listening_warms_the_local_brain_only_when_the_cloud_cannot_answer(make_app):
    from sam.events import VoiceState

    app, _ = brain_app(make_app)
    local = FakeLocal()
    app.llm.backends["ollama"] = local
    app.bus.publish(VoiceState(state="listening", engine="cascade"))
    await asyncio.sleep(0.05)
    assert local.warmed == []                          # Groq can answer
    for ref in ("groq:openai/gpt-oss-20b", "groq:openai/gpt-oss-120b"):
        app.llm._cool(ref, 600)                        # noqa: SLF001
    task = app.conversation.prewarm_local_brain()
    await task
    model, messages, tools = local.warmed[0]
    assert model == "qwen3:8b" and messages[0]["role"] == "system" and tools


# --- the server manager -------------------------------------------------------------------------------------

class FakePopen:
    instances: list["FakePopen"] = []

    def __init__(self, argv, **kwargs) -> None:
        self.argv = argv
        self.kwargs = kwargs
        self.pid = 4242
        self.returncode = None
        FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 1

    def wait(self, timeout=None):
        return self.returncode


async def test_the_server_is_found_under_sam_home_and_started_hidden(make_app, home: Path, monkeypatch):
    exe = home / "tools" / "ollama-v0.33.1" / "ollama.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    (home / "data" / "ollama-models").mkdir()
    app = make_app(backends={})
    app.config.set("llm.local.host", "127.0.0.1:11999")
    up = {"value": False}
    FakePopen.instances.clear()
    monkeypatch.setenv("OLLAMA_IGPU_ENABLE", "7")         # the parent's value never reaches the child
    server = OllamaServer(app.config, popen=FakePopen, listening=lambda host: up["value"])
    killed = []
    monkeypatch.setattr(server, "_kill_tree", lambda proc: killed.append(proc.pid) or True)
    assert server.exe_path() == exe and server.models_dir() == home / "data" / "ollama-models"

    async def come_up():
        await asyncio.sleep(0.3)
        up["value"] = True

    asyncio.get_running_loop().create_task(come_up())
    assert await server.ensure(wait_s=3)
    proc = FakePopen.instances[0]
    assert proc.argv == [str(exe), "serve"]
    flags = proc.kwargs["creationflags"]
    assert flags & subprocess.CREATE_NO_WINDOW and flags & subprocess.CREATE_NEW_PROCESS_GROUP
    assert not flags & subprocess.DETACHED_PROCESS
    env = proc.kwargs["env"]
    assert env["OLLAMA_HOST"] == "127.0.0.1:11999" and env["OLLAMA_MODELS"] == str(home / "data" / "ollama-models")
    assert env["OLLAMA_IGPU_ENABLE"] == "1"               # llm.local.gpu on by default
    app.config.set("llm.local.gpu", False)
    assert "OLLAMA_IGPU_ENABLE" not in server._env()
    assert await server.ensure() and len(FakePopen.instances) == 1     # already up: nothing new
    assert await server.stop() and killed == [4242]


async def test_a_server_that_already_listens_is_shared_and_never_stopped(make_app, monkeypatch):
    app = make_app(backends={})
    FakePopen.instances.clear()
    server = OllamaServer(app.config, popen=FakePopen, listening=lambda host: True)
    assert await server.ensure() and FakePopen.instances == []       # SAM v1's ollama serve on 11434
    assert await server.stop() is False and not server.started_by_sam


async def test_no_executable_means_no_local_brain(make_app, monkeypatch):
    app = make_app(backends={})
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(Path(app.config.home) / "nowhere"))
    server = OllamaServer(app.config, popen=FakePopen, listening=lambda host: False)
    assert server.exe_path() is None and not server.available()
    assert await server.ensure() is False and server.last_error == "ollama.exe not found"
    app.config.set("llm.local.ollama_exe", str(Path(app.config.home) / "missing.exe"))
    assert server.exe_path() is None


def test_split_host_forms():
    assert split_host("127.0.0.1:11434") == ("127.0.0.1", 11434)
    assert split_host("http://localhost:11435/") == ("localhost", 11435)
    assert split_host("") == ("127.0.0.1", 11434)


def test_default_backends_include_the_local_brain(make_app):
    from sam.brain.llm_backends import default_backends

    app = make_app(backends={})
    backends = default_backends(app.config, app.secrets)
    assert isinstance(backends["ollama"], OllamaBackend)
    assert app.config.get("llm.local.model") == "qwen3:8b" and app.config.get("llm.local.enabled") is True


# --- what the user sees -----------------------------------------------------------------------------------

def test_the_island_says_local_brain_until_the_cloud_is_back():
    from sam.ui.island_hints import LOCAL_WORD, MODELS_WORD, IslandHints

    hints = IslandHints(clock=lambda: 1000.0)
    assert hints.on_notice(VoiceNotice(kind="local", text_ckb=LOCAL_NOTICE_CKB)) == (LOCAL_NOTICE_CKB, "alert")
    assert hints.status_override("idle") == LOCAL_WORD == "مێشکی ناوخۆیی"
    assert hints.status_override("thinking") == LOCAL_WORD and hints.status_override("speaking") is None
    hints.on_answer("assistant", "سڵاو، فەرموو.")           # a local answer does not clear it
    assert hints.status_override("idle") == LOCAL_WORD
    hints.on_notice(VoiceNotice(kind="models", text_ckb="x", until=2000.0))
    assert hints.status_override("idle") == MODELS_WORD      # no model at all wins
    hints.clear_sticky()
    hints.on_notice(VoiceNotice(kind="cloud", text_ckb=CLOUD_BACK_CKB))
    assert hints.status_override("idle") is None


async def test_the_start_up_snapshot_reports_the_brain(make_app):
    from sam.ui.status_seed import snapshot

    app = make_app(backends={})
    events = [e for e in await snapshot(app) if isinstance(e, ComponentStatus) and e.component == "brain"]
    assert events and events[0].state == "ok"
    app.llm.brain_mode = "local"
    events = [e for e in await snapshot(app) if isinstance(e, ComponentStatus) and e.component == "brain"]
    assert events[0].state == "degraded" and events[0].detail.startswith("local")


@tool("list_alerts", description="alerts", params={"type": "object", "properties": {"status": {"type": "string"}}})
async def fake_alerts(ctx, status: str = "active") -> dict[str, Any]:
    return ok("0 ئاگادارکردنەوە (active).", alerts=[])


async def test_the_alert_list_is_read_out_without_a_second_local_round(make_app):
    """Live run 2026-09-25: qwen3:8b worded an empty list as «هیچ
    ئاگادارکردنەوەکەی نەدەرە. بۆ چی نەدەرە؟»; the list now has a fixed sentence."""
    app, _ = brain_app(make_app, [rate_limited("groq", "openai/gpt-oss-20b")] * 4, tools=(fake_alerts,))
    local = FakeLocal([Reply(calls=[("list_alerts", {})])])
    app.llm.backends["ollama"] = local
    chunks = [c async for c in app.conversation.respond_stream("ئاگادارکردنەوەکانم چین؟ هەموویان", source="text")]
    assert "".join(chunks).strip() == "هیچ ئاگادارکردنەوەیەکی چالاکت نییە." and len(local.calls) == 1


# --- review 2026-09-25: no silent cold local rounds, no false claims ----------------------------------------

async def test_a_failed_wording_round_says_the_tools_own_sentence_not_a_cold_local_answer(make_app):
    """The cloud picked get_price, then every rung rate-limited the wording round:
    the tool's own Sorani result at once (the local brain's 150 s timeout would
    have cost ~1.5 min of silence when cold)."""
    steps = [Reply(calls=[("get_price", {"symbol": "XAUUSD"})])] + [rate_limited("groq", "m")] * 6
    app, _ = brain_app(make_app, steps, tools=(fake_price,))
    local = FakeLocal(["نرخی زێڕ ٤٣١٢ە."], loaded=[])
    app.llm.backends["ollama"] = local
    chunks = [c async for c in app.conversation.respond_stream("gold?", source="cascade")]
    assert chunks[-1].strip() == "زێڕ ئێستا لەسەر ٤٣١٢ مامەڵە دەکرێت." and local.calls == []


async def test_a_read_result_is_worded_locally_and_a_cold_start_is_announced(make_app):
    """A result the user wants read (no own sentence) may go to the local brain,
    and a cold one mid-turn is announced: the start-of-turn check saw a healthy cloud."""
    steps = [Reply(calls=[("list_alerts", {"status": "all"})])] + [rate_limited("groq", "m")] * 6
    app, _ = brain_app(make_app, steps, tools=(fake_alerts,))
    local = FakeLocal(["هیچ ئاگادارکردنەوەیەکت نییە."], delay=0.05, loaded=[])
    app.llm.backends["ollama"] = local
    chunks = [c async for c in app.conversation.respond_stream("all my alerts", source="cascade")]
    assert LOCAL_ACK_CKB in chunks and chunks[-1].strip() == "هیچ ئاگادارکردنەوەیەکت نییە."
    assert len(local.calls) == 1


async def test_a_slow_cloud_cut_by_the_deadline_announces_the_cold_local_brain(make_app):
    app, groq = brain_app(make_app, ["late"] * 4)
    original = groq.complete

    async def slow(model, req):
        await asyncio.sleep(2.0)
        return await original(model, req)

    groq.complete = slow
    app.config.set("conversation.picker_deadline_s", 0.5)
    local = FakeLocal(["سڵاو."], delay=0.05, loaded=[])
    app.llm.backends["ollama"] = local
    chunks = [c async for c in app.conversation.respond_stream("سڵاو", source="cascade")]
    assert chunks[0] == LOCAL_ACK_CKB and chunks.count(LOCAL_ACK_CKB) == 1 and chunks[-1].strip() == "سڵاو."


@pytest.mark.parametrize("claim", ["نۆتپاد ئامادەیە.", "نۆتپاد بکەرەوە.", "Notepad is open."])
async def test_the_local_brain_never_claims_an_action_it_did_not_run(make_app, claim):
    app, _ = brain_app(make_app, [rate_limited("groq", "openai/gpt-oss-20b")] * 4)
    app.llm.backends["ollama"] = FakeLocal([claim], loaded=["qwen3:8b"])
    chunks = [c async for c in app.conversation.respond_stream("نۆتپاد بکەرەوە", source="text")]
    assert " ".join(c.strip() for c in chunks) == LOCAL_NOT_DONE_CKB


async def test_small_talk_and_explanations_from_the_local_brain_stay(make_app):
    app, _ = brain_app(make_app, [rate_limited("groq", "openai/gpt-oss-20b")] * 8)
    app.llm.backends["ollama"] = FakeLocal(["سڵاو، فەرموو. چۆنی؟"], loaded=["qwen3:8b"])
    chunks = [c async for c in app.conversation.respond_stream("سڵاو سام", source="text")]
    assert " ".join(c.strip() for c in chunks) == "سڵاو، فەرموو. چۆنی؟"


def test_the_local_request_has_a_short_history_and_the_local_rules():
    """Live run 2026-09-25: after ten fast-path answers in the history qwen3:8b
    answered two commands from memory, with no tool call."""
    from sam.brain.llm_local import LOCAL_RULES, trim_history, with_local_rules

    history = [{"role": "system", "content": f"rules\n\n{CONTEXT_HEADING}\nNow: 01:00."}]
    for n in range(10):
        history += [{"role": "user", "content": f"q{n}"}, {"role": "assistant", "content": f"a{n}"}]
    current = [{"role": "user", "content": "now"}, {"role": "assistant", "content": "", "tool_calls": []},
               {"role": "tool", "tool_call_id": "c1", "content": "{}"}]
    trimmed = trim_history(history + current, 2)
    assert [m["content"] for m in trimmed[1:3]] == ["q9", "a9"] and trimmed[3:] == current
    ruled = with_local_rules(trimmed)
    assert ruled[0]["content"].endswith(LOCAL_RULES) and split_context(ruled[0]["content"])[0] == "rules"
    worker = [{"role": "system", "content": "w"}, {"role": "user", "content": "goal"},
              {"role": "assistant", "content": "", "tool_calls": []}, {"role": "tool", "content": "{}"}]
    assert trim_history(worker, 0) == worker                     # one goal + its tool rounds: untouched


async def test_the_local_brain_gets_the_trimmed_request(make_app):
    app, _ = brain_app(make_app, [rate_limited("groq", "openai/gpt-oss-20b")] * 30)
    local = FakeLocal(["سڵاو."], loaded=["qwen3:8b"])
    app.llm.backends["ollama"] = local
    for text in ("یەک", "دوو", "سێ"):
        [c async for c in app.conversation.respond_stream(text, source="text")]
    request = local.calls[-1][1]
    roles = [m["role"] for m in request.messages]
    assert roles == ["system", "user"] and "SAM's local brain is answering" in request.messages[0]["content"]
    app.config.set("llm.local.history_messages", 2)
    [c async for c in app.conversation.respond_stream("چوار", source="text")]
    assert [m["role"] for m in local.calls[-1][1].messages] == ["system", "user", "assistant", "user"]


@pytest.mark.parametrize("user, reply, claim", [
    ("پێم بڵێ زێڕ ئێستا بە چەند مامەڵە دەکرێت", "زێڕ ئێستا لە ٤٢٦٩ دەبێت.", True),
    # live run 3: a window count answered with the gold price, an alert question from nothing
    ("ئەو پەنجەرانەی ئێستا کراونەتەوە بژمێرە", "چەندە زێڕ لەسەر 4272 مامەڵە دەکرێت.", True),
    ("چ ئاگادارکردنەوەیەکم بۆ زێڕ داناوە؟", "ئاگادارکردنەوەکان بۆ زێڕ نەداناوە.", True),
    ("سڵاو، ئەمڕۆ چۆنی؟", "سڵاو، باشم. تۆ چۆنی؟", False),
    ("what is gold doing", "Gold is around 4269 now.", True),
    ("ئەگەر زێڕ گەیشتە ٤٣٠٠ چی بکەم؟", "کە گەیشتە ٤٣٠٠ چاوەڕێی شکاندن بکە.", False),   # the user's own number
    ("ئۆردەر بلۆک چییە؟", "ئۆردەر بلۆک کۆتا مۆمی پێچەوانەیە پێش جووڵەیەکی بەهێز.", False),
])
def test_a_local_price_without_a_tool_is_not_said(user, reply, claim):
    from sam.brain.responder import Responder

    response = LLMResponse(text=reply, provider="ollama", model="qwen3:8b")
    assert Responder._local_false_claim(response, user) is claim          # noqa: SLF001


def test_a_number_from_the_library_passages_is_grounded():
    from sam.brain.responder import Responder

    response = LLMResponse(text="بەپێی کتێبەکەت، کاتێک زێڕ لە ژێر ٢٠٠ دایە کڕین مەکە.", provider="ollama")
    assert Responder._local_false_claim(response, "ستراتیژییەکەم بۆ زێڕ چی دەڵێت؟") is True           # noqa: SLF001
    assert Responder._local_false_claim(response, "ستراتیژییەکەم بۆ زێڕ چی دەڵێت؟",                    # noqa: SLF001
                                        "[1] «Book», p. 1: ... moving average 200 ...") is False


@tool("get_price", description="Current price.", params={"type": "object", "properties": {"symbol": {"type": "string"}}},
      risk="safe", blocking=True)
async def fake_two_prices(ctx, symbol: str = "") -> dict[str, Any]:
    if "زیو" in symbol:
        return ok("زیو ئێستا لەسەر ٥٢ مامەڵە دەکرێت.", symbol="XAGUSD")
    return ok("زێڕ ئێستا لەسەر ٤٣١٢ مامەڵە دەکرێت.", symbol="XAUUSD")


async def test_every_local_tool_result_is_said_not_only_the_last(make_app):
    app, _ = brain_app(make_app, [rate_limited("groq", "openai/gpt-oss-20b")] * 4, tools=(fake_two_prices,))
    app.llm.backends["ollama"] = FakeLocal([Reply(calls=[("get_price", {"symbol": "زێڕ"}),
                                                         ("get_price", {"symbol": "زیو"})])], loaded=["qwen3:8b"])
    chunks = [c async for c in app.conversation.respond_stream("نرخی زێڕ و زیو پێکەوە", source="text")]
    said = " ".join(c.strip() for c in chunks)
    assert said == "زێڕ ئێستا لەسەر ٤٣١٢ مامەڵە دەکرێت. زیو ئێستا لەسەر ٥٢ مامەڵە دەکرێت."


async def test_no_warm_up_while_a_turn_or_a_local_answer_runs(make_app):
    """Live run 2026-09-25: a voice-prompt warm-up (98 s) started while a typed
    turn waited for the local model, and that answer hit the 150 s timeout."""
    app, _ = brain_app(make_app, [rate_limited("groq", "openai/gpt-oss-20b")] * 4)
    local = FakeLocal(["سڵاو."], loaded=[])
    app.llm.backends["ollama"] = local
    for ref in ("groq:openai/gpt-oss-20b", "groq:openai/gpt-oss-120b"):
        app.llm._cool(ref, 600)                      # noqa: SLF001
    app.conversation.active_turns = 1
    assert app.conversation.prewarm_local_brain() is None
    app.conversation.active_turns = 0
    app.llm._local_inflight = 1                      # noqa: SLF001
    assert app.llm.prewarm_local([], []) is None
    app.llm._local_inflight = 0                      # noqa: SLF001
    task = app.conversation.prewarm_local_brain()
    assert task is not None and await task and local.warmed
