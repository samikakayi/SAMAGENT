from __future__ import annotations

import json

import httpx
import pytest
from conftest import FAKE_GROQ, FakeBackend, rate_limited

from sam.brain.llm import LLMClient, LLMError, LLMRequest, ToolCall, parse_json_text, split_ref
from sam.brain.llm_backends import GeminiBackend, OpenAICompatBackend, classify_status


class StubConfig:
    def __init__(self, **values):
        self.values = {"llm.ladder.chat": ["groq:m1", "omniroute:m2", "gemini:m3"], "llm.cooldown_429_s": 60,
                       "llm.max_tokens": 1000, "llm.reasoning": "low", "llm.timeout_s": 5, **values}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def all(self):
        return dict(self.values)


def client(backends, db=None, **config):
    return LLMClient(StubConfig(**config), secrets=None, db=db, backends=backends)


MSGS = [{"role": "user", "content": "سڵاو"}]


def test_split_ref_and_json_parsing():
    assert split_ref("groq:openai/gpt-oss-20b") == ("groq", "openai/gpt-oss-20b")
    with pytest.raises(ValueError):
        split_ref("nothing")
    assert parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_text('Here: {"b": [1]} ok') == {"b": [1]}


async def test_first_rung_answers_and_max_tokens_floor():
    groq = FakeBackend("groq")
    llm = client({"groq": groq, "omniroute": FakeBackend("omniroute")})
    response = await llm.chat(MSGS)
    assert response.text == "reply from groq:m1"
    request = groq.calls[0][1]
    assert request.max_tokens == 4096 and request.reasoning == "low"


async def test_429_cools_down_and_falls_through():
    groq = FakeBackend("groq", {"m1": [rate_limited("groq", "m1", retry_after=120)]})
    omni = FakeBackend("omniroute")
    llm = client({"groq": groq, "omniroute": omni})
    assert (await llm.chat(MSGS)).provider == "omniroute"
    assert len(groq.calls) == 1  # no retry on 429
    # Second call skips the cooling rung entirely.
    assert (await llm.chat(MSGS)).provider == "omniroute"
    assert len(groq.calls) == 1
    assert "groq:m1" in llm.status()["cooling"]


async def test_one_retry_on_server_error_then_success():
    groq = FakeBackend("groq", {"m1": [LLMError("server", "503", provider="groq", model="m1", status=503), "second"]})
    llm = client({"groq": groq})
    assert (await llm.chat(MSGS)).text == "second"
    assert len(groq.calls) == 2


async def test_only_one_retry_then_next_rung():
    err = LLMError("timeout", provider="groq", model="m1")
    groq = FakeBackend("groq", {"m1": [err, err, "never"]})
    omni = FakeBackend("omniroute")
    llm = client({"groq": groq, "omniroute": omni})
    assert (await llm.chat(MSGS)).provider == "omniroute"
    assert len(groq.calls) == 2


async def test_auth_failure_cools_whole_provider():
    groq = FakeBackend("groq", {"m1": [LLMError("auth", provider="groq", model="m1", status=401)]})
    llm = client({"groq": groq, "omniroute": FakeBackend("omniroute")},
                 **{"llm.ladder.chat": ["groq:m1", "groq:m9", "omniroute:m2"]})
    assert (await llm.chat(MSGS)).provider == "omniroute"
    assert [m for m, _ in groq.calls] == ["m1"]  # m9 skipped: provider cooling


async def test_bad_request_about_reasoning_retries_without_it():
    groq = FakeBackend("groq", {"m1": [LLMError("bad_request", "unknown field reasoning_effort", provider="groq",
                                                model="m1", status=400), "fine"]})
    llm = client({"groq": groq})
    assert (await llm.chat(MSGS)).text == "fine"
    assert groq.calls[0][1].reasoning == "low" and groq.calls[1][1].reasoning is None


async def test_unconfigured_skipped_and_exhausted_lists_attempts():
    llm = client({"groq": FakeBackend("groq", configured=False),
                  "omniroute": FakeBackend("omniroute", {"m2": [LLMError("network", provider="omniroute", model="m2")]})})
    with pytest.raises(LLMError) as info:
        await llm.chat(MSGS)
    assert info.value.kind == "exhausted"
    assert info.value.attempts[0] == "groq:m1: unconfigured"
    assert any(a.startswith("omniroute:m2: network") for a in info.value.attempts)
    assert "gemini:m3: unconfigured" in info.value.attempts


async def test_daily_cap_and_usage_counters(tmp_path):
    from sam.db import Database

    db = Database(tmp_path / "u.sqlite3")
    groq, omni = FakeBackend("groq"), FakeBackend("omniroute")
    llm = client({"groq": groq, "omniroute": omni}, db=db, **{"llm.daily_caps": {"groq:m1": 2}})
    for _ in range(3):
        await llm.chat(MSGS)
    assert len(groq.calls) == 2 and len(omni.calls) == 1
    assert db.usage_for("groq", "m1")["requests"] == 2
    assert db.usage_for("groq", "m1")["tokens_out"] == 10


async def test_stream_falls_back_before_first_chunk_only():
    groq = FakeBackend("groq", {"m1": [rate_limited("groq", "m1")]})
    omni = FakeBackend("omniroute", {"m2": ["hello there friend"]})
    llm = client({"groq": groq, "omniroute": omni})
    chunks = [c async for c in llm.stream(MSGS)]
    assert "".join(c.text for c in chunks if c.kind == "text").strip() == "hello there friend"
    assert chunks[-1].kind == "done" and chunks[-1].response.provider == "omniroute"


async def test_empty_stream_is_a_hidden_rate_limit_and_falls_back():
    """OmniRoute answered HTTP 200 + keepalives + nothing when rate-limited
    (measured 2026-09-24): the ladder must move on, not return ''."""
    from sam.brain.llm import LLMChunk, LLMResponse

    class Empty(FakeBackend):
        async def stream(self, model, req):
            self.calls.append((model, req))
            yield LLMChunk(kind="done", response=LLMResponse(text="", provider=self.provider, model=model))

    groq = Empty("groq")
    omni = FakeBackend("omniroute", {"m2": ["سڵاو"]})
    llm = client({"groq": groq, "omniroute": omni})
    chunks = [c async for c in llm.stream(MSGS)]
    assert chunks[-1].response.provider == "omniroute" and len(groq.calls) == 1   # no same-rung retry
    assert llm.status()["cooling"]                                                   # groq:m1 cools down

    class EmptyButFinished(FakeBackend):   # a real empty answer ("stop") is passed through
        async def stream(self, model, req):
            yield LLMChunk(kind="done", response=LLMResponse(text="", provider=self.provider, model=model,
                                                             finish_reason="stop"))

    chunks = [c async for c in client({"groq": EmptyButFinished("groq")}).stream(MSGS)]
    assert chunks[-1].response.finish_reason == "stop"


async def test_stream_error_after_text_is_raised():
    class Broken(FakeBackend):
        async def stream(self, model, req):
            from sam.brain.llm import LLMChunk
            yield LLMChunk(kind="text", text="نیو")
            raise LLMError("network", provider="groq", model=model)

    llm = client({"groq": Broken("groq"), "omniroute": FakeBackend("omniroute")})
    seen = []
    with pytest.raises(LLMError):
        async for chunk in llm.stream(MSGS):
            seen.append(chunk.text)
    assert seen == ["نیو"]


# --- OpenAI-compatible backend over a mock transport -------------------------------

def _backend(handler, provider="groq"):
    return OpenAICompatBackend(provider, lambda: "https://api.example/v1", lambda: FAKE_GROQ,
                               transport=httpx.MockTransport(handler))


async def test_openai_backend_tool_call_and_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "open_app", "arguments": "{\"name\": \"TradingView\"}"}}]}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7}})

    backend = _backend(handler)
    req = LLMRequest(messages=[{"role": "user", "content": "x", "_private": 1}],
                     tools=[{"type": "function", "function": {"name": "open_app", "parameters": {}}}],
                     reasoning="minimal")
    response = await backend.complete("openai/gpt-oss-20b", req)
    assert response.tool_calls == [ToolCall(id="call_1", name="open_app", arguments={"name": "TradingView"})]
    assert response.usage == {"tokens_in": 12, "tokens_out": 7}
    assert response.assistant_message()["tool_calls"][0]["function"]["name"] == "open_app"
    assert seen["auth"] == f"Bearer {FAKE_GROQ}"
    assert seen["body"]["reasoning_effort"] == "low" and seen["body"]["max_tokens"] == 4096
    assert "_private" not in seen["body"]["messages"][0]
    await backend.aclose()


async def test_openai_backend_sse_stream_with_tool_deltas():
    lines = [
        {"choices": [{"delta": {"content": "باشە، "}}]},
        {"choices": [{"delta": {"content": "دەیکەمەوە."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c9", "extra_content": {"google": {"thought_signature": "s"}},
                                                "function": {"name": "open_", "arguments": "{\"na"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "app", "arguments": "me\": \"Chrome\"}"}}]}, "finish_reason": "tool_calls"}]},
    ]
    body = "".join(f"data: {json.dumps(l, ensure_ascii=False)}\n\n" for l in lines) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=body.encode("utf-8"), headers={"content-type": "text/event-stream"})

    backend = _backend(handler)
    chunks = [c async for c in backend.stream("m", LLMRequest(messages=MSGS))]
    assert "".join(c.text for c in chunks if c.kind == "text") == "باشە، دەیکەمەوە."
    calls = [c.tool_call for c in chunks if c.kind == "tool_call"]
    assert calls == [ToolCall(id="c9", name="open_app", arguments={"name": "Chrome"})]
    assert chunks[-1].response.finish_reason == "tool_calls"
    kept = chunks[-1].response.assistant_message()["tool_calls"][0]
    assert kept["extra_content"] == {"google": {"thought_signature": "s"}}
    await backend.aclose()


async def test_openai_backend_keeps_provider_extras_and_strips_tool_names():
    seen = {}
    signature = {"google": {"thought_signature": "c2lnbmF0dXJl"}}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "extra_content": signature,
             "function": {"name": "chart_state", "arguments": "{}"}}]}}]})

    backend = _backend(handler, provider="omniroute")
    history = [{"role": "user", "content": "x"},
               {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function",
                                                                     "function": {"name": "t", "arguments": "{}"}}]},
               {"role": "tool", "tool_call_id": "c0", "name": "t", "content": "{}"}]
    response = await backend.complete("sam-fast", LLMRequest(messages=history))
    sent = seen["body"]["messages"]
    assert "name" not in sent[2] and sent[1]["content"] is None
    assert response.assistant_message()["tool_calls"][0]["extra_content"] == signature
    await backend.aclose()


async def test_openai_backend_429_maps_to_rate_limit_without_leaking_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "7"},
                              json={"error": {"message": f"Rate limit for key {FAKE_GROQ}"}})

    backend = _backend(handler)
    with pytest.raises(LLMError) as info:
        await backend.complete("m", LLMRequest(messages=MSGS))
    assert info.value.kind == "rate_limit" and info.value.retry_after == 7.0
    assert FAKE_GROQ not in str(info.value) and FAKE_GROQ not in info.value.message
    await backend.aclose()


async def test_openai_backend_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    backend = _backend(handler, provider="omniroute")
    with pytest.raises(LLMError) as info:
        await backend.complete("sam-fast", LLMRequest(messages=MSGS))
    assert info.value.kind == "network"
    await backend.aclose()


@pytest.mark.parametrize("status,body,kind", [(429, "", "rate_limit"), (401, "", "auth"), (403, "quota", "quota"),
                                              (404, "", "not_found"), (400, "model does not exist", "not_found"),
                                              (400, "bad", "bad_request"), (503, "", "server")])
def test_status_classification(status, body, kind):
    assert classify_status(status, body) == kind


# --- Gemini conversion (no network) -------------------------------------------------

def test_gemini_contents_conversion():
    from google.genai import types

    backend = GeminiBackend(lambda: None)
    image = "data:image/png;base64," + "iVBORw0KGgo="
    messages = [
        {"role": "system", "content": "RESPOND IN CENTRAL KURDISH (SORANI), ARABIC SCRIPT"},
        {"role": "user", "content": [{"type": "text", "text": "چی دەبینیت؟"}, {"type": "image_url", "image_url": {"url": image}}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_x", "type": "function", "function": {"name": "chart_state", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_x", "content": json.dumps({"ok": True, "summary": "XAUUSD M15"})},
    ]
    system, contents = backend.to_contents(messages)
    assert system.startswith("RESPOND IN CENTRAL KURDISH")
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[0].parts[1].inline_data.mime_type == "image/png"
    # Foreign tool history (no Gemini signature) is replayed as text.
    assert "chart_state" in contents[1].parts[0].text and "XAUUSD M15" in contents[2].parts[0].text
    # A native Gemini turn is replayed verbatim with a function_response.
    native = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="get_price", args={}))])
    messages2 = [{"role": "user", "content": "نرخ"},
                 {"role": "assistant", "content": "", "_gemini_content": native.model_dump(mode="json", exclude_none=True),
                  "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_price", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\": true}"}]
    _, contents2 = backend.to_contents(messages2)
    assert contents2[1].parts[0].function_call.name == "get_price"
    assert contents2[2].parts[0].function_response.name == "get_price"
    assert contents2[2].parts[0].function_response.id is None


def test_gemini_config_sets_thinking_and_tools():
    from google.genai import types

    backend = GeminiBackend(lambda: None)
    req = LLMRequest(messages=MSGS, reasoning="minimal", json_schema={"type": "object"},
                     tools=[{"type": "function", "function": {"name": "t", "description": "d",
                                                              "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}])
    config = backend._config(req, "sys")
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
    assert config.max_output_tokens == 4096
    assert config.tools[0].function_declarations[0].name == "t"
    assert config.tools[0].function_declarations[0].behavior is None
    assert config.response_mime_type == "application/json"


async def test_gemini_backend_with_fake_sdk_client():
    from google.genai import types

    class FakeModels:
        async def generate_content(self, model, contents, config):
            return types.GenerateContentResponse(
                candidates=[types.Candidate(content=types.Content(role="model", parts=[
                    types.Part(text="باشە"), types.Part(function_call=types.FunctionCall(name="open_app", args={"name": "Chrome"}),
                                                       thought_signature=b"sig")]))],
                usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=5, candidates_token_count=2,
                                                                          thoughts_token_count=3))

    class FakeClient:
        class aio:  # noqa: N801
            models = FakeModels()

    backend = GeminiBackend(lambda: "AIza" + "x" * 30, client_factory=lambda key: FakeClient())
    response = await backend.complete("gemini-3.5-flash-lite", LLMRequest(messages=MSGS))
    assert response.text == "باشە" and response.tool_calls[0].name == "open_app"
    assert response.usage == {"tokens_in": 5, "tokens_out": 5}
    native = response.raw_message["_gemini_content"]
    assert native["parts"][1]["thought_signature"]  # signature kept for the next turn


async def test_slow_server_failure_moves_on_without_a_retry_and_cools_the_rung():
    """Integration smoke 2026-09-24: OmniRoute answered 503 after 39 s and again
    after 34 s on the same-rung retry. A slow failure must not be retried."""
    err = LLMError("server", "503", provider="groq", model="m1", status=503)
    groq = FakeBackend("groq", {"m1": [err, "never"]})
    omni = FakeBackend("omniroute")
    llm = client({"groq": groq, "omniroute": omni}, **{"llm.slow_failure_s": 0})   # every failure counts as slow
    assert (await llm.chat(MSGS)).provider == "omniroute"
    assert len(groq.calls) == 1
    assert "groq:m1" in llm.status()["cooling"]
    # A quick 5xx still gets its single retry (test_one_retry_on_server_error_then_success).
