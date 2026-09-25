"""The local brain's backend: Ollama's native ``/api/chat`` on this PC.

Why the native API (not Ollama's OpenAI-compatible /v1): it takes ``think:
false`` (Qwen3 otherwise thinks before every tool call), ``keep_alive`` (the
model stays loaded a few minutes after use) and ``options`` (context size,
reply cap) per request, and it streams NDJSON.

Measured on this PC (2026-09-24, CPU only -- the Radeon 890M crashes Ollama's
Vulkan loader, see local_server.py; 12 threads, ~2 GB RAM free while other
work ran), with SAM's real voice prompt and the 18 compact core tools
(4.1-4.6k prompt tokens), 14 Sorani commands (lead scratchpad sam2/localbrain):

| model | load | generate | first prompt | next prompts | per command | tools right |
| qwen3:8b | 8.3 s | 8.4 tok/s | 86 s | 1-2.7 s (prefix cache) | 2.6-9.2 s | 12/14 |
| qwen3.5:4b | 6.5 s | 14 tok/s | 52 s | 12-16 s (no prefix reuse) | 13-19 s | 13/14 |

qwen3:8b is the default: its prompt cache makes every command after the first
3-5x faster, and its Sorani small talk was natural («سڵاو، فەرموو. چۆنی؟»)
where qwen3.5:4b answered «سڵاو، بە خۆشحاڵی. چیە ئەمڕۆ؟». Its two misses
(«نۆتپاد بکەرەوە», «کرۆم بکەرەوە» echoed as text) are commands the no-AI fast
path answers first (fastpath.py). The cache only helps while the start of the
prompt stays the same, so ``split_context`` moves the per-turn part of SAM's
instruction (time, memory, conversation) next to the user's words.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator, Callable

import httpx

from ..config import LOCAL_BRAIN_DEFAULTS
from ..secrets import redact
from .llm import LLMChunk, LLMError, LLMRequest, LLMResponse, ToolCall
from .persona import CONTEXT_HEADING

log = logging.getLogger("sam.local_brain")

DEFAULTS: dict[str, Any] = dict(LOCAL_BRAIN_DEFAULTS)   # settings llm.local.* (sam/config.py)
_DATA_URL = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.+)$", re.S)


def split_context(system: str) -> tuple[str, str]:
    """(stable part, per-turn part) of SAM's system instruction."""
    head, sep, tail = system.partition(CONTEXT_HEADING)
    if not sep:
        return system, ""
    return head.rstrip(), (sep + tail).strip()


def _text_and_images(content: Any) -> tuple[str, list[str]]:
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[str] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and part.get("text"):
            texts.append(str(part["text"]))
        elif part.get("type") == "image_url":
            match = _DATA_URL.match(str((part.get("image_url") or {}).get("url", "")))
            if match:
                images.append(match.group("data"))
    return "\n".join(texts), images


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"value": value}


def to_ollama_messages(messages: list[dict[str, Any]], *, images: bool = False) -> list[dict[str, Any]]:
    """OpenAI chat messages -> Ollama native messages.

    - the per-turn part of the first system message goes in front of the LAST
      user message (the stable prefix stays cached, see the module doc);
    - assistant tool calls carry their arguments as objects, tool results name
      their tool (``tool_name``); provider-private keys are dropped.
    """
    out: list[dict[str, Any]] = []
    context = ""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    for index, message in enumerate(messages):
        role = message.get("role")
        text, pictures = _text_and_images(message.get("content"))
        if role == "system":
            if not out and not context:
                stable, context = split_context(text)
                out.append({"role": "system", "content": stable})
            else:
                out.append({"role": "system", "content": text})
            continue
        if role == "user":
            if index == last_user and context:
                text = f"{context}\n\n---\n{text}"
            item: dict[str, Any] = {"role": "user", "content": text}
            if pictures and images:
                item["images"] = pictures
            out.append(item)
            continue
        if role == "assistant":
            item = {"role": "assistant", "content": text or ""}
            calls = []
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                calls.append({"function": {"name": function.get("name", ""),
                                           "arguments": _arguments(function.get("arguments"))}})
            if calls:
                item["tool_calls"] = calls
            out.append(item)
            continue
        if role == "tool":
            item = {"role": "tool", "content": text if isinstance(text, str) else json.dumps(text)}
            if message.get("name"):
                item["tool_name"] = message["name"]
            out.append(item)
    return out


class OllamaBackend:
    """``provider = "ollama"``; ``configured()`` is True when the local brain
    is enabled and a server answers or can be started."""

    provider = "ollama"

    def __init__(self, config: Any, server: Any, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self.server = server
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._available_at = 0.0
        self._available = False
        self.last_model = ""

    # -- settings ----------------------------------------------------------------------------------------
    def _setting(self, key: str) -> Any:
        try:
            value = self.config.get(key, DEFAULTS.get(key))
        except Exception:  # noqa: BLE001
            value = DEFAULTS.get(key)
        return DEFAULTS.get(key) if value is None else value

    def enabled(self) -> bool:
        return bool(self._setting("llm.local.enabled"))

    def configured(self) -> bool:
        if not self.enabled():
            return False
        now = time.monotonic()
        if now - self._available_at > 30.0:     # a socket probe + file checks, at most every 30 s
            self._available_at = now
            try:
                self._available = bool(self.server.available())
            except Exception:  # noqa: BLE001
                self._available = False
        return self._available

    def model(self) -> str:
        return str(self._setting("llm.local.model") or "qwen3:8b")

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=3.0), trust_env=False,
                                             transport=self._transport)
        return self._client

    def _url(self, path: str) -> str:
        return f"{self.server.base_url}{path}"

    # -- requests ----------------------------------------------------------------------------------------
    def _body(self, model: str, req: LLMRequest, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model, "stream": stream,
            "messages": to_ollama_messages(req.messages, images=bool(self._setting("llm.local.vision"))),
            "keep_alive": str(self._setting("llm.local.keep_alive")),
            "options": {"num_ctx": int(self._setting("llm.local.num_ctx")),
                        "num_predict": max(64, min(int(req.max_tokens or 1024), int(self._setting("llm.local.max_tokens")))),
                        "temperature": float(req.temperature if req.temperature is not None
                                             else self._setting("llm.local.temperature"))}}
        if not self._setting("llm.local.think"):
            body["think"] = False
        if req.tools and req.tool_choice != "none":
            body["tools"] = req.tools
        if req.json_schema is not None:
            body["format"] = req.json_schema
        return body

    def _error(self, model: str, status: int, text: str) -> LLMError:
        message = redact(text or "")[:300]
        lowered = message.lower()
        if status == 404 or ("not found" in lowered and "model" in lowered):
            return LLMError("not_found", message, provider="ollama", model=model, status=status)
        if status == 400:
            return LLMError("bad_request", message, provider="ollama", model=model, status=status)
        return LLMError("server", message, provider="ollama", model=model, status=status)

    async def _ready(self, model: str) -> None:
        if not await self.server.ensure():
            raise LLMError("network", getattr(self.server, "last_error", "") or "local server unavailable",
                           provider="ollama", model=model)

    @staticmethod
    def _calls(message: dict[str, Any]) -> list[ToolCall]:
        calls = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            if not function.get("name"):
                continue
            call_id = str(item.get("id") or "") or "call_" + uuid.uuid4().hex[:16]
            calls.append(ToolCall(id=call_id, name=str(function["name"]), arguments=_arguments(function.get("arguments"))))
        return calls

    def _response(self, model: str, text: str, calls: list[ToolCall], final: dict[str, Any], ttft: float | None,
                  started: float) -> LLMResponse:
        raw: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            raw["tool_calls"] = [c.as_openai() for c in calls]
        total = (time.perf_counter() - started) * 1000.0
        self.last_model = model
        return LLMResponse(text=text, tool_calls=calls, provider="ollama", model=model,
                           finish_reason=str(final.get("done_reason") or "stop"),
                           usage={"tokens_in": int(final.get("prompt_eval_count") or 0),
                                  "tokens_out": int(final.get("eval_count") or 0)},
                           ttft_ms=ttft if ttft is not None else total, total_ms=total, raw_message=raw)

    async def complete(self, model: str, req: LLMRequest) -> LLMResponse:
        await self._ready(model)
        started = time.perf_counter()
        body = self._body(model, req, False)
        try:
            response = await self._http().post(self._url("/api/chat"), json=body, timeout=req.timeout_s)
            if response.status_code == 400 and "think" in response.text.lower() and "think" in body:
                body.pop("think")          # a model without a thinking switch
                response = await self._http().post(self._url("/api/chat"), json=body, timeout=req.timeout_s)
        except httpx.TimeoutException as exc:
            raise LLMError("timeout", type(exc).__name__, provider="ollama", model=model) from None
        except httpx.HTTPError as exc:
            raise LLMError("network", type(exc).__name__, provider="ollama", model=model) from None
        if response.status_code >= 400:
            raise self._error(model, response.status_code, _error_text(response.text))
        try:
            payload = response.json()
        except ValueError:
            raise LLMError("server", "unparseable response", provider="ollama", model=model) from None
        message = payload.get("message") or {}
        return self._response(model, str(message.get("content") or ""), self._calls(message), payload, None, started)

    async def stream(self, model: str, req: LLMRequest) -> AsyncIterator[LLMChunk]:
        await self._ready(model)
        started = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        calls: list[ToolCall] = []
        final: dict[str, Any] = {}
        try:
            async with self._http().stream("POST", self._url("/api/chat"), json=self._body(model, req, True),
                                           timeout=req.timeout_s) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise self._error(model, response.status_code, _error_text(body))
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("error"):
                        raise self._error(model, 500, str(event["error"]))
                    message = event.get("message") or {}
                    content = message.get("content")
                    if content:
                        if ttft is None:
                            ttft = (time.perf_counter() - started) * 1000.0
                        parts.append(content)
                        yield LLMChunk(kind="text", text=content)
                    calls.extend(self._calls(message))
                    if event.get("done"):
                        final = event
        except httpx.TimeoutException as exc:
            raise LLMError("timeout", type(exc).__name__, provider="ollama", model=model) from None
        except httpx.HTTPError as exc:
            raise LLMError("network", type(exc).__name__, provider="ollama", model=model) from None
        for call in calls:
            if ttft is None:
                ttft = (time.perf_counter() - started) * 1000.0
            yield LLMChunk(kind="tool_call", tool_call=call)
        yield LLMChunk(kind="done", response=self._response(model, "".join(parts), calls, final, ttft, started))

    async def list_models(self) -> list[str]:
        if not await self.server.ensure():
            raise LLMError("network", "local server unavailable", provider="ollama")
        try:
            response = await self._http().get(self._url("/api/tags"), timeout=10)
        except httpx.HTTPError as exc:
            raise LLMError("network", type(exc).__name__, provider="ollama") from None
        if response.status_code >= 400:
            raise self._error("", response.status_code, _error_text(response.text))
        try:
            return [str(m.get("name") or m.get("model")) for m in response.json().get("models") or []]
        except (ValueError, AttributeError):
            return []

    async def loaded(self) -> list[str]:
        """Models in memory now (``/api/ps``) -- no start, no load."""
        if not self.server.listening():
            return []
        try:
            response = await self._http().get(self._url("/api/ps"), timeout=3)
            return [str(m.get("name")) for m in response.json().get("models") or []]
        except Exception:  # noqa: BLE001
            return []

    async def warm(self, model: str, messages: list[dict[str, Any]] | None = None,
                   tools: list[dict[str, Any]] | None = None) -> bool:
        """Load ``model`` and (with ``messages``) read the stable prompt into
        its cache: one token generated. A cold first answer cost ~94 s on this
        PC; after this the first real one costs a few seconds."""
        try:
            await self._ready(model)
            body: dict[str, Any] = {"model": model, "stream": False, "keep_alive": str(self._setting("llm.local.keep_alive")),
                                    "messages": [], "options": {"num_ctx": int(self._setting("llm.local.num_ctx")),
                                                                "num_predict": 1}}
            if messages:
                body["messages"] = to_ollama_messages(messages)
                body["think"] = False
                if tools:
                    body["tools"] = tools
            response = await self._http().post(self._url("/api/chat"), json=body, timeout=300)
            return response.status_code < 400
        except Exception as exc:  # noqa: BLE001 - best effort
            log.info("local brain warm-up failed: %s", type(exc).__name__)
            return False

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()
        if self._setting("llm.local.stop_on_quit"):
            try:
                await self.server.stop()
            except Exception:  # noqa: BLE001
                log.exception("stopping the local brain failed")


def _error_text(body: str) -> str:
    try:
        return str(json.loads(body).get("error") or body)
    except (ValueError, AttributeError):
        return body


def make_backend(config: Any, *, transport: httpx.AsyncBaseTransport | None = None,
                 server_factory: Callable[[Any], Any] | None = None) -> OllamaBackend:
    from .local_server import OllamaServer

    server = server_factory(config) if server_factory is not None else OllamaServer(config)
    return OllamaBackend(config, server, transport=transport)


__all__ = ["OllamaBackend", "make_backend", "to_ollama_messages", "split_context", "DEFAULTS"]
