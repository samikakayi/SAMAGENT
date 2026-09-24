"""LLM backends: OpenAI-compatible HTTP (OmniRoute, Groq, OpenRouter) and
Gemini direct (google-genai).

Keys are read at call time from ``Secrets`` and sent only to their own
provider; error messages are redacted and never include request headers.
``httpx`` clients use ``trust_env=False`` (no system proxy may see keys) and
keep connections alive: a fresh TLS handshake cost 0.25-0.35 s per call when
measured (reports/realtime-voice.json).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Callable

import httpx

from ..secrets import redact
from .llm import LLMChunk, LLMError, LLMRequest, LLMResponse, ToolCall

log = logging.getLogger("sam.llm")


def _strip_private(message: dict[str, Any]) -> dict[str, Any]:
    """Provider-private keys ("_gemini_content") never go to OpenAI-style
    APIs; neither does "name" on tool messages (not in the tool-message schema;
    the Gemini backend uses it, strict validators may reject it)."""
    out = {k: v for k, v in message.items() if not k.startswith("_")}
    if out.get("role") == "tool":
        out.pop("name", None)
    if out.get("role") == "assistant" and out.get("tool_calls") and not out.get("content"):
        out["content"] = None
    return out


def _openai_tool_calls(calls: list[ToolCall], originals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assistant tool_calls for the history, keeping any provider extras of the
    original call (e.g. Gemini's OpenAI-compat ``extra_content`` thought
    signature, which Gemini 3 requires back on the next turn)."""
    out = []
    for index, call in enumerate(calls):
        item = call.as_openai()
        original = originals[index] if index < len(originals) else {}
        for key, value in original.items():
            if key not in ("index", "id", "type", "function") and value is not None:
                item[key] = value
        out.append(item)
    return out


def _new_call_id() -> str:
    import uuid
    return "call_" + uuid.uuid4().hex[:16]


def _retry_after(headers: httpx.Headers | dict[str, str] | None, body: str = "") -> float | None:
    if headers is not None:
        value = headers.get("retry-after")
        if value:
            try:
                return float(value)
            except ValueError:
                pass
    match = re.search(r"retry(?:Delay| in)[\"':\s]+(\d+(?:\.\d+)?)s", body or "", re.I)
    return float(match.group(1)) if match else None


def classify_status(status: int, body: str) -> str:
    lowered = (body or "").lower()
    if status == 429:
        return "rate_limit"
    if status in (401, 403):
        if "quota" in lowered or "billing" in lowered:
            return "quota"
        return "auth"
    if status == 402:
        return "quota"
    if status == 404:
        return "not_found"
    if status in (400, 413, 422):
        if "model" in lowered and ("not found" in lowered or "does not exist" in lowered or "decommissioned" in lowered):
            return "not_found"
        return "bad_request"
    if status >= 500:
        return "server"
    return "bad_request"


# --- OpenAI-compatible ---------------------------------------------------------

class OpenAICompatBackend:
    """Chat Completions over httpx; SSE streaming; tool calls."""

    def __init__(self, provider: str, base_url: Callable[[], str], api_key: Callable[[], str | None], *,
                 extra_headers: dict[str, str] | None = None, transport: httpx.AsyncBaseTransport | None = None,
                 require_key: bool = True) -> None:
        self.provider = provider
        self._base_url = base_url
        self._api_key = api_key
        self._extra_headers = extra_headers or {}
        self._transport = transport
        self._require_key = require_key
        self._client: httpx.AsyncClient | None = None

    def configured(self) -> bool:
        return bool(self._api_key()) or not self._require_key

    def _http(self) -> httpx.AsyncClient:
        # keepalive_expiry 120 s: httpx's default 5 s closed the connection between
        # spoken turns, so each paid a new TLS handshake (Groq 476 vs 277 ms after
        # 8 s idle, repair review keepalive_probe.py, 2026-09-24).
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0), trust_env=False,
                                             transport=self._transport,
                                             limits=httpx.Limits(max_connections=20, max_keepalive_connections=8,
                                                                 keepalive_expiry=120.0))
        return self._client

    async def warm(self) -> bool:
        """Open the TLS connection ahead of the first turn: a HEAD on the API
        host without any key (no quota, nothing sent but the request line)."""
        if not self.configured():
            return False
        try:
            base = httpx.URL(self._base_url())
            response = await self._http().head(f"{base.scheme}://{base.host}/", timeout=5.0)
            return response.status_code < 500
        except Exception as exc:  # noqa: BLE001 - best effort
            log.debug("warm-up of %s failed: %s", self.provider, type(exc).__name__)
            return False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._extra_headers}
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _reasoning(self, model: str, effort: str | None) -> dict[str, Any]:
        if not effort:
            return {}
        if self.provider == "openrouter":
            return {"reasoning": {"effort": "low" if effort == "minimal" else effort}}
        if self.provider == "groq":
            # gpt-oss accepts low|medium|high; other Groq models reject the field.
            if "gpt-oss" in model:
                return {"reasoning_effort": "low" if effort == "minimal" else effort}
            return {}
        # OmniRoute forwards OpenAI params to Gemini (which knows "minimal").
        return {"reasoning_effort": effort}

    def _body(self, model: str, req: LLMRequest, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": [_strip_private(m) for m in req.messages],
                                "max_tokens": req.max_tokens, "stream": stream}
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.tools:
            body["tools"] = req.tools
            if req.tool_choice:
                body["tool_choice"] = req.tool_choice
        if req.json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "result", "schema": req.json_schema, "strict": False}}
        body.update(self._reasoning(model, req.reasoning))
        return body

    def _error(self, model: str, status: int, text: str, headers: httpx.Headers | None) -> LLMError:
        message = redact(text or "")[:300]
        return LLMError(classify_status(status, text), message, provider=self.provider, model=model,
                        status=status, retry_after=_retry_after(headers, text))

    async def complete(self, model: str, req: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = await self._http().post(f"{self._base_url().rstrip('/')}/chat/completions",
                                               headers=self._headers(), json=self._body(model, req, False),
                                               timeout=req.timeout_s)
        except httpx.TimeoutException as exc:
            raise LLMError("timeout", type(exc).__name__, provider=self.provider, model=model) from None
        except httpx.HTTPError as exc:
            raise LLMError("network", type(exc).__name__, provider=self.provider, model=model) from None
        if response.status_code >= 400:
            raise self._error(model, response.status_code, response.text, response.headers)
        try:
            payload = response.json()
            choice = (payload.get("choices") or [{}])[0]
            message = choice.get("message") or {}
        except (ValueError, AttributeError, IndexError):
            raise LLMError("server", "unparseable response", provider=self.provider, model=model) from None
        originals = [c for c in message.get("tool_calls") or [] if isinstance(c, dict)]
        calls = [self._tool_call(c) for c in originals]
        usage = payload.get("usage") or {}
        total = (time.perf_counter() - started) * 1000.0
        text = message.get("content") or ""
        raw: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            raw["tool_calls"] = _openai_tool_calls(calls, originals)
        return LLMResponse(text=text, tool_calls=calls, provider=self.provider, model=model,
                           finish_reason=choice.get("finish_reason"),
                           usage={"tokens_in": int(usage.get("prompt_tokens") or 0),
                                  "tokens_out": int(usage.get("completion_tokens") or 0)},
                           ttft_ms=total, total_ms=total, raw_message=raw)

    @staticmethod
    def _tool_call(item: dict[str, Any]) -> ToolCall:
        function = item.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        return ToolCall(id=item.get("id") or _new_call_id(), name=function.get("name") or "", arguments=args or {})

    async def stream(self, model: str, req: LLMRequest) -> AsyncIterator[LLMChunk]:
        started = time.perf_counter()
        ttft: float | None = None
        text_parts: list[str] = []
        pending: dict[int, dict[str, Any]] = {}
        finish: str | None = None
        usage: dict[str, Any] = {}
        try:
            async with self._http().stream("POST", f"{self._base_url().rstrip('/')}/chat/completions",
                                           headers=self._headers(), json=self._body(model, req, True),
                                           timeout=req.timeout_s) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise self._error(model, response.status_code, body, response.headers)
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if ttft is None:
                                ttft = (time.perf_counter() - started) * 1000.0
                            text_parts.append(content)
                            yield LLMChunk(kind="text", text=content)
                        for tc in delta.get("tool_calls") or []:
                            slot = pending.setdefault(int(tc.get("index", 0)),
                                                      {"id": None, "name": "", "args": "", "extra": {}})
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            function = tc.get("function") or {}
                            if function.get("name"):
                                slot["name"] += function["name"]
                            if function.get("arguments"):
                                slot["args"] += function["arguments"]
                            for key, value in tc.items():
                                if key not in ("index", "id", "type", "function") and value is not None:
                                    slot["extra"][key] = value
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
        except httpx.TimeoutException as exc:
            raise LLMError("timeout", type(exc).__name__, provider=self.provider, model=model) from None
        except httpx.HTTPError as exc:
            raise LLMError("network", type(exc).__name__, provider=self.provider, model=model) from None
        slots = [s for _, s in sorted(pending.items())]
        calls = [self._tool_call({"id": s["id"], "function": {"name": s["name"], "arguments": s["args"] or "{}"}})
                 for s in slots]
        for call in calls:
            if ttft is None:
                ttft = (time.perf_counter() - started) * 1000.0
            yield LLMChunk(kind="tool_call", tool_call=call)
        text = "".join(text_parts)
        raw: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            raw["tool_calls"] = _openai_tool_calls(calls, [s["extra"] for s in slots])
        yield LLMChunk(kind="done", response=LLMResponse(
            text=text, tool_calls=calls, provider=self.provider, model=model, finish_reason=finish,
            usage={"tokens_in": int(usage.get("prompt_tokens") or 0),
                   "tokens_out": int(usage.get("completion_tokens") or 0)},
            ttft_ms=ttft, total_ms=(time.perf_counter() - started) * 1000.0, raw_message=raw))

    async def list_models(self) -> list[str]:
        try:
            response = await self._http().get(f"{self._base_url().rstrip('/')}/models", headers=self._headers(),
                                              timeout=15)
        except httpx.HTTPError as exc:
            raise LLMError("network", type(exc).__name__, provider=self.provider) from None
        if response.status_code >= 400:
            raise self._error("", response.status_code, response.text, response.headers)
        try:
            return [str(m.get("id")) for m in (response.json().get("data") or []) if m.get("id")]
        except (ValueError, AttributeError):
            return []

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


# --- Gemini direct (google-genai) ------------------------------------------------

_DATA_URL = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.+)$", re.S)


class GeminiBackend:
    """``client.aio.models.generate_content(_stream)`` with thinking level set.

    Gemini 3 validates thought signatures on multi-turn function calling; the
    model's own turns are therefore kept verbatim in ``_gemini_content`` and
    replayed as-is. Tool history produced by another provider (no signature)
    is replayed as plain text instead of synthetic function-call parts.
    """

    provider = "gemini"

    def __init__(self, api_key: Callable[[], str | None], *, client_factory: Callable[[str], Any] | None = None) -> None:
        self._api_key = api_key
        self._client_factory = client_factory
        self._client: Any = None
        self._client_key_fp: str | None = None

    def configured(self) -> bool:
        return bool(self._api_key())

    def _genai(self) -> Any:
        key = self._api_key()
        if not key:
            raise LLMError("unconfigured", provider="gemini")
        import hashlib
        fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]
        if self._client is None or fingerprint != self._client_key_fp:
            if self._client_factory is not None:
                self._client = self._client_factory(key)
            else:
                from google import genai
                self._client = genai.Client(api_key=key)
            self._client_key_fp = fingerprint
        return self._client

    # -- conversion -------------------------------------------------------------
    @staticmethod
    def _parts_from_content(content: Any, types: Any) -> list[Any]:
        if isinstance(content, str):
            return [types.Part(text=content)] if content else []
        parts = []
        for item in content or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                parts.append(types.Part(text=item["text"]))
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                match = _DATA_URL.match(url)
                if match:
                    parts.append(types.Part.from_bytes(data=base64.b64decode(match.group("data")),
                                                       mime_type=match.group("mime")))
        return parts

    def to_contents(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[Any]]:
        from google.genai import types

        system: list[str] = []
        contents: list[Any] = []
        names_by_id: dict[str, str] = {}
        native_ids: set[str] = set()
        tool_block = -1  # index of the user Content collecting consecutive tool results
        for message in messages:
            role = message.get("role")
            if role == "system":
                if isinstance(message.get("content"), str):
                    system.append(message["content"])
                continue
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    names_by_id[call.get("id", "")] = (call.get("function") or {}).get("name", "")
                native = message.get("_gemini_content")
                if native:
                    contents.append(types.Content.model_validate(native))
                    native_ids.update(c.get("id", "") for c in message.get("tool_calls") or [])
                    continue
                parts = self._parts_from_content(message.get("content"), types)
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    parts.append(types.Part(text=f"[I called tool {function.get('name')} with "
                                                 f"{function.get('arguments', '{}')}]"))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue
            if role == "tool":
                call_id = message.get("tool_call_id", "")
                name = message.get("name") or names_by_id.get(call_id, "tool")
                raw = message.get("content")
                try:
                    response = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    response = {"result": raw}
                if not isinstance(response, dict):
                    response = {"result": response}
                if call_id in native_ids:
                    part = types.Part(function_response=types.FunctionResponse(
                        id=call_id if not call_id.startswith("call_") else None, name=name, response=response))
                else:
                    part = types.Part(text=f"[Result of tool {name}: {json.dumps(response, ensure_ascii=False)[:4000]}]")
                if tool_block == len(contents) - 1 and tool_block >= 0:
                    contents[-1].parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))
                    tool_block = len(contents) - 1
                continue
            parts = self._parts_from_content(message.get("content"), types)
            if parts:
                contents.append(types.Content(role="user", parts=parts))
        return ("\n\n".join(system) or None), contents

    def _config(self, req: LLMRequest, system: str | None) -> Any:
        from google.genai import types

        kwargs: dict[str, Any] = {"max_output_tokens": req.max_tokens}
        if system:
            kwargs["system_instruction"] = system
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.reasoning:
            level = {"minimal": types.ThinkingLevel.MINIMAL, "low": types.ThinkingLevel.LOW,
                     "medium": types.ThinkingLevel.MEDIUM, "high": types.ThinkingLevel.HIGH}.get(req.reasoning)
            if level is not None:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        if req.tools:
            declarations = []
            for item in req.tools:
                function = item.get("function") or {}
                decl: dict[str, Any] = {"name": function.get("name"), "description": function.get("description", "")}
                params = function.get("parameters") or {}
                if params.get("properties"):
                    decl["parameters_json_schema"] = params
                declarations.append(types.FunctionDeclaration(**decl))
            kwargs["tools"] = [types.Tool(function_declarations=declarations)]
            kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
            if req.tool_choice in ("none", "required"):
                mode = types.FunctionCallingConfigMode.NONE if req.tool_choice == "none" else types.FunctionCallingConfigMode.ANY
                kwargs["tool_config"] = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode))
        if req.json_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = req.json_schema
        return types.GenerateContentConfig(**kwargs)

    def _error(self, model: str, exc: Exception) -> LLMError:
        code = getattr(exc, "code", None)
        text = str(exc)
        if isinstance(code, int):
            kind = classify_status(code, text)
            if code == 400 and "api key" in text.lower():
                kind = "auth"
            return LLMError(kind, redact(text)[:300], provider="gemini", model=model, status=code,
                            retry_after=_retry_after(None, text))
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return LLMError("timeout", type(exc).__name__, provider="gemini", model=model)
        if isinstance(exc, (httpx.HTTPError, OSError)):
            return LLMError("network", type(exc).__name__, provider="gemini", model=model)
        return LLMError("server", redact(f"{type(exc).__name__}: {text}")[:300], provider="gemini", model=model)

    @staticmethod
    def _collect(parts: list[Any]) -> tuple[str, list[ToolCall]]:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for part in parts:
            if getattr(part, "thought", False):
                continue
            if getattr(part, "text", None):
                texts.append(part.text)
            call = getattr(part, "function_call", None)
            if call is not None and call.name:
                # The Gemini API often omits ids; a synthetic "call_" id is
                # used on our side only and never replayed as a Gemini id.
                calls.append(ToolCall(id=call.id or _new_call_id(), name=call.name, arguments=dict(call.args or {})))
        return "".join(texts), calls

    def _response(self, model: str, parts: list[Any], usage_meta: Any, finish: Any, ttft: float | None,
                  started: float) -> LLMResponse:
        from google.genai import types

        text, calls = self._collect(parts)
        native = types.Content(role="model", parts=[p for p in parts if not getattr(p, "thought", False) or
                                                    getattr(p, "thought_signature", None)])
        raw: dict[str, Any] = {"role": "assistant", "content": text,
                               "_gemini_content": native.model_dump(mode="json", exclude_none=True)}
        if calls:
            raw["tool_calls"] = [c.as_openai() for c in calls]
        usage = {"tokens_in": int(getattr(usage_meta, "prompt_token_count", 0) or 0),
                 "tokens_out": int((getattr(usage_meta, "candidates_token_count", 0) or 0) +
                                   (getattr(usage_meta, "thoughts_token_count", 0) or 0))}
        total = (time.perf_counter() - started) * 1000.0
        return LLMResponse(text=text, tool_calls=calls, provider="gemini", model=model,
                           finish_reason=str(finish) if finish is not None else None, usage=usage,
                           ttft_ms=ttft if ttft is not None else total, total_ms=total, raw_message=raw)

    async def complete(self, model: str, req: LLMRequest) -> LLMResponse:
        client = self._genai()
        system, contents = self.to_contents(req.messages)
        started = time.perf_counter()
        try:
            result = await client.aio.models.generate_content(model=model, contents=contents,
                                                              config=self._config(req, system))
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK raises several families
            raise self._error(model, exc) from None
        candidate = (result.candidates or [None])[0]
        parts = list(getattr(getattr(candidate, "content", None), "parts", None) or [])
        return self._response(model, parts, result.usage_metadata, getattr(candidate, "finish_reason", None),
                              None, started)

    async def stream(self, model: str, req: LLMRequest) -> AsyncIterator[LLMChunk]:
        client = self._genai()
        system, contents = self.to_contents(req.messages)
        started = time.perf_counter()
        ttft: float | None = None
        parts: list[Any] = []
        usage_meta: Any = None
        finish: Any = None
        try:
            iterator = await client.aio.models.generate_content_stream(model=model, contents=contents,
                                                                       config=self._config(req, system))
            async for chunk in iterator:
                usage_meta = chunk.usage_metadata or usage_meta
                candidate = (chunk.candidates or [None])[0]
                if candidate is None:
                    continue
                finish = candidate.finish_reason or finish
                for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
                    parts.append(part)
                    if getattr(part, "thought", False):
                        continue
                    if getattr(part, "text", None):
                        if ttft is None:
                            ttft = (time.perf_counter() - started) * 1000.0
                        yield LLMChunk(kind="text", text=part.text)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._error(model, exc) from None
        response = self._response(model, parts, usage_meta, finish, ttft, started)
        for call in response.tool_calls:
            yield LLMChunk(kind="tool_call", tool_call=call)
        yield LLMChunk(kind="done", response=response)

    async def list_models(self) -> list[str]:
        client = self._genai()
        names: list[str] = []
        try:
            pager = await client.aio.models.list()
            async for model in pager:
                name = str(getattr(model, "name", "") or "")
                names.append(name.removeprefix("models/"))
        except Exception as exc:  # noqa: BLE001
            raise self._error("", exc) from None
        return names

    async def aclose(self) -> None:
        client, self._client = self._client, None
        closer = getattr(getattr(client, "aio", None), "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception:  # noqa: BLE001
                pass


def default_backends(config: Any, secrets: Any) -> dict[str, Any]:
    """The four providers, reading URLs from settings and keys from Secrets."""
    return {
        "omniroute": OpenAICompatBackend(
            "omniroute", lambda: config.get("providers.omniroute.base_url"),
            lambda: secrets.get("litellm_api_key")),
        "groq": OpenAICompatBackend(
            "groq", lambda: config.get("providers.groq.base_url"), lambda: secrets.get("groq_api_key")),
        "openrouter": OpenAICompatBackend(
            "openrouter", lambda: config.get("providers.openrouter.base_url"),
            lambda: secrets.get("openrouter_api_key"), extra_headers={"X-Title": "SAM"}),
        "gemini": GeminiBackend(lambda: secrets.get("gemini_api_key")),
    }


__all__ = ["OpenAICompatBackend", "GeminiBackend", "default_backends", "classify_status"]
