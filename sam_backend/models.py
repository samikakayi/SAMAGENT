from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .config import Settings


class ModelError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssistantTurn:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None


class ModelAdapter(Protocol):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str) -> AssistantTurn: ...
    async def list_models(self) -> list[dict[str, Any]]: ...


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "{}")
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": value}
    return {}


class OllamaAdapter:
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url
        self.timeout = httpx.Timeout(180, connect=5)

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str) -> AssistantTurn:
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            # Qwen's hidden thinking mode is very slow on CPU-only Windows
            # machines and is unnecessary for SAM's deterministic policy loop.
            "think": False,
            "options": {"temperature": 0.2, "num_predict": 256, "repeat_penalty": 1.05},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        message = data.get("message") or {}
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            calls.append(ToolCall(str(item.get("id") or f"call_{uuid.uuid4().hex}"), str(function.get("name", "")), _arguments(function.get("arguments"))))
        return AssistantTurn(str(message.get("content") or ""), calls, raw={
            "model": data.get("model") or model,
            "done_reason": data.get("done_reason"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "load_duration": data.get("load_duration"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_duration": data.get("eval_duration"),
            "total_duration": data.get("total_duration"),
        })

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models") or []
            return [{"id": item.get("name") or item.get("model"), "name": item.get("name") or item.get("model"), "provider": "ollama", "size": item.get("size")} for item in models]
        except (httpx.HTTPError, ValueError):
            return []


class OpenAIResponsesAdapter:
    """Minimal OpenAI Responses API adapter; no SDK or global credentials required."""

    def __init__(self, settings: Settings):
        self.base_url = settings.openai_base_url
        self.api_key = settings.openai_api_key
        self.timeout = httpx.Timeout(180, connect=10)

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for item in tools:
            function = item.get("function") or {}
            converted.append({
                "type": "function", "name": function.get("name"), "description": function.get("description", ""),
                "parameters": function.get("parameters") or {"type": "object", "properties": {}}, "strict": False,
            })
        return converted

    @staticmethod
    def _input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "system":
                instructions.append(content)
            elif role == "tool":
                items.append({"type": "function_call_output", "call_id": message.get("tool_call_id"), "output": content})
            elif role in {"user", "assistant"}:
                if content:
                    items.append({"role": role, "content": content})
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    items.append({
                        "type": "function_call", "call_id": call.get("id"), "name": function.get("name"),
                        "arguments": json.dumps(function.get("arguments") or {}, ensure_ascii=False),
                    })
        return "\n\n".join(instructions), items

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str) -> AssistantTurn:
        if not self.api_key:
            raise ModelError("OPENAI_API_KEY is not configured")
        instructions, input_items = self._input(messages)
        payload: dict[str, Any] = {
            "model": model, "input": input_items, "tools": self._tools(tools),
            "tool_choice": "auto", "parallel_tool_calls": False, "store": False,
        }
        if instructions:
            payload["instructions"] = instructions
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.post(
                    f"{self.base_url}/responses", json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            message = str(exc)
            if isinstance(exc, httpx.HTTPStatusError):
                try:
                    message = exc.response.json().get("error", {}).get("message", message)
                except ValueError:
                    pass
            raise ModelError(f"OpenAI Responses request failed: {message}") from exc
        content_parts: list[str] = []
        calls: list[ToolCall] = []
        for item in data.get("output") or []:
            if item.get("type") == "function_call":
                calls.append(ToolCall(str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"), str(item.get("name", "")), _arguments(item.get("arguments"))))
            elif item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") in {"output_text", "text"}:
                        content_parts.append(str(part.get("text") or ""))
        return AssistantTurn("\n".join(part for part in content_parts if part), calls, raw={"id": data.get("id"), "status": data.get("status"), "usage": data.get("usage")})

    async def list_models(self) -> list[dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                response = await client.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"})
                response.raise_for_status()
                models = response.json().get("data") or []
            return [{"id": item.get("id"), "name": item.get("id"), "provider": "openai"} for item in models]
        except (httpx.HTTPError, ValueError):
            return []


class ChatCompletionsAdapter:
    """OpenAI-compatible chat adapter used by LiteLLM and OpenRouter."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str | None,
        key_required: bool,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.key_required = key_required
        self.extra_headers = {key: value for key, value in (extra_headers or {}).items() if value}
        self.timeout = httpx.Timeout(180, connect=10)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for raw in messages:
            message = dict(raw)
            calls = []
            for item in message.get("tool_calls") or []:
                converted_call = dict(item)
                function = dict(converted_call.get("function") or {})
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    function["arguments"] = json.dumps(arguments or {}, ensure_ascii=False)
                converted_call["function"] = function
                calls.append(converted_call)
            if calls:
                message["tool_calls"] = calls
            converted.append(message)
        return converted

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str) -> AssistantTurn:
        if self.key_required and not self.api_key:
            raise ModelError(f"{self.provider.upper()} credential is not configured")
        payload = {
            "model": model,
            "messages": self._messages(messages),
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            message = str(exc)
            if isinstance(exc, httpx.HTTPStatusError):
                try:
                    error = exc.response.json().get("error") or {}
                    message = str(error.get("message") or message)
                except ValueError:
                    pass
            raise ModelError(f"{self.provider} chat request failed: {message}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise ModelError(f"{self.provider} returned no completion choices")
        message = choices[0].get("message") or {}
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            calls.append(ToolCall(
                str(item.get("id") or f"call_{uuid.uuid4().hex}"),
                str(function.get("name") or ""),
                _arguments(function.get("arguments")),
            ))
        raw = {
            "id": data.get("id"),
            "model": data.get("model") or model,
            "provider": self.provider,
            "usage": data.get("usage") or {},
            "finish_reason": choices[0].get("finish_reason"),
        }
        return AssistantTurn(str(message.get("content") or ""), calls, raw=raw)

    async def list_models(self) -> list[dict[str, Any]]:
        if self.key_required and not self.api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
                response = await client.get(f"{self.base_url}/models", headers=self._headers())
                response.raise_for_status()
                data = response.json()
            return [
                {
                    "id": item.get("id"),
                    "name": item.get("name") or item.get("id"),
                    "provider": self.provider,
                    "context_length": item.get("context_length"),
                    "pricing": item.get("pricing"),
                    "architecture": item.get("architecture"),
                }
                for item in data.get("data") or [] if item.get("id")
            ]
        except (httpx.HTTPError, ValueError):
            return []


class OpenRouterAdapter(ChatCompletionsAdapter):
    def __init__(self, settings: Settings) -> None:
        headers = {"X-Title": settings.openrouter_title}
        if settings.openrouter_http_referer:
            headers["HTTP-Referer"] = settings.openrouter_http_referer
        super().__init__(
            provider="openrouter",
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            key_required=True,
            extra_headers=headers,
        )


class LiteLLMAdapter(ChatCompletionsAdapter):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            provider="litellm",
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
            key_required=False,
        )


class AdapterRegistry:
    def __init__(self, settings: Settings):
        self.adapters: dict[str, ModelAdapter] = {
            "ollama": OllamaAdapter(settings),
            "openai": OpenAIResponsesAdapter(settings),
            "openrouter": OpenRouterAdapter(settings),
            "litellm": LiteLLMAdapter(settings),
        }

    @property
    def providers(self) -> list[str]:
        return list(self.adapters)

    def get(self, provider: str) -> ModelAdapter:
        try:
            return self.adapters[provider]
        except KeyError as exc:
            raise ModelError(f"Unsupported provider: {provider}") from exc
