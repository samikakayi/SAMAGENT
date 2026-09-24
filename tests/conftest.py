"""Shared fixtures. Tests never use the network, speakers, the real SAM_HOME
or live apps; every App gets a temp home and an empty environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sam.brain.llm import LLMChunk, LLMError, LLMRequest, LLMResponse, ToolCall  # noqa: E402

# Fake credentials with the real shapes (never real keys).
FAKE_GROQ = "gsk_" + "T3st" * 10
FAKE_GEMINI = "AIza" + "FakeKey0123456789abcdefXYZ"
FAKE_GEMINI_AQ = "AQ." + "Ab8RN6Kfake_auth_key-0123456789xyz"
FAKE_OPENROUTER = "sk-or-v1-" + "0f" * 20


class FakeBackend:
    """Scripted backend: ``script`` maps model -> list of results/exceptions
    returned in order (last one repeats)."""

    def __init__(self, provider: str, script: dict[str, list[Any]] | None = None, configured: bool = True) -> None:
        self.provider = provider
        self.script = script or {}
        self._configured = configured
        self.calls: list[tuple[str, LLMRequest]] = []

    def configured(self) -> bool:
        return self._configured

    def _next(self, model: str) -> Any:
        items = self.script.get(model) or [f"reply from {self.provider}:{model}"]
        index = sum(1 for m, _ in self.calls if m == model) - 1
        return items[min(index, len(items) - 1)]

    async def complete(self, model: str, req: LLMRequest) -> LLMResponse:
        self.calls.append((model, req))
        item = self._next(model)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(text=str(item), provider=self.provider, model=model, usage={"tokens_in": 3, "tokens_out": 5},
                           total_ms=1.0)

    async def stream(self, model: str, req: LLMRequest) -> AsyncIterator[LLMChunk]:
        self.calls.append((model, req))
        item = self._next(model)
        if isinstance(item, BaseException):
            raise item
        text = str(item)
        for word in text.split(" "):
            yield LLMChunk(kind="text", text=word + " ")
        yield LLMChunk(kind="done", response=LLMResponse(text=text, provider=self.provider, model=model, total_ms=1.0))

    async def list_models(self) -> list[str]:
        return list(self.script)

    async def aclose(self) -> None:
        pass


def rate_limited(provider: str, model: str, retry_after: float | None = None) -> LLMError:
    return LLMError("rate_limit", "RESOURCE_EXHAUSTED", provider=provider, model=model, status=429,
                    retry_after=retry_after)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    (path / "data").mkdir(parents=True)
    return path


@pytest.fixture
def make_app(home: Path):
    """Factory: ``make_app(backends=..., env_text=...)`` -> App on a temp home."""
    from sam.app import App

    created: list[Any] = []

    def factory(backends: dict[str, Any] | None = None, env_text: str = "") -> Any:
        if env_text:
            (home / ".env").write_text(env_text, encoding="utf-8")
        app = App(home, environ={}, llm_backends=backends if backends is not None else {})
        created.append(app)
        return app

    yield factory
    for app in created:
        app.close()


__all__ = ["FakeBackend", "rate_limited", "FAKE_GROQ", "FAKE_GEMINI", "FAKE_GEMINI_AQ", "FAKE_OPENROUTER", "ToolCall"]
