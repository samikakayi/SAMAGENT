"""Test doubles for the voice package: no speakers, no microphone, no network.

- ``FakeOutputStream``: stands in for ``sounddevice.RawOutputStream``; tests
  call ``pull()`` to consume audio the way the PortAudio thread would.
- ``FakeMic``: MicStream-compatible; tests ``push()`` frames.
- ``FakeLiveClient`` / ``FakeLiveSession``: mimic ``client.aio.live.connect``
  and ``AsyncSession`` (send_realtime_input / send_client_content /
  send_tool_response / receive) and replay real ``types.LiveServerMessage``
  objects built from dicts, so attribute access matches the SDK.
- ``FakeStt`` / ``FakeTts`` / ``fake_llm``: scripted providers with call logs.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncIterator

from sam.voice.stt import SttError, SttResult
from sam.voice.tts import TtsError

FRAME = 960  # 30 ms at 16 kHz, 16-bit


def tone_frame(level: int = 8000) -> bytes:
    """A loud-ish non-silent frame (used where the VAD classifier is faked)."""
    import array
    samples = array.array("h", [level if (i // 8) % 2 else -level for i in range(FRAME // 2)])
    return samples.tobytes()


def quiet_frame() -> bytes:
    return bytes(FRAME)


class FakeOutputStream:
    """RawOutputStream stand-in. ``pull(frames)`` runs the callback once."""

    instances: list["FakeOutputStream"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = False
        self.closed = False
        self.aborted = False
        self.played = bytearray()
        FakeOutputStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True

    def pull(self, frames: int) -> bytes:
        out = bytearray(frames * 2)
        view = memoryview(out)
        self.callback(view, frames, None, None)
        self.played.extend(out)
        return bytes(out)


class FakeSpeaker:
    """Speaker-compatible sink: records writes and flushes, never plays."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.flushes = 0
        self.dropped = 0
        self.epoch = 0
        self.playing = False
        self.is_open = False
        self.device = None
        self.underflows = 0
        self.last_error = None
        self._pending = 0

    def write(self, pcm: bytes, *, epoch: int | None = None, rate: int = 24000) -> bool:
        if not pcm or (epoch is not None and epoch != self.epoch):
            return False
        self.chunks.append(pcm)
        self._pending += len(pcm)
        return True

    def flush(self) -> int:
        self.flushes += 1
        dropped, self._pending = self._pending, 0
        self.dropped += dropped
        self.epoch += 1
        return dropped

    def buffered_ms(self) -> float:
        return 0.0

    async def wait_idle(self, timeout: float | None = None, poll_s: float = 0.01) -> bool:
        return True

    async def open(self) -> bool:
        self.is_open = True
        return True

    def close(self) -> None:
        self.is_open = False

    async def close_if_idle(self) -> bool:
        return False

    @property
    def audio(self) -> bytes:
        return b"".join(self.chunks)


class FakeMic:
    """MicStream-compatible source fed by the test."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True
        self.queue.put_nowait(None)

    @property
    def is_open(self) -> bool:
        return self.started and not self.stopped

    def push(self, frame: bytes, count: int = 1) -> None:
        for _ in range(count):
            self.queue.put_nowait(frame)

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item


# -- Live ----------------------------------------------------------------------------------------------

def msg(data: dict[str, Any]) -> Any:
    """A real ``types.LiveServerMessage`` from a snake_case dict."""
    from google.genai import types
    return types.LiveServerMessage.model_validate(data)


def audio_msg(pcm: bytes = b"\x01\x00" * 480) -> Any:
    return msg({"server_content": {"model_turn": {"parts": [{"inline_data": {"data": pcm,
                                                                          "mime_type": "audio/pcm;rate=24000"}}]}}})


def idle_msg() -> Any:
    return msg({"server_content": {"turn_complete": True, "interaction_status": "IDLE"}})


DROP = object()


class FakeLiveSession:
    def __init__(self) -> None:
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.audio: list[bytes] = []
        self.realtime: list[dict[str, Any]] = []
        self.client_content: list[tuple[Any, bool]] = []
        self.tool_responses: list[list[Any]] = []
        self.closed = False

    def push(self, *messages: Any) -> None:
        for message in messages:
            self.inbox.put_nowait(message)

    def drop(self) -> None:
        self.inbox.put_nowait(DROP)

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.realtime.append(kwargs)
        audio = kwargs.get("audio")
        if audio is not None:
            self.audio.append(audio.data if hasattr(audio, "data") else audio["data"])

    async def send_client_content(self, *, turns: Any = None, turn_complete: bool = True) -> None:
        self.client_content.append((turns, turn_complete))

    async def send_tool_response(self, *, function_responses: Any) -> None:
        items = function_responses if isinstance(function_responses, list) else [function_responses]
        self.tool_responses.append(items)

    async def receive(self) -> AsyncIterator[Any]:
        from sam.voice.live_config import turn_is_idle
        while True:
            item = await self.inbox.get()
            if item is DROP:
                raise ConnectionError("socket dropped (1006)")
            yield item
            sc = getattr(item, "server_content", None)
            if sc is not None and turn_is_idle(sc):
                return  # like the SDK: receive() ends after a completed turn


class FakeLiveClient:
    """``client.aio.live.connect(model=, config=)``. ``plan`` items: a
    FakeLiveSession to hand out, or an Exception to raise on connect."""

    def __init__(self, plan: list[Any] | None = None) -> None:
        self.plan = list(plan or [])
        self.connects: list[tuple[str, Any]] = []
        self.sessions: list[FakeLiveSession] = []
        outer = self

        class _Live:
            def connect(self, *, model: str, config: Any) -> Any:
                return outer._connect(model, config)

        class _Aio:
            live = _Live()

        self.aio = _Aio()

    @contextlib.asynccontextmanager
    async def _connect(self, model: str, config: Any) -> AsyncIterator[FakeLiveSession]:
        self.connects.append((model, config))
        item = self.plan.pop(0) if self.plan else FakeLiveSession()
        if isinstance(item, BaseException):
            raise item
        self.sessions.append(item)
        try:
            yield item
        finally:
            item.closed = True


class ApiError(Exception):
    def __init__(self, code: int, message: str = "") -> None:
        super().__init__(f"{code} {message}")
        self.code = code


# -- STT / TTS / LLM -----------------------------------------------------------------------------------------

class FakeStt:
    provider = "fake-stt"

    def __init__(self, texts: list[Any] | None = None, *, configured: bool = True, delay: float = 0.0) -> None:
        self.texts = list(texts or ["سڵاو"])
        self._configured = configured
        self.delay = delay
        self.calls: list[bytes] = []
        self.last_provider = None

    def configured(self) -> bool:
        return self._configured

    def status(self) -> dict[str, Any]:
        return {"fake": {"configured": self._configured}}

    async def transcribe(self, pcm: bytes, rate: int = 16000) -> SttResult:
        self.calls.append(pcm)
        if self.delay:
            await asyncio.sleep(self.delay)
        item = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        if isinstance(item, BaseException):
            raise item
        self.last_provider = self.provider
        return SttResult(text=item, provider=self.provider, ms=1.0)

    async def aclose(self) -> None:
        pass


class FakeTts:
    """Yields ``chunks`` PCM chunks per request; records every text."""

    def __init__(self, *, chunks: int = 2, chunk_bytes: int = 960, configured: bool = True, delay: float = 0.0,
                 fail: BaseException | None = None) -> None:
        self.chunks = chunks
        self.chunk_bytes = chunk_bytes
        self._configured = configured
        self.delay = delay
        self.fail = fail
        self.texts: list[str] = []
        self.last_provider = "fake-tts"
        self.started = asyncio.Event()

    def configured(self) -> bool:
        return self._configured

    def status(self) -> dict[str, Any]:
        return {"fake": {"configured": self._configured}}

    def max_chars(self) -> int:
        return 480

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        self.texts.append(text)
        self.started.set()
        if self.fail is not None:
            raise self.fail
        for _ in range(self.chunks):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield b"\x10\x00" * (self.chunk_bytes // 2)

    async def synthesize(self, text: str) -> bytes:
        return b"".join([c async for c in self.stream(text)])

    async def aclose(self) -> None:
        pass


def fake_llm(pieces: list[Any], log: list[str] | None = None):
    """``llm_stream(text, turn)`` yielding ``pieces``; an asyncio.Event in the
    list pauses the stream until it is set (to prove pipelining)."""

    async def stream(text: str, turn: Any) -> AsyncIterator[str]:
        if log is not None:
            log.append(text)
        for piece in pieces:
            if isinstance(piece, asyncio.Event):
                await piece.wait()
                continue
            if isinstance(piece, BaseException):
                raise piece
            yield piece
    return stream


async def settle(predicate: Any = None, timeout: float = 2.0, step: float = 0.005) -> bool:
    """Let the loop run until ``predicate()`` is true (or just a few ticks)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        await asyncio.sleep(step)
        if predicate is None or predicate():
            return True
        if loop.time() >= deadline:
            return False


__all__ = ["FakeOutputStream", "FakeSpeaker", "FakeMic", "FakeLiveSession", "FakeLiveClient", "FakeStt", "FakeTts", "fake_llm",
           "msg", "audio_msg", "idle_msg", "settle", "tone_frame", "quiet_frame", "ApiError", "SttError", "TtsError"]
