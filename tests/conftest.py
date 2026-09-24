from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_backend.app import create_app  # noqa: E402
from sam_backend.config import Settings  # noqa: E402
from sam_backend.models import AssistantTurn, ToolCall  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def supported_runtime():
    """Tests describe SAM running the way it is supported: not elevated.

    A Windows CI runner is an Administrator, and `Settings.prepare()` rightly
    refuses to start there, so every test that builds an application would fail
    for a reason that has nothing to do with what it is testing. This states the
    runtime being described rather than weakening the guard: the tests that own
    that guard set this back to True themselves and still prove the refusal.
    """
    import sam_backend.config as config

    # Session scope: module-scoped fixtures build applications before any
    # function-scoped patch would apply.
    patch = pytest.MonkeyPatch()
    patch.setattr(config, "is_elevated_windows_process", lambda: False)
    yield
    patch.undo()


class _SilentStream:
    """Takes audio the way a PortAudio output stream does, and plays none of it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        pass

    def write(self, data):
        return False

    def stop(self):
        pass

    def abort(self):
        pass

    def close(self):
        pass


class _SilentSapiVoice:
    """A Windows voice that finishes at once without a sound."""

    def __init__(self):
        from types import SimpleNamespace

        self.Rate = 0
        self.Volume = 100
        self.Status = SimpleNamespace(RunningState=1)

    def Speak(self, text, flags=0):  # noqa: N802 - the COM method's name
        return 1


@pytest.fixture(autouse=True)
def silent_speakers(monkeypatch):
    """No test makes a sound.

    The suite runs on the machine SAM speaks from, with its owner in the
    room, and `service.speak("Gold is trading at 408.")` used to reach the
    real Windows voice. Every speaker path is swapped for one that accepts
    audio and plays nothing; a test that inspects playback installs its own.
    """
    import sam_backend.voice as voice

    monkeypatch.setattr(voice, "_output_stream", lambda **kwargs: _SilentStream(**kwargs))
    monkeypatch.setattr(voice.TextToSpeech, "_sapi_speaker", lambda self: _SilentSapiVoice())
    try:
        import sounddevice
    except Exception:  # noqa: BLE001 - no PortAudio, nothing to guard
        return

    def refuse(*args, **kwargs):
        raise AssertionError("A test tried to play audio through the real speakers.")

    monkeypatch.setattr(sounddevice, "play", refuse)
    monkeypatch.setattr(sounddevice, "OutputStream", refuse)


class FakeAdapter:
    async def list_models(self):
        return [{"id": "fake", "name": "fake", "provider": "ollama"}]

    async def complete(self, messages, tools, model):
        if messages and messages[-1].get("role") == "tool":
            return AssistantTurn("The approved action finished." if "denied" not in messages[-1].get("content", "").lower() else "I respected the denial.")
        last_user = next((message.get("content", "") for message in reversed(messages) if message.get("role") == "user"), "")
        if last_user.startswith("overwrite "):
            _, path, content = last_user.split(" ", 2)
            return AssistantTurn("I need approval to overwrite that file.", [ToolCall("call_overwrite", "write_file", {"path": path, "content": content})])
        if last_user == "run python":
            return AssistantTurn("I need approval to run Python.", [ToolCall("call_python", "run_python", {"code": "print('ran')"})])
        if last_user == "read secret":
            return AssistantTurn("I need approval to access that sensitive file.", [ToolCall("call_secret", "read_file", {"path": ".env"})])
        if last_user.startswith("create "):
            _, path, content = last_user.split(" ", 2)
            return AssistantTurn("Creating it.", [ToolCall("call_create", "write_file", {"path": path, "content": content})])
        return AssistantTurn(f"SAM heard: {last_user}")


class FakeRegistry:
    def __init__(self):
        self.adapter = FakeAdapter()

    def get(self, provider):
        return self.adapter


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        workspace_root=tmp_path / "workspace",
        data_dir=tmp_path / "data",
        default_provider="ollama",
        default_model="fake",
        cors_origins=["http://127.0.0.1:8765"],
    )


@pytest.fixture()
def app(settings: Settings):
    return create_app(settings, FakeRegistry())


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client
