from __future__ import annotations

import asyncio
import importlib
from pathlib import Path


def test_health_and_public_config_do_not_expose_secrets(client, settings):
    settings.openai_api_key = "never-return-this"
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["audit_chain_valid"] is True

    response = client.get("/api/config")
    assert response.status_code == 200
    assert response.json()["openai_configured"] is True
    assert "never-return-this" not in response.text
    assert response.json()["security"]["loopback_only"] is True


def test_origin_guard_rejects_untrusted_sites(client):
    response = client.post("/api/chat", json={"message": "hello"}, headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


def test_chat_persists_conversation_and_messages(client):
    response = client.post("/api/chat", json={"message": "hello SAM"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["message"]["content"] == "SAM heard: hello SAM"

    conversation_id = body["conversation_id"]
    messages = client.get(f"/api/conversations/{conversation_id}/messages").json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert client.get("/api/conversations").json()["conversations"][0]["message_count"] == 2


def test_conversation_title_can_be_renamed_with_narrow_validation(client):
    conversation = client.post("/api/conversations", json={"title": "Before"}).json()["conversation"]
    conversation_id = conversation["id"]

    renamed = client.patch(f"/api/conversations/{conversation_id}", json={"title": "  Gold review  "})
    assert renamed.status_code == 200
    assert renamed.json()["conversation"]["title"] == "Gold review"
    assert client.get(f"/api/conversations/{conversation_id}").json()["conversation"]["title"] == "Gold review"

    assert client.patch(f"/api/conversations/{conversation_id}", json={"title": "   "}).status_code == 422
    assert client.patch(f"/api/conversations/{conversation_id}", json={"title": "x" * 161}).status_code == 422
    assert client.patch("/api/conversations/missing", json={"title": "Valid"}).status_code == 404

    preflight = client.options(
        f"/api/conversations/{conversation_id}",
        headers={
            "Origin": "http://127.0.0.1:8765",
            "Access-Control-Request-Method": "PATCH",
        },
    )
    assert preflight.status_code == 200
    assert "PATCH" in preflight.headers["access-control-allow-methods"]
    assert any(entry["status"] == "renamed" for entry in client.get("/api/audit", params={"event_type": "conversation"}).json()["entries"])


def test_safe_new_workspace_file_is_created_without_approval(client, settings):
    response = client.post("/api/chat", json={"message": "create note.txt hello"})
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert (settings.workspace_root / "note.txt").read_text(encoding="utf-8") == "hello"


def test_overwrite_requires_exact_single_use_approval(client, settings):
    target = settings.workspace_root / "note.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old", encoding="utf-8")

    requested = client.post("/api/chat", json={"message": "overwrite note.txt new"})
    assert requested.status_code == 200
    assert requested.json()["status"] == "awaiting_approval"
    approval = requested.json()["approvals"][0]
    assert approval["status"] == "pending"
    assert target.read_text(encoding="utf-8") == "old"

    decided = client.post(f"/api/approvals/{approval['id']}/decision", json={"decision": "approved", "note": "Expected overwrite"})
    assert decided.status_code == 200
    assert decided.json()["approval"]["status"] == "executed"
    assert target.read_text(encoding="utf-8") == "new"

    reused = client.post(f"/api/approvals/{approval['id']}/decision", json={"decision": "approved"})
    assert reused.status_code == 409
    assert client.get("/api/audit/verify").json() == {"valid": True}


def test_denied_python_is_not_executed(client):
    requested = client.post("/api/chat", json={"message": "run python"}).json()
    assert requested["status"] == "awaiting_approval"
    approval_id = requested["approvals"][0]["id"]

    denied = client.post(f"/api/approvals/{approval_id}/decision", json={"decision": "denied"})
    assert denied.status_code == 200
    assert denied.json()["approval"]["status"] == "denied"
    assert denied.json()["agent_response"]["message"]["content"] == "I respected the denial."


def test_memory_and_workspace_tree(client):
    created = client.post("/api/memories", json={"content": "The user prefers Sorani Kurdish.", "tags": ["preference"], "importance": 0.9})
    assert created.status_code == 201
    assert client.get("/api/memories", params={"query": "Sorani"}).json()["memories"][0]["content"].startswith("The user")

    tree = client.get("/api/workspace/tree")
    assert tree.status_code == 200
    assert Path(tree.json()["root"]).name == "workspace"


def test_likely_secret_is_rejected_from_memory(client):
    response = client.post("/api/memories", json={"content": "api_key=super-secret-value"})
    assert response.status_code == 400
    assert "super-secret-value" not in str(client.get("/api/audit").json())


def test_permission_mode_is_validated_and_applied(client, settings):
    response = client.put("/api/settings", json={"permission_mode": "strict"})
    assert response.status_code == 200
    assert response.json()["runtime"]["permission_mode"] == "strict"
    assert settings.permission_mode == "strict"
    assert client.put("/api/settings", json={"permission_mode": "approve-everything"}).status_code == 422


def test_realtime_trading_and_router_capabilities_are_public(client):
    voice = client.get("/api/voice/capabilities")
    assert voice.status_code == 200
    capabilities = voice.json()
    # The endpoint reports what the local engines can actually do, per subsystem.
    for section in ("vad", "stt", "tts", "devices", "language", "barge_in"):
        assert section in capabilities, f"voice capabilities missing {section}"
    assert capabilities["barge_in"]["stop_words"]
    assert capabilities["mode"] in capabilities["modes"]
    assert capabilities["browser_fallback"]["barge_in"] is True
    theories = client.get("/api/trading/theories").json()["built_in"]
    assert any(item["id"] == "snr" and item["health"] == "AVAILABLE" for item in theories)
    assert any(item["id"] == "footprint" and item["health"] == "UNAVAILABLE" for item in theories)
    router = client.get("/api/router/status").json()
    assert router["mode"] in {"AUTO", "LOCAL_ONLY", "CLOUD_ONLY", "MANUAL"}


def test_transcribe_without_a_sorani_provider_is_refused(client):
    import numpy

    from sam_backend.sorani import pcm_to_wav

    wav = pcm_to_wav(numpy.zeros(1600, dtype=numpy.float32), 16_000)
    response = client.post(
        "/api/voice/transcribe",
        files={"file": ("utterance.wav", wav, "audio/wav")},
        data={"language": "ckb"},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "KurdishTTS" in detail or "سۆرانی" in detail


def test_transcribe_endpoint_accepts_a_browser_wav(client, monkeypatch):
    import numpy

    from sam_backend.sorani import pcm_to_wav

    voice = client.app.state.voice
    monkeypatch.setattr(voice, "sorani_input_configured", lambda: True)
    monkeypatch.setattr(
        voice,
        "transcribe_audio_bytes",
        lambda payload, language=None: {
            "text": "سڵاو",
            "language": "ckb",
            "engine": "sorani:kurdishtts",
            "captured": True,
            "seconds": 1.0,
        },
    )
    wav = pcm_to_wav(numpy.zeros(1600, dtype=numpy.float32), 16_000)
    response = client.post(
        "/api/voice/transcribe",
        files={"file": ("utterance.wav", wav, "audio/wav")},
        data={"language": "ckb-IQ"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "سڵاو"
    assert body["engine"].startswith("sorani")
    assert "api" not in body.get("engine", "").lower()


def test_transcribe_rejects_unreadable_audio(client, monkeypatch):
    voice = client.app.state.voice
    monkeypatch.setattr(voice, "sorani_input_configured", lambda: True)
    response = client.post(
        "/api/voice/transcribe",
        files={"file": ("utterance.wav", b"not-a-wav", "audio/wav")},
        data={"language": "ckb"},
    )
    assert response.status_code == 400


def test_tts_endpoint_returns_wav_bytes(client, monkeypatch):
    import numpy

    from sam_backend.sorani import pcm_to_wav

    voice = client.app.state.voice
    wav = pcm_to_wav(numpy.zeros(800, dtype=numpy.float32), 16_000)
    monkeypatch.setattr(
        voice,
        "synthesize",
        lambda text, language=None: {
            "audio": wav,
            "engine": "sorani:kurdishtts",
            "language": "ckb",
            "speaker_id": "speaker-1",
        },
    )
    response = client.post("/api/voice/tts", json={"text": "سڵاو", "language": "ckb"})
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"
    assert "wav" in response.headers["content-type"]
    assert response.headers.get("x-sam-engine", "").startswith("sorani")


def test_tts_without_audio_is_refused_not_spoken_in_english(client, monkeypatch):
    from sam_backend.sorani import MESSAGE_NO_PROVIDER

    voice = client.app.state.voice
    monkeypatch.setattr(
        voice,
        "synthesize",
        lambda text, language=None: {"audio": None, "engine": "sorani", "error": MESSAGE_NO_PROVIDER},
    )
    response = client.post("/api/voice/tts", json={"text": "سڵاو", "language": "ckb"})
    assert response.status_code == 503
    assert response.json()["detail"] == MESSAGE_NO_PROVIDER


def test_model_discovery_endpoints_bound_slow_offline_providers(client, monkeypatch):
    class SlowOfflineAdapter:
        def __init__(self):
            self.cancelled = 0

        async def list_models(self):
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            return [{"id": "should-not-arrive"}]

    app_module = importlib.import_module("sam_backend.app")
    monkeypatch.setattr(app_module, "MODEL_DISCOVERY_TIMEOUT_SECONDS", 0.01)
    adapter = SlowOfflineAdapter()
    client.app.state.adapters.adapter = adapter

    models = client.get("/api/models")
    router = client.get("/api/router/status")

    assert models.status_code == 200
    assert all(not items for items in models.json()["providers"].values())
    assert router.status_code == 200
    assert router.json()["configured"]["ollama"] is False
    assert router.json()["configured"]["litellm"] is False
    assert adapter.cancelled >= 4


def test_settings_can_enable_scoped_desktop_permissions(client, settings):
    response = client.put("/api/settings", json={"computer_control_enabled": True, "screen_access_enabled": True, "voice_mode": "CONVERSATION"})
    assert response.status_code == 200
    assert settings.computer_control_enabled is True
    assert settings.screen_access_enabled is True
    assert settings.voice_mode == "CONVERSATION"


def test_approved_sensitive_file_never_enters_history_audit_or_public_result(client, settings):
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    secret = "OPENAI_API_KEY=never-persist-this-value"
    (settings.workspace_root / ".env").write_text(secret, encoding="utf-8")

    requested = client.post("/api/chat", json={"message": "read secret"}).json()
    approval_id = requested["approvals"][0]["id"]
    response = client.post(f"/api/approvals/{approval_id}/decision", json={"decision": "approved"})
    assert response.status_code == 200
    assert secret not in response.text
    assert secret not in str(client.get("/api/audit").json())
    conversation_id = requested["conversation_id"]
    assert secret not in str(client.get(f"/api/conversations/{conversation_id}/messages").json())
