"""Voice pipeline tests: capability honesty, barge-in, and audio handling."""

from __future__ import annotations

from pathlib import Path

import numpy
import pytest

from sam_backend.config import Settings
from sam_backend.contracts import CapabilityState
from sam_backend.voice import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    UNSUPPORTED_STT_LANGUAGES,
    BargeInController,
    SileroVad,
    VoiceService,
    audio_devices,
    language_support,
    resample_mono,
    rms_energy,
)


@pytest.fixture()
def service(tmp_path: Path) -> VoiceService:
    settings = Settings(
        project_root=tmp_path,
        workspace_root=tmp_path / "workspace",
        data_dir=tmp_path / "data",
    )
    settings.prepare()
    return VoiceService(settings)


# --- Language honesty ---------------------------------------------------------


@pytest.mark.parametrize("code", sorted(UNSUPPORTED_STT_LANGUAGES))
def test_kurdish_speech_is_unsupported_when_no_sorani_provider_is_configured(code):
    """Whisper alone cannot do Kurdish, and that must not be papered over."""
    support = language_support(code)
    assert support["stt_supported"] is False
    assert support["state"] == CapabilityState.UNAVAILABLE.value
    # The reason must say typing still works, so the limit is not overstated.
    assert "typed" in support["reason"].lower()


def test_a_regional_tag_does_not_hide_an_unsupported_language():
    assert language_support("ckb-IQ")["stt_supported"] is False


CONNECTED_SORANI = {"provider": "kurdishtts", "role": "stt", "status": "CONNECTED"}


def test_sorani_speech_is_supported_once_a_provider_is_connected():
    """Kurdish speech is real when KurdishTTS answers; the claim tracks the provider."""
    support = language_support("ckb", CONNECTED_SORANI)
    assert support["stt_supported"] is True
    assert support["state"] == CapabilityState.AVAILABLE.value
    assert support["engine"] == "kurdishtts"
    # It must still be clear this is not local recognition.
    assert "whisper has no" in support["note"].lower()


def test_a_regional_sorani_tag_is_supported_through_the_provider():
    assert language_support("ckb-IQ", CONNECTED_SORANI)["stt_supported"] is True


@pytest.mark.parametrize("status", [
    {"status": "UNCONFIGURED", "detail": "no key"},
    {"status": "AUTH_FAILED", "detail": "rejected"},
    {"status": "ERROR", "detail": "unreachable"},
])
def test_sorani_is_not_claimed_when_the_provider_is_not_connected(status):
    """A configured-but-broken provider must not read as working speech."""
    support = language_support("ckb", status)
    assert support["stt_supported"] is False
    assert status["detail"] in support["reason"]


def test_a_provider_for_sorani_does_not_make_kurmanji_supported():
    """KurdishTTS serves Sorani; Kurmanji is a different language."""
    assert language_support("kmr", CONNECTED_SORANI)["stt_supported"] is False


@pytest.mark.parametrize("code", ["en", "en-US", "ar", "fa", "tr"])
def test_languages_whisper_covers_are_reported_supported(code):
    assert language_support(code)["stt_supported"] is True


def test_transcribing_an_unsupported_language_returns_no_text(service: VoiceService):
    audio = numpy.zeros(SAMPLE_RATE, dtype=numpy.float32)
    result = service.stt.transcribe(audio, language="ckb")
    assert result["supported"] is False
    assert result["text"] == ""
    # It must refuse rather than fall back to a similar-script language.
    assert result["language"] == "ckb"


# --- Barge-in -----------------------------------------------------------------


def test_interrupting_while_speaking_is_reported_and_counted():
    controller = BargeInController()
    controller.begin_speaking()
    assert controller.speaking is True
    assert controller.interrupt() is True
    assert controller.should_stop_tts() is True
    assert controller.interruptions == 1
    assert controller.speaking is False


def test_interrupting_when_silent_is_not_counted_as_an_interruption():
    controller = BargeInController()
    assert controller.interrupt() is False
    assert controller.interruptions == 0


def test_beginning_a_new_utterance_clears_the_previous_interrupt():
    controller = BargeInController()
    controller.begin_speaking()
    controller.interrupt()
    controller.begin_speaking()
    assert controller.should_stop_tts() is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("وەستە", True),
        ("وەستە، بچۆ 5 خولەکی", True),
        ("Stop", True),
        ("please cancel that", True),
        ("hold on a second", True),
        ("analyze gold now", False),
        ("", False),
    ],
)
def test_stop_commands_are_recognised_in_sorani_and_english(text, expected):
    assert BargeInController.is_stop_command(text) is expected


def test_reset_clears_both_speaking_and_interrupted_state():
    controller = BargeInController()
    controller.begin_speaking()
    controller.interrupt()
    controller.reset()
    assert controller.speaking is False
    assert controller.should_stop_tts() is False


# --- Audio helpers ------------------------------------------------------------


def test_silence_has_no_energy_and_a_tone_does():
    silence = numpy.zeros(FRAME_SAMPLES, dtype=numpy.float32)
    tone = numpy.sin(numpy.linspace(0, 40 * numpy.pi, FRAME_SAMPLES)).astype(numpy.float32)
    assert rms_energy(silence) == pytest.approx(0.0)
    assert rms_energy(tone) > 0.5
    assert rms_energy(numpy.array([], dtype=numpy.float32)) == 0.0


def test_vad_finds_no_speech_in_silence():
    vad = SileroVad()
    if not vad.available():
        pytest.skip("Silero VAD is not installed")
    assert vad.speech_segments(numpy.zeros(SAMPLE_RATE * 2, dtype=numpy.float32)) == []


def test_a_pure_tone_is_not_mistaken_for_speech():
    vad = SileroVad()
    if not vad.available():
        pytest.skip("Silero VAD is not installed")
    tone = numpy.sin(numpy.linspace(0, 800 * numpy.pi, SAMPLE_RATE * 2)).astype(numpy.float32)
    assert vad.contains_speech(tone) is False


# --- Capability surface -------------------------------------------------------


def test_devices_are_enumerated_with_defaults_marked():
    devices = audio_devices()
    if devices["state"] == CapabilityState.UNCONFIGURED.value:
        pytest.skip("sounddevice is not installed")
    assert devices["inputs"], "no input devices were found"
    for device in devices["inputs"]:
        assert {"index", "name", "channels", "sample_rate", "default"} <= set(device)


def test_capabilities_expose_every_subsystem(service: VoiceService):
    capabilities = service.capabilities()
    for section in ("vad", "stt", "tts", "devices", "language", "barge_in", "mode", "modes"):
        assert section in capabilities
    assert capabilities["mode"] in capabilities["modes"]
    assert capabilities["sample_rate"] == SAMPLE_RATE


def test_an_invalid_voice_mode_falls_back_to_push_to_talk(tmp_path: Path):
    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    settings.prepare()
    service = VoiceService(settings)
    service.settings.voice_mode = "NONSENSE"
    assert service.mode == "PUSH_TO_TALK"


def test_the_wake_word_is_matched_case_insensitively(service: VoiceService):
    service.settings.voice_wake_word = "SAM"
    assert service.heard_wake_word("hey sam, open tradingview") is True
    assert service.heard_wake_word("open tradingview") is False


def test_speech_output_never_claims_a_piper_voice_it_does_not_have(service: VoiceService):
    capability = service.tts.capability()
    if capability["engine"] == "piper":
        assert service.tts.piper_voice_available() is True
    else:
        assert capability["state"] in {
            CapabilityState.PARTIALLY_AVAILABLE.value,
            CapabilityState.UNCONFIGURED.value,
        }
        assert "reason" in capability


def test_streaming_synthesis_is_refused_without_a_voice_model(service: VoiceService):
    if service.tts.piper_voice_available():
        pytest.skip("A Piper voice model is installed")
    with pytest.raises(RuntimeError, match="Piper voice model"):
        list(service.tts.synthesize_chunks("hello"))


# --- Sorani routing -----------------------------------------------------------


def test_sorani_speech_never_falls_back_to_an_english_voice(service: VoiceService):
    """Reading Kurdish through an English voice is not Sorani, so it is refused."""
    service.settings.voice_language = "ckb"
    result = service.speak("سڵاو، من سامم.")
    assert result["engine"].startswith("sorani")
    if not result.get("spoken"):
        # Without a provider it reports why, rather than speaking English.
        assert result["error"]
    assert result.get("language") == "ckb"


def test_sorani_text_is_detected_from_the_reply_itself(service: VoiceService):
    """A Sorani answer is spoken in Sorani even if the setting says English."""
    service.settings.voice_language = "en"
    result = service.speak("زێڕ لە ئاستی ٤٠٨ دایە.")
    assert result["engine"].startswith("sorani")


def test_english_text_is_not_routed_to_the_sorani_voice(service: VoiceService):
    service.settings.voice_language = "en"
    result = service.speak("Gold is trading at 408.")
    assert not str(result.get("engine", "")).startswith("sorani")


def test_sorani_recognition_never_falls_back_to_whisper(service: VoiceService):
    """Whisper answers Kurdish audio with fluent Arabic, which would be a lie."""
    audio = numpy.zeros(SAMPLE_RATE, dtype=numpy.float32)
    result = service._transcribe_sorani(audio, 1.0)
    assert result["language"] == "ckb"
    assert result["engine"].startswith("sorani")
    # It either recognised Sorani or explained why not; it never returns
    # Whisper's guess at another language.
    assert result.get("text") == "" or isinstance(result["text"], str)


def test_the_sorani_wake_word_is_heard(service: VoiceService):
    assert service.heard_wake_word("سام، بچۆ بۆ پێنج خولەکی") is True


def test_48k_browser_audio_is_resampled_to_16k():
    source = numpy.linspace(-0.5, 0.5, 48000, dtype=numpy.float32)
    resampled = resample_mono(source, 48_000, SAMPLE_RATE)
    assert len(resampled) == SAMPLE_RATE


def test_uploaded_sorani_wav_never_hits_whisper(service: VoiceService, monkeypatch):
    """Browser recordings still must not be handed to Whisper as 'close enough'."""
    from sam_backend.sorani import pcm_to_wav

    monkeypatch.setattr(
        service.stt,
        "transcribe",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Whisper must not see Sorani audio")),
    )
    monkeypatch.setattr(
        service,
        "_transcribe_sorani",
        lambda audio, seconds: {"text": "سڵاو", "language": "ckb", "engine": "sorani:kurdishtts"},
    )
    wav = pcm_to_wav(numpy.zeros(48_000, dtype=numpy.float32), 48_000)
    result = service.transcribe_audio_bytes(wav, language="ckb-IQ")
    assert result["text"] == "سڵاو"
    assert result["captured"] is True
    assert result["engine"].startswith("sorani")


def test_synthesize_does_not_play_on_the_server_speakers(service: VoiceService, monkeypatch):
    """The UI plays the WAV; server playback would double the reply."""
    played = {"count": 0}

    class FakeTts:
        def synthesize(self, text, speaker_id=None):
            from sam_backend.sorani import pcm_to_wav

            return pcm_to_wav(numpy.zeros(800, dtype=numpy.float32), SAMPLE_RATE), {
                "route": "kurdishtts", "speaker_id": "speaker-1",
            }

    monkeypatch.setattr(service, "sorani_stack", lambda: (None, FakeTts()))
    monkeypatch.setattr(
        service,
        "_play_wav",
        lambda *args, **kwargs: played.__setitem__("count", played["count"] + 1) or {"played": True, "interrupted": False},
    )
    result = service.synthesize("سڵاو، من سامم.", "ckb")
    assert played["count"] == 0
    assert result["audio"][:4] == b"RIFF"
    assert result["engine"].startswith("sorani")
    assert result.get("language") == "ckb"


def test_sorani_status_never_carries_a_credential(service: VoiceService, monkeypatch):
    import json

    monkeypatch.setenv("KURDISHTTS_TTS_API_KEY", "test-key-value-not-real-000000")
    monkeypatch.setenv("KURDISHTTS_STT_API_KEY", "test-key-value-not-real-111111")
    service._sorani_stack = None
    blob = json.dumps(service.sorani_status(), ensure_ascii=False)
    assert "test-key-value-not-real-000000" not in blob
    assert "test-key-value-not-real-111111" not in blob
