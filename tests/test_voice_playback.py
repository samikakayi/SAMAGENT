"""Speaker output: one continuous stream per reply, one reply at a time, honest results.

The user heard SAM's voice "completely jumbled and broken up". Two causes, both
measured on the machine it runs on (digital silence only): every 200 ms slice
of a reply opened and closed its own PortAudio stream -- 466 ms of wall time
per 200 ms of voice -- and sounddevice.play() shares one module-global stream,
so a second reply stopped the first on every slice and the two interleaved.

Nothing here touches a real device: the conftest guard makes the speakers
silent, and these tests install a recording stream in their place.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

import sam_backend.voice as voice_module
from sam_backend.config import Settings
from sam_backend.sorani import MESSAGE_NO_PROVIDER, pcm_to_wav, wav_to_float32, wav_to_frames
from sam_backend.voice import SPEAKERS, VoiceService, play_audio

KURDISHTTS_RATE = 22_050  # what the WAVs KurdishTTS returned on this machine use


class RecordingStream:
    def __init__(self, speakers: "Speakers", **kwargs):
        self.speakers = speakers
        self.kwargs = kwargs
        self.writes: list[int] = []
        self.started = self.stopped = self.aborted = self.closed = False

    def start(self):
        self.started = True
        with self.speakers.lock:
            self.speakers.active += 1
            self.speakers.most_at_once = max(self.speakers.most_at_once, self.speakers.active)
            self.speakers.events.append(("start", id(self)))

    def write(self, data):
        assert data.dtype == numpy.float32
        assert data.ndim == 2 and data.shape[1] == self.kwargs["channels"]
        assert not data.any(), "only digital silence is ever used here"
        if self.speakers.fail_after is not None and len(self.writes) >= self.speakers.fail_after:
            raise RuntimeError("Unanticipated host error [PaErrorCode -9999]")
        self.writes.append(int(data.shape[0]))
        if self.speakers.write_delay:
            time.sleep(self.speakers.write_delay)
        return False

    def stop(self):
        self.stopped = True

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True
        with self.speakers.lock:
            if self.started:
                self.speakers.active -= 1
            self.speakers.events.append(("close", id(self)))


class Speakers:
    """Stands in for the output device and remembers everything done to it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.streams: list[RecordingStream] = []
        self.events: list[tuple[str, int]] = []
        self.active = 0
        self.most_at_once = 0
        self.write_delay = 0.0
        self.fail_after: int | None = None
        self.refuse_rates: set[int] = set()

    def open(self, **kwargs):
        if kwargs["samplerate"] in self.refuse_rates:
            raise RuntimeError("Invalid sample rate [PaErrorCode -9997]")
        stream = RecordingStream(self, **kwargs)
        self.streams.append(stream)
        return stream


@pytest.fixture()
def speakers(monkeypatch) -> Speakers:
    fake = Speakers()
    monkeypatch.setattr(voice_module, "_output_stream", fake.open)
    yield fake
    assert not SPEAKERS.busy, "a test left somebody waiting for the speakers"


@pytest.fixture()
def service(tmp_path: Path) -> VoiceService:
    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "workspace", data_dir=tmp_path / "data")
    settings.prepare()
    return VoiceService(settings)


def silence_wav(seconds: float, rate: int = KURDISHTTS_RATE) -> bytes:
    return pcm_to_wav(numpy.zeros(int(rate * seconds), dtype=numpy.float32), rate)


def stereo_silence_wav(frames: int, rate: int) -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(numpy.zeros(frames * 2, dtype=numpy.int16).tobytes())
    return buffer.getvalue()


def sorani_voice(service: VoiceService, monkeypatch, seconds: float = 1.0, *, fail: str | None = None):
    """KurdishTTS without the network: it answers with silence of a known length."""

    class FakeTts:
        calls = 0

        def synthesize(self, text, speaker_id=None):
            FakeTts.calls += 1
            if fail:
                raise RuntimeError(fail)
            return silence_wav(seconds), {"route": "kurdishtts", "speaker_id": "speaker-1"}

    monkeypatch.setattr(service, "sorani_stack", lambda: (None, FakeTts()))
    return FakeTts


# --- One continuous stream ------------------------------------------------------


def test_a_reply_plays_through_one_stream_and_every_sample_is_written(service, speakers):
    """One open, one drain -- not a stream torn down five times a second."""
    result = service._play_wav(silence_wav(2.0))

    assert len(speakers.streams) == 1, "a new stream was opened between blocks"
    stream = speakers.streams[0]
    assert stream.kwargs["samplerate"] == KURDISHTTS_RATE
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "float32"
    assert sum(stream.writes) == 2 * KURDISHTTS_RATE, "samples were dropped"
    # Blocks are short enough to notice a barge-in within about 100 ms.
    assert max(stream.writes) <= int(KURDISHTTS_RATE * voice_module.PLAYBACK_BLOCK_SECONDS)
    assert stream.started and stream.stopped and stream.closed
    assert not stream.aborted, "the end of the reply was cut off instead of drained"
    assert result["played"] is True and result["interrupted"] is False and "error" not in result
    assert result["seconds"] == pytest.approx(2.0, abs=0.01)


def test_the_wav_decides_the_rate_and_the_channels(service, speakers):
    frames = 16_000
    result = service._play_wav(stereo_silence_wav(frames, 16_000))

    stream = speakers.streams[0]
    assert stream.kwargs["samplerate"] == 16_000
    assert stream.kwargs["channels"] == 2
    assert sum(stream.writes) == frames
    assert result["played"] is True


def test_a_rate_the_device_refuses_is_converted_not_dropped(service, speakers, monkeypatch):
    speakers.refuse_rates = {KURDISHTTS_RATE}
    monkeypatch.setattr(voice_module, "_output_defaults", lambda device: (48_000, 2))
    result = service._play_wav(silence_wav(1.0))

    assert [stream.kwargs["samplerate"] for stream in speakers.streams] == [48_000]
    assert speakers.streams[0].kwargs["channels"] == 1
    assert sum(speakers.streams[0].writes) == pytest.approx(48_000, abs=2)
    assert result["played"] is True and "error" not in result


def test_piper_sentences_share_one_stream(service, speakers, monkeypatch):
    """Piper used sounddevice.play() per sentence too; now it streams into one."""
    chunk = numpy.zeros(11_025, dtype=numpy.int16).tobytes()

    class FakePiper:
        config = SimpleNamespace(sample_rate=22_050)

        def synthesize(self, text):
            for _ in range(3):
                yield SimpleNamespace(audio_int16_bytes=chunk, sample_rate=22_050)

    monkeypatch.setattr(service.tts, "piper_voice_available", lambda: True)
    monkeypatch.setattr(service.tts, "_load_piper", lambda: FakePiper())
    result = service.speak("Gold is trading at 408.", language="en")

    assert len(speakers.streams) == 1
    assert sum(speakers.streams[0].writes) == 3 * 11_025
    assert result["engine"] == "piper" and result["chunks"] == 3
    assert result["ok"] is True and result["spoken"] is True


# --- Barge-in -------------------------------------------------------------------


def test_barge_in_aborts_the_stream_mid_clip(speakers):
    writes_before_cancel = 3
    stream_writes = lambda: len(speakers.streams[0].writes) if speakers.streams else 0  # noqa: E731
    result = play_audio([numpy.zeros(KURDISHTTS_RATE * 5, dtype=numpy.float32)], KURDISHTTS_RATE,
                        cancel=lambda: stream_writes() >= writes_before_cancel)

    stream = speakers.streams[0]
    assert len(stream.writes) == writes_before_cancel, "it kept writing after the cancel"
    assert stream.aborted and stream.closed
    assert not stream.stopped, "stop() would have played out the queued audio first"
    assert result["interrupted"] is True and result["played"] is True
    assert result["seconds"] == pytest.approx(0.3, abs=0.01)


def test_a_cancel_before_the_first_word_opens_nothing(speakers):
    result = play_audio([numpy.zeros(KURDISHTTS_RATE, dtype=numpy.float32)], KURDISHTTS_RATE,
                        cancel=lambda: True)
    assert speakers.streams == []
    assert result["interrupted"] is True and result["played"] is False


def test_an_interrupted_reply_is_ok_and_says_it_was_interrupted(service, speakers, monkeypatch):
    """Being stopped is what the user asked for, not a failure to report."""
    sorani_voice(service, monkeypatch, seconds=3.0)
    speakers.write_delay = 0.01
    outcome: dict = {}
    thread = threading.Thread(target=lambda: outcome.update(service.speak("سڵاو، من سامم.", "ckb")))
    thread.start()
    deadline = time.monotonic() + 5
    while not (speakers.streams and speakers.streams[0].writes) and time.monotonic() < deadline:
        time.sleep(0.005)
    assert service.barge_in.interrupt("user_speech") is True
    thread.join(timeout=5)

    assert outcome["interrupted"] is True
    assert outcome["ok"] is True and "error" not in outcome
    assert speakers.streams[0].aborted


# --- Never two at once ------------------------------------------------------------


def test_two_replies_never_play_at_the_same_time(service, speakers, monkeypatch):
    """Hands-free and /api/voice/speak together used to interleave slice by slice."""
    sorani_voice(service, monkeypatch, seconds=0.5)
    speakers.write_delay = 0.01
    results: list[dict] = []
    threads = [threading.Thread(target=lambda: results.append(service.speak("سڵاو، من سامم.", "ckb")))
               for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert speakers.most_at_once == 1, "two replies were on the speakers together"
    assert len(speakers.streams) == 3
    # Each reply ran start-to-close before the next one started.
    kinds = [kind for kind, _ in speakers.events]
    assert kinds == ["start", "close"] * 3
    for stream in speakers.streams:
        assert sum(stream.writes) == KURDISHTTS_RATE // 2, "a reply was cut short by another"
    assert [result["ok"] for result in results] == [True, True, True]


def test_the_listener_stays_deaf_until_the_last_overlapping_reply_ends(service, speakers, monkeypatch):
    """Suppression is a flag: the first reply ending must not wake it during the second."""
    sorani_voice(service, monkeypatch, seconds=0.5)
    speakers.write_delay = 0.01
    hook: list[bool] = []
    service.on_speaking = hook.append
    threads = [threading.Thread(target=service.speak, args=("سڵاو، من سامم.", "ckb")) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert hook == [True, False]
    assert service.barge_in.speaking is False


def test_barge_in_also_stops_the_reply_waiting_behind(service, speakers, monkeypatch):
    """"Stop" means stop, not "next one please"."""
    sorani_voice(service, monkeypatch, seconds=5.0)
    # Slow enough (about 2.5 s for the first reply) that it is certainly still
    # playing when the user says stop; at 10 ms a block it could finish first,
    # and the second reply then rightly took its turn before the barge-in.
    speakers.write_delay = 0.05
    results: list[dict] = []
    threads = [threading.Thread(target=lambda: results.append(service.speak("سڵاو، من سامم.", "ckb")))
               for _ in range(2)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5
    while not (speakers.streams and speakers.streams[0].writes) and time.monotonic() < deadline:
        time.sleep(0.005)
    # Wait until the second reply has synthesised and is really queued behind
    # the first. A fixed sleep here was flaky on a loaded machine: the second
    # reply could reach the queue only after the barge-in had been handled.
    while len(SPEAKERS._line) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(SPEAKERS._line) == 2, "the second reply never queued behind the first"
    service.barge_in.interrupt("user_speech")
    for thread in threads:
        thread.join(timeout=10)

    assert len(speakers.streams) == 1, "the queued reply started after the user said stop"
    assert all(result["interrupted"] for result in results)
    assert not SPEAKERS.busy


def test_the_windows_voice_waits_its_turn_too(service, speakers, monkeypatch):
    sorani_voice(service, monkeypatch, seconds=0.5)
    speakers.write_delay = 0.01
    overlap: list[bool] = []

    class RecordingSapi:
        Rate = 0
        Volume = 100
        Status = SimpleNamespace(RunningState=1)

        def Speak(self, text, flags=0):  # noqa: N802
            overlap.append(speakers.active > 0)

    monkeypatch.setattr(service.tts, "_sapi_speaker", lambda: RecordingSapi())
    first = threading.Thread(target=service.speak, args=("سڵاو، من سامم.", "ckb"))
    first.start()
    deadline = time.monotonic() + 5
    while not speakers.active and time.monotonic() < deadline:
        time.sleep(0.005)
    english = service.speak("Gold is trading at 408.", "en")
    first.join(timeout=10)

    assert english["engine"] == "sapi" and english["ok"] is True
    assert overlap == [False], "the Windows voice spoke over a Sorani reply"


# --- Truthful results ---------------------------------------------------------------


def test_a_spoken_sorani_reply_is_ok(service, speakers, monkeypatch):
    sorani_voice(service, monkeypatch, seconds=1.0)
    result = service.speak("سڵاو، من سامم.", "ckb")
    assert result["ok"] is True and result["spoken"] is True and result["interrupted"] is False
    assert result["engine"].startswith("sorani") and result["language"] == "ckb"
    assert "audio" not in result, "the WAV bytes must not travel back to the API caller"


def test_no_sorani_provider_is_not_ok(service, speakers, monkeypatch):
    for name in ("KURDISHTTS_TTS_API_KEY", "KURDISHTTS_STT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    service._sorani_stack = None
    result = service.speak("سڵاو، من سامم.", "ckb")
    assert result["ok"] is False and result["spoken"] is False
    assert result["error"]
    assert speakers.streams == []


def test_a_synthesis_failure_is_not_ok(service, speakers, monkeypatch):
    sorani_voice(service, monkeypatch, fail="KurdishTTS داواکارییەکەی ڕەتکردەوە (400).")
    result = service.speak("سڵاو، من سامم.", "ckb")
    assert result["ok"] is False
    assert "400" in result["error"]


def test_a_playback_failure_is_carried_up_not_dropped(service, speakers, monkeypatch):
    """_play_wav's error used to stop at _speak_sorani, so a dead device read as spoken."""
    sorani_voice(service, monkeypatch, seconds=1.0)
    speakers.fail_after = 2
    result = service.speak("سڵاو، من سامم.", "ckb")

    assert result["ok"] is False
    assert "PaErrorCode -9999" in result["error"]
    assert speakers.streams[0].closed, "a failed stream was left open"
    assert not SPEAKERS.busy


def test_a_device_that_will_not_open_is_not_ok(service, speakers, monkeypatch):
    sorani_voice(service, monkeypatch, seconds=1.0)
    speakers.refuse_rates = {KURDISHTTS_RATE}
    monkeypatch.setattr(voice_module, "_output_defaults", lambda device: (KURDISHTTS_RATE, 1))
    result = service.speak("سڵاو، من سامم.", "ckb")
    assert result["ok"] is False and "PaErrorCode -9997" in result["error"]


def test_an_unreadable_clip_is_not_ok(service, speakers, monkeypatch):
    class Broken:
        def synthesize(self, text, speaker_id=None):
            return b"RIFF-not-really-a-wav", {"route": "kurdishtts"}

    monkeypatch.setattr(service, "sorani_stack", lambda: (None, Broken()))
    result = service.speak("سڵاو، من سامم.", "ckb")
    assert result["ok"] is False and result["error"]
    assert speakers.streams == []


def test_an_empty_clip_is_not_ok(service, speakers, monkeypatch):
    class Empty:
        def synthesize(self, text, speaker_id=None):
            return pcm_to_wav(numpy.zeros(0, dtype=numpy.float32), KURDISHTTS_RATE), {"route": "kurdishtts"}

    monkeypatch.setattr(service, "sorani_stack", lambda: (None, Empty()))
    result = service.speak("سڵاو، من سامم.", "ckb")
    assert result["ok"] is False and result["error"]


def test_a_piper_failure_is_not_ok(service, speakers, monkeypatch):
    def broken():
        raise RuntimeError("voice model is corrupt")

    monkeypatch.setattr(service.tts, "piper_voice_available", lambda: True)
    monkeypatch.setattr(service.tts, "_load_piper", broken)
    result = service.speak("Gold is trading at 408.", "en")
    assert result["ok"] is False and "corrupt" in result["error"]


def test_a_windows_voice_failure_is_not_ok(service, speakers, monkeypatch):
    def broken():
        raise OSError("CoInitialize has not been called.")

    monkeypatch.setattr(service.tts, "_sapi_speaker", broken)
    result = service.speak("Gold is trading at 408.", "en")
    assert result["engine"] == "sapi"
    assert result["ok"] is False and "CoInitialize" in result["error"]
    assert not SPEAKERS.busy


def test_the_windows_voice_speaking_is_ok(service, speakers):
    result = service.speak("Gold is trading at 408.", "en")
    assert result["engine"] == "sapi" and result["ok"] is True and result["spoken"] is True


def test_an_engine_that_throws_is_reported_and_the_listener_woken(service, speakers, monkeypatch):
    hook: list[bool] = []
    service.on_speaking = hook.append

    def explode(text):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(service, "_speak_sorani", explode)
    result = service.speak("سڵاو، من سامم.", "ckb")
    assert result["ok"] is False and "exploded" in result["error"]
    assert hook == [True, False]


def test_blank_text_is_not_spoken(service, speakers):
    result = service.speak("   ")
    assert result["ok"] is False and result["error"]
    assert speakers.streams == []


def test_the_speak_endpoint_returns_the_outcome(client, speakers, monkeypatch):
    """The settings page shows `error` when there is one; now `ok` agrees with it."""
    voice = client.app.state.voice
    monkeypatch.setattr(voice, "_speak_sorani", lambda text: {"engine": "sorani:kurdishtts", "language": "ckb",
                                                               "error": MESSAGE_NO_PROVIDER, "spoken": False})
    response = client.post("/api/voice/speak", json={"text": "سڵاو", "language": "ckb"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False and body["error"] == MESSAGE_NO_PROVIDER
    assert speakers.streams == []


def test_the_speak_endpoint_plays_through_the_continuous_stream(client, speakers, monkeypatch):
    voice = client.app.state.voice
    sorani_voice(voice, monkeypatch, seconds=0.5)
    body = client.post("/api/voice/speak", json={"text": "سڵاو", "language": "ckb"}).json()
    assert body["ok"] is True and body["spoken"] is True
    assert len(speakers.streams) == 1
    assert sum(speakers.streams[0].writes) == KURDISHTTS_RATE // 2


# --- Decoding -------------------------------------------------------------------------


def test_8_bit_wav_silence_decodes_to_silence():
    """8-bit WAV is unsigned; read as int8 its silence was full scale."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(8_000)
        handle.writeframes(bytes([128]) * 800)
    frames, rate = wav_to_frames(buffer.getvalue())
    assert rate == 8_000 and frames.shape == (800, 1)
    assert not frames.any()


def test_stereo_wav_keeps_its_channels_for_playback_and_folds_for_recognition():
    wav = stereo_silence_wav(1_000, 16_000)
    frames, _ = wav_to_frames(wav)
    mono, _ = wav_to_float32(wav)
    assert frames.shape == (1_000, 2)
    assert mono.shape == (1_000,)


# --- What gets read aloud ---------------------------------------------------------

# The hands-free reply that went silent on 2026-09-24 (552 characters, raw
# Markdown): KurdishTTS's free plan refuses a request over 500 characters.
THE_REPLY_THAT_WENT_SILENT = (
    "ناوم سامە. یاریدەدەرێکی زیرەکی دەستکردی ناوخۆییم و دەتوانم لە کارکردن لەگەڵ فایل، کۆد و کۆمپیوتەر یارمەتیت بدەم.  \n\n"
    "بۆ ئەوەی بتوانم بە شێوەیەکی باشتر یارمەتیت بدەم، تکایە زانیاری زیاتر بدە:  \n\n"
    "- ئەگەر پێویستت بە نووسینی سکریپتی بۆ تحلیل داتای بازاڕی، کەیسی کۆدێک هەیە؟  \n"
    "- ئەگەر دەتەوێت فایلی CSV یان Excel بەکاردەهێنیت، کەیسی فایلی هەیە؟  \n"
    "- ئەگەر پێویستت بە ڕێنمایی لەسەر ستراتیژیەکانی تریەنگ، کەیسی بڕیارەکانت چییە؟  \n\n"
    "تکایە ئەم زانیاریانە بدە، من دەتوانم فایلی کۆد بنووسم، داتاکان تحلیل بکەم، "
    "یان ڕێنماییەکانت بە شێوەیەکی فەرمی و بەسەرچاوەیەک بدەم."
)


def test_markdown_is_not_read_aloud():
    text = "**سەلم** بە مانای سڵاو.\n- یەکەم خاڵ\n2. دووەم خاڵ\n```python\nprint(1)\n```\n# سەردێڕ"
    spoken = voice_module.speakable_text(text)

    for mark in ("*", "`", "#", "print(1)", "- ", "2."):
        assert mark not in spoken, f"{mark!r} would be read aloud"
    for words in ("سەلم", "یەکەم خاڵ", "دووەم خاڵ", "سەردێڕ"):
        assert words in spoken


def test_no_piece_of_a_long_reply_is_over_the_free_plan_cap():
    assert len(THE_REPLY_THAT_WENT_SILENT) > 500
    chunks = voice_module.speakable_chunks(THE_REPLY_THAT_WENT_SILENT)

    assert len(chunks) > 1
    assert all(len(chunk) <= voice_module.SPEECH_CHUNK_CHARS for chunk in chunks)
    # Only Markdown is lost; every word is still said, in order.
    assert " ".join(chunks).split() == voice_module.speakable_text(THE_REPLY_THAT_WENT_SILENT).split()


def test_a_sentence_longer_than_a_piece_is_cut_at_a_space():
    sentence = " ".join(["وشە"] * 120) + "."
    chunks = voice_module.speakable_chunks(sentence)

    assert len(chunks) > 1
    assert all(len(chunk) <= voice_module.SPEECH_CHUNK_CHARS for chunk in chunks)
    assert all(word == "وشە" for chunk in chunks for word in chunk.rstrip(".").split())


def test_a_long_sorani_reply_is_synthesised_piece_by_piece_into_one_stream(service, speakers, monkeypatch):
    requests: list[str] = []

    class PiecewiseTts:
        def synthesize(self, text, speaker_id=None):
            requests.append(text)
            return silence_wav(0.3), {"route": "kurdishtts", "speaker_id": "speaker-1"}

    monkeypatch.setattr(service, "sorani_stack", lambda: (None, PiecewiseTts()))
    result = service.speak(THE_REPLY_THAT_WENT_SILENT, "ckb")

    assert result["ok"] is True, result
    assert len(requests) > 1 and all(len(text) <= voice_module.SPEECH_CHUNK_CHARS for text in requests)
    assert not any("*" in text or text.lstrip().startswith("-") for text in requests)
    assert len(speakers.streams) == 1, "each piece opened its own stream"
    assert sum(speakers.streams[0].writes) == len(requests) * int(0.3 * KURDISHTTS_RATE)
    assert result["chunks"] == result["chunks_total"] == len(requests)


def test_a_piece_that_fails_later_is_reported_not_swallowed(service, speakers, monkeypatch):
    calls = {"count": 0}

    class FailsOnTheSecondPiece:
        def synthesize(self, text, speaker_id=None):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("KurdishTTS refused the request (4xx).")
            return silence_wav(0.3), {"route": "kurdishtts", "speaker_id": "speaker-1"}

    monkeypatch.setattr(service, "sorani_stack", lambda: (None, FailsOnTheSecondPiece()))
    result = service.speak(THE_REPLY_THAT_WENT_SILENT, "ckb")

    assert result["ok"] is False
    assert result["spoken"] is True, "the first sentence was already heard"
    assert result["error"].startswith("Stopped after 1 of")
    assert len(speakers.streams) == 1 and speakers.streams[0].stopped


def test_an_empty_reply_speaks_nothing_and_says_so(service, speakers, monkeypatch):
    tts = sorani_voice(service, monkeypatch)
    result = service.speak("```\ncode only\n```", "ckb")

    assert result["ok"] is False and result["spoken"] is False
    assert tts.calls == 0 and not speakers.streams
