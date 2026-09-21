"""Voice acceptance test (spec section 20).

Exercises the real engines end to end: synthesize speech to a WAV with the
installed TTS voice, read that audio back, run it through Silero VAD and the
faster-whisper model, and confirm the words survive the round trip. Then verify
barge-in actually stops speech that is already playing.

Run:  .venv\\Scripts\\python.exe tools\\voice_acceptance.py
"""

from __future__ import annotations

import sys
import threading
import time
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: E402

from sam_backend.config import Settings  # noqa: E402
from sam_backend.voice import SileroVad, VoiceService, rms_energy  # noqa: E402

PHRASE = "Analyze gold on the one hour chart and find an entry on the five minute."
results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def synthesize_to_wav(text: str, path: Path) -> bool:
    """Render speech to a file with the installed Windows voice."""
    try:
        import win32com.client

        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(path), 3)  # 3 = SSFMCreateForWrite
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.AudioOutputStream = stream
        speaker.Speak(text)
        stream.Close()
        return path.is_file() and path.stat().st_size > 1000
    except Exception as exc:
        print(f"   synthesis failed: {exc}")
        return False


def load_wav_16k_mono(path: Path) -> numpy.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels, width, rate, frames = handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()
        raw = handle.readframes(frames)
    dtype = {1: numpy.int8, 2: numpy.int16, 4: numpy.int32}[width]
    audio = numpy.frombuffer(raw, dtype=dtype).astype(numpy.float32)
    audio /= float(numpy.iinfo(dtype).max)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16_000:
        # Linear resample to the 16 kHz the model expects.
        target = int(len(audio) * 16_000 / rate)
        audio = numpy.interp(
            numpy.linspace(0, len(audio) - 1, target), numpy.arange(len(audio)), audio
        ).astype(numpy.float32)
    return audio


def main() -> int:
    print("=" * 74)
    print("VOICE ACCEPTANCE TEST")
    print("=" * 74)

    root = PROJECT_ROOT / "work" / "voice"
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(project_root=root, workspace_root=root / "ws", data_dir=root / "data", voice_language="en-US")
    service = VoiceService(settings)

    print("\n1) Capability probe")
    capabilities = service.capabilities()
    step("Microphone devices enumerated", bool(capabilities["devices"]["inputs"]),
         f"{len(capabilities['devices']['inputs'])} inputs, {len(capabilities['devices']['outputs'])} outputs")
    step("Silero VAD available", capabilities["vad"]["state"] == "AVAILABLE", capabilities["vad"]["engine"])
    step("Speech recognition available", capabilities["stt"]["state"] == "AVAILABLE",
         f"{capabilities['stt']['engine']} / {capabilities['stt']['model']}")
    step("Speech output available", capabilities["tts"]["state"] in {"AVAILABLE", "PARTIALLY_AVAILABLE"},
         capabilities["tts"]["engine"])

    print("\n2) Synthesize real speech audio")
    wav_path = root / "acceptance.wav"
    if not step("TTS rendered audio to a file", synthesize_to_wav(PHRASE, wav_path),
                f"{wav_path.stat().st_size if wav_path.exists() else 0} bytes"):
        return summarize()
    audio = load_wav_16k_mono(wav_path)
    step("Audio loaded as 16 kHz mono", audio.size > 16_000,
         f"{audio.size} samples = {audio.size / 16_000:.2f}s, rms={rms_energy(audio):.4f}")

    print("\n3) Voice activity detection on the real audio")
    vad = SileroVad()
    segments = vad.speech_segments(audio)
    step("VAD found speech in speech", bool(segments), f"{len(segments)} segment(s)")
    silence = numpy.zeros(16_000 * 2, dtype=numpy.float32)
    step("VAD finds no speech in silence", not vad.speech_segments(silence))

    print("\n4) Transcribe the audio (this loads the Whisper model, first run is slow)")
    started = time.perf_counter()
    transcript = service.stt.transcribe(audio, language="en")
    elapsed = (time.perf_counter() - started)
    text = (transcript.get("text") or "").lower()
    print(f"   heard: {text!r}  ({elapsed:.1f}s, lang={transcript.get('language')})")
    keywords = ["gold", "hour", "entry"]
    hits = [word for word in keywords if word in text]
    step("Speech was transcribed", bool(text), f"{len(text)} characters")
    step("Transcript recovers the spoken keywords", len(hits) >= 2, f"matched {hits} of {keywords}")

    print("\n5) Sorani language honesty")
    sorani = service.stt.transcribe(audio, language="ckb")
    step("Sorani speech is refused rather than mistranscribed",
         sorani.get("supported") is False and not sorani.get("text"),
         sorani.get("reason", "")[:90])

    print("\n6) Barge-in stops speech that is already playing")
    service.barge_in.reset()
    outcome: dict = {}

    def speak() -> None:
        outcome.update(service.speak(
            "This is a long sentence that SAM begins reading aloud so that an interruption "
            "can be demonstrated while the audio is still playing."
        ))

    thread = threading.Thread(target=speak, daemon=True)
    thread.start()
    time.sleep(1.2)
    was_speaking = service.barge_in.interrupt("user_speech")
    thread.join(timeout=10)
    step("SAM was speaking when the user interrupted", was_speaking)
    step("Speech stopped on interruption", bool(outcome.get("interrupted")),
         f"engine={outcome.get('engine')} interrupted={outcome.get('interrupted')}")
    step("Interruption was counted", service.barge_in.interruptions == 1,
         f"count={service.barge_in.interruptions}")

    return summarize()


def summarize() -> int:
    print("\n" + "=" * 74)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
    print("=" * 74)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
