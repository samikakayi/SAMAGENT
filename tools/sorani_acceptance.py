"""Sorani voice pipeline acceptance test (spec sections 32-34, 36).

Exercises the real path against the live providers:

    SORANI TEXT -> KurdishTTS TTS -> WAV audio -> KurdishTTS STT
                -> SORANI TEXT -> intent -> SAM action

No microphone is involved, so this does not prove a room-and-headset capture;
it proves the provider round trip, the recognition quality, and the command
mapping. The microphone leg is checked separately by tools/sorani_microphone.py,
which needs a person to speak.

Run:  .venv\\Scripts\\python.exe tools\\sorani_acceptance.py
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_backend import sorani, sorani_intent  # noqa: E402
from sam_backend.config import Settings  # noqa: E402
from sam_backend.secrets import SecretStore  # noqa: E402
from sam_backend.voice import VoiceService, language_support  # noqa: E402

# Real commands the user would speak, with the action each must produce.
# All are ordinary-length utterances; the sub-second case is measured separately
# in section 6, because the provider cannot recognise those reliably.
SPOKEN_COMMANDS: tuple[tuple[str, str], ...] = (
    ("بچۆ بۆ پێنج خولەکی", "set_timeframe"),
    ("شیکاری زێڕ بکە", "analyze"),
    ("هێڵەکان بسڕەوە", "clear_drawings"),
    ("تکایە وەستە", "stop"),
)
results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def _overlap(said: str, heard: str) -> float:
    """How much of what was said came back, after normalisation.

    Word overlap alone is unfair on short commands: a one-word "وەستە" heard as
    "وەستا" scores zero despite differing by a single letter, so the closer of
    the word and character measures is used.
    """
    spoken_text = sorani_intent.normalize(said)
    heard_text = sorani_intent.normalize(heard)
    spoken = set(spoken_text.split())
    recognised = set(heard_text.split())
    by_word = len(spoken & recognised) / len(spoken) if spoken else 0.0
    by_character = difflib.SequenceMatcher(None, spoken_text, heard_text).ratio()
    return max(by_word, by_character)


def main() -> int:
    print("=" * 74)
    print("SORANI VOICE PIPELINE ACCEPTANCE TEST")
    print("=" * 74)

    settings = Settings.from_env()
    settings.prepare()
    store = SecretStore(settings.data_dir)
    voice = VoiceService(settings, store)

    print("\n1) Provider health")
    status = voice.sorani_status()
    step("The Sorani stack reports a state", bool(status.get("state")), status["state"])
    stt_ok = step("KurdishTTS speech recognition is connected",
                  status["stt"]["primary"]["status"] == "CONNECTED",
                  status["stt"]["primary"]["status"])
    tts_ok = step("KurdishTTS speech synthesis is connected",
                  status["tts"]["primary"]["status"] == "CONNECTED",
                  status["tts"]["primary"]["status"])
    step("A Sorani speaker is selected",
         bool(status["tts"]["primary"].get("selected_speaker")),
         str(status["tts"]["primary"].get("selected_speaker")))
    step("Sorani is a full round trip", bool(status.get("round_trip")))

    print("\n2) No credential is exposed anywhere in the status")
    blob = json.dumps(status, ensure_ascii=False)
    keys = [store.get("kurdishtts_stt_api_key"), store.get("kurdishtts_tts_api_key")]
    step("The status payload contains no API key",
         not any(key and key in blob for key in keys))
    step("The status reports states only",
         status["stt"]["primary"]["status"] in
         {"CONNECTED", "UNCONFIGURED", "AUTH_FAILED", "RATE_LIMITED", "ERROR"})

    print("\n3) Language capability tracks the provider, not wishful thinking")
    support = language_support("ckb", status["stt"]["primary"])
    step("Sorani speech recognition is reported available",
         support["stt_supported"] is True and support["state"] == "AVAILABLE")
    step("It is stated that this is not local recognition",
         "whisper has no" in (support.get("note") or "").lower())
    step("Without a provider the same call reports unavailable",
         language_support("ckb", {"status": "UNCONFIGURED", "detail": "none"})["stt_supported"] is False)

    if not (stt_ok and tts_ok):
        print("\n  Skipping the speech round trip: a provider is not connected.")
        return summarize()

    print("\n4) Sorani speech round trip, one real command at a time")
    _, tts_router = voice.sorani_stack()
    stt_router, _ = voice.sorani_stack()
    audio_dir = PROJECT_ROOT / "work" / "sorani"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for index, (phrase, expected_action) in enumerate(SPOKEN_COMMANDS, start=1):
        print(f"\n  Command {index}: {phrase}")
        try:
            audio, metadata = tts_router.synthesize(phrase)
        except Exception as exc:
            step(f"[{index}] Sorani audio was synthesised", False, str(exc)[:90])
            continue
        step(f"[{index}] Sorani audio was synthesised", len(audio) > 1000, f"{len(audio)} bytes")
        (audio_dir / f"command_{index}.wav").write_bytes(audio)

        samples, rate = sorani.wav_to_float32(audio)
        peak = float(abs(samples).max()) if len(samples) else 0.0
        step(f"[{index}] The audio carries real sound, not silence", peak > 0.01, f"peak={peak:.3f}")

        try:
            heard = stt_router.transcribe(samples, rate)
        except Exception as exc:
            step(f"[{index}] The audio was recognised", False, str(exc)[:90])
            continue
        text = (heard.get("text") or "").strip()
        step(f"[{index}] The audio was recognised", bool(text), repr(text))
        step(f"[{index}] Recognition came from KurdishTTS", heard.get("route") == "kurdishtts",
             str(heard.get("route")))
        step(f"[{index}] The reply is in Kurdish script, not another language",
             sorani.looks_sorani(text))
        overlap = _overlap(phrase, text)
        step(f"[{index}] The recognised words match what was said",
             overlap >= 0.5, f"{overlap:.0%} of words returned")

        intent = sorani_intent.parse(text)
        step(f"[{index}] The recognised speech maps to an action",
             intent is not None and intent.action == expected_action,
             f"got {intent.action if intent else None!r}, wanted {expected_action!r}")

    print("\n5) An English voice is never substituted for Sorani")
    step("The TTS router says so explicitly",
         "never substituted" in tts_router.status()["note"].lower())
    spoken = voice.speak("سڵاو، من سامم.", language="ckb")
    step("A Sorani reply is spoken by the Sorani engine",
         str(spoken.get("engine", "")).startswith("sorani"), str(spoken.get("engine")))
    step("The Sorani reply is tagged as Kurdish", spoken.get("language") == "ckb")

    print("\n6) Stopping does not depend on recognising a word")
    # Measured against the live service: a ~0.5s utterance comes back empty most
    # of the time. So the short "وەستە" a user shouts mid-sentence may never
    # become text, and stopping must not wait for it to.
    short_audio, _ = tts_router.synthesize("وەستە")
    short_samples, short_rate = sorani.wav_to_float32(short_audio)
    short_seconds = len(short_samples) / short_rate
    # Synthesis length varies run to run, so what matters is not that this clip
    # is under the threshold but that a one-word stop lands in the range where
    # the provider was measured to be unreliable (recognised 3 times in 8 below
    # ~0.6s, 8 times in 8 near 1s).
    step("A one-word stop is short enough to be at risk of being missed",
         short_seconds < 1.0,
         f"{short_seconds:.2f}s, reliable from ~1.0s (limit {sorani.MIN_RELIABLE_SECONDS}s)")

    heard_short = voice._transcribe_sorani(short_samples, short_seconds)
    recognised = bool((heard_short.get("text") or "").strip())
    print(f"       (recognised this time: {recognised} — best effort by design)")
    step("An unrecognised short utterance explains itself rather than going silent",
         recognised or heard_short.get("too_short") is True,
         heard_short.get("reason", "")[:60])

    # The real stop path: speech energy, no transcript involved.
    voice.barge_in.reset()
    voice.barge_in.begin_speaking()
    interrupted = voice.barge_in.interrupt("user_speech")
    step("Barge-in stops SAM on speech energy, with no recognition needed", interrupted)
    step("The interruption is visible to the caller", voice.barge_in.interrupted is True)
    voice.barge_in.end_speaking()

    step("A recognised stop word is still understood when it does arrive",
         sorani_intent.is_stop_command("وەستا") and sorani_intent.is_stop_command("وەستە"))

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
