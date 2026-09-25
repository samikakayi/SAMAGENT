"""The real CAM++ speaker model on this PC (skipped where the model or the
offline corpus is absent: they live in the git-ignored ``work/`` folder, or set
``SAM_TEST_SPEAKER_MODEL``). Measures what voiceprint.py promises: <= 150 ms
per utterance (loudness normalisation included), loudness does not change who
the speaker is, and after the owner's turns (first utterances after a click)
the adapted profile and its owner-calibrated threshold keep the owner in and
other voices out. Synthetic voices only (no real recordings exist on this PC);
the real user's own scores (0.055-0.346 against a quiet enrollment) are what
the lower base threshold and the adaptation are for."""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from pathlib import Path

import pytest

from sam.voice.voiceprint import (MODEL_FILE, SENSITIVITY, SherpaEmbedder, SpeakerCheck, VoiceprintStore, cosine,
                                  mean_vector, sherpa_available)

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("SAM_TEST_SPEAKER_MODEL") or ROOT / "work" / "models" / MODEL_FILE)
CORPUS = ROOT / "work" / "voiceeval" / "corpus"

pytestmark = pytest.mark.skipif(not (sherpa_available() and MODEL.is_file() and CORPUS.is_dir()),
                                reason="speaker model / offline corpus not present")


def audio(stem: str):
    import numpy
    return numpy.load(CORPUS / f"{stem}.npy")


def pcm(samples, db: float | None = None) -> bytes:
    import numpy
    if db is not None:
        rms = float(numpy.sqrt(numpy.mean(samples ** 2))) or 1e-9
        samples = samples * (10 ** (db / 20) / rms)
    return (numpy.clip(samples, -1, 1) * 32767).astype(numpy.int16).tobytes()


def voice(prefix: str, db: float | None = None) -> list[bytes]:
    return [pcm(audio(f.stem), db) for f in sorted(CORPUS.glob(f"{prefix}__*.npy"))]


def test_real_model_speed_and_loudness(make_app):
    app = make_app()
    embedder = SherpaEmbedder(MODEL, threads=2)
    began = time.perf_counter()
    embedder.load()
    load_ms = (time.perf_counter() - began) * 1000
    check = SpeakerCheck(app, embedder_factory=lambda path: embedder, store=VoiceprintStore(app.db))
    clips = [audio(f.stem) for f in sorted(CORPUS.glob("sapi_david__*.npy"))][:6]
    times, quiet, shout = [], [], []
    for clip in clips:
        began = time.perf_counter()
        reference = check._embed_sync(embedder, pcm(clip, -26.0))  # noqa: SLF001 - what verify() runs in its thread
        times.append((time.perf_counter() - began) * 1000)
        quiet.append(cosine(reference, check._embed_sync(embedder, pcm(clip, -36.4))))  # noqa: SLF001
        shout.append(cosine(reference, check._embed_sync(embedder, pcm(clip, -13.4))))  # noqa: SLF001 - clipped
    assert statistics.median(times) < 150, times
    assert load_ms < 5000
    # The real user: enrollment -33.9 dBFS, then -36.4 and (frustrated) -13.4 dBFS.
    assert min(quiet) > 0.98 and min(shout) > 0.9, (quiet, shout)


async def test_real_model_owner_adaptation_keeps_other_voices_out(make_app):
    app = make_app()
    embedder = SherpaEmbedder(MODEL, threads=2)
    store = VoiceprintStore(app.db, protect=lambda b: b[::-1], unprotect=lambda b: bytes(b)[::-1])
    check = SpeakerCheck(app, embedder_factory=lambda path: embedder, store=store)
    user = voice("sapi_david", -34.0)
    enrolled = mean_vector([await check.embed(clip) for clip in user[:4]])
    store.save(enrolled, model="campplus", level_db=-34.0, clips=4, consistency=0.7)
    check.forget_cache()
    assert check.threshold() == SENSITIVITY["normal"]
    for clip in voice("sapi_david", -18.0)[4:6]:                      # the owner's turns after two clicks (louder)
        assert await check.learn_owner(await check.embed(clip))
    threshold = check.threshold()
    assert SENSITIVITY["normal"] <= threshold <= 0.35
    targets = [(await check.verify(clip, speech_ms=2000)).score for clip in voice("sapi_david", -26.0)[6:]]
    others = [(await check.verify(clip, speech_ms=2000)).score for prefix in ("onecore_mark", "sapi_zira",
                                                                             "kurdishtts_sorani1", "other_en_tts")
              for clip in voice(prefix, -26.0)]
    assert min(targets) >= threshold, (min(targets), threshold)
    assert sum(score >= threshold for score in others) <= 1, (sorted(others)[-3:], threshold)
    assert check.status()["owner_utterances"] == 2 and "vector" not in repr(check.status())
    stored = store.load()
    assert len(stored["owners"]) == 2 and stored["enroll"] == pytest.approx(enrolled, abs=1e-5)
    await asyncio.sleep(0)
