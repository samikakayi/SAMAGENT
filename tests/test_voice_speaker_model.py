"""The real CAM++ speaker model on this PC (skipped where the model or the
offline corpus is absent: they live in the git-ignored ``work/`` folder, or set
``SAM_TEST_SPEAKER_MODEL``). Measures what voiceprint.py promises: <= 150 ms
per utterance and a clear gap between the enrolled voice and other voices at
the "normal" threshold 0.50. Synthetic voices only (no real recordings exist
on this PC), so real-voice margins will be smaller."""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import pytest

from sam.voice.voiceprint import MODEL_FILE, SENSITIVITY, SherpaEmbedder, cosine, mean_vector, sherpa_available

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("SAM_TEST_SPEAKER_MODEL") or ROOT / "work" / "models" / MODEL_FILE)
CORPUS = ROOT / "work" / "voiceeval" / "corpus"

pytestmark = pytest.mark.skipif(not (sherpa_available() and MODEL.is_file() and CORPUS.is_dir()),
                                reason="speaker model / offline corpus not present")


def pcm(stem: str) -> bytes:
    import numpy
    audio = numpy.load(CORPUS / f"{stem}.npy")
    return (numpy.clip(audio, -1, 1) * 32767).astype(numpy.int16).tobytes()


def voice(prefix: str) -> list[bytes]:
    return [pcm(f.stem) for f in sorted(CORPUS.glob(f"{prefix}__*.npy"))]


def test_real_model_speed_and_separation():
    embedder = SherpaEmbedder(MODEL, threads=2)
    began = time.perf_counter()
    embedder.load()
    load_ms = (time.perf_counter() - began) * 1000
    user = voice("sapi_david")
    enrolled = mean_vector([embedder.embed(clip) for clip in user[:4]])
    times, targets = [], []
    for clip in user[4:]:
        began = time.perf_counter()
        vector = embedder.embed(clip)
        times.append((time.perf_counter() - began) * 1000)
        targets.append(cosine(enrolled, vector))
    others = [cosine(enrolled, embedder.embed(clip)) for prefix in ("onecore_mark", "sapi_zira", "kurdishtts_sorani1")
              for clip in voice(prefix)]
    threshold = SENSITIVITY["normal"]
    assert statistics.median(times) < 150, times
    assert min(targets) >= threshold > max(others), (min(targets), max(others))
    assert load_ms < 5000
