"""«تەنها دەنگی من» -- local speaker verification (the user asked for it, 2026-09-24).

Model: 3D-Speaker CAM++ ``3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced``
(Apache-2.0) through sherpa-onnx 1.13.8 (``SpeakerEmbeddingExtractor``, cp313
Windows wheel), downloaded once from the official k2-fsa release (28.3 MB,
SHA-256 checked) into ``%LOCALAPPDATA%\\SAM2\\models`` -- only when the user
starts the enrollment. Nothing leaves the PC: the voiceprint is one 192-float
embedding, DPAPI-protected (current user) in ``data/sam2.sqlite3`` table
``voice_profile``, deleted with «سڕینەوەی دەنگی من».

Chosen by measurement on this PC (work/voiceeval/eval_spk*.py, offline corpus
of 83 clips from 8 synthetic voices -- KurdishTTS sorani_1, Windows David /
Zira / Mark in two engines, another English TTS -- there are no recordings of
real people on this PC, so real-voice margins will be smaller):

    model (sherpa-onnx, 1 thread)     ms/utt median  target min  non-target max
    CAM++ zh-en common_advanced            83           0.672         0.383
    CAM++ en VoxCeleb                      84           0.272         0.843 (!)
    ERes2Net base zh-cn                   201           0.595         0.464
    WeSpeaker ResNet34 en                 207           0.790         0.746

CAM++ common_advanced with 2 threads on the first 3 s of speech: median 60 ms,
p90 78 ms, max 100 ms per utterance (budget 150 ms); load 1.2 s (done in a
thread when listening starts), +45 MB RSS. Equal-error threshold on the corpus
0.385; different-persona pairs never above 0.383; the same voice with a second
talker mixed in at +10 dB SNR still scored >= 0.589 (at 0 dB: 0.248). Hence
thresholds: low 0.40 / normal 0.50 / high 0.60 (setting
``voice.only_my_voice_sensitivity``). Utterances under 1 s of speech (a quick
«بەڵێ») score lower (0.5 s crops: target min 0.397, non-target max 0.319), so
they use threshold - 0.10.

Without a voiceprint (or without sherpa-onnx) every check passes and the
near-field gate alone decides (gate.py).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .audio import MIC_RATE

log = logging.getLogger("sam.voice.voiceprint")

MODEL_FILE = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/" + MODEL_FILE)
MODEL_SHA256 = "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"
MODEL_BYTES = 28_281_164
SENSITIVITY = {"low": 0.40, "normal": 0.50, "high": 0.60}
SHORT_SPEECH_MS = 1000.0
SHORT_RELIEF = 0.10
VERIFY_MAX_S = 3.0
SCHEMA = [(1, """CREATE TABLE IF NOT EXISTS voice_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    blob BLOB NOT NULL,
    model TEXT NOT NULL,
    dims INTEGER NOT NULL,
    level_db REAL,
    clips INTEGER NOT NULL DEFAULT 0,
    consistency REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL)""")]


class VoiceprintError(Exception):
    """kind: unavailable | download | integrity | storage | model."""

    def __init__(self, kind: str, message: str = "") -> None:
        super().__init__(f"{kind}: {message}".strip(": "))
        self.kind = kind


@dataclass
class VerifyResult:
    ok: bool
    score: float | None = None
    threshold: float | None = None
    ms: float = 0.0
    reason: str = ""          # match | other_voice | off | not_enrolled | unavailable | error


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm else list(vector)


def mean_vector(vectors: list[list[float]]) -> list[float]:
    count = len(vectors)
    return normalize([sum(column) / count for column in zip(*vectors)])


# -- model file ----------------------------------------------------------------------------------

def model_path(app: Any) -> Path:
    custom = str(app.config.get("voice.speaker_model_path", "") or "").strip()
    if custom:
        return Path(custom)
    return Path(app.config.log_dir).parent / "models" / MODEL_FILE


def model_ready(app: Any) -> bool:
    path = model_path(app)
    if not path.is_file():
        return False
    return bool(app.config.get("voice.speaker_model_path", "")) or path.stat().st_size == MODEL_BYTES


def sherpa_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("sherpa_onnx") is not None


async def download_model(app: Any, progress: Callable[[float], None] | None = None, *,
                         transport: Any = None) -> Path:
    """Fetch the model from the official k2-fsa release (user-started enrollment only)."""
    import httpx

    target = model_path(app)
    if model_ready(app):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    digest = hashlib.sha256()
    received = 0
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=15.0),
                                     transport=transport, trust_env=True) as client:
            async with client.stream("GET", MODEL_URL) as response:
                if response.status_code != 200:
                    raise VoiceprintError("download", f"HTTP {response.status_code}")
                with part.open("wb") as handle:
                    async for chunk in response.aiter_bytes(1 << 16):
                        handle.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                        if progress is not None:
                            progress(min(1.0, received / MODEL_BYTES))
    except VoiceprintError:
        part.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 - network problems become a Sorani message
        part.unlink(missing_ok=True)
        raise VoiceprintError("download", type(exc).__name__) from None
    if digest.hexdigest() != MODEL_SHA256 or received != MODEL_BYTES:
        part.unlink(missing_ok=True)
        raise VoiceprintError("integrity", "checksum mismatch")
    part.replace(target)
    return target


class SherpaEmbedder:
    """CAM++ speaker embeddings (192 floats). Thread-safe; call from a worker thread."""

    name = "campplus-zh-en-common-advanced"

    def __init__(self, path: Path, *, threads: int = 2) -> None:
        self.path = Path(path)
        self.threads = threads
        self._extractor: Any = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._extractor is not None:
                return
            import sherpa_onnx

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(self.path), num_threads=self.threads)
            if not config.validate():
                raise VoiceprintError("model", "invalid model file")
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embed(self, pcm16: bytes, rate: int = MIC_RATE) -> list[float]:
        import numpy

        self.load()
        samples = numpy.frombuffer(pcm16[: len(pcm16) - len(pcm16) % 2], dtype=numpy.int16)
        samples = samples[: int(VERIFY_MAX_S * rate)].astype(numpy.float32) / 32768.0
        with self._lock:
            stream = self._extractor.create_stream()
            stream.accept_waveform(rate, samples)
            stream.input_finished()
            vector = self._extractor.compute(stream)
        return normalize([float(x) for x in vector])


# -- storage --------------------------------------------------------------------------------------

def _protect(payload: bytes) -> bytes:
    from ..secrets import _dpapi  # the same DPAPI helper as the key store (current user scope)

    blob = _dpapi("protect", payload)
    if blob is None:
        raise VoiceprintError("storage", "DPAPI is not available")
    return blob


def _unprotect(blob: bytes) -> bytes | None:
    from ..secrets import _dpapi

    return _dpapi("unprotect", bytes(blob))


class VoiceprintStore:
    def __init__(self, db: Any, *, protect: Callable[[bytes], bytes] = _protect,
                 unprotect: Callable[[bytes], bytes | None] = _unprotect) -> None:
        self.db = db
        self._protect = protect
        self._unprotect = unprotect
        self._ready = False

    def _ensure(self) -> None:
        if not self._ready:
            self.db.ensure_schema("voiceprint", SCHEMA)
            self._ready = True

    def save(self, vector: list[float], *, model: str, level_db: float | None, clips: int,
             consistency: float | None) -> None:
        self._ensure()
        blob = self._protect(json.dumps({"v": 1, "vector": [round(x, 6) for x in vector]}).encode("utf-8"))
        now = time.time()
        self.db.execute("INSERT OR REPLACE INTO voice_profile (id, blob, model, dims, level_db, clips, consistency, "
                        "created_at, updated_at) VALUES (1,?,?,?,?,?,?,?,?)",
                        (blob, model, len(vector), level_db, clips, consistency, now, now))

    def load(self) -> dict[str, Any] | None:
        self._ensure()
        row = self.db.query_one("SELECT * FROM voice_profile WHERE id=1")
        if row is None:
            return None
        plain = self._unprotect(row["blob"])
        if plain is None:
            log.warning("the voiceprint could not be decrypted (another Windows user?)")
            return None
        data = json.loads(plain.decode("utf-8"))
        return {"vector": [float(x) for x in data.get("vector", [])], "model": row["model"],
                "level_db": row.get("level_db"), "clips": row.get("clips"), "consistency": row.get("consistency"),
                "created_at": row.get("created_at")}

    def exists(self) -> bool:
        self._ensure()
        return self.db.query_one("SELECT 1 AS x FROM voice_profile WHERE id=1") is not None

    def delete(self) -> bool:
        self._ensure()
        existed = self.exists()
        self.db.execute("DELETE FROM voice_profile WHERE id=1")
        return existed


# -- the check used by the engine ----------------------------------------------------------------------

class SpeakerCheck:
    """``app.voice.speaker``: is this utterance the enrolled user?"""

    def __init__(self, app: Any, *, embedder_factory: Callable[[Path], Any] | None = None,
                 store: VoiceprintStore | None = None) -> None:
        self.app = app
        self.store = store or VoiceprintStore(app.db)
        self._custom_embedder = embedder_factory is not None
        self._embedder_factory = embedder_factory or (lambda path: SherpaEmbedder(path))
        self._embedder: Any = None
        self._profile: dict[str, Any] | None = None
        self._profile_loaded = False
        self.unavailable_reason = ""
        self.last: VerifyResult | None = None

    # -- state ------------------------------------------------------------------------------------
    def profile(self) -> dict[str, Any] | None:
        if not self._profile_loaded:
            self._profile_loaded = True
            try:
                self._profile = self.store.load()
            except Exception:  # noqa: BLE001 - a broken profile = not enrolled
                log.warning("could not read the voiceprint", exc_info=True)
                self._profile = None
        return self._profile

    def forget_cache(self) -> None:
        self._profile_loaded = False
        self._profile = None

    @property
    def enrolled(self) -> bool:
        return self.profile() is not None

    @property
    def enabled(self) -> bool:
        return bool(self.app.config.get("voice.only_my_voice", True)) and self.enrolled

    @property
    def usable(self) -> bool:
        """Enabled AND able to run: the model file is there (or a test
        embedder) and loading/embedding has not failed. When enrolled but not
        usable, ``verify`` lets every voice through ("unavailable"): the
        island and the Settings card say so (review 2026-09-24)."""
        if not self.enabled or self.unavailable_reason:
            return False
        return self._custom_embedder or model_ready(self.app)

    def threshold(self, speech_ms: float | None = None) -> float:
        level = str(self.app.config.get("voice.only_my_voice_sensitivity", "normal") or "normal")
        value = SENSITIVITY.get(level, SENSITIVITY["normal"])
        if speech_ms is not None and speech_ms < SHORT_SPEECH_MS:
            value -= SHORT_RELIEF
        return round(value, 3)

    def embedder(self) -> Any:
        if self._embedder is None:
            if not self._custom_embedder and not sherpa_available():
                raise VoiceprintError("unavailable", "sherpa-onnx is not installed")
            self._embedder = self._embedder_factory(model_path(self.app))
        return self._embedder

    async def warm(self) -> bool:
        """Load the model off the loop (1.2 s measured) when it will be needed."""
        if not self.enabled:
            return False
        try:
            embedder = self.embedder()
            await asyncio.to_thread(embedder.load)
            return True
        except Exception as exc:  # noqa: BLE001
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"[:160]
            log.warning("speaker model unavailable: %s", self.unavailable_reason)
            return False

    async def embed(self, pcm: bytes) -> list[float]:
        embedder = self.embedder()
        return await asyncio.to_thread(embedder.embed, pcm)

    async def verify(self, pcm: bytes, *, speech_ms: float | None = None) -> VerifyResult:
        began = time.perf_counter()
        if not bool(self.app.config.get("voice.only_my_voice", True)):
            return VerifyResult(True, reason="off")
        profile = self.profile()
        if profile is None:
            return VerifyResult(True, reason="not_enrolled")
        try:
            vector = await self.embed(pcm)
        except Exception as exc:  # noqa: BLE001 - never lock the user out: fall back to the gate
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"[:160]
            log.warning("voiceprint check failed, letting the utterance through: %s", self.unavailable_reason)
            return VerifyResult(True, reason="unavailable", ms=(time.perf_counter() - began) * 1000.0)
        score = cosine(vector, profile["vector"])
        threshold = self.threshold(speech_ms)
        result = VerifyResult(score >= threshold, round(score, 3), threshold, round(
            (time.perf_counter() - began) * 1000.0, 1), "match" if score >= threshold else "other_voice")
        self.last = result
        return result

    def status(self) -> dict[str, Any]:
        profile = self.profile()
        return {"enrolled": profile is not None, "enabled": self.enabled,
                "sensitivity": self.app.config.get("voice.only_my_voice_sensitivity", "normal"),
                "threshold": self.threshold(), "model_ready": model_ready(self.app), "usable": self.usable,
                "sherpa": sherpa_available(), "unavailable": self.unavailable_reason,
                "clips": profile.get("clips") if profile else None,
                "created_at": profile.get("created_at") if profile else None,
                "last": vars(self.last) if self.last else None}


__all__ = ["SpeakerCheck", "VoiceprintStore", "SherpaEmbedder", "VerifyResult", "VoiceprintError", "download_model",
           "model_path", "model_ready", "sherpa_available", "cosine", "mean_vector", "normalize", "SENSITIVITY",
           "MODEL_URL", "MODEL_SHA256", "MODEL_FILE", "MODEL_BYTES"]
