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
0.385; different-persona pairs never above 0.383.

The REAL user (2026-09-25 morning, sam2.log + activity): enrolled with 5 clips
(consistency 0.686, level -33.9 dBFS), then his own voice scored 0.055 / 0.346
/ 0.212 / 0.224 at -36.4 / -25.9 / -13.4 / -25.6 dBFS and was rejected four
times by the old threshold 0.40 -- calibrated on synthetic TTS voices read
calmly; real speech through a real headset in a real room, louder and
frustrated, scores far lower against a quiet enrollment. The policy is now
(listening.py, frames.py):

- the FIRST utterance after an explicit activation (island click / hotkey) is
  the owner by definition: never rejected here (only the near-field gate
  applies). Once its words are admitted it adapts the profile
  (``learn_owner``): the stored vector is the weighted mean of the enrollment
  (weight = 2 x its clips) and the last ``OWNER_KEEP`` (10) owner utterances
  -- a bounded running average, stored DPAPI-protected like the enrollment;
- the voiceprint gates only follow-ups, barge-ins and always-listening. Its
  threshold is the base -- low 0.15 / normal 0.20 / high 0.28 (3 of the 4
  rejected real scores pass 0.20) -- raised with how the owner really scores
  against the CURRENT profile: ``max(base, min(q25 - margin, cap))`` over the
  leave-one-out scores of the kept owner utterances (unbiased: they were never
  selected by score; the minimum below 4 of them; normal: margin 0.15, cap
  0.35); under 1 s of speech -0.05; never below 0.10;
- loudness does not matter: every clip is normalised to -24 dBFS of active
  speech (peak-limited) before embedding (``normalize_level``), enrollment and
  verification alike.

Measured with the real model on the offline corpus (no real recordings exist
here; enrollment clean at -34 dBFS; use through a simulated room + headset
channel -- band-limit, brighter vocal effort, reverb, noise at 18 dB SNR,
optionally shouted at -13.4 dBFS and clipped -- which brings the owner's
scores against the enrollment to 0.23-0.55, the real user's range; 6 voices,
every rotation of which clips are the turns after a click; other voices =
other personas, through the same channel / clean):

    owner turns after clicks        0      1      2      4
    room: owner rejected (FRR)   0.0 %  2.7 %  1.9 %  1.7 %   (old 0.40: 45.5 %)
    room: other voice accepted  11.1 %  9.1 % 14.1 %  9.4 %   same room
    clean/shout: FRR / FAR       0 % / 37.8 %  0 % / 0.1-0.3 %   (old 0.40: 0 % / 0 %)
    threshold (median)           0.20   0.27   0.28   0.35

A first draft that kept pre-adaptation owner scores (lagging) and compared
follow-ups with the last owner utterance as well let 93-99 % of the other
voices of the same room in: the adapted profile learns the room/channel too,
so the threshold must rise with the owner's scores against the adapted
profile. Loudness alone barely moves CAM++ (cosine to the same clip: -36 dBFS
0.995, -13.4 dBFS clipped 0.964; normalised 0.999 / 0.970): clipping and the
room/channel mismatch lowered the real scores, adaptation restores them.
Synthetic voices overstate impostor similarity (clean other voices score a
median 0.19 against an enrolled synthetic voice), so the 0-turn FAR is a
pessimistic figure; the near-field gate filters far voices before this.

Without a voiceprint (or without sherpa-onnx) every check passes and the
near-field gate alone decides (gate.py).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import threading
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .audio import MIC_RATE

log = logging.getLogger("sam.voice.voiceprint")

MODEL_FILE = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/" + MODEL_FILE)
MODEL_SHA256 = "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"
MODEL_BYTES = 28_281_164
# Follow-up / barge-in / always-listening thresholds (see the module docstring).
SENSITIVITY = {"low": 0.15, "normal": 0.20, "high": 0.28}
# How far under the owner's own low (q25) leave-one-out score a voice must fall to be "clearly other",
# and how high the threshold may rise when the owner scores high (grid on real embeddings: 0.15 / 0.35;
# enrollment weight 2 x clips: overall owner FRR ~2 %, other-voice FAR ~1 % over the scratch grid).
ADAPTIVE_MARGIN = {"low": 0.20, "normal": 0.15, "high": 0.10}
ADAPTIVE_CAP = {"low": 0.30, "normal": 0.35, "high": 0.40}
ADAPTIVE_QUANTILE = 0.25         # (the minimum while fewer than ADAPTIVE_MIN_SAMPLES owner utterances)
ADAPTIVE_MIN_SAMPLES = 4
SHORT_SPEECH_MS = 1000.0
SHORT_RELIEF = 0.05
THRESHOLD_FLOOR = 0.10           # never below: that is where unrelated voices score
OWNER_KEEP = 10                  # owner utterances kept in the profile (a bounded running average)
ENROLL_WEIGHT_PER_CLIP = 2.0     # the enrollment keeps >= half of the profile's weight
ADAPT_MIN_SPEECH_MS = 800.0      # shorter utterances give noisy embeddings: not used to adapt
LEVEL_TARGET_DB = -24.0          # loudness every clip is brought to before embedding
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
    vector: list[float] | None = field(default=None, repr=False)   # in memory only, never in status()

    def public(self) -> dict[str, Any]:
        return {"ok": self.ok, "score": self.score, "threshold": self.threshold, "ms": self.ms,
                "reason": self.reason}


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


def _total(enroll: list[float], weight: float, owners: list[list[float]]) -> list[float]:
    total = [weight * e for e in enroll]
    for vector in owners:
        total = [t + v for t, v in zip(total, vector)]
    return total


def weighted_profile(enroll: list[float], weight: float, owners: list[list[float]] | None) -> list[float]:
    """The vector verification uses: the enrollment (``weight``) plus the kept
    owner utterances (1 each, at most ``OWNER_KEEP``): the enrollment always
    keeps a real share, one odd utterance moves the profile only a little."""
    return normalize(_total(enroll, weight, list(owners or [])[-OWNER_KEEP:]))


def loo_scores(enroll: list[float], weight: float, owners: list[list[float]] | None) -> list[float]:
    """How each kept owner utterance scores against the profile built WITHOUT
    it: how an unseen owner utterance scores against the current profile
    (with one kept utterance: its score against the enrollment)."""
    kept = list(owners or [])[-OWNER_KEEP:]
    total = _total(enroll, weight, kept)
    return [cosine([t - v for t, v in zip(total, vector)], vector) for vector in kept]


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    low = int(math.floor(position))
    high = min(len(ordered) - 1, low + 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def normalize_level(pcm16: bytes, *, target_db: float = LEVEL_TARGET_DB, frame: int = 480) -> bytes:
    """Scale 16-bit PCM so its ACTIVE speech (30 ms frames within 30 dB of the
    loudest) has ``target_db`` RMS, peak-limited to -0.3 dBFS: the same voice
    whispered at -36 dBFS or shouted at -13 dBFS reaches the model alike (real
    use 2026-09-25). Pure Python (~15 ms for 3 s; runs in the embedding thread)."""
    samples = array("h")
    samples.frombytes(pcm16[: len(pcm16) - len(pcm16) % 2])
    if not samples:
        return b""
    energies = []
    for start in range(0, len(samples), frame):
        chunk = samples[start:start + frame]
        energies.append(sum(v * v for v in chunk) / max(1, len(chunk)))
    loudest = max(energies)
    if loudest <= 0:
        return samples.tobytes()
    floor = loudest * 1e-3                                  # 30 dB under the loudest frame
    active = [e for e in energies if e >= floor]
    rms = math.sqrt(sum(active) / len(active)) / 32768.0
    peak = max(abs(v) for v in samples) / 32768.0
    if rms <= 0 or peak <= 0:
        return samples.tobytes()
    gain = min(10 ** (target_db / 20.0) / rms, 0.966 / peak)
    if 0.98 <= gain <= 1.02:
        return samples.tobytes()
    return array("h", (max(-32768, min(32767, int(v * gain))) for v in samples)).tobytes()


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

    def _blob(self, state: dict[str, Any]) -> bytes:
        """Everything about the voice (the profile, the enrollment, the kept
        owner utterances) in ONE DPAPI-protected blob: nothing voice-related
        is stored in clear."""
        payload = {"v": 2, "vector": [round(x, 6) for x in state["vector"]],
                   "enroll": [round(x, 6) for x in state.get("enroll") or state["vector"]],
                   "owners": [[round(x, 6) for x in vector] for vector in (state.get("owners") or [])][-OWNER_KEEP:],
                   "owners_total": int(state.get("owners_total") or 0)}
        return self._protect(json.dumps(payload).encode("utf-8"))

    def save(self, vector: list[float], *, model: str, level_db: float | None, clips: int,
             consistency: float | None) -> None:
        """A new enrollment: the adaptation and the owner scores start over."""
        self._ensure()
        blob = self._blob({"vector": vector, "enroll": vector})
        now = time.time()
        self.db.execute("INSERT OR REPLACE INTO voice_profile (id, blob, model, dims, level_db, clips, consistency, "
                        "created_at, updated_at) VALUES (1,?,?,?,?,?,?,?,?)",
                        (blob, model, len(vector), level_db, clips, consistency, now, now))

    def update(self, state: dict[str, Any]) -> bool:
        """Store an adapted profile (same protection; the enrollment row's
        model/level/clips/created_at stay)."""
        self._ensure()
        blob = self._blob(state)
        self.db.execute("UPDATE voice_profile SET blob=?, updated_at=? WHERE id=1", (blob, time.time()))
        return self.exists()

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
        vector = [float(x) for x in data.get("vector", [])]
        return {"vector": vector, "enroll": [float(x) for x in (data.get("enroll") or vector)],
                "owners": [[float(x) for x in owner] for owner in (data.get("owners") or [])],
                "owners_total": int(data.get("owners_total") or 0),
                "model": row["model"], "level_db": row.get("level_db"), "clips": row.get("clips"),
                "consistency": row.get("consistency"), "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at")}

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
        self._adapt_lock: asyncio.Lock | None = None
        self.unavailable_reason = ""
        self.last: VerifyResult | None = None
        self.last_owner: dict[str, Any] | None = None

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

    def _sensitivity(self) -> str:
        level = str(self.app.config.get("voice.only_my_voice_sensitivity", "normal") or "normal")
        return level if level in SENSITIVITY else "normal"

    def base_threshold(self) -> float:
        return SENSITIVITY[self._sensitivity()]

    @staticmethod
    def _enroll_weight(profile: dict[str, Any]) -> float:
        return ENROLL_WEIGHT_PER_CLIP * float(profile.get("clips") or 5)

    def owner_scores(self) -> list[float]:
        """Leave-one-out scores of the kept owner utterances against the
        current profile: how the owner really scores now."""
        profile = self.profile()
        if not profile or not profile.get("owners"):
            return []
        return loo_scores(profile.get("enroll") or profile["vector"], self._enroll_weight(profile), profile["owners"])

    def threshold(self, speech_ms: float | None = None) -> float:
        """The score a follow-up / barge-in / always-listening utterance needs:
        the base, raised towards just under the owner's own low scores against
        the current profile (never selected by score; at most the cap) as the
        adapted profile fits the owner better -- other voices heard through
        the same room then stay out (see the module docstring). It never
        drops below the base: on the measured data that cost no owner
        rejections and let fewer other voices in."""
        base = self.base_threshold()
        value = base
        scores = self.owner_scores()
        if scores:
            level = self._sensitivity()
            low = quantile(scores, ADAPTIVE_QUANTILE) if len(scores) >= ADAPTIVE_MIN_SAMPLES else min(scores)
            value = max(base, min(low - ADAPTIVE_MARGIN[level], ADAPTIVE_CAP[level]))
        if speech_ms is not None and speech_ms < SHORT_SPEECH_MS:
            value -= SHORT_RELIEF
        return round(max(THRESHOLD_FLOOR, value), 3)

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

    def _embed_sync(self, embedder: Any, pcm: bytes) -> list[float]:
        clip = pcm[: int(VERIFY_MAX_S * MIC_RATE) * 2]
        return embedder.embed(normalize_level(clip))

    async def embed(self, pcm: bytes) -> list[float]:
        """Speaker embedding of ``pcm`` after loudness normalisation (the
        enrollment and every check alike: loudness never changes the score)."""
        embedder = self.embedder()
        return await asyncio.to_thread(self._embed_sync, embedder, pcm)

    async def verify(self, pcm: bytes, *, speech_ms: float | None = None) -> VerifyResult:
        """The follow-up / barge-in / always-listening check (the first
        utterance after a click is never verified: ``learn_owner``)."""
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
            (time.perf_counter() - began) * 1000.0, 1), "match" if score >= threshold else "other_voice",
            vector=vector)
        self.last = result
        return result

    async def learn_owner(self, vector: list[float] | None = None, *, pcm: bytes | None = None,
                          speech_ms: float | None = None) -> dict[str, Any] | None:
        """The owner spoke (the first utterance after an explicit activation,
        admitted as a request): keep it in the profile (the last
        ``OWNER_KEEP``, a bounded running average with the enrollment, stored
        DPAPI-protected). Returns {"score" (against the profile before it),
        "n", "threshold" (the follow-up threshold now)} or None."""
        if not self.enabled or self.unavailable_reason:
            return None
        if speech_ms is not None and speech_ms < ADAPT_MIN_SPEECH_MS:
            return None
        try:
            if vector is None:
                if pcm is None:
                    return None
                vector = await self.embed(pcm)
        except Exception as exc:  # noqa: BLE001 - adaptation is optional
            log.info("owner utterance not embedded: %s", type(exc).__name__)
            return None
        if self._adapt_lock is None:
            self._adapt_lock = asyncio.Lock()
        async with self._adapt_lock:
            profile = self.profile()
            if profile is None:
                return None
            score = cosine(vector, profile["vector"])
            enroll = profile.get("enroll") or profile["vector"]
            owners = [*(profile.get("owners") or []), normalize(list(vector))][-OWNER_KEEP:]
            state = {**profile, "enroll": enroll, "owners": owners,
                     "owners_total": int(profile.get("owners_total") or 0) + 1,
                     "vector": weighted_profile(enroll, self._enroll_weight(profile), owners)}
            try:
                await asyncio.to_thread(self.store.update, state)
            except Exception as exc:  # noqa: BLE001 - keep the in-memory profile; storage is retried next time
                log.warning("adapted voiceprint not stored: %s", type(exc).__name__)
            self._profile, self._profile_loaded = state, True
            self.last_owner = {"score": round(score, 3), "n": len(owners), "threshold": self.threshold()}
            return dict(self.last_owner)

    def status(self) -> dict[str, Any]:
        profile = self.profile()
        scores = self.owner_scores()
        return {"enrolled": profile is not None, "enabled": self.enabled,
                "sensitivity": self.app.config.get("voice.only_my_voice_sensitivity", "normal"),
                "threshold": self.threshold(), "base_threshold": self.base_threshold(),
                "model_ready": model_ready(self.app), "usable": self.usable,
                "sherpa": sherpa_available(), "unavailable": self.unavailable_reason,
                "clips": profile.get("clips") if profile else None,
                "created_at": profile.get("created_at") if profile else None,
                "owner_utterances": len(profile.get("owners") or []) if profile else 0,
                "owner_scores": {"n": len(scores), "min": round(min(scores), 3) if scores else None,
                                 "q25": round(quantile(scores, ADAPTIVE_QUANTILE), 3) if scores else None},
                "last": self.last.public() if self.last else None, "last_owner": self.last_owner}


__all__ = ["SpeakerCheck", "VoiceprintStore", "SherpaEmbedder", "VerifyResult", "VoiceprintError", "download_model",
           "model_path", "model_ready", "sherpa_available", "cosine", "mean_vector", "normalize", "normalize_level",
           "weighted_profile", "loo_scores", "quantile", "SENSITIVITY", "MODEL_URL", "MODEL_SHA256", "MODEL_FILE", "MODEL_BYTES"]
