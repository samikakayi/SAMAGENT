"""«ناساندنی دەنگی من»: record the user's voiceprint (VoiceEngine mixin).

The ONLY time SAM records on purpose, and only after the user pressed Start
in the dialog (opened by the Settings button or by saying «دەنگم بناسە»): the
dialog shows 5 short Sorani sentences (strings.ENROLL_SENTENCES); each is
recorded as one utterance (webrtcvad + the near-field gate + endpointer, up to
12 s; speech already going on when the sentence appeared is skipped), kept in
memory only, checked (>= 1 s of speech, not too quiet), turned into a CAM++
speaker embedding (voiceprint.py) and dropped. The mean embedding is stored
DPAPI-protected in ``voice_profile``; the median speech level of the clips
becomes the near-field gate's ``voice.gate_user_level_db`` (gate.py: with a
known user level TV/family 12+ dB quieter are rejected), and «تەنها دەنگی من»
(``voice.only_my_voice``) turns on. Normal listening is closed while the
enrollment runs.

Consistency check (adversarial review 2026-09-24: the old "each clip against
the mean of ALL clips >= 0.45" accepted 4 clips of the user + 1 of another
man, and 3 + 2 -- the other voice then passed "only my voice" for good):
at least ``MIN_CLIPS`` (4) clips, EVERY PAIR of clips must score >=
``MIN_PAIR`` (0.40), and no clip may be 8 dB quieter than the others.
Measured with the real model on the offline corpus (work/final/
enroll_measure.py): same voice, 5 clips, lowest pair 0.60-0.75; any set with
one or two clips of another voice, lowest pair <= 0.22. The clip that fits
worst is dropped and only that sentence is read again (at most 3 times).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from . import strings
from .audio import pcm_rms
from .gate import GateSettings, NearFieldGate, level_db
from .notices import VoiceEnrollRequest, VoiceNotice
from .vad import Endpointer, FrameClassifier
from .voiceprint import (SpeakerCheck, VoiceprintError, cosine, download_model, mean_vector, model_ready,
                         sherpa_available)


def _pair_scores(vectors: dict[int, list[float]]) -> tuple[float, int]:
    """(lowest cosine of any two clips, the clip that fits the others worst)."""
    keys = list(vectors)
    lowest = 1.0
    fit: dict[int, float] = {}
    for i in keys:
        scores = [cosine(vectors[i], vectors[j]) for j in keys if j != i]
        fit[i] = sum(scores) / len(scores) if scores else 1.0
        lowest = min([lowest, *scores])
    return lowest, min(keys, key=lambda k: fit[k])

log = logging.getLogger("sam.voice.enroll")

MIN_SPEECH_MS = 1000.0
MIN_LEVEL_DB = -55.0
MIN_CLIPS = 4
MIN_PAIR = 0.40                # lowest cosine between any two clips (see the module docstring)
MIN_CONSISTENCY = MIN_PAIR     # kept for callers of the old name
LEVEL_SPREAD_DB = 8.0          # a clip this much quieter than the median is someone/something else
MAX_REDOS = 3
ALREADY_TALKING_FRAMES = 10    # speech that starts in the first 0.3 s began before the sentence appeared
RECORD_TIMEOUT_S = 12.0


def _speech_level(frames: list[bytes], voiced: list[bool]) -> float | None:
    levels = sorted(level_db(pcm_rms(f)) for f, v in zip(frames, voiced) if v)
    return levels[len(levels) // 2] if levels else None


class EnrollmentSupport:
    app: Any
    speaker_check: SpeakerCheck
    listening: bool

    def _init_enrollment(self) -> None:
        self._enroll_clips: dict[int, tuple[bytes, float]] = {}
        self._enrolling = False
        self._enroll_mic: Any = None
        self._enroll_redos = 0

    # -- requests ---------------------------------------------------------------------------------------
    def request_enrollment(self, source: str = "voice") -> None:
        """The user asked for it («دەنگم بناسە»): the Settings voice card opens the dialog."""
        self.app.bus.publish(VoiceEnrollRequest(source=source))

    # -- the dialog's calls (core loop) -----------------------------------------------------------------------
    async def enroll_begin(self) -> dict[str, Any]:
        if not sherpa_available() and getattr(self.speaker_check, "_custom_embedder", False) is False:
            return {"ok": False, "reason": "unavailable", "message_ckb": strings.ENROLL_UNAVAILABLE}
        if self.listening:
            await self.stop_listening()  # type: ignore[attr-defined]
        self._enrolling = True
        self._enroll_clips = {}
        self._enroll_redos = 0
        if not getattr(self.speaker_check, "_custom_embedder", False) and not model_ready(self.app):
            def progress(share: float) -> None:
                percent = int(share * 100)
                if percent % 10 == 0:
                    self.app.bus.publish(VoiceNotice(kind="enroll", text_ckb=strings.ENROLL_DOWNLOADING.format(
                        percent=str(percent).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))),
                        detail=f"download {percent}%"))
            try:
                await download_model(self.app, progress)
            except VoiceprintError as exc:
                self._enrolling = False
                return {"ok": False, "reason": exc.kind, "message_ckb": strings.ENROLL_DOWNLOAD_FAILED}
        try:
            await asyncio.to_thread(self.speaker_check.embedder().load)
            self.speaker_check.unavailable_reason = ""
        except Exception as exc:  # noqa: BLE001
            self._enrolling = False
            log.warning("speaker model failed to load: %s", type(exc).__name__)
            return {"ok": False, "reason": "model", "message_ckb": strings.ENROLL_UNAVAILABLE}
        return {"ok": True, "sentences": list(strings.ENROLL_SENTENCES)}

    async def enroll_record(self, index: int, *, timeout_s: float = RECORD_TIMEOUT_S) -> dict[str, Any]:
        """Record one sentence (index into ENROLL_SENTENCES). Audio stays in memory."""
        if not self._enrolling:
            return {"ok": False, "reason": "not_started"}
        mic = await asyncio.to_thread(self._mic_factory)  # type: ignore[attr-defined]
        self._enroll_mic = mic
        classifier = FrameClassifier(aggressiveness=2, energy_floor=0.002)
        frame_ms = int(self.app.config.get("voice.mic_block_ms", 30))
        gate = NearFieldGate(GateSettings(), frame_ms=frame_ms)   # far/quiet background never starts a clip
        endpointer = Endpointer(frame_ms=frame_ms, silence_ms=900, min_speech_ms=250, max_utterance_s=timeout_s)
        frames: list[bytes] = []
        voiced: list[bool] = []
        pcm = b""
        speech_ms = 0.0
        level: float | None = None
        self._publish("listening", detail="enroll", force=True)  # type: ignore[attr-defined]
        try:
            await mic.start()
            deadline = time.monotonic() + timeout_s

            async def collect() -> None:
                nonlocal pcm, speech_ms, level
                skipping = False
                async for frame in mic.frames():
                    rms = pcm_rms(frame)
                    speech = classifier.is_speech(frame, rms)
                    near = gate.classify(rms, speech)
                    frames.append(frame)
                    voiced.append(near)
                    event = endpointer.process(frame, near)
                    if event is not None and event.kind == "start":
                        gate.utterance_started()
                        # Someone was already talking when the sentence appeared: not the reading.
                        skipping = len(frames) <= ALREADY_TALKING_FRAMES
                    if event is not None and event.kind == "end" and not event.too_short:
                        if skipping:
                            skipping = False
                        else:
                            summary = gate.utterance_levels()
                            pcm, speech_ms = event.pcm, event.speech_ms
                            level = summary.get("p50_db") if summary.get("frames") else None
                            return
                    if time.monotonic() >= deadline:
                        return
            await asyncio.wait_for(collect(), timeout_s + 2.0)
        except asyncio.TimeoutError:
            pass
        except Exception as exc:  # noqa: BLE001
            log.warning("enrollment mic failed: %s", type(exc).__name__)
            return {"ok": False, "reason": "mic", "message_ckb": strings.MIC_FAILED}
        finally:
            self._enroll_mic = None
            await mic.stop()
            self._publish("idle", detail="enroll", force=True)  # type: ignore[attr-defined]
        if level is None:
            level = _speech_level(frames, voiced)
        frames.clear()  # nothing is kept but the accepted utterance, and only until finish()
        if not pcm:
            return {"ok": False, "reason": "no_speech", "message_ckb": strings.ENROLL_NO_SPEECH}
        if speech_ms < MIN_SPEECH_MS:
            return {"ok": False, "reason": "too_short", "message_ckb": strings.ENROLL_TOO_SHORT,
                    "speech_ms": speech_ms}
        if level is None or level < MIN_LEVEL_DB:
            return {"ok": False, "reason": "too_quiet", "message_ckb": strings.ENROLL_TOO_QUIET, "level_db": level}
        self._enroll_clips[int(index)] = (pcm, float(level))
        return {"ok": True, "speech_ms": speech_ms, "level_db": round(level, 1), "clips": len(self._enroll_clips)}

    def _redo(self, index: int, reason: str, **extra: Any) -> dict[str, Any]:
        """Drop one clip and ask for that sentence again (or give up after MAX_REDOS)."""
        self._enroll_clips.pop(index, None)
        self._enroll_redos += 1
        if self._enroll_redos > MAX_REDOS:
            self._enroll_clips = {}
            self._enrolling = False
            return {"ok": False, "reason": reason, "message_ckb": strings.ENROLL_INCONSISTENT, **extra}
        return {"ok": False, "reason": reason, "retry_index": index, "message_ckb": strings.ENROLL_REPEAT_ONE,
                **extra}

    async def enroll_finish(self) -> dict[str, Any]:
        clips = dict(self._enroll_clips)
        if len(clips) < MIN_CLIPS:
            self._enroll_clips = {}
            self._enrolling = False
            return {"ok": False, "reason": "too_few", "message_ckb": strings.ENROLL_TOO_SHORT}
        levels = sorted(level for _, level in clips.values())
        median = levels[len(levels) // 2]
        quiet = min(clips, key=lambda k: clips[k][1])
        if clips[quiet][1] < median - LEVEL_SPREAD_DB:
            return self._redo(quiet, "too_quiet_clip", level_db=round(clips[quiet][1], 1))
        try:
            vectors = {index: await self.speaker_check.embed(pcm) for index, (pcm, _) in clips.items()}
        except Exception as exc:  # noqa: BLE001
            log.warning("enrollment embedding failed: %s", type(exc).__name__)
            self._enroll_clips = {}
            self._enrolling = False
            return {"ok": False, "reason": "model", "message_ckb": strings.ENROLL_UNAVAILABLE}
        consistency, worst = _pair_scores(vectors)
        if consistency < MIN_PAIR:
            return self._redo(worst, "inconsistent", consistency=round(consistency, 3))
        self._enroll_clips = {}
        self._enrolling = False
        profile = mean_vector(list(vectors.values()))
        user_level = round(median, 1)
        try:
            self.speaker_check.store.save(profile, model=getattr(self.speaker_check.embedder(), "name", "model"),
                                          level_db=user_level, clips=len(clips), consistency=round(consistency, 3))
        except Exception as exc:  # noqa: BLE001
            log.warning("voiceprint not stored: %s", type(exc).__name__)
            return {"ok": False, "reason": "storage", "message_ckb": strings.ENROLL_UNAVAILABLE}
        self.speaker_check.forget_cache()
        self.speaker_check.unavailable_reason = ""   # the model just ran for every clip
        self.app.config.set("voice.gate_user_level_db", user_level)
        self.app.config.set("voice.only_my_voice", True)
        self.app.db.log_activity("voice", "voiceprint_saved", ok=True, source="voice",
                                 summary=f"clips={len(clips)} consistency={consistency:.3f} level={user_level}")
        self.app.bus.publish(VoiceNotice(kind="enroll", text_ckb=strings.ENROLL_SAVED, detail="saved"))
        return {"ok": True, "message_ckb": strings.ENROLL_SAVED, "consistency": round(consistency, 3),
                "level_db": user_level, "clips": len(clips)}

    async def enroll_cancel(self) -> None:
        self._enroll_clips = {}
        self._enrolling = False
        mic = self._enroll_mic
        if mic is not None:
            await mic.stop()

    async def voiceprint_delete(self) -> dict[str, Any]:
        existed = await asyncio.to_thread(self.speaker_check.store.delete)
        self.speaker_check.forget_cache()
        self.app.config.set("voice.gate_user_level_db", None)
        self.app.db.log_activity("voice", "voiceprint_deleted", ok=True, source="voice", summary=f"existed={existed}")
        self.app.bus.publish(VoiceNotice(kind="enroll", text_ckb=strings.ENROLL_DELETED, detail="deleted"))
        return {"ok": True, "existed": existed, "message_ckb": strings.ENROLL_DELETED}

    def voiceprint_status(self) -> dict[str, Any]:
        return self.speaker_check.status()


__all__ = ["EnrollmentSupport", "MIN_CONSISTENCY", "MIN_SPEECH_MS"]
