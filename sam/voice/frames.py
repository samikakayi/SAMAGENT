"""The microphone pipeline of ``VoiceEngine`` (mixin; kept out of engine.py).

Per 30 ms frame: level meter -> echo guard -> webrtcvad -> near-field gate
(gate.py) -> endpointer (vad.py). Per utterance:

- cascade: at the end of the utterance "only my voice" is checked on its first
  3 s (voiceprint.py, ~60 ms in a thread) BEFORE STT; a rejected utterance
  costs no STT and no model call (island hint only, and its level becomes
  the gate's "background" so that talker stays below the threshold);
- Live: NOTHING is sent until the utterance is accepted -- near-field for
  ``voice.gate_min_voiced_ms`` and, with a voiceprint, verified on its first
  1.2 s -- then the held frames (pre-roll included) are sent, the rest
  streams, and ``audio_stream_end`` closes it (live.py). Tonight the Live
  session would have heard the TV too: the old code streamed every frame;
- barge-in over SAM's voice: a voice start only ducks the speaker; the reply
  is cut after ``voice.barge_in_ms`` of near-field speech and only when that
  speech is the user's: verified by the voiceprint, or (no voiceprint) about
  as loud as the user's known level. With neither, speech over SAM's voice
  only ducks it (a TV must not cut the answer off -- review 2026-09-24);
- every finished utterance asks the listening policy first
  (``may_take_utterance``, listening.py): one utterance per click, one per
  follow-up window, the user's own voice while SAM thinks, a yes/no while a
  confirmation waits. Anything else costs no STT and no model call.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..events import LevelMeter
from . import strings
from .audio import level_from_rms, pcm_rms
from .listening import FOLLOWUP_LEVEL_DROP_DB
from .notices import VoiceNotice
from .vad import VadEvent

log = logging.getLogger("sam.voice")

EARLY_VERIFY_MS = 1200.0      # speech needed for an early (Live / barge-in) voiceprint check


@dataclass
class _Utt:
    """The utterance in progress."""

    started: float
    barge: bool = False                     # began while SAM was speaking
    held: list[bytes] = field(default_factory=list)   # Live: frames waiting for acceptance
    accepted: bool | None = None            # Live: decided mid-utterance
    verify_task: "asyncio.Task[Any] | None" = None
    verify_ms: float = 0.0                  # speech covered by verify_task
    cut: bool = False                       # it cut SAM's reply
    in_conversation: bool = False           # began while SAM answered or right after (no «سام» needed)


class FramePipeline:
    app: Any
    speaker: Any
    cascade: Any
    live: Any
    engine_name: str

    def _init_frames(self) -> None:
        self._utt: _Utt | None = None
        self._barging = False
        self._vp_unavailable_noted = False
        self._verifying = 0
        self._last_level_at = 0.0
        self._echo_until = 0.0

    # -- helpers ------------------------------------------------------------------------------------
    def _verify_needed(self) -> bool:
        speaker = getattr(self, "speaker_check", None)
        return bool(speaker is not None and speaker.enabled)

    def _min_voiced_ms(self) -> float:
        return float(self.app.config.get("voice.gate_min_voiced_ms", 300))

    def _spawn_verify(self, pcm: bytes, speech_ms: float) -> "asyncio.Task[Any]":
        self._verifying += 1

        async def run() -> Any:
            try:
                return await self.speaker_check.verify(pcm, speech_ms=speech_ms)  # type: ignore[attr-defined]
            finally:
                self._verifying -= 1
        return asyncio.ensure_future(run())

    def _verified(self, result: Any) -> bool:
        """The voiceprint said "match" (a check that could not run is not a match)."""
        if getattr(result, "reason", "") == "unavailable":
            self._voiceprint_unavailable()
        return bool(getattr(result, "ok", False)) and getattr(result, "reason", "") == "match"

    def _voiceprint_unavailable(self) -> None:
        """Enrolled, but the model could not run: tell the user once per session
        (the check then lets speech through -- "only my voice" is off in effect)."""
        if self._vp_unavailable_noted:
            return
        self._vp_unavailable_noted = True
        reason = str(getattr(getattr(self, "speaker_check", None), "unavailable_reason", "") or "")[:160]
        self.app.bus.publish(VoiceNotice(kind="voiceprint", text_ckb=strings.VOICEPRINT_UNAVAILABLE,
                                         detail="unavailable"))
        try:
            self.app.db.log_activity("voice", "voiceprint_unavailable", ok=False, source="voice",
                                     summary=self.app.redact(reason) or "unavailable")
        except Exception:  # noqa: BLE001
            pass

    def _by_level(self) -> bool:
        """No voiceprint: this speech is about as loud as the user's known level."""
        level = self.user_level()  # type: ignore[attr-defined]
        levels = self.gate.utterance_levels()  # type: ignore[attr-defined]
        p50 = levels.get("p50_db") if levels.get("frames") else None
        return level is not None and isinstance(p50, (int, float)) and p50 >= level - FOLLOWUP_LEVEL_DROP_DB

    def _dropped(self, levels: dict[str, Any], why: str) -> None:
        """Speech that the listening policy did not take: no STT, no model call."""
        log.debug("utterance not taken (%s): %s", why, levels)

    def _ignored(self, result: Any, levels: dict[str, Any]) -> None:
        """Speech that was not the user's: silent, one small island hint."""
        self.gate.note_background(levels.get("p50_db") if levels.get("frames") else None)  # type: ignore[attr-defined]
        self.app.bus.publish(VoiceNotice(kind="ignored", text_ckb=strings.IGNORED_NOT_YOU,
                                         detail=f"score={getattr(result, 'score', None)}"))
        try:
            self.app.db.log_activity("voice", "not_my_voice", ok=True, source="voice",
                                     summary=f"score={getattr(result, 'score', None)} "
                                             f"threshold={getattr(result, 'threshold', None)} "
                                             f"level={levels.get('p50_db')}"[:200])
        except Exception:  # noqa: BLE001
            pass

    # -- per frame ----------------------------------------------------------------------------------------
    async def _on_frame(self, frame: bytes) -> None:
        now = time.perf_counter()
        mono = time.monotonic()
        rms = pcm_rms(frame)
        if mono - self._last_level_at >= 0.05:  # <= 20 Hz for the orb
            self._last_level_at = mono
            self.app.bus.publish(LevelMeter(source="mic", level=level_from_rms(rms)))
        if self._echo_guard:  # type: ignore[attr-defined]
            if self.speaker.playing:
                self._echo_until = mono + 0.3
            if mono < self._echo_until and rms < float(self.app.config.get("voice.barge_in_rms", 0.05)):
                frame, rms = bytes(len(frame)), 0.0  # digital silence of the same length
        endpointer = self._endpointer  # type: ignore[attr-defined]
        voiced = self._classifier.is_speech(frame, rms)  # type: ignore[attr-defined]
        near = self.gate.classify(rms, voiced)  # type: ignore[attr-defined]
        event = endpointer.process(frame, near, now)
        live = self.live if self.engine_name == "live" else None
        if event is not None and event.kind == "start":
            self._utterance_started(live)
        utt = self._utt
        if utt is not None and endpointer.in_speech:
            await self._during_speech(utt, frame, event, live, endpointer.current_speech_ms)
        if event is not None and event.kind == "end":
            self._utt = None
            if utt is not None and utt.barge and not utt.cut:
                # Too short (or not the user) to be a barge-in: SAM keeps talking, no STT spent.
                self.speaker.gain = 1.0
                if utt.verify_task is None or event.too_short:
                    return
            await self._on_utterance(event, live, utt)

    def _utterance_started(self, live: Any) -> None:
        self.gate.utterance_started()  # type: ignore[attr-defined]
        utt = _Utt(started=time.monotonic(), in_conversation=self.in_conversation())  # type: ignore[attr-defined]
        self._utt = utt
        if live is not None:
            utt.held = self._endpointer.frames_so_far()  # type: ignore[attr-defined]
        elif self.speaker.playing:
            # Maybe a barge-in, maybe a backchannel or the TV: duck now, decide on voiced time.
            utt.barge = True
            self.speaker.gain = float(self.app.config.get("voice.duck_gain", 0.35))

    async def _during_speech(self, utt: _Utt, frame: bytes, event: VadEvent | None, live: Any,
                             speech_ms: float) -> None:
        if event is None and live is not None:
            if utt.accepted:
                await live.send_audio(frame)
            elif utt.accepted is None:
                utt.held.append(frame)
        if utt.barge and not utt.cut and speech_ms >= float(self.app.config.get("voice.barge_in_ms", 400)):
            if not self._verify_needed():
                if self._by_level():
                    self._cut_reply(utt)      # else: only ducked (maybe the TV)
            elif utt.verify_task is None:
                # Only the user's own voice may cut SAM off (the TV must not):
                # check what was said so far (short speech: threshold - 0.10).
                pcm = b"".join(self._endpointer.frames_so_far())  # type: ignore[attr-defined]
                utt.verify_task, utt.verify_ms = self._spawn_verify(pcm, speech_ms), speech_ms
            elif utt.verify_task.done() and self._task_ok(utt.verify_task):
                self._cut_reply(utt)
        if live is not None and utt.accepted is None:
            await self._live_decide(utt, live, speech_ms)

    def _cut_reply(self, utt: _Utt) -> None:
        utt.cut = True
        self.speaker.gain = 1.0
        self._barge()

    def _barge(self) -> bool:
        """Cut / replace SAM's reply. The state goes back to "listening", which
        is not the end of an answer (no follow-up window is opened for it)."""
        self._barging = True
        try:
            return bool(self.cascade.barge_in())
        finally:
            self._barging = False

    @staticmethod
    def _task_ok(task: "asyncio.Task[Any]") -> bool:
        try:
            return bool(task.result().ok)
        except Exception:  # noqa: BLE001 - a failed check never locks the user out
            return True

    def _task_verified(self, task: "asyncio.Task[Any]") -> bool:
        try:
            return self._verified(task.result())
        except Exception:  # noqa: BLE001
            return False

    async def _live_decide(self, utt: _Utt, live: Any, speech_ms: float) -> None:
        if speech_ms < self._min_voiced_ms():
            return
        verified = False
        if self._verify_needed():
            if utt.verify_task is None:
                if speech_ms < EARLY_VERIFY_MS:
                    return
                utt.verify_task, utt.verify_ms = self._spawn_verify(b"".join(utt.held), speech_ms), speech_ms
                return
            if not utt.verify_task.done():
                return
            if not self._task_ok(utt.verify_task):
                utt.accepted = False
                utt.held = []
                self._ignored(utt.verify_task.result(), self.gate.utterance_levels())  # type: ignore[attr-defined]
                return
            verified = self._task_verified(utt.verify_task)
        levels = self.gate.utterance_levels()  # type: ignore[attr-defined]
        if not self.may_take_utterance(verified=verified,  # type: ignore[attr-defined]
                                       meta={"levels": levels, "verified": verified}):
            utt.accepted = False
            utt.held = []
            self._dropped(levels, "live")
            return
        utt.accepted = True
        self.note_accepted()  # type: ignore[attr-defined]
        live.note_speech_start()
        held, utt.held = utt.held, []
        for chunk in held:
            await live.send_audio(chunk)

    # -- per utterance -------------------------------------------------------------------------------------
    async def _on_utterance(self, event: VadEvent, live: Any, utt: _Utt | None) -> None:
        levels = self.gate.utterance_levels()  # type: ignore[attr-defined]
        if event.too_short:
            if live is None:
                self.cascade.resume_carry()
            return
        if live is not None:
            if utt is not None and utt.accepted:
                live.note_end_of_speech(event.pcm, event.eos_at)
            elif utt is None or utt.accepted is None:
                self.app.spawn(self._finish_live(event, live, utt, levels), "voice-live-utterance")
            return
        cut = bool(utt.cut) if utt is not None else False
        if not self._verify_needed():
            self._submit(event, cut, levels, utt=utt)
            return
        self.app.spawn(self._verify_then_submit(event, utt, cut, levels), "voice-verify")

    async def _verify_then_submit(self, event: VadEvent, utt: _Utt | None, cut: bool,
                                  levels: dict[str, Any]) -> None:
        task = utt.verify_task if utt is not None and utt.verify_ms >= EARLY_VERIFY_MS else None
        if task is None:
            task = self._spawn_verify(event.pcm, event.speech_ms)
        result = await task
        if not result.ok:
            self._ignored(result, levels)
            if utt is not None and utt.barge:
                self.speaker.gain = 1.0
            return
        verified = self._verified(result)
        if utt is not None and utt.barge and not cut and self.speaker.playing and verified:
            cut = self._barge()  # the user's own voice over SAM's: cut now
        self._submit(event, cut, levels, verified=verified, utt=utt)

    def _submit(self, event: VadEvent, cut: bool, levels: dict[str, Any], *, verified: bool = False,
                utt: _Utt | None = None) -> None:
        meta: dict[str, Any] = {"levels": levels, "verified": verified,
                                "in_conversation": bool(utt.in_conversation) if utt else False}
        # A cut utterance was already judged to be the user (voiceprint / level) when it cut SAM off.
        if not cut and not self.may_take_utterance(verified=verified, meta=meta):  # type: ignore[attr-defined]
            if utt is not None and utt.barge:
                self.speaker.gain = 1.0
            self._dropped(levels, "policy")
            return
        self.note_accepted()  # type: ignore[attr-defined]
        first, self._first_pending = self._first_pending, False  # type: ignore[attr-defined]
        pending = bool(getattr(self.app.confirm, "has_pending", False))
        if not cut and not self.speaker.playing and verified and not pending:
            # The user's own voice while SAM still thinks: the new words replace
            # that request (carried over). Never while a confirmation waits: that
            # would silence the reply waiting for the yes/no (review 2026-09-24).
            self._barge()
        self.cascade.submit_utterance(event.pcm, event.eos_at, cut_reply=cut, meta={**meta, "first": first})

    async def _finish_live(self, event: VadEvent, live: Any, utt: _Utt | None, levels: dict[str, Any]) -> None:
        """A Live utterance that ended before it was accepted (short, or the
        voiceprint check was still running): decide on the whole utterance."""
        verified = False
        if self._verify_needed():
            task = utt.verify_task if utt is not None and utt.verify_task is not None else None
            result = await (task or self._spawn_verify(event.pcm, event.speech_ms))
            if not result.ok:
                self._ignored(result, levels)
                return
            verified = self._verified(result)
        if self.live is not live or not getattr(live, "ready", False):
            return
        if not self.may_take_utterance(verified=verified,  # type: ignore[attr-defined]
                                       meta={"levels": levels, "verified": verified}):
            self._dropped(levels, "live-end")
            return
        self.note_accepted()  # type: ignore[attr-defined]
        live.note_speech_start()
        step = 960
        for offset in range(0, len(event.pcm), step):
            await live.send_audio(event.pcm[offset:offset + step])
        live.note_end_of_speech(event.pcm, event.eos_at)


__all__ = ["FramePipeline", "EARLY_VERIFY_MS"]
