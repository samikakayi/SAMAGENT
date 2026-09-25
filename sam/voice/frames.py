"""The microphone pipeline of ``VoiceEngine`` (mixin; kept out of engine.py).

Per 30 ms frame: level meter -> echo guard -> webrtcvad -> near-field gate
(gate.py) -> endpointer (vad.py). Per utterance:

- the FIRST utterance after a click / the hotkey (``_first_pending``,
  listening.py) is the owner by definition: only the near-field gate applies,
  no voiceprint check (real use 2026-09-25: the user's own voice scored 0.055
  -0.346 and was rejected four times). Its embedding is computed next to STT
  and, once its transcript is admitted, adapts the voiceprint (engine.py);
- cascade: every other utterance, when a voiceprint exists, is checked on its
  first 3 s (voiceprint.py, ~60 ms in a thread) BEFORE STT with the low,
  owner-calibrated follow-up threshold; a rejected utterance costs no STT and
  no model call; the island shows «دەنگەکەت نەناسرایەوە — کلیک بکە» once per
  episode and the utterance's level becomes the gate's "background";
- one utterance split by a pause: speech that starts within
  ``voice.merge_window_s`` (1.2 s) of the end of an accepted utterance -- whose
  reply has not started yet -- at about the same level (not more than 8 dB
  quieter; a TV across the room is not a continuation) is its continuation:
  its reply is held (cascade.py) and both become ONE request. The
  continuation of the owner's turn is the owner too; a follow-up's
  continuation is checked like a follow-up;
- Live: NOTHING is sent until the utterance is accepted -- near-field for
  ``voice.gate_min_voiced_ms`` and, with a voiceprint (not for the owner's
  turn), verified on its first 1.2 s -- then the held frames (pre-roll
  included) are sent, the rest streams, and ``audio_stream_end`` closes it
  (live.py);
- barge-in over SAM's voice: a voice start only ducks the speaker; the reply
  is cut after ``voice.barge_in_ms`` of near-field speech and only when that
  speech is the user's: verified by the voiceprint, or (no voiceprint) about
  as loud as the user's known level. With neither, speech over SAM's voice
  only ducks it (a TV must not cut the answer off -- review 2026-09-24).
  Speech too SHORT to cut it (a quick «بەسە») that sounds like the user is
  transcribed with SAM kept ducked (``short_barge``): a stop phrase stops SAM
  at once, a yes/no answers a pending question, anything else is ignored;
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
from .gate import level_db
from .listening import FOLLOWUP_LEVEL_DROP_DB
from .notices import VoiceNotice
from .vad import VadEvent
from .voiceprint import ADAPT_MIN_SPEECH_MS

log = logging.getLogger("sam.voice")

EARLY_VERIFY_MS = 1200.0      # speech needed for an early (Live / barge-in) voiceprint check
CONTINUE_LEVEL_DROP_DB = 8.0  # a continuation is at most this much quieter than what it continues
START_LEVEL_FRAMES = 8        # frames that started an utterance: its first level estimate


@dataclass
class _Sent:
    """The last utterance handed to the cascade (a continuation may join it)."""

    item: Any                    # cascade.Utterance
    p50: float | None
    owner: bool                  # the owner's turn after a click (or its continuation)
    verified: bool


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
    first: bool = False                     # the owner's turn after a click: never voiceprint-checked
    continues: _Sent | None = None          # began right after an accepted utterance: part of it
    continue_mode: str = ""                 # "hold" | "carry" (cascade.hold_for_continuation)


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
        self._last_sent: _Sent | None = None

    # -- helpers ------------------------------------------------------------------------------------
    def _verify_needed(self) -> bool:
        speaker = getattr(self, "speaker_check", None)
        return bool(speaker is not None and speaker.enabled)

    def _min_voiced_ms(self) -> float:
        return float(self.app.config.get("voice.gate_min_voiced_ms", 300))

    def _merge_window_s(self) -> float:
        try:
            return max(0.0, float(self.app.config.get("voice.merge_window_s", 1.2)))
        except (TypeError, ValueError):
            return 1.2

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

    def _ignored(self, result: Any, levels: dict[str, Any], context: str = "followup") -> None:
        """The voiceprint says this was not the user: no STT, no model call;
        «دەنگەکەت نەناسرایەوە — کلیک بکە» once per episode (a click then
        re-opens the owner's turn, listening.py), every rejection logged."""
        self.gate.note_background(levels.get("p50_db") if levels.get("frames") else None)  # type: ignore[attr-defined]
        if self.note_not_recognized():  # type: ignore[attr-defined]
            self.app.bus.publish(VoiceNotice(kind="not_recognized", text_ckb=strings.VOICE_NOT_RECOGNIZED,
                                             detail=f"score={getattr(result, 'score', None)}"))
        try:
            self.app.db.log_activity("voice", "not_my_voice", ok=True, source="voice",
                                     summary=f"score={getattr(result, 'score', None)} "
                                             f"threshold={getattr(result, 'threshold', None)} "
                                             f"level={levels.get('p50_db')} context={context}"[:200])
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
                # Too short (or not the user) to be a barge-in: SAM keeps talking.
                self.speaker.gain = 1.0
                if event.too_short:
                    return
                if utt.verify_task is None:
                    self._short_barge(event)
                    return
            await self._on_utterance(event, live, utt)

    def _start_level(self) -> float | None:
        """Median level of the frames that started the utterance (the gate's
        own per-utterance levels begin after them)."""
        frames = self._endpointer.frames_so_far()[-START_LEVEL_FRAMES:]  # type: ignore[attr-defined]
        loud = sorted(level_db(pcm_rms(f)) for f in frames)[-5:]    # the voiced frames of the trigger window
        return loud[len(loud) // 2] if loud else None

    def _continuation(self) -> tuple[_Sent, str] | None:
        """Is the utterance that just started the rest of the last one? Right
        after it ended (``voice.merge_window_s``), before SAM answered it, and
        not clearly quieter (the TV across the room is not the user going on)."""
        sent = self._last_sent
        if sent is None:
            return None
        start = self._start_level()
        if sent.p50 is not None and start is not None and start < sent.p50 - CONTINUE_LEVEL_DROP_DB:
            return None
        mode = self.cascade.hold_for_continuation(sent.item, self._merge_window_s())
        if mode is None:
            return None
        if mode == "carry":
            self._barge()      # its silent reply is cancelled now; the words are carried over
        return sent, mode

    def _utterance_started(self, live: Any) -> None:
        self.gate.utterance_started()  # type: ignore[attr-defined]
        utt = _Utt(started=time.monotonic(), in_conversation=self.in_conversation())  # type: ignore[attr-defined]
        self._utt = utt
        if live is not None:
            utt.held = self._endpointer.frames_so_far()  # type: ignore[attr-defined]
            utt.first = bool(self._first_pending)  # type: ignore[attr-defined]
            return
        continued = self._continuation()
        if continued is not None:
            utt.continues, utt.continue_mode = continued
            return
        if self.speaker.playing:
            # Maybe a barge-in, maybe a backchannel or the TV: duck now, decide on voiced time.
            utt.barge = True
            self.speaker.gain = float(self.app.config.get("voice.duck_gain", 0.35))
        else:
            utt.first = bool(self._first_pending)  # type: ignore[attr-defined]

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
                # check what was said so far (short speech: a lower threshold).
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
        if self._verify_needed() and not utt.first:
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
                self._ignored(utt.verify_task.result(), self.gate.utterance_levels(), "live")  # type: ignore[attr-defined]
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
        if utt.first:
            self._first_pending = False  # type: ignore[attr-defined]
        live.note_speech_start()
        held, utt.held = utt.held, []
        for chunk in held:
            await live.send_audio(chunk)

    # -- per utterance -------------------------------------------------------------------------------------
    async def _on_utterance(self, event: VadEvent, live: Any, utt: _Utt | None) -> None:
        levels = self.gate.utterance_levels()  # type: ignore[attr-defined]
        if event.too_short:
            if utt is not None and utt.continues is not None:
                self._continuation_failed(utt)
            elif live is None:
                self.cascade.resume_carry()
            return
        if live is not None:
            if utt is not None and utt.accepted:
                live.note_end_of_speech(event.pcm, event.eos_at)
            elif utt is None or utt.accepted is None:
                self.app.spawn(self._finish_live(event, live, utt, levels), "voice-live-utterance")
            return
        cut = bool(utt.cut) if utt is not None else False
        if utt is not None and utt.continues is not None:
            self.app.spawn(self._continue_then_submit(event, utt, levels), "voice-continue")
            return
        if utt is not None and utt.first and not cut:
            self._submit(event, cut, levels, utt=utt, owner=True)   # the owner by definition
            return
        if not self._verify_needed():
            self._submit(event, cut, levels, utt=utt)
            return
        self.app.spawn(self._verify_then_submit(event, utt, cut, levels), "voice-verify")

    def _continuation_failed(self, utt: _Utt) -> None:
        """Not a continuation after all (a cough, another voice): the utterance
        it would have joined is answered alone."""
        sent = utt.continues
        if sent is None:
            return
        if utt.continue_mode == "carry":
            self.cascade.resume_carry()
        else:
            self.cascade.release_hold(sent.item)

    async def _continue_then_submit(self, event: VadEvent, utt: _Utt, levels: dict[str, Any]) -> None:
        sent = utt.continues
        assert sent is not None
        p50 = levels.get("p50_db") if levels.get("frames") else None
        if sent.p50 is not None and p50 is not None and p50 < sent.p50 - CONTINUE_LEVEL_DROP_DB:
            self._continuation_failed(utt)
            self._dropped(levels, "continuation-level")
            return
        verified = sent.verified
        if not sent.owner and self._verify_needed():
            result = await self._spawn_verify(event.pcm, event.speech_ms)
            if not result.ok:
                self._continuation_failed(utt)
                self._ignored(result, levels, "continuation")
                return
            verified = self._verified(result)
        self.activity()  # type: ignore[attr-defined]
        meta: dict[str, Any] = {"levels": levels, "verified": verified, "owner": sent.owner,
                                "in_conversation": utt.in_conversation, "continues": sent.item}
        item = self.cascade.submit_utterance(event.pcm, event.eos_at, meta=meta)
        self._last_sent = _Sent(item=item, p50=sent.p50 if sent.p50 is not None else p50, owner=sent.owner,
                                verified=verified)

    def _short_barge(self, event: VadEvent) -> None:
        """Speech over SAM's voice that did not cut it: too short (a quick
        «بەسە») or untrusted. Only the user's (voiceprint / known level) short
        words are transcribed -- SAM stays ducked meanwhile -- and then only a
        stop phrase or a yes/no to a pending question counts (cascade.py)."""
        levels = self.gate.utterance_levels()  # type: ignore[attr-defined]
        if event.speech_ms >= float(self.app.config.get("voice.barge_in_ms", 400)):
            return                        # long enough to cut, yet it did not: not trusted (maybe the TV)
        if self._verify_needed():
            self.app.spawn(self._verify_short_barge(event, levels), "voice-verify-short")
        elif self._by_level():
            self._submit_short_barge(event, levels, verified=False)

    async def _verify_short_barge(self, event: VadEvent, levels: dict[str, Any]) -> None:
        result = await self._spawn_verify(event.pcm, event.speech_ms)
        if not result.ok:
            self._ignored(result, levels, "short_barge")
            return
        self._submit_short_barge(event, levels, verified=self._verified(result))

    def _submit_short_barge(self, event: VadEvent, levels: dict[str, Any], *, verified: bool) -> None:
        if self.speaker.playing:
            self.speaker.gain = float(self.app.config.get("voice.duck_gain", 0.35))   # restored by the cascade
        self.cascade.submit_utterance(event.pcm, event.eos_at, meta={"levels": levels, "verified": verified,
                                                                     "short_barge": True, "in_conversation": True})

    async def _verify_then_submit(self, event: VadEvent, utt: _Utt | None, cut: bool,
                                  levels: dict[str, Any]) -> None:
        task = utt.verify_task if utt is not None and utt.verify_ms >= EARLY_VERIFY_MS else None
        if task is None:
            task = self._spawn_verify(event.pcm, event.speech_ms)
        result = await task
        if not result.ok:
            self._ignored(result, levels, "barge" if utt is not None and utt.barge else "followup")
            if utt is not None and utt.barge:
                self.speaker.gain = 1.0
            return
        verified = self._verified(result)
        if utt is not None and utt.barge and not cut and self.speaker.playing and verified:
            cut = self._barge()  # the user's own voice over SAM's: cut now
        self._submit(event, cut, levels, verified=verified, utt=utt)

    def _submit(self, event: VadEvent, cut: bool, levels: dict[str, Any], *, verified: bool = False,
                utt: _Utt | None = None, owner: bool = False) -> None:
        meta: dict[str, Any] = {"levels": levels, "verified": verified, "owner": owner,
                                "in_conversation": bool(utt.in_conversation) if utt else False}
        # A cut utterance was already judged to be the user (voiceprint / level) when it cut SAM off.
        if not cut and not self.may_take_utterance(verified=verified or owner, meta=meta):  # type: ignore[attr-defined]
            if utt is not None and utt.barge:
                self.speaker.gain = 1.0
            self._dropped(levels, "policy")
            return
        self.note_accepted()  # type: ignore[attr-defined]
        first, self._first_pending = self._first_pending, False  # type: ignore[attr-defined]
        first = first or owner
        pending = bool(getattr(self.app.confirm, "has_pending", False))
        if not cut and not self.speaker.playing and verified and not pending:
            # The user's own voice while SAM still thinks: the new words replace
            # that request (carried over). Never while a confirmation waits: that
            # would silence the reply waiting for the yes/no (review 2026-09-24).
            self._barge()
        meta["first"] = first
        check = getattr(self, "speaker_check", None)
        if owner and check is not None and check.usable and event.speech_ms >= ADAPT_MIN_SPEECH_MS:
            # Embedded next to STT; it adapts the voiceprint only if the words are admitted (engine.py).
            embed = asyncio.ensure_future(check.embed(event.pcm))
            embed.add_done_callback(lambda t: t.cancelled() or t.exception())   # never "never retrieved"
            meta["owner_embed"] = embed
            meta["speech_ms"] = event.speech_ms
        item = self.cascade.submit_utterance(event.pcm, event.eos_at, cut_reply=cut, meta=meta)
        p50 = levels.get("p50_db") if levels.get("frames") else None
        self._last_sent = _Sent(item=item, p50=p50, owner=owner, verified=verified)

    async def _finish_live(self, event: VadEvent, live: Any, utt: _Utt | None, levels: dict[str, Any]) -> None:
        """A Live utterance that ended before it was accepted (short, or the
        voiceprint check was still running): decide on the whole utterance."""
        verified = False
        first = bool(utt.first) if utt is not None else False
        if self._verify_needed() and not first:
            task = utt.verify_task if utt is not None and utt.verify_task is not None else None
            result = await (task or self._spawn_verify(event.pcm, event.speech_ms))
            if not result.ok:
                self._ignored(result, levels, "live")
                return
            verified = self._verified(result)
        if self.live is not live or not getattr(live, "ready", False):
            return
        if not self.may_take_utterance(verified=verified,  # type: ignore[attr-defined]
                                       meta={"levels": levels, "verified": verified}):
            self._dropped(levels, "live-end")
            return
        self.note_accepted()  # type: ignore[attr-defined]
        if first:
            self._first_pending = False  # type: ignore[attr-defined]
        live.note_speech_start()
        step = 960
        for offset in range(0, len(event.pcm), step):
            await live.send_audio(event.pcm[offset:offset + step])
        live.note_end_of_speech(event.pcm, event.eos_at)


__all__ = ["FramePipeline", "EARLY_VERIFY_MS", "CONTINUE_LEVEL_DROP_DB"]
