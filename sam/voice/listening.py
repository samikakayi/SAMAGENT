"""When SAM listens: push-to-talk turns by default, "always listening" by opt-in.

Real use on 2026-09-24 (sam2.log, turns/usage_counters): with the old
conversation window (open while anything sounded like speech, 45 s after the
last activity) the TV and a family conversation kept it open for minutes and
became commands: 149 KurdishTTS STT calls and a model request per sound. Now
(setting ``voice.always_listening`` False, the default):

- a click on the island or the hotkey opens listening for ONE utterance; with
  no accepted speech within ``voice.start_timeout_s`` (8 s) listening closes
  quietly (island: «هیچ قسەیەکم نەبیست، گوێگرتن داخرا»);
- while SAM thinks or talks about that request, other speech is NOT taken
  (it would cancel the request or become a request itself) -- except the
  user's own voice when a voiceprint exists (voiceprint.py) and a yes/no to
  a pending confirmation;
- after SAM's answer a follow-up window of ``voice.followup_s`` (6 s) takes
  ONE more utterance, and at most ``voice.followup_turns`` (2) follow-ups
  happen per click -- but only when SAM can tell the user from the room: a
  voiceprint, or the user's speech level from the enrollment / several
  consistent turns (then the follow-up must be at most 6 dB quieter than
  the user). Without either, every request needs a click.

  Why so strict (adversarial review 2026-09-24, closed-loop simulation with
  the real engine, webrtcvad and gate on a virtual clock): with no
  voiceprint and a nearly continuous TV at -40 dBFS, the old follow-up
  window fed itself -- TV utterance -> STT -> model -> answer -> new window
  -> next TV utterance -- 61 STT and 61 model calls in 300 s from 3 clicks,
  the mic open 270 s. With the user's level known it was 3 STT calls.
- the conversation (one ``conversations`` row, the brain's memory) stays open
  across these turns and ends only after ``voice.conversation_timeout_s`` of
  quiet with the mic closed (then ``VoiceState("sleeping")``: fact extraction
  runs once per conversation, not once per turn).

"Always listening" keeps the mic open but still requires the near-field gate
AND the spoken name «سام» / SAM at the start of the utterance (checked on the
transcript, BEFORE any model call). Exceptions: a clear yes/no to a pending
confirmation, and the single utterance right after SAM answered -- only with
the same trust test as the push-to-talk follow-ups, and at most
``voice.followup_turns`` times per «سام». It uses the cascade: Live would hand
the audio to the model before the name could be checked.

Who is the owner (real use 2026-09-25: the user's own voice was rejected four
times by the voiceprint, voiceprint.py): the FIRST utterance after an explicit
activation -- a click on the island or the hotkey (``start_listening(
explicit=True)``; a window opened for a confirmation question or by unmuting
is not one) -- is the owner by definition and never voiceprint-checked; a
continuation that starts within ``voice.merge_window_s`` of it is part of it
(frames.py). The voiceprint gates only follow-ups, barge-ins and
always-listening. When it rejects one, the island shows «دەنگەکەت نەناسرایەوە
— کلیک بکە» once per episode (``note_not_recognized``), and a click while that
hint is fresh re-opens an explicit window instead of closing listening
(``rearm_on_click``).

``ListeningPolicy`` is a VoiceEngine mixin; the engine owns ``listening``,
``_busy()``, ``_stop_listening()``, ``_publish()``, ``gate``, ``speaker``,
``speaker_check``.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from ..events import VoiceState
from ..textnorm import normalize_ckb
from . import strings
from .notices import VoiceNotice

log = logging.getLogger("sam.voice")

# The name at the start of an utterance, optionally after a greeting word.
# KurdishTTS STT writes the English name in Arabic script; Gemini may keep "SAM".
_GREETINGS = {"هێی", "هەی", "ئەی", "های", "هێ", "hey", "hi", "ok", "okay", "یا"}
_NAMES = {"سام", "سامی", "سەم", "صام", "sam", "sam.", "sam,",
          # KurdishTTS STT writes «هێی سام» as one word (live clip 2026-09-24: «هەیسام نرخی زێڕ چەندە؟»)
          "هەیسام", "هێیسام", "هێسام", "هایسام", "ئەیسام", "heysam"}
# «دەنگم بناسە» / «دەنگی من بناسەوە» ... but NOT «دەنگم تۆمار بکە» (that can
# mean "record my voice" for a voice note -- review 2026-09-24).
_ENROLL = re.compile(r"(دەنگ(?:ی)?\s*(?:من|م)|دەنگەکەم|دەنگم)\s*(?:بناسە|بناسەوە|فێربە)"
                     r"|learn my voice|recognize my voice|enroll my voice", re.IGNORECASE)
# A follow-up (no click, no «سام») must be about as loud as the user: at most
# this much under the user's level (the gate already rejects user - 8 dB).
FOLLOWUP_LEVEL_DROP_DB = 6.0
# A click within this long after «دەنگەکەت نەناسرایەوە — کلیک بکە» means "it is me".
REARM_S = 20.0
# Still rejected this long after the hint was shown: show it again (never a flood, never silence).
HINT_REPEAT_S = 15.0


def starts_with_name(text: str) -> bool:
    words = normalize_ckb(text or "", strip_punct=True).split()
    for index, word in enumerate(words[:3]):
        if word in _NAMES or word.rstrip("،,.!؟?") in _NAMES:
            return all(w in _GREETINGS for w in words[:index])
        if word not in _GREETINGS:
            return False
    return False


def asks_enrollment(text: str) -> bool:
    return bool(_ENROLL.search(normalize_ckb(text or "")))


class ListeningPolicy:
    app: Any
    listening: bool
    muted: bool
    speaker: Any
    gate: Any

    def _init_listening(self) -> None:
        self._window_until = 0.0
        self._window_kind = "start"         # start | followup | always
        self._accepted_in_window = 0
        self._followup_open = False          # the follow-up window may still take ONE utterance
        self._followups_left = 0             # follow-up turns left since the last click / «سام»
        self._followup_grace_until = 0.0
        self._conversation_open = False
        self._last_turn_at = 0.0
        self._first_pending = False          # the next accepted utterance follows an explicit click
        self._explicit_window = False        # this window was opened by a click / the hotkey
        self._reopen_owner = False           # a re-opened window («دووبارەی بکەرەوە») is the owner's turn again
        self._prev_state = "idle"
        self._confirm_takes: tuple[str, int] = ("", 0)
        self._untrusted_noted = False
        self._reopens_left = 0
        self._reopen_pending = False
        self._not_recognized_at = 0.0        # monotonic time of the last «دەنگەکەت نەناسرایەوە» episode
        self._not_recognized_noted = False   # the hint was shown in the current episode
        self._hint_shown_at = 0.0

    # -- settings -------------------------------------------------------------------------------------
    def always_listening(self) -> bool:
        return bool(self.app.config.get("voice.always_listening", False))

    def _seconds(self, key: str, default: float) -> float:
        try:
            return max(1.0, float(self.app.config.get(key, default)))
        except (TypeError, ValueError):
            return default

    def _followup_budget(self) -> int:
        try:
            return max(0, min(10, int(self.app.config.get("voice.followup_turns", 2))))
        except (TypeError, ValueError):
            return 2

    # -- who is speaking? -------------------------------------------------------------------------------
    def user_level(self) -> float | None:
        level = self.app.config.get("voice.gate_user_level_db", None)
        return float(level) if isinstance(level, (int, float)) else None

    def voice_trusted(self) -> bool:
        """SAM can tell the user from the room: a working voiceprint, or the
        user's level from the enrollment / several consistent turns (gate.py)."""
        check = getattr(self, "speaker_check", None)
        if check is not None and bool(getattr(check, "usable", False)):
            return True
        return self.user_level() is not None

    def _sounds_like_user(self, meta: dict[str, Any]) -> bool:
        """A follow-up without a click (or without «سام») is taken only when it
        is verified by the voiceprint or about as loud as the user."""
        if meta.get("verified"):
            return True
        check = getattr(self, "speaker_check", None)
        if check is not None and bool(getattr(check, "usable", False)):
            return False  # the voiceprint ran and did not say "match"
        level = self.user_level()
        levels = meta.get("levels") or {}
        p50 = levels.get("p50_db") if levels.get("frames") else None
        return level is not None and isinstance(p50, (int, float)) and p50 >= level - FOLLOWUP_LEVEL_DROP_DB

    # -- window bookkeeping ------------------------------------------------------------------------------------
    def _open_window(self, *, explicit: bool = True) -> None:
        """Listening just opened: by the user (click / hotkey: ``explicit``,
        the next utterance is the owner by definition) or for a confirmation
        question / after unmuting (not explicit: the voiceprint still gates)."""
        now = time.monotonic()
        self._window_kind = "always" if self.always_listening() else "start"
        self._window_until = now + self._seconds("voice.start_timeout_s", 8.0)
        self._accepted_in_window = 0
        self._followup_open = False
        self._followups_left = self._followup_budget()
        self._followup_grace_until = 0.0
        self._first_pending = bool(explicit)
        self._explicit_window = bool(explicit)
        self._reopen_owner = False
        self._reopens_left = 1
        self._reopen_pending = False
        self._prev_state = "listening"
        if explicit:
            self._not_recognized_noted = False
            self._not_recognized_at = 0.0

    # -- «دەنگەکەت نەناسرایەوە — کلیک بکە» ----------------------------------------------------------------
    def note_not_recognized(self) -> bool:
        """The voiceprint rejected a follow-up / barge-in / always-listening
        utterance. True when the island hint should be published now: once
        per episode (until a click or an accepted utterance), so repeated
        tries are neither silent nor a flood of notices."""
        now = time.monotonic()
        self._not_recognized_at = now
        if self._not_recognized_noted and now - self._hint_shown_at < HINT_REPEAT_S:
            return False
        self._not_recognized_noted = True
        self._hint_shown_at = now
        return True

    def rearm_on_click(self) -> bool:
        """A click while listening normally closes it; right after «کلیک بکە»
        it means "this is me": a new explicit window opens instead."""
        return bool(self._not_recognized_at) and time.monotonic() - self._not_recognized_at <= REARM_S

    def may_take_utterance(self, *, verified: bool = False, meta: dict[str, Any] | None = None) -> bool:
        """May this finished utterance go to STT / Live at all? (checked
        BEFORE any provider is paid; ``verified`` = the voiceprint matched)."""
        if bool(getattr(self.app.confirm, "has_pending", False)):
            return self._confirm_take()
        if self._window_kind == "always":
            return True               # the name «سام» is checked on the transcript (engine.admit_transcript)
        if verified:
            return True               # the user's own voice: a correction or an addition
        if self._window_kind == "start":
            return self._accepted_in_window == 0
        if self._window_kind == "followup" and self._followup_open:
            return meta is None or self._sounds_like_user(meta)
        return False

    def _confirm_take(self) -> bool:
        """At most 3 utterances go to STT while one confirmation waits (the
        answer, a repeat, one more): a TV cannot spend STT for 20 s."""
        latest = getattr(self.app.confirm, "pending", lambda: [])()
        confirm_id = str(latest[-1].get("confirm_id", "")) if latest else ""
        seen_id, count = self._confirm_takes
        count = count + 1 if seen_id == confirm_id else 1
        self._confirm_takes = (confirm_id, count)
        return count <= 3

    def note_accepted(self) -> None:
        """An utterance passed every check and goes to STT / Live."""
        now = time.monotonic()
        self._accepted_in_window += 1
        self._conversation_open = True
        self._last_turn_at = now
        self._not_recognized_noted = False    # a new rejection is a new episode (hint again)
        self._not_recognized_at = 0.0
        if self._window_kind == "followup":
            self._followup_open = False   # one utterance per follow-up window; never re-extended
        elif self._window_kind == "start":
            # Until STT / the reply make the engine busy (``_busy`` keeps it open then).
            self._window_until = max(self._window_until, now + 2.0)

    def activity(self) -> None:
        """SAM or the user is doing something in this conversation: the
        conversation timeout counts from when it ends."""
        self._last_turn_at = time.monotonic()

    def _followup_began(self) -> None:
        """SAM finished answering: open the follow-up window -- or close
        listening when follow-ups are not allowed (no trust, budget used)."""
        now = time.monotonic()
        self._last_turn_at = now
        followup = self._seconds("voice.followup_s", 6.0)
        allowed = self._followups_left > 0 and self.voice_trusted()
        if self.always_listening():
            self._window_kind = "always"
            self._followup_open = allowed
            self._followup_grace_until = now + followup if allowed else 0.0
            return
        self._window_kind = "followup"
        if allowed:
            self._followups_left -= 1
            self._followup_open = True
            self._window_until = now + followup
            self._followup_grace_until = now + followup
        else:
            # The watcher closes listening on its next tick (0.25 s).
            self._followup_open = False
            self._window_until = now
            self._followup_grace_until = 0.0

    def note_state(self, state: str) -> None:
        # A barge-in also returns to "listening": that is not the end of an answer.
        answered = state == "listening" and self._prev_state in ("speaking", "thinking", "working")
        if state == "listening" and self._reopen_pending:
            self._reopen_pending = False
            if answered:
                self._reopen_start()   # SAM said «دووبارەی بکەرەوە»: the click's window again, from now
        elif answered and not getattr(self, "_barging", False):
            self._followup_began()
        self._prev_state = state

    def utterance_void(self, *, reopen: bool = False) -> None:
        """The last taken utterance produced no request (empty transcript, STT
        failure, not admitted, SAM's own echo): it is not an answer. With
        ``reopen`` the user gets the click's window again -- once per click,
        so a TV cannot keep it open this way."""
        self._prev_state = "listening"
        self._reopen_pending = False
        if reopen and self._window_kind != "always" and self._reopens_left > 0:
            self._reopens_left -= 1
            self._reopen_pending = True     # SAM may first say «دووبارەی بکەرەوە»
            # The owner's turn after a click that produced no words: the repeat is still his turn.
            self._reopen_owner = self._explicit_window and self._window_kind == "start"
            self._reopen_start()

    def _reopen_start(self) -> None:
        self._window_kind = "start"
        self._accepted_in_window = 0
        self._followup_open = False
        self._first_pending = bool(getattr(self, "_reopen_owner", False))
        self._window_until = time.monotonic() + self._seconds("voice.start_timeout_s", 8.0)

    def in_conversation(self) -> bool:
        """An utterance starting now may continue the exchange without «سام»:
        SAM is answering or has just answered (always-listening mode)."""
        cascade = getattr(self, "cascade", None)
        active = bool(getattr(cascade, "reply_active", False)) or bool(self.speaker.playing)
        return active or time.monotonic() <= self._followup_grace_until

    def requires_name(self) -> bool:
        """Always-listening mode: an utterance must start with «سام»
        (exceptions in ``name_exempt``)."""
        return self.always_listening()

    def name_exempt(self, meta: dict[str, Any]) -> bool:
        """Always-listening: the ONE utterance right after SAM's answer may skip
        «سام» -- when follow-ups are allowed and it sounds like the user."""
        if not meta.get("in_conversation") or self._followups_left <= 0:
            return False
        if not (self._followup_open or meta.get("verified")):   # the user's own voice may interrupt
            return False
        if not self._sounds_like_user(meta):
            return False
        self._followup_open = False
        self._followups_left -= 1
        return True

    def named_request(self) -> None:
        """An always-listening utterance with «سام»: a fresh follow-up budget."""
        self._followups_left = self._followup_budget()

    # -- the watcher's decision ------------------------------------------------------------------------------
    async def _window_tick(self) -> None:
        now = time.monotonic()
        if not self.listening:
            await self._maybe_end_conversation(now)
            return
        if self.muted or self.always_listening():
            return
        if self._busy():  # type: ignore[attr-defined]
            self._window_until = max(self._window_until, now + 1.0)
            return
        if now >= self._window_until:
            nothing = self._accepted_in_window == 0 and self._window_kind == "start"
            await self._close_window("no_speech" if nothing else "turn_end")

    async def _close_window(self, reason: str) -> None:
        await self._stop_listening(reason=reason, publish=False)  # type: ignore[attr-defined]
        self._publish("idle", detail=reason, force=True)  # type: ignore[attr-defined]
        if reason == "no_speech":
            text = strings.LISTEN_NO_SPEECH
        elif not self.voice_trusted() and not self._untrusted_noted:
            # Once per session: why there was no follow-up, and how to get one.
            self._untrusted_noted = True
            text = strings.LISTEN_CLOSED_ENROLL
        else:
            text = strings.LISTEN_CLOSED
        self.app.bus.publish(VoiceNotice(kind="closed", text_ckb=text, detail=reason))

    async def _maybe_end_conversation(self, now: float) -> None:
        if not self._conversation_open:
            return
        if self._busy():  # type: ignore[attr-defined]
            self._last_turn_at = now
            return
        if now - self._last_turn_at >= self._seconds("voice.conversation_timeout_s", 45.0):
            self._conversation_open = False
            self.app.bus.publish(VoiceState(state="sleeping", engine="", detail="conversation_end"))
            self._published = ("sleeping", "")  # type: ignore[attr-defined]
            self.state = "sleeping"  # type: ignore[attr-defined]

    def conversation_ended(self) -> None:
        """The user closed listening themselves ("sleeping" was published)."""
        self._conversation_open = False

    def listening_status(self) -> dict[str, Any]:
        left = max(0.0, self._window_until - time.monotonic()) if self.listening else 0.0
        return {"mode": "always" if self.always_listening() else "turn", "window": self._window_kind,
                "window_left_s": round(left, 1), "accepted_in_window": self._accepted_in_window,
                "followup_open": self._followup_open, "followups_left": self._followups_left,
                "trusted": self.voice_trusted(), "conversation_open": self._conversation_open,
                "explicit": self._explicit_window, "owner_turn_pending": self._first_pending,
                "not_recognized": bool(self._not_recognized_at)}


__all__ = ["ListeningPolicy", "starts_with_name", "asks_enrollment", "FOLLOWUP_LEVEL_DROP_DB", "REARM_S"]
