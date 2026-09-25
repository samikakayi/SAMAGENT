"""One continuous conversation: typed turns, the cascade voice path, and the
single writer of the ``turns`` table.

v1 set ``conversation_id=None`` on every voice turn, so 28 user turns became 24
conversations and the model forgot the previous sentence; a keyword gate sent
Sorani commands no tools, so TradingView opened 0 of 7 times
(reports/audit-latency.json). Here:

- ONE conversation lives until the voice window sleeps or the user is idle for
  ``conversation.idle_timeout_s``; older turns are folded into a summary and
  the newest ``conversation.history_turns`` are sent verbatim.
- Summaries and fact extraction are optional model calls: they run in a pause
  (never during a live exchange), one request each, and only when no rung
  rests and the day's budget allows (``budget.py``, measured on the user's
  first evening test); otherwise the summary is folded without a model and the
  extraction waits.
- The turn loop itself (tool tiers, ladders, deadlines, rewording, Sorani
  outcomes) is ``responder.Responder``; model choice is ``ladders.py``.
- Every ``Transcript`` event (Live, cascade, typed, worker) and every finished
  tool call is persisted here and nowhere else.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from typing import Any

from ..events import Caption, SpeakRequest, ToolFinished, ToolStarted, Transcript, VoiceState
from ..textnorm import normalize_ckb
from .budget import DEFAULTS as BUDGET_DEFAULTS
from .budget import BackgroundBudget, quota_reason
from .confirm import ASK_AGAIN_CKB
from .library_context import DEFAULTS as LIBRARY_DEFAULTS
from .llm import LLMError
from .responder import (ACKS_DO, ACKS_LOOK, CORE_TOOLS, SORANI_CUT_OFF, SORANI_DONE, SORANI_NO_MODEL,
                        SORANI_NOT_DONE, SORANI_NOT_UNDERSTOOD, Responder, clean_tool_args, more_tools, stop_speaking)
from .ladders import FAST_TOOL_PICKERS

log = logging.getLogger("sam.conversation")

# While the text/cascade path builds its system prompt, the history travels as
# chat messages, so the persona must not repeat the recent turns inside it.
_HISTORY_IN_MESSAGES: contextvars.ContextVar[bool] = contextvars.ContextVar("sam_history_in_messages",
                                                                            default=False)

DEDUP_WINDOW_S = 8.0
ASSISTANT_DEDUP_WINDOW_S = 90.0
SUMMARY_BATCH = 8
PREVIOUS_CONVERSATION_S = 2 * 3600
BACKGROUND_TICK_S = 15.0
EXTRACT_DEFER_MAX_S = 24 * 3600.0   # a deferred extraction older than a day is dropped
BACKGROUND_TIMEOUT_S = 20.0

SUMMARY_PROMPT = (
    "Summarise this part of a conversation between a Kurdish user and his assistant SAM for SAM's own "
    "memory. English, at most 80 words, keep names, symbols, numbers and Sorani key terms as said. Merge "
    "it with the previous summary if one is given. Output only the summary text.")

DEFAULTS: dict[str, Any] = {
    # "auto" = per-turn ladders from key presence + live health (ladders.py);
    # a list of refs / core ladder names overrides.
    "conversation.ladder.voice": "auto",
    "conversation.ladder.reply": "auto",
    "conversation.ladder.text": "auto",
    "conversation.summary_ladder": "extract",
    "conversation.idle_timeout_s": 900,
    "conversation.stream": False,   # see Responder._ask for the measurement
    "conversation.voice_ack": True,
    "conversation.llm_timeout_s": 15,   # smoke 2026-09-24: answers 1-10.5 s; busy rungs 17-40 s to fail
    "conversation.picker_deadline_s": 14,
    "conversation.wording_deadline_s": 8,
    "conversation.reword_deadline_s": 5,
    "conversation.reasoning": "minimal",   # Gemini rungs; Groq maps it to "low"; a 400 retries without
    "conversation.tool_tier": "core",      # core (+ more_tools) | all
    "conversation.core_tools": list(CORE_TOOLS),
}


def drop_abandoned(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Leave requests that got SORANI_NO_MODEL out of the prompt (they stay in
    the DB). Integration smoke 2026-09-24: after three such replies the next
    question ("نرخی زێڕ چەندە؟") ran the stale "هێڵەکانت بسڕەوە" from history."""
    out: list[dict[str, Any]] = []
    for turn in turns:
        if turn["role"] == "assistant" and str(turn["text"]).strip().endswith(SORANI_NO_MODEL):
            while out and out[-1]["role"] == "user":
                out.pop()
            continue
        out.append(turn)
    return out


def _same_text(a: str, b: str) -> bool:
    return normalize_ckb(a, strip_punct=True) == normalize_ckb(b, strip_punct=True)


class Conversation(Responder):
    """``app.conversation`` (docs/CONTRACTS.md 3.2)."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.conversation_id: int | None = None
        self._last_activity = 0.0
        self._tool_args: dict[str, dict[str, Any]] = {}
        self._unsubscribe: list[Any] = []
        self._summarizing = False
        self._idle_task: asyncio.Task[Any] | None = None
        self._last_ack = ""
        self._ack_index = -1
        self.active_turns = 0                          # respond_stream calls running now
        self.budget = BackgroundBudget(app)
        self._summary_due: set[int] = set()            # conversations with old turns to fold
        self._extract_due: dict[int, float] = {}       # ended conversations whose extraction waits

    @property
    def last_activity(self) -> float:
        """Wall time of the last transcript or tool result (budget.py)."""
        return self._last_activity

    # -- wiring ----------------------------------------------------------------------------
    def attach(self) -> None:
        bus = self.app.bus
        self._unsubscribe = [
            bus.subscribe(Transcript, self._on_transcript),
            bus.subscribe(ToolStarted, self._on_tool_started),
            bus.subscribe(ToolFinished, self._on_tool_finished),
            bus.subscribe(VoiceState, self._on_voice_state),
        ]

    def detach(self) -> None:
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        self._unsubscribe = []

    @property
    def memory(self) -> Any:
        return self.app.memory

    # -- lifecycle ---------------------------------------------------------------------------
    def new_conversation(self, source: str = "voice") -> int:
        """Start a fresh conversation (the previous one is left as is)."""
        self.conversation_id = int(self.memory.start_conversation(source))
        self._last_activity = time.time()
        return self.conversation_id

    def ensure_conversation(self, source: str = "voice") -> int:
        """The current conversation; after a restart, the latest one continues
        if it is still open and was active within the idle timeout."""
        if self.conversation_id is not None:
            return self.conversation_id
        latest = self.memory.latest_conversation()
        if latest is not None and latest.get("ended_at") is None:
            last_turn = self.app.db.scalar("SELECT MAX(at) FROM turns WHERE conversation_id=?", (latest["id"],))
            if last_turn and time.time() - float(last_turn) < self._idle_timeout():
                self.conversation_id = int(latest["id"])
                self._last_activity = float(last_turn)
                return self.conversation_id
        return self.new_conversation("text" if source == "text" else "voice")

    def _idle_timeout(self) -> float:
        return float(self.app.config.get("conversation.idle_timeout_s", 900) or 900)

    async def on_sleep(self) -> None:
        """The voice window slept (or the user went idle): close the
        conversation and extract durable facts with one cheap call -- now if
        the budget allows, else later in a pause (``background_tick``)."""
        conversation_id = self.conversation_id
        if conversation_id is None:
            return
        self.conversation_id = None
        self.memory.end_conversation(conversation_id)
        if self.app.config.get("memory.extract_on_sleep", True):
            await self._extract(conversation_id, ended=True)

    def _extract_refs(self) -> list[str]:
        ladder = str(self.app.config.get("memory.extract_ladder", "extract") or "extract")
        try:
            return list(self.app.llm.ladder(ladder))
        except (ValueError, AttributeError):
            return []

    async def _extract(self, conversation_id: int, *, ended: bool = False) -> None:
        """Fact extraction under the background budget: one rung, or wait."""
        refs, reason = self.budget.pick("extract", self._extract_refs(), ended=ended)
        if not refs:
            self._extract_due.setdefault(conversation_id, time.time())
            self.budget.record("extract", "deferred", reason or "")
            return
        self._extract_due.pop(conversation_id, None)
        self.budget.record("extract", "ran")
        try:
            await self.memory.extract_facts(conversation_id, ladder=refs)
        except Exception:  # noqa: BLE001 - memory must never break the voice loop
            log.exception("fact extraction failed")

    async def background_tick(self) -> None:
        """One pass over the waiting background jobs (summaries first: they keep
        the prompt small; then at most one deferred extraction)."""
        for conversation_id in sorted(self._summary_due):
            if not self._unsummarized(conversation_id):
                self._summary_due.discard(conversation_id)
                continue
            refs, reason = self.budget.pick("summary", self._summary_refs())
            if refs:
                await self.summarize(conversation_id, refs=refs)
            elif quota_reason(reason) or len(self._unsummarized(conversation_id)) >= 2 * SUMMARY_BATCH:
                # Quota pressure, or a long exchange without a pause: fold without a model now.
                self.budget.record("summary", "skipped", reason or "")
                await self.summarize(conversation_id, refs=[])
            else:
                continue
            self._summary_due.discard(conversation_id)
        now = time.time()
        for conversation_id, since in sorted(self._extract_due.items(), key=lambda item: item[1]):
            if now - since > EXTRACT_DEFER_MAX_S:
                self._extract_due.pop(conversation_id, None)
                continue
            if self.budget.reason_to_skip("extract", self._extract_refs()) is None:
                await self._extract(conversation_id)
            break

    async def idle_watch(self, interval_s: float = BACKGROUND_TICK_S) -> None:
        """Background task: end a conversation nobody touched for a while and
        run the background jobs that wait for a pause."""
        while True:
            await asyncio.sleep(interval_s)
            try:
                if self.conversation_id is not None and time.time() - self._last_activity > self._idle_timeout():
                    await self.on_sleep()
                await self.background_tick()
            except Exception:  # noqa: BLE001 - the watcher must keep running
                log.exception("conversation background tick failed")

    # -- persistence (single writer of `turns`) -----------------------------------------------
    def _mix_source(self, conversation_id: int, source: str) -> None:
        kind = "text" if source == "text" else ("voice" if source in ("live", "cascade") else "")
        if not kind:
            return
        self.app.db.execute(
            "UPDATE conversations SET source='mixed' WHERE id=? AND source NOT IN (?, 'mixed')",
            (conversation_id, kind))

    def _on_transcript(self, event: Transcript) -> None:
        text = (event.text or "").strip()
        if not text or event.role not in ("user", "assistant", "system"):
            return
        conversation_id = event.conversation_id or self.ensure_conversation(event.source)
        now = time.time()
        # The voice engine and respond_stream may both report the same
        # utterance (the user's at once; the reply when playback ends, and a
        # barge-in reports a cut-off prefix of it). Only the last turn of the
        # same role is compared, so a real repeat later is still stored.
        last = self.memory.recent_turns(conversation_id, limit=1, roles=(event.role,))
        if last:
            previous = last[0]
            window = DEDUP_WINDOW_S if event.role == "user" else ASSISTANT_DEDUP_WINDOW_S
            same = _same_text(previous["text"], text) or (event.role == "assistant" and (
                previous["text"].startswith(text) or text.startswith(previous["text"])))
            if same and now - float(previous["at"]) < window:
                return
        meta = {"turn_id": event.turn_id} if event.turn_id else None
        self.memory.add_turn(conversation_id, event.role, text, source=event.source, meta=meta)
        self._last_activity = now
        self._mix_source(conversation_id, event.source)
        if event.role == "user":
            self.app.db.execute("UPDATE conversations SET title=? WHERE id=? AND title=''",
                                (" ".join(text.split())[:80], conversation_id))
        elif event.role == "assistant":
            self._maybe_summarize(conversation_id)

    def _on_tool_started(self, event: ToolStarted) -> None:
        self._tool_args[event.call_id] = event.args
        if len(self._tool_args) > 200:
            self._tool_args.pop(next(iter(self._tool_args)))

    def _on_tool_finished(self, event: ToolFinished) -> None:
        args = self._tool_args.pop(event.call_id, None)
        # Worker steps stay in the activity log; its final summary arrives as
        # a Transcript. Everything else is part of the conversation's story.
        if event.source == "worker":
            return
        conversation_id = self.ensure_conversation(event.source or "voice")
        text = f"{event.name}: {'ok' if event.ok else 'failed'} - {event.summary}"[:300]
        self.memory.add_turn(conversation_id, "tool", text, source=event.source or "system",
                             meta={"name": event.name, "ok": event.ok, "call_id": event.call_id, "args": args})
        self._last_activity = time.time()

    def _on_voice_state(self, event: VoiceState) -> Any:
        if event.state == "sleeping" and self.conversation_id is not None:
            return self.on_sleep()
        if event.state == "listening":
            self.prewarm_local_brain()
        return None

    def prewarm_local_brain(self, mode: str = "voice") -> Any:
        """The user starts talking while no cloud rung can answer: load the
        local model and read SAM's stable prompt into its cache now, while
        they speak (load 8 s + a cold 4.1k-token prompt 86 s on this PC,
        llm_ollama.py), instead of after the transcript. Returns the task."""
        llm = self.app.llm
        if int(getattr(self, "active_turns", 0) or 0):
            return None           # a turn is using the local model now: a warm-up would only make it wait
        try:
            if llm.cloud_usable(self._ladder(mode)) or not llm.local_ready():
                return None
            system = self.app.persona.system_instruction(mode) if self.app.persona is not None else ""
            messages = [{"role": "system", "content": system}, {"role": "user", "content": "."}] if system else None
            task = llm.prewarm_local(messages, self._tools_for_round(set()))
            if task is not None and not task.done():
                self.app.spawn(task, "local-brain-warm")    # tracked: cancelled on quit
            return task
        except Exception:  # noqa: BLE001 - a warm-up is optional
            log.debug("local brain warm-up not started", exc_info=True)
            return None

    # -- older turns -> summary ------------------------------------------------------------------
    def _summary_upto(self, conversation_id: int) -> int:
        value = self.app.db.scalar("SELECT summary_upto FROM brain_conversation_state WHERE conversation_id=?",
                                   (conversation_id,))
        return int(value or 0)

    def _unsummarized(self, conversation_id: int) -> list[dict[str, Any]]:
        """User/assistant turns older than the verbatim history window that
        are not in the summary yet."""
        window = int(self.app.config.get("conversation.history_turns", 12) or 12)
        turns = self.memory.recent_turns(conversation_id, limit=400, roles=("user", "assistant"),
                                         after_id=self._summary_upto(conversation_id))
        return turns[:-window] if len(turns) > window else []

    def _maybe_summarize(self, conversation_id: int) -> None:
        """Old turns wait to be folded; ``background_tick`` does it in a pause
        (it used to run a model call right after every assistant turn, in the
        middle of the exchange -- four of them at 20:55-20:58 on 2026-09-24)."""
        if len(self._unsummarized(conversation_id)) >= SUMMARY_BATCH:
            self._summary_due.add(conversation_id)

    def _summary_refs(self) -> list[str]:
        try:
            return list(self.app.llm.ladder(str(self.app.config.get("conversation.summary_ladder", "extract"))))
        except (ValueError, AttributeError):
            return []

    async def summarize(self, conversation_id: int, *, refs: list[str] | None = None) -> str:
        """Fold old turns into ``conversations.summary``: one request to
        ``refs`` (default: the summary ladder's first rung), or an extractive
        summary without a model when ``refs`` is empty or the call fails."""
        if self._summarizing:
            return ""
        self._summarizing = True
        try:
            turns = self._unsummarized(conversation_id)
            if not turns:
                return ""
            conv = self.memory.get_conversation(conversation_id) or {}
            previous = str(conv.get("summary") or "")
            lines = [f"{'USER' if t['role'] == 'user' else 'SAM'}: {' '.join(str(t['text']).split())[:300]}"
                     for t in turns]
            summary = ""
            rungs = self._summary_refs()[:1] if refs is None else list(refs)
            if rungs:
                self.budget.record("summary", "ran")
                try:
                    response = await self.app.llm.chat(
                        [{"role": "system", "content": SUMMARY_PROMPT},
                         {"role": "user", "content": (f"Previous summary: {previous}\n\n" if previous else "")
                          + "\n".join(lines)}],
                        ladder=rungs, reasoning="low", timeout_s=BACKGROUND_TIMEOUT_S, retry_transient=False,
                        local=False)
                    summary = " ".join((response.text or "").split())[:800]
                except LLMError as err:
                    self.budget.record("summary", "failed", err.kind)
                    log.info("summary call failed (%s); using extractive summary", err.kind)
            if not summary:
                asks = "; ".join(" ".join(str(t["text"]).split())[:60] for t in turns if t["role"] == "user")
                summary = (previous + " | " if previous else "") + "User asked: " + asks
                summary = summary[-800:]
            self.app.db.execute("UPDATE conversations SET summary=? WHERE id=?", (summary, conversation_id))
            self.app.db.execute(
                "INSERT INTO brain_conversation_state(conversation_id, summary_upto, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET summary_upto=excluded.summary_upto, "
                "updated_at=excluded.updated_at", (conversation_id, int(turns[-1]["id"]), time.time()))
            return summary
        finally:
            self._summarizing = False

    # -- context for the persona -------------------------------------------------------------------
    def context_for_prompt(self, mode: str = "voice", *, max_chars: int = 1200) -> str:
        """Summary of older turns, recent actions and (for a fresh Live
        session) the last turns, clipped to ``max_chars``."""
        parts: list[str] = []
        conversation_id = self.conversation_id
        history_in_messages = _HISTORY_IN_MESSAGES.get()
        if conversation_id is not None:
            conv = self.memory.get_conversation(conversation_id) or {}
            if conv.get("summary"):
                parts.append("Earlier in this conversation: " + str(conv["summary"])[:600])
            actions = self.memory.recent_turns(conversation_id, limit=6, roles=("tool",))
            if actions:
                parts.append("Recent actions: " + "; ".join(
                    f"{time.strftime('%H:%M', time.localtime(float(a['at'])))} {str(a['text'])[:90]}" for a in actions))
            if not history_in_messages and mode != "worker":
                turns = self.memory.recent_turns(conversation_id, limit=6, roles=("user", "assistant"))
                if turns:
                    parts.append("Last turns (do not repeat them):\n" + "\n".join(
                        f"{'User' if t['role'] == 'user' else 'SAM'}: {' '.join(str(t['text']).split())[:160]}"
                        for t in turns))
        if (conversation_id is None or not self.memory.count_turns(conversation_id)) and mode != "worker":
            previous = self._previous_conversation()
            if previous:
                parts.append(previous)
        text = "\n".join(parts)
        return text[:max_chars]

    def _previous_conversation(self) -> str:
        row = self.app.db.query_one(
            "SELECT * FROM conversations WHERE ended_at IS NOT NULL AND id != ? ORDER BY id DESC LIMIT 1",
            (self.conversation_id or -1,))
        if row is None or time.time() - float(row["ended_at"]) > PREVIOUS_CONVERSATION_S:
            return ""
        minutes = int((time.time() - float(row["ended_at"])) // 60)
        if row.get("summary"):
            body = str(row["summary"])[:400]
        else:
            turns = self.memory.recent_turns(int(row["id"]), limit=4, roles=("user", "assistant"))
            body = " / ".join(f"{'User' if t['role'] == 'user' else 'SAM'}: {' '.join(str(t['text']).split())[:100]}"
                              for t in turns)
        return f"Previous conversation ({minutes} min ago): {body}" if body else ""

    # -- building a request ------------------------------------------------------------------------
    def _messages(self, conversation_id: int, text: str, mode: str) -> list[dict[str, Any]]:
        token = _HISTORY_IN_MESSAGES.set(True)
        try:
            system = self.app.persona.system_instruction(mode) if self.app.persona is not None else ""
        finally:
            _HISTORY_IN_MESSAGES.reset(token)
        window = int(self.app.config.get("conversation.history_turns", 12) or 12)
        turns = self.memory.recent_turns(conversation_id, limit=window + 1, roles=("user", "assistant"))
        if turns and turns[-1]["role"] == "user" and _same_text(turns[-1]["text"], text):
            turns = turns[:-1]  # the current utterance, already stored
        turns = drop_abandoned(turns)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        for turn in turns[-window:]:
            content = str(turn["text"])
            if messages and messages[-1]["role"] == turn["role"]:
                messages[-1]["content"] += "\n" + content  # strict APIs want alternating roles
            else:
                messages.append({"role": turn["role"], "content": content})
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += "\n" + text
        else:
            messages.append({"role": "user", "content": text})
        return messages

    # -- typed input ---------------------------------------------------------------------------------------
    def _live_forwarder(self) -> Any:
        """``app.voice.send_text`` when a Live session is open (typed text
        then goes into that session, as speech would), else None."""
        voice = getattr(self.app, "voice", None)
        send = getattr(voice, "send_text", None)
        if voice is None or send is None:
            return None
        live_open = getattr(voice, "live_session_open", None)
        if live_open is None:
            live_open = getattr(voice, "engine_name", "") == "live" and getattr(voice, "state", "") in (
                "listening", "thinking", "speaking", "working")
        return send if live_open else None

    async def handle_text(self, text: str, *, source: str = "text") -> str:
        """Typed input from the panel: same brain as speech. Returns the full
        reply ('' when the text answered a confirmation or went to Live)."""
        text = (text or "").strip()
        if not text:
            return ""
        if self.app.confirm.offer_transcript(text):
            self.app.bus.publish(Transcript(role="user", text=text, source=source))
            return ""
        needs_answer = getattr(self.app.confirm, "needs_clear_answer", None)
        if callable(needs_answer) and needs_answer(text):
            # «باشە» while a question waits: ask again instead of starting a new turn
            # beside the one that waits for this answer (verify review 2026-09-24).
            self.app.bus.publish(Transcript(role="user", text=text, source=source))
            self.app.bus.publish(Caption(text=ASK_AGAIN_CKB, role="assistant", final=True))
            return ASK_AGAIN_CKB
        forward = self._live_forwarder()
        if forward is not None:
            try:
                if await forward(text):
                    return ""
            except Exception:  # noqa: BLE001 - fall back to the text path
                log.exception("sending typed text to Live failed")
        bus = self.app.bus
        bus.publish(Caption(text=text, role="user", final=True))
        voice = getattr(self.app, "voice", None)
        previous_state = str(getattr(voice, "state", "idle") or "idle") if voice is not None else "idle"
        show_state = previous_state in ("idle", "sleeping", "muted")
        if show_state:
            bus.publish(VoiceState(state="thinking", engine="", detail="typed"))
        parts: list[str] = []
        try:
            async for piece in self.respond_stream(text, source=source):
                parts.append(piece)
                bus.publish(Caption(text=" ".join(parts), role="assistant", final=False))
        finally:
            if show_state:
                bus.publish(VoiceState(state=previous_state, engine="", detail="typed"))  # type: ignore[arg-type]
        reply = " ".join(parts).strip()
        if reply:
            bus.publish(Caption(text=reply, role="assistant", final=True))
            if self.app.config.get("conversation.speak_typed_replies", False):
                bus.publish(SpeakRequest(text_ckb=reply, source="system"))
        return reply


BRAIN_MIGRATIONS = [(1, """
CREATE TABLE IF NOT EXISTS brain_conversation_state (
    conversation_id INTEGER PRIMARY KEY,
    summary_upto INTEGER NOT NULL DEFAULT 0,   -- last turn id folded into conversations.summary
    updated_at REAL NOT NULL
);
""")]


def register(app: Any) -> None:
    app.config.register_defaults(DEFAULTS)
    app.config.register_defaults(BUDGET_DEFAULTS)
    app.config.register_defaults(LIBRARY_DEFAULTS)
    app.db.ensure_schema("brain", BRAIN_MIGRATIONS)
    app.conversation = Conversation(app)
    app.conversation.attach()
    for fn in (more_tools, stop_speaking):
        if app.tools.get(fn.tool_spec.name) is None:
            app.tools.add(fn, owner="brain")


async def start(app: Any) -> None:
    conversation = app.conversation
    if conversation is not None:
        conversation._idle_task = app.spawn(conversation.idle_watch(), "conversation-idle")


async def stop(app: Any) -> None:
    conversation = app.conversation
    if conversation is None:
        return
    task = conversation._idle_task
    if task is not None:
        task.cancel()
    conversation.detach()


__all__ = ["Conversation", "register", "start", "stop", "DEFAULTS", "drop_abandoned", "clean_tool_args",
           "ACKS_DO", "ACKS_LOOK", "SORANI_NO_MODEL", "SORANI_CUT_OFF", "SORANI_DONE", "SORANI_NOT_DONE",
           "SORANI_NOT_UNDERSTOOD", "FAST_TOOL_PICKERS", "LLMError"]
