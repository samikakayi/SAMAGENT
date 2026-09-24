"""The turn loop behind ``Conversation.respond_stream`` (typed text and the
cascade voice path), split out of conversation.py.

One turn: round 1 goes to the picker ladder (chooses tools; small talk is
answered here), tool calls run through ``app.tools.dispatch`` (risk gating,
confirmations, timings, taint), and rounds after a tool result go to the
quality-first wording ladder, at most ``conversation.max_tool_rounds`` rounds
and then one forced wording round. Model choice and the measurements behind
it live in ``ladders.py``.

Rules added by the repair review (2026-09-24, measured):
- Groq's words are never spoken first when a better model is reachable:
  small talk Groq answered is worded again by the wording ladder (bounded by
  ``REWORD_DEADLINE_S``; Groq's own text only if nothing else answers), and
  text that comes together with a tool call is not spoken -- the cached
  acknowledgement is (the user heard two answers before).
- Two tool tiers: ~15 core tools with compact descriptions on every round,
  and ``more_tools`` to attach the others to the next round (33 full schemas
  were 73% of every request, 3,863 of 5,312 tokens, and drove Groq's 429s).
- Rounds have deadlines; when no model can word a result, ``outcome.py``
  says the tool's own honest result in Sorani.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator

from ..events import Error, Transcript
from ..textnorm import fix_letters, is_arabic_script, normalize_ckb
from . import fastpath, ladders, taint
from .library_context import add_to_messages, library_block, sources_line
from .llm import LLMError, LLMResponse, ToolCall
from .llm_local import LOCAL_PROVIDER
from .outcome import own_sentence, tool_sentence
from .speech import SentenceChunker
from .tools import ToolContext, ok, tool

log = logging.getLogger("sam.conversation")

SORANI_NO_MODEL = "ببورە، ئێستا ناتوانم پەیوەندی بە مۆدێلەکانەوە بکەم. تکایە کەمێکی تر هەوڵ بدەرەوە."
SORANI_CUT_OFF = "ببورە، وەڵامەکەم پچڕا."
SORANI_DONE = "تەواو بوو."
SORANI_NOT_DONE = "ببورە، نەکرا."
SORANI_NOT_UNDERSTOOD = "ببورە، تێنەگەیشتم. دەتوانیت جارێکی تر بیڵێیتەوە؟"
LOCAL_ACK_CKB = "یەک چرکە، بە مێشکی ناوخۆیی بیری لێ دەکەمەوە."
# The local brain answered an action with words only (qwen3:8b said «نۆتپاد ئامادەیە» with no
# tool call, 2026-09-25): SAM never claims an action it did not run.
LOCAL_NOT_DONE_CKB = "ببورە، ئەمەم نەکرد؛ مێشکی ناوخۆیی تێی نەگەیشت. تکایە بە شێوەیەکی تر بیڵێوە."

# Short acknowledgements spoken while tools run (cascade voice only).
ACKS_DO = ("باشە.", "بەسەرچاو.", "ئێستا دەیکەم.", "با بیکەم.", "باشە، ئێستا.")
ACKS_LOOK = ("با سەیری بکەم.", "یەک چرکە.", "با بزانم.")
_LOOKING = frozenset({"analyze_market", "screen_look", "chart_state", "web_search", "recall", "get_price",
                      "strategy_get", "list_alerts", "fetch_page"})

MORE_TOOLS = "more_tools"
# The review's core tier (measured 10,361 characters full, 8,533 compact).
# list_alerts / cancel_alert joined it in the acceptance review: "cancel my gold
# alert" needed a more_tools round first, and when that round found no model
# the user heard «تەواو بوو.» while the alert stayed active.
# knowledge_search joined when the user's library of books arrived (sam/knowledge,
# 2026-09-25): questions about his own documents must find their pages at once.
CORE_TOOLS = ["open_app", "tv_open", "tv_set_chart", "analyze_market", "draw_on_chart", "clear_my_drawings",
              "get_price", "set_alert", "list_alerts", "cancel_alert", "web_search", "remember", "recall",
              "knowledge_search", "delegate_task", "window_control", "system_control", "stop_all"]
SHORT_REPLY_CHARS = 40


class AckText(str):
    """A yielded chunk that is only the acknowledgement (the cascade times the
    first sound of the real answer separately: first_answer_audio)."""


class AnswerText(str):
    """A yielded chunk of the answer itself."""


@tool(MORE_TOOLS,
      description="Attach more of SAM's tools to your next step when none of the tools you have fits: files, "
                  "run_powershell, run_python, screen_look, click, type_text, press_keys, screen_act, open_url, "
                  "fetch_page, build_project, chart_state, strategy_save, strategy_list, strategy_get, theory_info, "
                  "knowledge_add, knowledge_list, knowledge_remove, forget. Name the ones you need.",
      params={"type": "object", "properties": {
          "tools": {"type": "array", "items": {"type": "string"}, "description": "tool names you need"},
          "need": {"type": "string", "description": "what you want to do, if unsure which tool"}}},
      description_ckb="هێنانی ئامرازی زیاتر", risk="safe", blocking=True, timeout_s=5)
async def more_tools(ctx: ToolContext, tools: list[str] | None = None, need: str = "") -> dict[str, Any]:
    registry = ctx.registry
    core = set(core_tool_names(ctx.app))
    known = [n for n in registry.names() if n not in core and n != MORE_TOOLS]
    wanted = [n for n in (tools or []) if n in known] or known
    return ok("These tools are attached to your next step: " + ", ".join(wanted) + ". Call the one you need now.",
              tools=wanted)


def core_tool_names(app: Any) -> list[str]:
    names = app.config.get("conversation.core_tools", CORE_TOOLS) or CORE_TOOLS
    return [str(n) for n in names]


def clean_tool_args(registry: Any, name: str, args: Any) -> Any:
    """Drop arguments a tool does not declare (Gemini sent ``{"reason": ...}``
    to the parameterless ``tv_open``); ``validate_args`` does the same since the
    integration, this copy keys the same-call dedup in ``_run_tool``."""
    spec = registry.get(name) if registry is not None else None
    if spec is None or not isinstance(args, dict):
        return args
    params = spec.params or {}
    if params.get("additionalProperties") is True:
        return args
    declared = params.get("properties") or {}
    return {key: value for key, value in args.items() if key in declared}


# Imperatives that ask SAM to DO something on the computer (not "tell me" / «پێم بڵێ»).
_ACTION_EN = frozenset({"open", "close", "start", "launch", "run", "draw", "delete", "remove", "clear", "set", "put",
                        "play", "pause", "mute", "unmute", "type", "send", "click", "save", "create", "make", "move",
                        "copy", "write", "switch", "change", "turn", "cancel", "minimize", "maximize", "install"})
_ACTION_CKB = ("بکەرەوە", "بکەوە", "دابخە", "داخە", "بکێشە", "بسڕەوە", "لابە", "بنووسە", "بنێرە", "بگۆڕە", "دابنێ",
               "هەڵبکە", "بکوژێنەوە", "لێبدە", "بخە", "بگرە", "بهێنە", "دروستبکە", "هەڵبگرە", "کپبکە")
_DONE_MARKERS = ("ئامادەیە", "کرایەوە", "کرا.", "کرا،", "کێشرا", "سڕایەوە", "سڕدرایەوە", "دانرا", "گۆڕا", "لابرا",
                 "داخرا", "نێردرا", "نووسرا", "تەواو بوو", "کردمەوە", "done", "opened", "is open", "is ready",
                 "closed", "i have", "i've")


def _is_action_command(text: str) -> bool:
    words = normalize_ckb(text, strip_punct=True).split()
    if not words:
        return False
    if any(w in _ACTION_EN for w in words[:3]):
        return True
    return "بڵێ" not in words and any(w.endswith(v) for w in words[-3:] for v in _ACTION_CKB)


_NUMBER = re.compile(r"[0-9\u0660-\u0669]{3,}")
_EASTERN_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _unverified_number(reply: str, user_text: str, grounding: str = "") -> bool:
    """A price-like number (3+ digits) with no tool result behind it: not said
    by the user and not in ``grounding`` (the system prompt: time, facts,
    library passages). Live runs 2026-09-25: qwen3:8b said «زێڕ ئێستا لە ٤٢٦٩
    دەبێت» from memory, and answered a window count with the gold price."""
    said = {n.translate(_EASTERN_TO_ASCII) for n in _NUMBER.findall(f"{user_text} {grounding}")}
    return bool({n.translate(_EASTERN_TO_ASCII) for n in _NUMBER.findall(reply)} - said)


# Things only a tool can know (alerts, windows, the chart, SAM's drawings, files).
_STATE_STEMS = tuple(normalize_ckb(s) for s in (
    "ئاگادارکردنەوە", "ئاگاداری", "ئالێرت", "ئەلێرت", "پەنجەرە", "چارت", "کێشراو", "فایل", "alert", "alarm",
    "window", "chart", "drawing", "file"))


def _asks_state(text: str) -> bool:
    return any(w.startswith(stem) for w in normalize_ckb(text, strip_punct=True).split() for stem in _STATE_STEMS)


def _claims_done(reply: str) -> bool:
    lowered = normalize_ckb(reply)
    return any(normalize_ckb(marker) in lowered for marker in _DONE_MARKERS)


def _echoes(reply: str, user_text: str) -> bool:
    """The reply mostly repeats the user's words (Groq answered «ئێستا چی کار
    بکەین؟» to «چۆنی؟ ئەمڕۆ چی بکەین؟», review 2026-09-24)."""
    said = set(normalize_ckb(user_text, strip_punct=True).split())
    words = normalize_ckb(reply, strip_punct=True).split()
    return bool(words) and sum(w in said for w in words) / len(words) >= 0.6


class Responder:
    """Mixin for ``Conversation``: needs ``self.app``, ``ensure_conversation``
    and ``_messages``."""

    app: Any
    _last_ack: str
    _ack_index: int

    # -- ladders ------------------------------------------------------------------------------
    def _ladder(self, mode: str, *, after_tools: bool = False) -> list[str]:
        """Refs for a round, health-ordered (see ladders.py for the measurements)."""
        if mode == "text":
            key = "conversation.ladder.reply" if after_tools else "conversation.ladder.text"
        else:
            key = "conversation.ladder.reply" if after_tools else "conversation.ladder.voice"
        fallback = ladders.auto_wording(self.app) if after_tools else ladders.auto_picker(self.app, mode)
        return ladders.ordered(self.app, ladders.resolve(self.app, self.app.config.get(key, "auto"), fallback))

    def _wording_refs(self) -> list[str]:
        """Rewording rungs: the wording ladder without the fast pickers, only
        rungs that can answer now and did not fail lately (a reword is an
        optional improvement: it must not cost a spoken turn a 5 s timeout)."""
        refs = ladders.resolve(self.app, self.app.config.get("conversation.ladder.reply", "auto"),
                               ladders.auto_wording(self.app))
        out = []
        for ref in refs:
            if ladders.is_fast_picker(ref) or self.app.llm.cooling(ref) or self.app.llm.strikes(ref):
                continue
            backend = self.app.llm.backends.get(ref.split(":", 1)[0])
            if backend is not None and backend.configured():
                out.append(ref)
        return ladders.ordered(self.app, out)

    def _deadline(self, key: str, default: float) -> float:
        return float(self.app.config.get(key, default) or default)

    def _llm_timeout(self) -> float:
        return float(self.app.config.get("conversation.llm_timeout_s", 15) or 15)

    def _ack(self, calls: list[ToolCall]) -> str:
        """A short spoken acknowledgement while tools and the wording round
        run (cascade only); rotates so the same words never come twice in a
        row."""
        pool = ACKS_LOOK if any(c.name in _LOOKING for c in calls) else ACKS_DO
        choices = [a for a in pool if a != self._last_ack] or list(pool)
        self._ack_index = (self._ack_index + 1) % len(choices)
        self._last_ack = choices[self._ack_index]
        return self._last_ack

    def _tools_for_round(self, extra: set[str]) -> list[dict[str, Any]]:
        registry = self.app.tools
        if str(self.app.config.get("conversation.tool_tier", "core")) == "all":
            return registry.openai_tools([n for n in registry.names() if n != MORE_TOOLS])
        registered = set(registry.names())
        core = [n for n in core_tool_names(self.app) if n in registered]
        if MORE_TOOLS in registered:
            core.append(MORE_TOOLS)
        added = sorted(n for n in extra if n in registered and n not in core)
        return registry.openai_tools(core, compact=True) + (registry.openai_tools(added) if added else [])

    # -- the turn -------------------------------------------------------------------------------
    async def respond_stream(self, text: str, *, source: str = "cascade", turn: Any = None) -> AsyncIterator[str]:
        """Answer ``text``: yields sentence-sized chunks (speakable for voice
        sources), runs tool calls in between, stores both turns."""
        text = (text or "").strip()
        if not text:
            return
        own_turn = turn is None
        if turn is None:
            turn = self.app.timing.turn("text" if source == "text" else source)
        conversation_id = self.ensure_conversation(source)  # type: ignore[attr-defined]
        # Voice engines publish the user's Transcript before calling us; the
        # dedup in _on_transcript makes this a no-op for them.
        self.app.bus.publish(Transcript(role="user", text=text, source=source, conversation_id=conversation_id))
        scope = taint.begin(text)
        mode = "text" if source == "text" else "voice"
        # Common commands need no model at all (fastpath.py / intents.py).
        intent = fastpath.intent_for(self.app, text)
        messages = [] if intent is not None else self._messages(conversation_id, text, mode)  # type: ignore[attr-defined]
        max_rounds = max(1, int(self.app.config.get("conversation.max_tool_rounds", 6) or 6))
        streaming = bool(self.app.config.get("conversation.stream", False))
        acknowledge = mode == "voice" and bool(self.app.config.get("conversation.voice_ack", True))
        chunker = SentenceChunker(speech=mode == "voice")
        parts: list[str] = []            # everything yielded (heard / shown)
        said = False                     # the model itself produced words (not only an acknowledgement)
        seen: dict[str, dict[str, Any]] = {}
        last: tuple[str, dict[str, Any], dict[str, Any]] | None = None
        extra: set[str] = set()
        completed = False
        # Trading/strategy questions: the user's own books in the prompt (library_context.py).
        passages: list[dict[str, Any]] = []
        block = ""
        if intent is None:
            block, passages = await library_block(self.app, text, mode)
            if block:
                add_to_messages(messages, block)
                scope.mark("knowledge_search")        # the passages are the user's files: data

        def emit(piece: str, **mark: Any) -> str:
            if not parts:
                turn.mark("first_chunk", **mark)
            if not mark.get("ack"):
                turn.mark_once("first_answer")
                piece = AnswerText(piece)
            else:
                piece = AckText(piece)
            parts.append(piece)
            return piece

        def outcome() -> str:
            """The honest result when the model gave no words (outcome.py)."""
            return SORANI_NOT_UNDERSTOOD if last is None else tool_sentence(*last)

        # A turn in progress: background model calls wait for a pause (budget.py).
        self.active_turns = int(getattr(self, "active_turns", 0) or 0) + 1
        try:
            if intent is not None:
                turn.mark("fastpath", intent=intent.name)
                async for kind, piece in fastpath.run(self.app, intent, source=source):
                    if kind == "ack":
                        if acknowledge:
                            yield emit(piece, ack=True)
                        continue
                    for chunk in chunker.feed(piece):
                        yield emit(chunk)
                for chunk in chunker.flush():
                    yield emit(chunk)
                completed = True
                return
            local_ack = await self._local_ack(mode) if acknowledge else ""
            local_acked = bool(local_ack)
            if local_ack:
                yield emit(local_ack, ack=True)
            for round_no in range(max_rounds + 1):
                tool_choice = "none" if round_no == max_rounds else None
                after_tools = round_no > 0
                tools = self._tools_for_round(extra)
                extra = set()
                ladder = self._ladder(mode, after_tools=after_tools)
                deadline = (self._deadline("conversation.wording_deadline_s", ladders.WORDING_DEADLINE_S) if after_tools
                            else self._deadline("conversation.picker_deadline_s", ladders.PICKER_DEADLINE_S))
                response: LLMResponse | None = None
                calls: list[ToolCall] = []
                try:
                    if streaming:
                        async for piece, response, calls in self._streamed_round(
                                messages, ladder, tools, tool_choice, turn, chunker):
                            if piece:
                                said = True
                                yield emit(piece)
                    else:
                        # After a tool with its own Sorani result, no cold local round (up to
                        # 150 s): if no cloud model can word it, that result is said at once.
                        local = False if after_tools and last is not None and own_sentence(*last) else None
                        async for kind, value in self._ask_announcing(
                                messages, ladder, tools, tool_choice, turn, deadline, local,
                                announce=acknowledge and not local_acked):
                            if kind == "ack":
                                local_acked = True
                                yield emit(value, ack=True)
                            else:
                                response = value
                        if response is None:           # _ask_announcing always yields one or raises
                            raise LLMError("exhausted", "no answer")
                        calls = list(response.tool_calls) if tool_choice != "none" else []
                        reply = response.text or ""
                        if not calls and last is None and self._local_false_claim(
                                response, text, str(messages[0].get("content") or "") if messages else block):
                            reply = LOCAL_NOT_DONE_CKB
                        if not calls and not after_tools and self._should_reword(response, text):
                            better = await self._reword(messages, tools, turn)
                            if better is not None and better.tool_calls:
                                response, calls, reply = better, list(better.tool_calls), ""
                            elif better is not None and (better.text or "").strip():
                                reply = better.text
                        if not calls:  # text next to a tool call is not spoken (the ack is)
                            for piece in chunker.feed(reply):
                                said = True
                                yield emit(piece)
                except LLMError as err:
                    for piece in chunker.flush():
                        said = True
                        yield emit(piece)
                    # After a tool ran, its honest outcome beats "no model".
                    message = SORANI_CUT_OFF if said else (SORANI_NO_MODEL if last is None else outcome())
                    passages = []                      # no model answer: nothing was answered from the books
                    self.app.bus.publish(Error(where="conversation", message_ckb=SORANI_NO_MODEL,
                                               detail=self.app.redact(str(err))[:300]))
                    yield emit(message)
                    completed = True
                    return
                for piece in chunker.flush():
                    said = True
                    yield emit(piece)
                if not calls or tool_choice == "none":
                    break
                if response is None:
                    response = LLMResponse(text="", tool_calls=calls)
                if acknowledge and not parts:
                    # Speak at once instead of staying silent through the tool
                    # and the (slower, quality-first) wording round.
                    yield emit(self._ack(calls), ack=True)
                messages.append(response.assistant_message())
                taint.use(scope)
                done_now: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
                for call in calls:
                    args = clean_tool_args(self.app.tools, call.name, call.arguments)
                    result = await self._run_tool(call, source, seen)
                    if call.name == MORE_TOOLS:
                        # Only attaches tools: it is never the outcome the user hears.
                        extra |= set(((result.get("data") or {}).get("tools")) or [])
                    else:
                        last = (call.name, args if isinstance(args, dict) else {}, result)
                        done_now.append(last)
                    messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                     "content": json.dumps(result, ensure_ascii=False, default=str)})
                own = self._local_outcome(response, done_now, extra)
                if own:
                    # The local brain picked the tool: its own Sorani result is the answer
                    # (a second local round costs 3-10 s on this PC's CPU and says the same).
                    said = True
                    for piece in [*chunker.feed(own), *chunker.flush()]:
                        yield emit(piece)
                    break
            if not said:
                # Never a blanket "done" when no tool ran.
                yield emit(outcome())
            completed = True
        finally:
            self.active_turns = max(0, int(getattr(self, "active_turns", 1) or 1) - 1)
            reply_text = " ".join(parts).strip()
            if reply_text and passages and said:
                # the pages for the panel, under the answer; never spoken (not a chunk)
                reply_text += "\n\n" + sources_line(passages)
            if reply_text:
                self.app.bus.publish(Transcript(role="assistant", text=reply_text, source=source,
                                                conversation_id=conversation_id))
            if own_turn:
                turn.finish(completed=completed)

    async def _local_ack(self, mode: str) -> str:
        """A spoken «one moment» when this turn will wait for a cold local
        brain (no cloud rung can answer and the model is not loaded: ~10 s
        to load, and a cold first prompt took 86 s on this PC)."""
        llm = self.app.llm
        try:
            if llm.cloud_usable(self._ladder(mode)) or not llm.local_ready():
                return ""
            backend = llm.local_backend()
            loaded = await backend.loaded() if hasattr(backend, "loaded") else []
        except Exception:  # noqa: BLE001 - an acknowledgement is optional
            return ""
        return "" if loaded else LOCAL_ACK_CKB

    async def _ask_announcing(self, messages: list[dict[str, Any]], ladder: list[str], tools: list[dict[str, Any]],
                              tool_choice: str | None, turn: Any, deadline: float, local: bool | None, *,
                              announce: bool) -> AsyncIterator[tuple[str, Any]]:
        """``_ask`` that yields ("ack", LOCAL_ACK_CKB) when the round falls to a
        COLD local brain (every cloud rung failed mid-turn, or the deadline cut
        a slow one: the user would otherwise hear ~1.5 min of silence), then
        ("response", LLMResponse). LLMError propagates."""
        if not announce or local is False:
            yield "response", await self._ask(messages, ladder, tools, tool_choice, turn, deadline_s=deadline,
                                              local=local)
            return
        cold = asyncio.Event()

        def on_local(is_cold: bool) -> None:
            if is_cold:
                cold.set()

        task = asyncio.ensure_future(self._ask(messages, ladder, tools, tool_choice, turn, deadline_s=deadline,
                                               local=local, on_local=on_local))
        waiter = asyncio.ensure_future(cold.wait())
        try:
            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            if cold.is_set() and not task.done():
                yield "ack", LOCAL_ACK_CKB
            yield "response", await task
        finally:
            waiter.cancel()
            if not task.done():
                task.cancel()

    @staticmethod
    def _local_false_claim(response: LLMResponse, user_text: str, grounding: str = "") -> bool:
        """The local brain answered with words only where a tool was needed: a
        short 'done/ready' claim or an echo of an action command (qwen3:8b:
        «نۆتپاد ئامادەیە», «کرۆم بکەرەوە.»), a price-like number that no tool,
        the user or ``grounding`` (the system prompt with this turn's library
        passages) gave it («زێڕ ئێستا لە ٤٢٦٩ دەبێت»), or a short statement about
        alerts / windows / the chart / files that only a tool can know."""
        if response.provider != LOCAL_PROVIDER or response.tool_calls:
            return False
        reply = (response.text or "").strip()
        if not reply:
            return False
        if _unverified_number(reply, user_text, grounding):
            return True
        if _asks_state(user_text) and len(reply) <= 160:
            return True       # «ئاگادارکردنەوەکان بۆ زێڕ نەداناوە» with no list_alerts call (live run)
        if len(reply) > 90 or not _is_action_command(user_text):
            return False
        return _claims_done(reply) or _echoes(reply, user_text)

    @staticmethod
    def _local_outcome(response: LLMResponse | None, done: list[Any], extra: set[str]) -> str:
        """The tools' own Sorani sentences after local-brain tool calls, one per
        call ('' = let the model word it: a list the user wants read, more tools
        needed). Live run 2026-09-25: «نرخی زێڕ و زیو» made two get_price calls
        and only the last result was said."""
        if response is None or response.provider != LOCAL_PROVIDER or not done or extra:
            return ""
        sentences: list[str] = []
        for item in done:
            sentence = own_sentence(*item)
            if not sentence:
                return ""
            if sentence not in sentences:
                sentences.append(sentence)
        return " ".join(sentences)

    async def _streamed_round(self, messages: list[dict[str, Any]], ladder: list[str], tools: list[dict[str, Any]],
                              tool_choice: str | None, turn: Any, chunker: SentenceChunker) -> AsyncIterator[Any]:
        """Setting ``conversation.stream``: yields (piece, response, calls)."""
        response: LLMResponse | None = None
        calls: list[ToolCall] = []
        async for chunk in self.app.llm.stream(messages, ladder=ladder, tools=tools, timeout_s=self._llm_timeout(),
                                               tool_choice=tool_choice, turn=turn):
            if chunk.kind == "text":
                for piece in chunker.feed(chunk.text):
                    yield piece, None, []
            elif chunk.kind == "tool_call" and chunk.tool_call is not None:
                calls.append(chunk.tool_call)
            elif chunk.kind == "done":
                response = chunk.response
        if response is None or (not response.text.strip() and not response.tool_calls and not calls):
            # Silent empty stream: redo the round unstreamed on the next rungs.
            response = await self._ask(messages, ladder, tools, tool_choice, turn,
                                       after=response.model_ref if response else None)
            for piece in chunker.feed(response.text or ""):
                yield piece, None, []
        if response.tool_calls:
            calls = list(response.tool_calls)
        yield "", response, calls if tool_choice != "none" else []

    def _should_reword(self, response: LLMResponse, user_text: str) -> bool:
        """Small talk a fast picker (Groq) answered is worded again by a better
        model when one can answer now -- unless it is a short, clean Sorani
        sentence that does not just echo the user."""
        if not ladders.is_fast_picker(response.model_ref):
            return False
        reply = fix_letters(response.text or "").strip()
        refs = self._wording_refs()
        if not reply or not refs:
            return False
        wanted = (len(reply) > SHORT_REPLY_CHARS or _echoes(reply, user_text)
                  or (is_arabic_script(user_text) and not is_arabic_script(reply)))
        if not wanted:
            return False
        # A reword is a second request for words the user already has: it is
        # skipped while any rung rests or the day's budget is low (budget.py).
        budget = getattr(self, "budget", None)
        if budget is not None:
            reason = budget.reason_to_skip("reword", refs, allow_live=True)
            budget.record("reword", "skipped" if reason else "ran", reason or "")
            if reason:
                return False
        return True

    async def _reword(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                      turn: Any) -> LLMResponse | None:
        refs = self._wording_refs()[:1]     # one request at most (budget.py): else Groq's own words stand
        if not refs:
            return None
        try:
            return await self._ask(messages, refs, tools, None, turn,
                                   deadline_s=self._deadline("conversation.reword_deadline_s",
                                                             ladders.REWORD_DEADLINE_S), local=False)
        except LLMError as err:
            log.info("rewording failed (%s); keeping the fast answer", err.kind)
            return None

    async def _ask(self, messages: list[dict[str, Any]], ladder: Any, tools: list[dict[str, Any]],
                   tool_choice: str | None, turn: Any, *, after: str | None = None,
                   deadline_s: float | None = None, local: bool | None = None,
                   on_local: Any = None) -> LLMResponse:
        """One non-streamed round. An answer with neither text nor a tool call
        is treated as a silent failure and the remaining rungs are asked once.

        Measured 2026-09-24 through OmniRoute: a streamed request with tools
        returned HTTP 200, keepalive chunks and then nothing (7.0 s) right when
        the upstream free model was rate limited; the same request unstreamed
        got a proper 429 (so the ladder falls back) or the tool call in 2.0 s.
        For 1-3 sentence replies streaming saved only ~0.2 s, so rounds are
        unstreamed by default (setting ``conversation.stream``)."""
        refs = self.app.llm.ladder(ladder)
        if after in refs:
            refs = refs[refs.index(after) + 1:]
        if not refs:
            raise LLMError("exhausted", "no model left after an empty answer")
        kwargs = {"tools": tools, "tool_choice": tool_choice, "turn": turn, "timeout_s": self._llm_timeout(),
                  "rung_timeouts": ladders.rung_caps(), "deadline_s": deadline_s, "retry_transient": False,
                  "reasoning": str(self.app.config.get("conversation.reasoning", "minimal") or "minimal")}
        if local is not None:
            kwargs["local"] = local
        if on_local is not None:
            kwargs["on_local"] = on_local
        response = await self._chat_reserving(messages, refs, kwargs)
        if (response.text or "").strip() or response.tool_calls:
            return response
        rest = refs[refs.index(response.model_ref) + 1:] if response.model_ref in refs else []
        if rest:
            log.info("empty answer from %s; asking the next rung", response.model_ref)
            try:
                return await self.app.llm.chat(messages, ladder=rest, **kwargs)
            except LLMError:
                pass
        return response

    async def _chat_reserving(self, messages: list[dict[str, Any]], refs: list[str],
                              kwargs: dict[str, Any]) -> LLMResponse:
        """``llm.chat`` over ``refs``, but the slow rungs ahead of the first
        healthy fast picker share only ``ladders.head_budget`` of the round's
        deadline, and one is not started with less than MIN_SLOW_RUNG_S left:
        a healthy Groq is always asked before the deadline (ladders.py has the
        measurements)."""
        deadline_s = kwargs.get("deadline_s")
        head, rest = ladders.split_head(self.app, refs) if deadline_s else ([], refs)
        if not head:
            return await self.app.llm.chat(messages, ladder=refs, **kwargs)
        started = time.monotonic()
        budget = ladders.head_budget(float(deadline_s))
        for ref in head:
            left = budget - (time.monotonic() - started)
            if left < ladders.MIN_SLOW_RUNG_S:
                log.info("skipping %s: %.1f s left for the slow rungs", ref, left)
                break
            try:
                # never the local brain here: the fast rungs after the head have not been asked yet
                return await self.app.llm.chat(messages, ladder=[ref], **{**kwargs, "deadline_s": left, "local": False,
                                                                          "on_local": None})
            except LLMError as err:
                log.info("%s gave no answer (%s); next rung", ref, self.app.redact(str(err))[:160])
        left = float(deadline_s) - (time.monotonic() - started)
        return await self.app.llm.chat(messages, ladder=rest,
                                       **{**kwargs, "deadline_s": max(left, ladders.FAST_RESERVE_S)})

    async def _run_tool(self, call: ToolCall, source: str, seen: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Dispatch one call; an identical repeat inside the same turn is not
        executed again (models sometimes loop on 'open it' after success)."""
        args = clean_tool_args(self.app.tools, call.name, call.arguments)
        key = f"{call.name}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
        if key in seen:
            previous = seen[key]
            return {"ok": previous.get("ok", False), "data": None,
                    "summary": "Already called with the same arguments in this turn; do not repeat it. "
                               f"Previous result: {previous.get('summary', '')}"[:600]}
        result = await self.app.tools.dispatch(call.name, args, source=source, call_id=call.id)
        seen[key] = result
        return result


__all__ = ["Responder", "AckText", "AnswerText", "more_tools", "core_tool_names", "clean_tool_args", "CORE_TOOLS", "MORE_TOOLS",
           "ACKS_DO", "ACKS_LOOK", "SORANI_NO_MODEL", "SORANI_CUT_OFF", "SORANI_DONE", "SORANI_NOT_DONE",
           "SORANI_NOT_UNDERSTOOD"]
