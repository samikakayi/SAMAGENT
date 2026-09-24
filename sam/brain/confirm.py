"""ConfirmBroker: the only way a risky action gets a YES.

Flow: a tool whose risk is ``confirm`` calls ``await broker.confirm(question)``.
The broker publishes ``ConfirmRequest`` (the island shows a card; the voice
engine makes the question heard) and waits. It resolves from

- the user's own speech: the voice engine passes EVERY final user transcript
  to ``offer_transcript(text)`` first; a short yes/no answer resolves the most
  recent pending confirmation and is consumed (returns True);
- a click: the UI calls ``resolve(confirm_id, approved, via="click")`` (thread
  safe);
- expiry: after ``timeout_s`` (default 20 s, design section 1) the answer is NO.

The model can never approve its own action: there is deliberately no
"confirm_pending" model tool (text read from screens/web pages could otherwise
talk the model into approving -- prompt-injection risk). Unclear answers are
not consumed and simply let the timer run out (default NO).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..events import ConfirmRequest, ConfirmResult, EventBus, new_id
from ..textnorm import normalize_ckb

log = logging.getLogger("sam.confirm")

def _norm_all(items: tuple[str, ...]) -> tuple[frozenset[str], tuple[str, ...]]:
    """Normalise like transcripts are normalised; entries that become several
    tokens (e.g. "don't" -> "don t") are matched as phrases."""
    words: set[str] = set()
    phrases: list[str] = []
    for item in items:
        norm = normalize_ckb(item, strip_punct=True)
        (phrases.append(norm) if " " in norm else words.add(norm))
    return frozenset(words), tuple(phrases)


# Compared after normalize_ckb (ي->ی, ك->ک, no ZWNJ/tatweel, punctuation off).
#
# YES is accepted only when the WHOLE utterance is a yes-phrase (every word a
# yes-word or a filler, at most MAX_YES_WORDS words). The repair review
# (2026-09-24, confirm_probe.py) showed the old "any yes-token in <= 8 words"
# rule approving a pending delete/send on «چاوەڕێ بکە» (wait), «ئا نازانم»
# (uh, I don't know), «باشە دواتر» (ok, later), «تەواو بەسە» (ok, enough) and
# on a NEW command such as «شیکاری گۆڵد بکە». So «بکە», «ئا»/«آ», «ئەها»,
# «تەواو» and «باشە» are not a yes on their own any more («باشە بیکە» is).
YES_WORDS, YES_PHRASES = _norm_all((
    "بەڵێ", "بەلێ", "بەڵی", "بەلی", "ئەرێ", "ئەری", "هەڵبەت", "ئەڵبەت", "بێگومان", "بیکە", "ڕاستە", "دروستە",
    "ئۆکەی", "ئۆکێ",
    "بەڵێ بیکە", "دەی بیکە", "باشە بیکە", "ئا بیکە", "بێ گومان", "بەردەوام بە",
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed", "proceed", "approve",
    "approved", "affirmative", "go ahead", "do it", "of course",
))
# Words that may stand next to a yes without changing it («ئا بەڵێ», «بەڵێ تکایە»).
# The name too: in always-listening mode the user may say «سام، بەڵێ».
FILLER_WORDS, _ = _norm_all(("ئا", "ئاا", "آ", "ئەها", "دەی", "تکایە", "باشە", "تەواو", "زۆر", "باش", "please", "sure",
                             "oh", "uh", "um", "well", "سام", "sam", "هێی", "hey"))
NO_WORDS, NO_PHRASES = _norm_all((
    "نا", "نەخێر", "نەخیر", "نەء", "نە", "مەیکە", "مەکە", "نەیکەی", "نەکەی", "وازبێنە", "بوەستە", "ڕاوەستە",
    "هەڵیوەشێنەوە", "ناوێت", "نامەوێت", "ڕەتیدەکەمەوە", "پاشگەزبوومەوە", "بەسە", "لێگەڕێ", "چاوەڕێ",
    "چاوەڕوان", "دواتر", "پاشان", "نازانم", "ناکات",
    "واز بێنە", "هیچ مەکە", "پاشگەز بوومەوە", "پێویست ناکات", "وازی لێ بێنە", "لێی گەڕێ",
    "no", "nope", "nah", "cancel", "stop", "abort", "negative", "dont", "don't", "do not", "never mind",
    "not", "wait", "later", "enough", "hold", "hold on",
    # negative imperatives of the actions SAM asks about (never a yes)
    "مەینێرە", "مەنێرە", "مەیسڕەوە", "مەسڕەوە", "دامەخە", "دایمەخە", "مەیکەرەوە", "مەکەرەوە", "دامەگرە",
    "دایمەگرە", "مەیگوازەوە", "مەینووسە",
))
MAX_ANSWER_WORDS = 8
MAX_YES_WORDS = 3
# The action's own verb as an answer («ئەم نامەیە بنێرم؟» -> «بەڵێ بینێرە» /
# «بینێرە»). Verify review 2026-09-24: these natural answers were "unclear",
# so the cascade started a new turn that muted the waiting reply and the
# question timed out to NO. They count as YES only while the pending question
# itself contains the same verb stem (first item), so «بیسڕەوە» can never
# approve sending a message.
ACTION_VERBS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("نێر", ("بنێرە", "بینێرە", "send", "send it")),
    ("سڕ", ("بسڕەوە", "بیسڕەوە", "delete", "delete it")),
    ("دابخ", ("دابخە", "دایبخە", "close", "close it")),
    ("جێبەجێ", ("جێبەجێی بکە", "جێبەجێ بکە", "run", "run it")),
    ("بکەمەوە", ("بیکەرەوە", "بکەرەوە", "open", "open it")),
    ("دابگر", ("دابگرە", "دایگرە", "press", "press it")),
    ("کلیک", ("کلیکی بکە", "کلیک بکە", "click", "click it")),
    ("گواز", ("بگوازەوە", "بیگوازەوە", "move", "move it")),
    ("بنووس", ("بنووسە", "بینووسە", "write", "type it")),
    ("پاشەکەوت", ("پاشەکەوتی بکە", "save", "save it")),
)
ASK_AGAIN_CKB = "بەڵێ یان نەخێر؟"


def _action_yes(question: str) -> tuple[frozenset[str], tuple[str, ...]]:
    """(words, phrases) that answer YES to this particular question."""
    asked = normalize_ckb(question or "", strip_punct=True)
    items: list[str] = []
    if asked:
        for stem, answers in ACTION_VERBS:
            if normalize_ckb(stem) in asked:
                items.extend(answers)
    return _norm_all(tuple(items)) if items else (frozenset(), ())


def classify_answer(text: str, question: str = "") -> bool | None:
    """True (yes), False (no) or None (not a clear short answer).

    'No' wins over 'yes' when both appear (safer default); utterances longer
    than ``MAX_ANSWER_WORDS`` are a new request, not an answer. A yes must be
    the whole utterance (``MAX_YES_WORDS`` words at most, fillers allowed);
    anything else is unclear and is NOT consumed, so it reaches the brain as a
    normal request and the pending question simply times out (default NO).
    With ``question`` (the pending confirmation), that action's own verb is a
    yes too (``ACTION_VERBS``).
    """
    norm = normalize_ckb(text, strip_punct=True)
    if not norm:
        return None
    tokens = norm.split()
    if len(tokens) > MAX_ANSWER_WORDS:
        return None
    padded = f" {norm} "
    if any(t in NO_WORDS for t in tokens) or any(f" {p} " in padded for p in NO_PHRASES):
        return False
    if len(tokens) > MAX_YES_WORDS:
        return None
    verb_words, verb_phrases = _action_yes(question)
    rest = padded
    for phrase in sorted(YES_PHRASES + verb_phrases, key=len, reverse=True):
        if f" {phrase} " in rest:
            rest = rest.replace(f" {phrase} ", " YES ")
    words = rest.split()
    yes_words = YES_WORDS | verb_words
    has_yes = any(w == "YES" or w in yes_words for w in words)
    only_yes = all(w == "YES" or w in yes_words or w in FILLER_WORDS for w in words)
    return True if has_yes and only_yes else None


@dataclass
class PendingConfirm:
    confirm_id: str
    question_ckb: str
    detail: str
    tool_name: str
    created_at: float
    expires_at: float
    future: "asyncio.Future[bool]" = field(repr=False)
    loop: asyncio.AbstractEventLoop = field(repr=False)

    def public(self) -> dict[str, Any]:
        return {"confirm_id": self.confirm_id, "question_ckb": self.question_ckb, "detail": self.detail,
                "tool_name": self.tool_name, "expires_at": self.expires_at}


class ConfirmBroker:
    def __init__(self, bus: EventBus | None = None, timeout_s: float = 20.0, *, db: Any = None) -> None:
        self.bus = bus
        self.timeout_s = float(timeout_s)
        self.db = db
        self._pending: dict[str, PendingConfirm] = {}

    async def confirm(self, question_ckb: str, detail: str = "", *, tool_name: str = "",
                      timeout_s: float | None = None) -> bool:
        """Ask the user; True only on an explicit yes before expiry."""
        loop = asyncio.get_running_loop()
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        now = time.time()
        item = PendingConfirm(confirm_id=new_id(), question_ckb=question_ckb, detail=detail, tool_name=tool_name,
                              created_at=now, expires_at=now + timeout, future=loop.create_future(), loop=loop)
        self._pending[item.confirm_id] = item
        if self.bus is not None:
            self.bus.publish(ConfirmRequest(confirm_id=item.confirm_id, question_ckb=question_ckb, detail=detail,
                                            tool_name=tool_name, expires_at=item.expires_at))
        via = "timeout"
        approved = False
        try:
            approved, via = await asyncio.wait_for(asyncio.shield(item.future), timeout)
        except asyncio.TimeoutError:
            approved, via = False, "timeout"
        except asyncio.CancelledError:
            approved, via = False, "cancel"
            raise
        finally:
            self._pending.pop(item.confirm_id, None)
            if not item.future.done():
                item.future.cancel()
            if self.bus is not None:
                self.bus.publish(ConfirmResult(confirm_id=item.confirm_id, approved=approved, via=via))
            if self.db is not None:
                try:
                    self.db.log_activity("confirm", tool_name, ok=approved, summary=question_ckb[:300], source=via)
                except Exception:  # noqa: BLE001
                    log.exception("confirm activity log failed")
        return approved

    def _latest(self) -> PendingConfirm | None:
        if not self._pending:
            return None
        return max(self._pending.values(), key=lambda p: p.created_at)

    def resolve(self, confirm_id: str | None, approved: bool, via: str = "click") -> bool:
        """Answer a pending confirmation (``None`` = the most recent one).
        Safe to call from any thread. Returns False if nothing was pending."""
        item = self._pending.get(confirm_id) if confirm_id else self._latest()
        if item is None:
            return False

        def _set() -> None:
            if not item.future.done():
                item.future.set_result((bool(approved), via))

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is item.loop:
            _set()
        else:
            item.loop.call_soon_threadsafe(_set)
        return True

    def offer_transcript(self, text: str) -> bool:
        """Feed a final user transcript. Consumed (True) only when a
        confirmation is pending and the text is a clear yes/no."""
        latest = self._latest()
        if latest is None:
            return False
        answer = classify_answer(text, latest.question_ckb)
        if answer is None:
            return False
        return self.resolve(latest.confirm_id, answer, via="voice")

    def classify_pending(self, text: str) -> bool | None:
        """``classify_answer`` against the most recent pending question
        (None when nothing is pending or ``text`` is not a clear yes/no)."""
        latest = self._latest()
        return None if latest is None else classify_answer(text, latest.question_ckb)

    def needs_clear_answer(self, text: str) -> bool:
        """A confirmation is pending and ``text`` is a SHORT utterance that is
        neither yes nor no («باشە», «ئا»): the caller should ask
        ``ASK_AGAIN_CKB`` instead of starting a new turn with it (a new
        cascade turn mutes the reply that waits for this answer)."""
        latest = self._latest()
        if latest is None:
            return False
        norm = normalize_ckb(text, strip_punct=True)
        return bool(norm) and len(norm.split()) <= MAX_YES_WORDS and classify_answer(text, latest.question_ckb) is None

    def pending(self) -> list[dict[str, Any]]:
        return [p.public() for p in sorted(self._pending.values(), key=lambda p: p.created_at)]

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def cancel_all(self) -> int:
        """Answer NO to everything pending (stop_all / shutdown)."""
        count = 0
        for item in list(self._pending.values()):
            if self.resolve(item.confirm_id, False, via="cancel"):
                count += 1
        return count


__all__ = ["ConfirmBroker", "classify_answer", "YES_WORDS", "NO_WORDS", "PendingConfirm", "ACTION_VERBS", "ASK_AGAIN_CKB"]
