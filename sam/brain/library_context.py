"""The user's own library in the model's context for trading/strategy questions.

The user asked for answers from his trading books (2026-09-25). A question
like «ئۆردەر بلۆک چییە؟» or "how should I place a stop loss?" gets the best
passages of his library (``sam.knowledge.passages_for``: local FTS search, a
few ms, no model) in the per-turn part of the prompt, so the answer follows
his books even when the model would not think of calling knowledge_search.

- Only questions about trading ideas: a trading term of the glossary that is
  more than an instrument/price/market word, or a strategy/theory/book word,
  AND a question or an "explain/tell me" request. Commands, prices and small
  talk never pay the extra ~500 prompt tokens (Groq's free tier allows ~8k
  tokens a minute, CONTRACTS section 5).
- Passages below ``conversation.library.min_score`` (0.5; the library's own "weak"
  line is 0.35) stay out of the prompt.
- The passages are the user's files: DATA, never instructions (the block says
  so, and the turn's taint scope is marked like a knowledge_search result).
- Citations: speech names at most the book title in a few words; the panel
  gets the full «title»، لاپەڕە N line under the answer (``sources_line``),
  which is never spoken.
"""

from __future__ import annotations

import logging
from typing import Any

from ..textnorm import normalize_ckb

log = logging.getLogger("sam.conversation")

DEFAULTS: dict[str, Any] = {
    "conversation.library.enabled": True,        # library passages for trading/strategy questions
    "conversation.library.min_score": 0.5,
    "conversation.library.max_chars": 1500,
    "conversation.library.k": 3,
}
# Glossary groups that alone do not make a question about trading IDEAS
# («نرخی زێڕ چەندە؟» is a price question, not a question for the books).
_GENERIC = frozenset({"gold", "dollar", "price", "market", "analysis", "buy", "sell", "timeframe", "high",
                      "bottom", "asia", "london", "new york", "news"})
_TOPIC_WORDS = frozenset(normalize_ckb(w) for w in (
    "ستراتیژی", "ستراتیژییەکەم", "ستراتیژییەکە", "ستراتیژییەکان", "تیۆری", "تیۆرییەکە", "کتێب", "کتێبەکەم",
    "کتێبەکانم", "کتێبەکە", "تێبینییەکانم", "strategy", "strategies", "theory", "book", "books", "notes"))
_QUESTION_WORDS = frozenset(normalize_ckb(w) for w in (
    "چی", "چییە", "چیە", "چۆن", "چۆنە", "بۆچی", "کەی", "کام", "کامە", "کوێ", "لەکوێ", "ئایا", "مانای", "واتای",
    "باسی", "ڕوونی", "ڕوون", "بڵێ", "پێم", "فێرم", "what", "how", "why", "when", "which", "where", "explain",
    "should", "tell", "does", "is", "are", "can"))
BLOCK_RULES = {
    "voice": ("If a passage answers it, answer from it in your own short words and name the book once by its "
              "title in a few words (never page numbers: the panel shows them). Do not call knowledge_search "
              "again for this question."),
    "text": ("If a passage answers it, answer from it and cite it as «title», page N. Do not call "
             "knowledge_search again for this question."),
}
SOURCES_LABEL_CKB = "پەیوەندیدار لە کتێبخانەکەت:"


def _generic_group(concept: Any) -> bool:
    variants = set(getattr(concept, "variants", ()) or ())
    return bool(variants & _GENERIC)


def is_library_question(text: str) -> bool:
    """A question about a trading idea, strategy, theory or the user's books."""
    words = normalize_ckb(text or "", strip_punct=True).split()
    if not words:
        return False
    asks = "؟" in text or "?" in text or any(w in _QUESTION_WORDS for w in words)
    if not asks:
        return False
    if any(w in _TOPIC_WORDS for w in words):
        return True
    try:
        from ..knowledge.query import analyse
    except Exception:  # noqa: BLE001 - no library package
        return False
    return any(c.glossary and not _generic_group(c) for c in analyse(text))


async def library_block(app: Any, text: str, mode: str) -> tuple[str, list[dict[str, Any]]]:
    """(prompt block, passages) for ``text``; ("", []) when the library has
    nothing relevant, is empty, missing or switched off."""
    library = getattr(app, "knowledge", None)
    if library is None or not app.config.get("conversation.library.enabled", True):
        return "", []
    try:
        if not library.has_documents() or not is_library_question(text):
            return "", []
        from ..knowledge import context_for_prompt, passages_for

        get = app.config.get
        passages = await passages_for(app, text, k=int(get("conversation.library.k", 3) or 3),
                                      max_chars=int(get("conversation.library.max_chars", 1500) or 1500),
                                      min_score=float(get("conversation.library.min_score", 0.5) or 0.5))
    except Exception:  # noqa: BLE001 - the library must never cost the user an answer
        log.warning("library context failed", exc_info=True)
        return "", []
    if not passages:
        return "", []
    block = context_for_prompt(passages) + "\n" + BLOCK_RULES["voice" if mode == "voice" else "text"]
    return app.redact(block), passages


def sources_line(passages: list[dict[str, Any]]) -> str:
    """«پەیوەندیدار لە کتێبخانەکەت: «Title»، لاپەڕە ١٢؛ ...» for the panel (never spoken)."""
    seen: list[str] = []
    for passage in passages:
        cite = str(passage.get("citation_ckb") or passage.get("citation") or "").strip()
        if cite and cite not in seen:
            seen.append(cite)
    return f"{SOURCES_LABEL_CKB} " + "؛ ".join(seen) if seen else ""


def add_to_messages(messages: list[dict[str, Any]], block: str) -> None:
    """Append the block to the system message (its per-turn part, after the
    persona's CONTEXT_HEADING: the local brain's prompt cache keeps the
    stable part)."""
    if messages and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": f"{messages[0].get('content', '')}\n\n{block}"}
    else:
        messages.insert(0, {"role": "system", "content": block})


__all__ = ["DEFAULTS", "SOURCES_LABEL_CKB", "add_to_messages", "is_library_question", "library_block",
           "sources_line"]
