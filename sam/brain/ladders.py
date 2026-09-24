"""Which models answer a conversation round, in what order, with what deadlines.

Measured on this PC (2026-09-24; brain builder, integration smoke, repair
review -- reports and ``review2-voice-brain-speed``):

- Groq gpt-oss-20b/120b pick the right tool for Sorani commands in 0.75-1.4 s,
  but their Sorani wording is broken (Arabic words, «یه‌کێ», the user's
  question echoed back, the same greeting twice). They have their own token
  bucket (8k tokens/minute), so they are the fast *tool picker* and the last
  resort for wording -- never the first choice for words the user hears.
- OmniRoute's combo (sam-fast) and its Gemini models word naturally when they
  answer (1-5 s), but while SAM v1 shares OmniRoute they returned 429 (sam-fast
  15 of 38 requests failed today) or nothing until the 15 s cap: two such rungs
  made the first typed reply take 24-39 s. Their first reply is capped at
  ``OMNIROUTE_CAP_S`` and failing rungs rest longer each time (``LLMClient``).
- Gemini direct (gemini-3.5-flash-lite, then 3.1-flash-lite, ~500 requests a
  day free each) is used first whenever the user has pasted a Gemini key: it
  picks tools and words Sorani in one call, so no rewording is needed. On
  2026-09-24 both were overloaded (3.5: all 9 requests 503 or a 30 s timeout in
  sam2.log; repair probe: 20 s timeout, then 503 twice on 3.1), so they are
  capped at ``GEMINI_CAP_S``, not retried within a turn, and demoted by health.

Every ladder is re-ordered by live health (``LLMClient.healthy_order``): a
rung that failed recently moves behind the ones that did not.

Settings ``conversation.ladder.voice/reply/text`` stay overrides: "auto" (the
default) builds the lists below per turn; a list or a core ladder name is used
as given (still health-ordered).
"""

from __future__ import annotations

from typing import Any

GEMINI_DIRECT = "gemini:gemini-3.5-flash-lite"
# A second direct Gemini model with its own free quota: 3.5-flash-lite answered
# "503 Service Unavailable" to all 9 requests on 2026-09-24 (sam2.log).
GEMINI_DIRECT_2 = "gemini:gemini-3.1-flash-lite"
GROQ_20B = "groq:openai/gpt-oss-20b"
GROQ_120B = "groq:openai/gpt-oss-120b"
FAST_TOOL_PICKERS = [GROQ_20B, GROQ_120B]
OMNIROUTE_GEMINI = ["omniroute:gemini/gemini-3-flash-preview", "omniroute:gemini/gemini-3.1-flash-lite"]
OMNIROUTE_CAP_S = 6.0         # good OmniRoute answers came in 1-5 s; busy ones failed after 15-40 s
GEMINI_CAP_S = 7.0            # repair probe 2026-09-24: 3.5-flash-lite timed out at 20 s, 3.1 said 503 twice
PICKER_DEADLINE_S = 14.0      # the first round of a turn, all rungs together
WORDING_DEADLINE_S = 8.0      # a round after a tool result, or a rewording: then the tool's own outcome
REWORD_DEADLINE_S = 5.0       # small talk worded again by a better model; else Groq's own text is used


def _omniroute(config: Any) -> list[str]:
    """The user's OmniRoute fast combo (first rung of the core 'sorani' ladder)
    plus the Gemini models OmniRoute lists directly."""
    sorani = list(config.get("llm.ladder.sorani") or [])
    fast = [ref for ref in sorani if ref.startswith("omniroute:")][:1] or ["omniroute:sam-fast"]
    return fast + OMNIROUTE_GEMINI


def has_gemini(app: Any) -> bool:
    try:
        backend = app.llm.backends.get("gemini")
        return bool(backend is not None and backend.configured())
    except Exception:  # noqa: BLE001
        return False


def _gemini(app: Any) -> list[str]:
    return [GEMINI_DIRECT, GEMINI_DIRECT_2] if has_gemini(app) else []


def auto_picker(app: Any, mode: str) -> list[str]:
    """First round of a turn (chooses tools; small talk is answered here)."""
    groq = [GROQ_20B, GROQ_120B] if mode == "voice" else [GROQ_120B, GROQ_20B]
    return _gemini(app) + groq + _omniroute(app.config)


def auto_wording(app: Any) -> list[str]:
    """Rounds after a tool result and rewording: quality first, Groq last."""
    return _gemini(app) + _omniroute(app.config) + [GROQ_120B, GROQ_20B]


def resolve(app: Any, value: Any, fallback: list[str]) -> list[str]:
    """A setting value -> refs. "auto"/empty -> ``fallback``; a list may mix
    refs and core ladder names ("sorani")."""
    if value in (None, "", "auto"):
        return list(fallback)
    items = value if isinstance(value, (list, tuple)) else [value]
    refs: list[str] = []
    for item in items:
        try:
            expanded = [str(item)] if ":" in str(item) else app.llm.ladder(str(item))
        except ValueError:
            expanded = []
        refs.extend(ref for ref in expanded if ref not in refs)
    return refs or list(fallback)


def ordered(app: Any, refs: list[str]) -> list[str]:
    order = getattr(app.llm, "healthy_order", None)
    return order(refs) if callable(order) else list(refs)


def rung_caps() -> dict[str, float]:
    return {"omniroute": OMNIROUTE_CAP_S, "gemini": GEMINI_CAP_S}


def is_fast_picker(ref: str) -> bool:
    return ref.startswith("groq:")


__all__ = ["auto_picker", "auto_wording", "resolve", "ordered", "rung_caps", "is_fast_picker", "has_gemini",
           "FAST_TOOL_PICKERS", "GEMINI_DIRECT", "PICKER_DEADLINE_S", "WORDING_DEADLINE_S", "REWORD_DEADLINE_S"]
