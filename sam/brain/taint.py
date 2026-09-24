"""Per-turn taint: after untrusted text entered a turn, risky follow-ups ask first.

Web pages, screen text (OCR/UIA), files and chart labels come back to the model
under ``untrusted`` (docs/CONTRACTS.md 0). The persona tells the model never to
follow them, but a prompt is not a guarantee: the repair review (2026-09-24)
found that after one injected page, safe-tier tools could still send data out
(open_url / fetch_page with a query string, web_search) or plant a "fact" that
enters every later prompt (remember). So the code tracks it:

- a *scope* is one user turn (typed, cascade or Live) or one worker task;
  ``begin(user_text)`` opens it (a ContextVar, so concurrent turns and the
  worker's own task each see their own state);
- ``ToolRegistry.dispatch`` marks the scope tainted when a result carries
  ``untrusted`` data, and before a SAFE call in a tainted scope asks
  ``check(name, args, state)`` whether it now needs the user's yes.

What still runs freely while tainted, on purpose: reading more (screen_look,
recall, get_price, chart tools), clicks without danger words and typing
without Enter -- the desktop flow is "screen_look, then click by number",
and a screen read would otherwise make every click a question. Clicks on
danger words, Enter in chat apps, deletions and shell changes already ask.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from ..textnorm import normalize_ckb


@dataclass
class TaintState:
    user_text: str = ""
    tainted: bool = False
    sources: list[str] = field(default_factory=list)

    def mark(self, tool_name: str) -> None:
        self.tainted = True
        if tool_name not in self.sources:
            self.sources.append(tool_name)


_SCOPE: contextvars.ContextVar[TaintState | None] = contextvars.ContextVar("sam_taint_scope", default=None)


def begin(user_text: str = "", *, inherit: bool = False) -> TaintState:
    """Open a new scope in the current context (a turn or a worker task).
    ``inherit`` keeps the taint of the scope that started it (a worker task
    delegated from a tainted turn starts tainted)."""
    parent = _SCOPE.get()
    state = TaintState(user_text=user_text or "")
    if inherit and parent is not None and parent.tainted:
        state.tainted = True
        state.sources = list(parent.sources)
    _SCOPE.set(state)
    return state


def use(state: TaintState | None) -> None:
    """Re-enter an open scope. The cascade iterates ``respond_stream`` through
    a new asyncio task per chunk (``CascadeVoice._deltas``), and a ContextVar
    set in one of those tasks is not seen by the next, so the turn re-enters
    its scope right before it dispatches tools."""
    _SCOPE.set(state)


def current() -> TaintState | None:
    return _SCOPE.get()


def carries_untrusted(result: Any) -> bool:
    data = result.get("data") if isinstance(result, dict) else None
    return isinstance(data, dict) and bool(data.get("untrusted"))


def _host_named(url: str, user_text: str) -> bool:
    host = (urlparse(str(url or "").strip()).hostname or "").lower()
    if not host:
        return False
    said = normalize_ckb(user_text)
    bare = host.removeprefix("www.")
    return bare in said or bare.split(".")[0] in said.split()


def _words_from_user(query: str, user_text: str) -> bool:
    said = set(normalize_ckb(user_text, strip_punct=True).split())
    words = [w for w in normalize_ckb(query, strip_punct=True).split() if len(w) > 2]
    return bool(words) and all(w in said for w in words)


_WRITE_FILES = frozenset({"write", "append", "copy", "move", "rename"})


def check(name: str, args: dict[str, Any], state: TaintState) -> tuple[str, str] | None:
    """(risk, question) when ``name(args)`` must be confirmed in a tainted
    scope, else None. Questions never quote model or page text (it is read
    aloud); the arguments are on the confirmation card's detail line."""
    if not state.tainted:
        return None
    after = "دوای خوێندنەوەی دەقێکی دەرەکی"
    if name in ("open_url", "fetch_page"):
        url = str(args.get("url") or "")
        if _host_named(url, state.user_text):
            return None
        host = urlparse(url).hostname or "?"
        return "confirm", f"{after}، ماڵپەڕی {host} بکەمەوە؟"
    if name == "web_search":
        if _words_from_user(str(args.get("query") or ""), state.user_text):
            return None
        return "confirm", f"{after}، گەڕانێکی نوێ بکەم؟ دەقی گەڕانەکە لەسەر شاشەیە."
    if name in ("remember", "strategy_save"):
        return "confirm", f"{after}، زانیارییەکی نوێ لە بیرەوەری پاشەکەوت بکەم؟ دەقەکە لەسەر شاشەیە."
    if name == "files" and str(args.get("action") or "").lower() in _WRITE_FILES:
        return "confirm", f"{after}، ئەم گۆڕانکارییە لە فایلەکاندا بکەم؟"
    if name == "type_text" and args.get("press_enter"):
        return "confirm", f"{after}، دەقەکە بنووسم و ئینتەر دابگرم؟"
    if name in ("delegate_task", "build_project", "screen_act"):
        return "confirm", f"{after}، ئەم ئەرکە دەست پێ بکەم؟ وردەکارییەکەی لەسەر شاشەیە."
    return None


__all__ = ["TaintState", "begin", "use", "current", "check", "carries_untrusted"]
