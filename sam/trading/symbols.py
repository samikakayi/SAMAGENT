"""One instrument resolver for every trading tool, and "did the user name it?".

Measured problems it fixes (repair review, 2026-09-24):

- The chart bridge used ``tv_parse.canonical_request`` (strips Sorani suffixes)
  but get_price / set_alert / analyze_market used ``canonical_symbol``, which
  does not: «زێڕەکە» reached MetaTrader as the Sorani word itself. Neither knew
  the user's own «گۆڵت», the letters of XAUUSD as speech-to-text wrote them
  («خاو یوئێزدی»), or near-misses the model invents ('XUUSD').
- Groq transliterated Sorani names into Latin "symbols" (ZAR, ZEUR, ZIW, NAWT,
  Bitkoin, altun): only the gold spellings were mapped.
- A one-letter "symbol" the model made up ('z', for a request that named no
  instrument at all) went to TradingView as a real ticker and switched the
  user's chart to a Nasdaq stock (BATS:Z). ``user_named`` lets the chart tool
  refuse a symbol the user never said.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..textnorm import normalize_ckb
from .common import SYMBOL_ALIASES, TV_SYMBOL_EQUIVALENTS, canonical_symbol

SUFFIXES = ("یەکە", "ەکان", "ەکە", "یەک", "ێک", "انە", "ی", "ە")
EXTRA_ALIASES: dict[str, str] = {normalize_ckb(k): v for k, v in {
    # the user's own spoken forms
    "گۆڵت": "XAUUSD", "گۆڵتی": "XAUUSD", "گولت": "XAUUSD", "گۆلت": "XAUUSD", "گۆڵدی": "XAUUSD",
    "زێڕ و دۆلار": "XAUUSD", "gold spot": "XAUUSD", "spot gold": "XAUUSD", "xau usd": "XAUUSD",
    "ئیتریۆم": "ETHUSD", "ئیسریۆم": "ETHUSD", "ئێسریۆم": "ETHUSD", "ئەسریۆم": "ETHUSD", "ئیسریوم": "ETHUSD",
    "یۆرۆ دۆلار": "EURUSD", "eur usd": "EURUSD", "داو جۆنز": "US30", "dow": "US30", "dow jones": "US30",
    "us30": "US30", "نەوت": "USOIL", "زیو": "XAGUSD",
}.items()}
# XAUUSD spoken letter by letter, as KurdishTTS STT wrote it on 2026-09-24 («خاو یوئێزدی») and similar.
SPELLED: dict[str, str] = {normalize_ckb(k).replace(" ", ""): v for k, v in {
    "خاو یوئێزدی": "XAUUSD", "خاو یو ئێس دی": "XAUUSD", "زاو یو ئێس دی": "XAUUSD", "ئێکس ئەی یو یو ئێس دی": "XAUUSD",
    "ئێکس ئەی یو": "XAUUSD", "ئێکس ئەیو یوئێسدی": "XAUUSD", "خاویوئێسدی": "XAUUSD", "ئێکس ئەی جی یو ئێس دی": "XAGUSD",
    "بی تی سی": "BTCUSD", "ئی یو ئاڕ یو ئێس دی": "EURUSD",
}.items()}
# Latin spellings of Sorani names that models send as a "symbol".
LATIN: dict[str, str] = {
    "ziw": "XAGUSD", "ziv": "XAGUSD", "zew": "XAGUSD", "zeew": "XAGUSD",
    "nawt": "USOIL", "newt": "USOIL", "naft": "USOIL", "nafte": "USOIL", "nawet": "USOIL",
    "bitkoin": "BTCUSD", "bitkoyn": "BTCUSD", "bitcoyn": "BTCUSD", "bitkon": "BTCUSD",
    "altun": "XAUUSD", "altin": "XAUUSD", "altoon": "XAUUSD", "altwn": "XAUUSD", "gold": "XAUUSD",
    "zer": "XAUUSD", "zar": "XAUUSD", "zeur": "XAUUSD",
}
# Names a near-miss (edit distance 1, >= 4 letters) is mapped to.
FUZZY_NAMES: dict[str, str] = {
    "xauusd": "XAUUSD", "xagusd": "XAGUSD", "btcusd": "BTCUSD", "ethusd": "ETHUSD", "eurusd": "EURUSD",
    "gbpusd": "GBPUSD", "usdjpy": "USDJPY", "usoil": "USOIL", "nas100": "NAS100", "us100": "NAS100",
    "usdx": "USDX", "gold": "XAUUSD", "silver": "XAGUSD", "bitcoin": "BTCUSD", "ethereum": "ETHUSD",
    "nasdaq": "NAS100", "btcusdt": "BTCUSD", "ethusdt": "ETHUSD",
}
KNOWN: frozenset[str] = frozenset(set(SYMBOL_ALIASES.values()) | set(TV_SYMBOL_EQUIVALENTS.values())
                                  | set(FUZZY_NAMES.values()) | {"GBPUSD", "USDJPY", "US30", "USOIL"})
_EXPLICIT = re.compile(r"^([A-Za-z0-9_]{1,24}):([A-Za-z0-9_.!/-]{1,32})$")
_GOLD_LATIN = re.compile(r"^zh?[aeêiîuûy]{1,3}r{1,2}$")


def _edit1(a: str, b: str) -> bool:
    """Levenshtein distance <= 1."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _alias(word: str) -> str | None:
    return EXTRA_ALIASES.get(word) or SYMBOL_ALIASES.get(word) or SPELLED.get(word.replace(" ", ""))


def _token_hits(tokens: list[str]) -> set[str]:
    hits: set[str] = set()
    for token in tokens:
        for form in [token] + [token[: -len(s)] for s in SUFFIXES if token.endswith(s) and len(token) > len(s) + 1]:
            target = _alias(form)
            if target:
                hits.add(target)
                break
    return hits


def resolve_instrument(text: str | None) -> str | None:
    """Spoken/typed/model-sent instrument -> canonical symbol ('زێڕەکە',
    'گۆڵت', 'خاو یوئێزدی', 'XUUSD', 'ZIW', 'OANDA:XAUUSD', 'BINANCE:BTCUSDT'),
    or None when it is not an instrument SAM knows (never guesses from free
    text or from one or two letters)."""
    raw = (text or "").strip()
    if not raw:
        return None
    if _EXPLICIT.match(raw):
        bare = canonical_symbol(raw)
        return bare if bare in KNOWN else None
    norm = normalize_ckb(raw)
    stripped = normalize_ckb(raw, strip_punct=True)
    for candidate in (norm, stripped, stripped.replace(" ", "")):
        target = _alias(candidate)
        if target:
            return target
    hits = _token_hits(stripped.split())
    if len(hits) == 1:
        return hits.pop()
    compact = re.sub(r"[\s/_\-.]", "", raw).lower()
    if compact in LATIN:
        return LATIN[compact]
    if _GOLD_LATIN.match(compact):
        return "XAUUSD"
    upper = canonical_symbol(raw)
    if upper in KNOWN:
        return upper
    if len(compact) >= 4 and compact.isascii():
        near = {v for k, v in FUZZY_NAMES.items() if _edit1(compact, k)}
        if len(near) == 1:
            return near.pop()
    return None


def same_instrument(a: str | None, b: str | None) -> bool:
    """'BINANCE:BTCUSDT' ~ 'BTCUSD', 'OANDA:NAS100USD' ~ 'NAS100', 'TVC:GOLD' ~ 'XAUUSD'."""
    if not a or not b:
        return False
    return instrument(a) == instrument(b)


def instrument(symbol: str) -> str:
    known = resolve_instrument(symbol)
    if known:
        return known
    key = canonical_symbol(re.sub(r"[/\s]", "", symbol or ""))
    return key[:-1] if key.endswith("USDT") and len(key) > 4 else key


# --- did the user name it? ---------------------------------------------------------------------------------------
def recent_user_text(app: Any, *, turns: int = 3, within_s: float = 600.0) -> str | None:
    """The user's last few utterances of the open conversation (None when there
    is no conversation to check against, e.g. a worker task or a test)."""
    conversation = getattr(app, "conversation", None)
    memory = getattr(app, "memory", None)
    conversation_id = getattr(conversation, "conversation_id", None)
    if memory is None or conversation_id is None:
        return None
    try:
        rows = memory.recent_turns(conversation_id, limit=turns, roles=("user",))
    except Exception:  # noqa: BLE001
        return None
    now = time.time()
    texts = [str(r.get("text") or "") for r in rows if now - float(r.get("at") or now) <= within_s]
    return " ".join(texts) if texts else None


def mentioned_instruments(text: str) -> set[str]:
    """Every instrument named in ``text`` (single words and 2-3 word runs)."""
    words = normalize_ckb(text, strip_punct=True).split()
    found: set[str] = set()
    for size in (1, 2, 3, 6):
        for i in range(0, max(0, len(words) - size + 1)):
            target = resolve_instrument(" ".join(words[i:i + size]))
            if target:
                found.add(target)
    return found


def user_named(app: Any, symbol: str, source: str = "") -> bool | None:
    """True when the user's recent words name this instrument (or the symbol
    literally); False when they do not; None when there is nothing to check
    (worker/UI calls, no conversation)."""
    if source in ("worker", "ui"):
        return None
    said = recent_user_text(app)
    if said is None:
        return None
    wanted = resolve_instrument(symbol)
    if wanted and wanted in mentioned_instruments(said):
        return True
    literal = re.sub(r"[^a-z0-9]", "", str(symbol).split(":")[-1].lower())
    spoken = re.sub(r"[^a-z0-9]", "", normalize_ckb(said))
    return bool(literal) and len(literal) >= 3 and literal in spoken


__all__ = ["resolve_instrument", "same_instrument", "instrument", "user_named", "recent_user_text",
           "mentioned_instruments", "KNOWN"]
