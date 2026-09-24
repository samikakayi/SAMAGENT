"""Persistent cache of synthesized SHORT fixed phrases (cascade TTS).

Why: measured on this PC 2026-09-24 (acceptance/voice_live.py, KurdishTTS
STT + brain + KurdishTTS TTS on a 2.5 s Sorani clip): STT 2.9 s, tool pick
1.3 s, then the brain's acknowledgement "یەک چرکە." took 2.0 s to its first
KurdishTTS audio -> time to first audio 6.3 s against the 4.5 s target. The
brain speaks one of eight fixed acknowledgements (sam/brain/conversation.py
ACKS_DO/ACKS_LOOK) at the start of every tool-using voice turn, so replaying
them from here removes that 2 s and the characters they cost on KurdishTTS's
20,000-a-month free plan.

Only REGISTERED phrases of <= ``MAX_CHARS`` are cached (the brain's
acknowledgements and the voice package's fixed sentences). The first version
cached every short reply sentence; the live run then stored one-off
sentences such as a spoken gold price (~180 KB each), which never repeat --
so ordinary replies are never cached. Keys are provider + voice + exact text
(punctuation changes the intonation, so it is kept). What is stored is SAM's
OWN synthesized voice, never microphone audio. Rows live in
``voice_tts_cache`` (namespace ``voice``), pruned to the ``MAX_ROWS`` most
recently used.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any

log = logging.getLogger("sam.voice.tts")

MAX_CHARS = 120
MAX_ROWS = 64
SCHEMA = [(1, """CREATE TABLE IF NOT EXISTS voice_tts_cache (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    voice TEXT NOT NULL,
    text TEXT NOT NULL,
    pcm BLOB NOT NULL,
    rate INTEGER NOT NULL DEFAULT 24000,
    created_at REAL NOT NULL,
    last_used_at REAL NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0)""")]

_SPACES = re.compile(r"\s+")


def ensure_schema(db: Any) -> None:
    db.ensure_schema("voice", SCHEMA)


def clean_phrase(text: str) -> str:
    return _SPACES.sub(" ", text or "").strip()


def phrase_key(provider: str, voice: str, text: str) -> str:
    clean = clean_phrase(text)
    return hashlib.sha256(f"{provider}\x1f{voice}\x1f{clean}".encode("utf-8")).hexdigest()[:32]


class PhraseCache:
    """get/put 24 kHz PCM of short phrases. Never raises: a broken cache only
    costs the synthesis it would have saved."""

    def __init__(self, db: Any, *, max_chars: int = MAX_CHARS, max_rows: int = MAX_ROWS) -> None:
        self.db = db
        self.max_chars = max_chars
        self.max_rows = max_rows
        self.hits = 0
        self.misses = 0
        self._ready = False
        self._memory: dict[str, bytes] = {}
        self._phrases: set[str] = set()

    def register(self, phrases: Any) -> int:
        """Declare fixed phrases worth caching; returns how many are accepted."""
        added = 0
        for text in phrases or ():
            clean = clean_phrase(str(text))
            if 0 < len(clean) <= self.max_chars and any(ch.isalnum() for ch in clean):
                self._phrases.add(clean)
                added += 1
        return added

    def cacheable(self, text: str) -> bool:
        return clean_phrase(text) in self._phrases

    def _ensure(self) -> bool:
        if not self._ready and self.db is not None:
            try:
                ensure_schema(self.db)
                self._ready = True
            except Exception:  # noqa: BLE001
                log.debug("tts cache schema failed", exc_info=True)
        return self._ready

    def get(self, provider: str, voice: str, text: str) -> bytes | None:
        if not self.cacheable(text):
            return None
        key = phrase_key(provider, voice, text)
        pcm = self._memory.get(key)
        if pcm is None and self._ensure():
            try:
                row = self.db.query_one("SELECT pcm FROM voice_tts_cache WHERE key=?", (key,))
                if row is not None and row.get("pcm"):
                    pcm = bytes(row["pcm"])
                    self._remember(key, pcm)
            except Exception:  # noqa: BLE001
                log.debug("tts cache read failed", exc_info=True)
        if pcm is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            self.db.execute("UPDATE voice_tts_cache SET last_used_at=?, uses=uses+1 WHERE key=?", (time.time(), key))
        except Exception:  # noqa: BLE001
            pass
        return pcm

    def put(self, provider: str, voice: str, text: str, pcm: bytes, rate: int = 24_000) -> bool:
        if not pcm or not self.cacheable(text) or not self._ensure():
            return False
        key = phrase_key(provider, voice, text)
        now = time.time()
        try:
            with self.db.transaction():
                self.db.execute(
                    "INSERT OR REPLACE INTO voice_tts_cache (key, provider, voice, text, pcm, rate, created_at, "
                    "last_used_at, uses) VALUES (?,?,?,?,?,?,?,?,0)",
                    (key, provider, voice, clean_phrase(text), bytes(pcm), rate, now, now))
                self.db.execute(
                    "DELETE FROM voice_tts_cache WHERE key NOT IN (SELECT key FROM voice_tts_cache "
                    "ORDER BY last_used_at DESC, rowid DESC LIMIT ?)", (self.max_rows,))
        except Exception:  # noqa: BLE001
            log.debug("tts cache write failed", exc_info=True)
            return False
        self._remember(key, bytes(pcm))
        return True

    def _remember(self, key: str, pcm: bytes) -> None:
        self._memory[key] = pcm
        while len(self._memory) > self.max_rows:  # same bound as the table
            self._memory.pop(next(iter(self._memory)))

    def has(self, provider: str, voice: str, text: str) -> bool:
        key = phrase_key(provider, voice, text)
        if key in self._memory:
            return True
        if not self._ensure():
            return False
        try:
            return self.db.query_one("SELECT 1 AS x FROM voice_tts_cache WHERE key=?", (key,)) is not None
        except Exception:  # noqa: BLE001
            return False

    def status(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "phrases": len(self._phrases)}


__all__ = ["PhraseCache", "phrase_key", "clean_phrase", "ensure_schema", "SCHEMA", "MAX_CHARS", "MAX_ROWS"]
