"""Credential storage (Windows DPAPI) and secret redaction.

Compatibility contract with SAM v1 (sam_backend/secrets.py, tag v1-final):
the SAME file ``<data_dir>/secrets.json``, the SAME ``{"_format": "dpapi-v1",
"data": base64(CryptProtectData(json))}`` layout and the SAME key names, so every
key the user already entered keeps working without being touched.

Differences from v1, all deliberate:
- Opening the store never writes it. v1 re-encrypted a legacy plaintext file on
  start-up; SAM 2 only rewrites the file when the user saves or clears a key
  (the file is shared with v1 while both exist).
- Gemini accepts the new ``AQ.`` auth-key shape as well as ``AIza...``.
- Values are resolved process env -> ``.env`` -> store, but ``.env`` values are
  never copied into ``os.environ`` (child processes such as PowerShell or VS Code
  must not inherit keys).
- ``redact()`` masks every provider key shape plus the exact values currently
  stored, and is installed as a logging filter.

No function here ever returns or logs a value except ``Secrets.get`` /
``SecretStore.get``, whose callers send it only to its own provider.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import logging
import os
import re
import stat
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

SECRETS_FILENAME = "secrets.json"
DPAPI_FORMAT = "dpapi-v1"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _dpapi(call: str, payload: bytes) -> bytes | None:
    """One DPAPI round trip (current user scope), or None when unavailable."""
    if os.name != "nt":
        return None
    try:
        crypt = ctypes.windll.crypt32
        buffer = ctypes.create_string_buffer(payload, len(payload))
        source = _Blob(len(payload), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
        result = _Blob()
        function = crypt.CryptProtectData if call == "protect" else crypt.CryptUnprotectData
        # CRYPTPROTECT_UI_FORBIDDEN (0x1): never prompt.
        if not function(ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(result)):
            return None
        try:
            return ctypes.string_at(result.pbData, result.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(result.pbData)
    except Exception:  # noqa: BLE001 - any OS failure means "not available"
        return None


# Shapes only -- enough to reject an obvious paste error before a network call.
# Names and patterns are v1's, plus the Gemini ``AQ.`` auth-key shape.
KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    "openrouter_api_key": re.compile(r"^sk-or-[A-Za-z0-9._\-]{20,200}$"),
    "openai_api_key": re.compile(r"^sk-[A-Za-z0-9._\-]{20,200}$"),
    "litellm_api_key": re.compile(r"^[A-Za-z0-9._\-]{8,200}$"),
    "groq_api_key": re.compile(r"^gsk_[A-Za-z0-9._\-]{20,200}$"),
    "gemini_api_key": re.compile(r"^(?:AIza[A-Za-z0-9._\-]{20,200}|AQ\.[A-Za-z0-9._\-]{20,400})$"),
    "n8n_api_key": re.compile(r"^[A-Za-z0-9._\-]{20,600}$"),
    "kurdishtts_stt_api_key": re.compile(r"^[A-Za-z0-9._\-]{16,200}$"),
    "kurdishtts_tts_api_key": re.compile(r"^[A-Za-z0-9._\-]{16,200}$"),
    "google_stt_credentials_path": re.compile(r"^[^\r\n]{3,400}$"),
}
SUPPORTED_KEYS: tuple[str, ...] = tuple(KEY_PATTERNS)

# Where a key may also come from outside the store (process env / .env).
# The first name found wins. LITELLM_* is the OmniRoute client key (.env).
ENV_NAMES: dict[str, tuple[str, ...]] = {
    "litellm_api_key": ("LITELLM_API_KEY", "LITELLM_MASTER_KEY"),
    "openrouter_api_key": ("OPENROUTER_API_KEY",),
    "openai_api_key": ("OPENAI_API_KEY",),
    "groq_api_key": ("GROQ_API_KEY",),
    "gemini_api_key": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "kurdishtts_stt_api_key": ("KURDISHTTS_STT_API_KEY",),
    "kurdishtts_tts_api_key": ("KURDISHTTS_TTS_API_KEY",),
    "n8n_api_key": ("N8N_API_KEY",),
    "google_stt_credentials_path": ("GOOGLE_STT_CREDENTIALS_PATH",),
}


class SecretStore:
    """The v1-compatible DPAPI file. Thread-safe; never logs a value."""

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / SECRETS_FILENAME
        self._lock = threading.RLock()

    # -- file format -------------------------------------------------------
    def _read(self) -> dict[str, str]:
        with self._lock:
            if not self.path.is_file():
                return {}
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                return {}
            if not isinstance(payload, dict):
                return {}
            if payload.get("_format") == DPAPI_FORMAT:
                try:
                    blob = base64.b64decode(payload.get("data", ""))
                except (ValueError, TypeError):
                    return {}
                plain = _dpapi("unprotect", blob)
                if plain is None:
                    # Another account's file, or DPAPI is blocked: refuse
                    # rather than return half-decoded material.
                    return {}
                try:
                    payload = json.loads(plain.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return {}
                if not isinstance(payload, dict):
                    return {}
            return {k: str(v) for k, v in payload.items() if isinstance(v, str) and not k.startswith("_")}

    def _write(self, values: dict[str, str]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            blob = _dpapi("protect", json.dumps(values).encode("utf-8"))
            document: dict[str, Any] = (
                {"_format": DPAPI_FORMAT, "data": base64.b64encode(blob).decode("ascii")}
                if blob is not None else values
            )
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
            try:
                self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
            self._restrict_windows_acl()

    def _restrict_windows_acl(self) -> None:
        """Owner-only ACL (chmod is a no-op for other accounts on Windows)."""
        if os.name != "nt":
            return
        account = os.environ.get("USERNAME")
        if not account:
            return
        domain = os.environ.get("USERDOMAIN")
        principal = f"{domain}\\{account}" if domain else account
        try:
            subprocess.run(
                ["icacls", str(self.path), "/inheritance:r", "/grant:r", f"{principal}:F"],
                capture_output=True, timeout=15, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            pass

    # -- public API --------------------------------------------------------
    def encrypted_at_rest(self) -> bool:
        with self._lock:
            if not self.path.is_file():
                return _dpapi("protect", b"probe") is not None
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                return False
            return isinstance(payload, dict) and payload.get("_format") == DPAPI_FORMAT

    def storage_status(self) -> dict[str, Any]:
        enc = self.encrypted_at_rest()
        return {"encrypted_at_rest": enc, "mechanism": "windows-dpapi" if enc else "owner-only-file",
                "path": str(self.path)}

    @staticmethod
    def fingerprint(value: str | None) -> str | None:
        """Stable, non-reversible id so two keys can be told apart in the UI."""
        if not value:
            return None
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]

    def get(self, name: str) -> str | None:
        return self._read().get(name)

    def names(self) -> list[str]:
        return sorted(self._read())

    def values(self) -> list[str]:
        """All stored values -- only for exact-value redaction, never output."""
        return [v for v in self._read().values() if v]

    def set(self, name: str, value: str) -> dict[str, Any]:
        if name not in SUPPORTED_KEYS:
            raise ValueError(f"Unsupported credential: {name}")
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("The credential is empty.")
        if not KEY_PATTERNS[name].match(cleaned):
            # The message never contains the value.
            raise ValueError(f"That does not look like a {name.replace('_', ' ')}. "
                             "Check for a truncated paste or stray whitespace.")
        with self._lock:
            values = self._read()
            values[name] = cleaned  # other entries (even unknown ones) are preserved
            self._write(values)
        return {"name": name, "stored": True, "fingerprint": self.fingerprint(cleaned)}

    def clear(self, name: str) -> bool:
        with self._lock:
            values = self._read()
            if name not in values:
                return False
            values.pop(name)
            self._write(values)
            return True

    def public_status(self) -> dict[str, Any]:
        values = self._read()
        return {n: {"configured": n in values, "fingerprint": self.fingerprint(values.get(n))}
                for n in SUPPORTED_KEYS}


class Secrets:
    """Resolution facade used by the whole app: env -> .env -> DPAPI store.

    ``env_lookup(NAME)`` is ``Config.env_value`` (process env, then .env);
    it is injected so tests can use fake environments.
    """

    KNOWN_VALUES_TTL_S = 30.0

    def __init__(self, store: SecretStore, env_lookup: Callable[[str], str | None] | None = None) -> None:
        self.store = store
        self._env_lookup = env_lookup or (lambda name: os.environ.get(name) or None)
        self._known: tuple[float, list[str]] | None = None

    def _env(self, name: str) -> tuple[str | None, str | None]:
        for env_name in ENV_NAMES.get(name, (name.upper(),)):
            value = self._env_lookup(env_name)
            if value and value.strip():
                return value.strip(), env_name
        return None, None

    def get(self, name: str) -> str | None:
        """The effective value. Send it only to its own provider."""
        value, _ = self._env(name)
        return value or self.store.get(name)

    def source(self, name: str) -> str:
        value, _ = self._env(name)
        if value:
            return "environment"
        return "secret_store" if self.store.get(name) else "unset"

    def has(self, name: str) -> bool:
        return bool(self.get(name))

    def set(self, name: str, value: str) -> dict[str, Any]:
        """Save a key the USER pasted in Settings. Returns no value."""
        result = self.store.set(name, value)
        self._known = None
        _REDACTION.refresh(self)
        return result

    def clear(self, name: str) -> bool:
        self._known = None
        return self.store.clear(name)

    def status(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name in SUPPORTED_KEYS:
            value = self.get(name)
            out[name] = {"configured": bool(value), "source": self.source(name),
                         "fingerprint": SecretStore.fingerprint(value)}
        return out

    def known_values(self) -> list[str]:
        """Exact values to mask (cached briefly: each read is a DPAPI decrypt)."""
        import time as _time

        now = _time.monotonic()
        if self._known is not None and now - self._known[0] < self.KNOWN_VALUES_TTL_S:
            return self._known[1]
        vals = set(self.store.values())
        for name in SUPPORTED_KEYS:
            env_value, _ = self._env(name)
            if env_value:
                vals.add(env_value)
        result = [v for v in vals if len(v) >= 8]
        self._known = (now, result)
        return result

    def redact(self, text: str) -> str:
        return redact(text, extra_values=self.known_values())


# --- redaction -----------------------------------------------------------------

# Provider key shapes. Order matters: specific prefixes before generic runs.
_SECRET_SHAPES = re.compile(
    r"(?:sk-or-[A-Za-z0-9._\-]{16,}"            # OpenRouter
    r"|sk-ant-[A-Za-z0-9._\-]{16,}"              # Anthropic
    r"|sk-proj-[A-Za-z0-9._\-]{16,}"             # OpenAI project
    r"|sk-[A-Za-z0-9._\-]{20,}"                  # OpenAI classic / LiteLLM virtual keys
    r"|gsk_[A-Za-z0-9._\-]{16,}"                 # Groq
    r"|AIza[0-9A-Za-z._\-]{20,}"                 # Google API key
    r"|AQ\.[0-9A-Za-z._\-]{16,}"                 # Google auth key (new shape)
    r"|gh[pousr]_[A-Za-z0-9]{30,}"               # GitHub
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"            # Slack
    r"|AKIA[0-9A-Z]{16}"                         # AWS access key id
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"  # JWT
    r")"
)
_BEARER = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/\-=]{8,}")
_ASSIGNMENT = re.compile(
    r"(?i)((?:\"|')?[A-Za-z0-9_\-]*(?:api[_\-]?key|secret|token|password|passwd|x-goog-api-key|authorization)"
    r"[A-Za-z0-9_\-]*(?:\"|')?\s*[:=]\s*(?:\"|')?)([^\s\"',}]{6,})"
)
_LONG_HEX = re.compile(r"\b[0-9a-fA-F]{32,}\b")
# Long base64/base64url runs with mixed character classes ('/' excluded so URL
# paths survive).
_LONG_B64 = re.compile(r"(?<![A-Za-z0-9+_\-])[A-Za-z0-9+_\-]{40,}={0,2}(?![A-Za-z0-9+_\-])")
MASK = "[REDACTED]"


def _looks_random(token: str) -> bool:
    return (any(c.isdigit() for c in token) and any(c.isupper() for c in token)
            and any(c.islower() for c in token))


def redact_count(text: str, extra_values: Iterable[str] = ()) -> tuple[str, int]:
    """Mask secrets in ``text``; return (masked_text, number_of_masks)."""
    if not text:
        return text, 0
    count = 0
    result = str(text)
    for value in sorted({v for v in extra_values if v and len(v) >= 8}, key=len, reverse=True):
        if value in result:
            count += result.count(value)
            result = result.replace(value, MASK)

    def sub(pattern: re.Pattern[str], repl: Callable[[re.Match[str]], str], value: str) -> str:
        def _r(m: re.Match[str]) -> str:
            nonlocal count
            out = repl(m)
            if out != m.group(0):
                count += 1
            return out
        return pattern.sub(_r, value)

    result = sub(_SECRET_SHAPES, lambda m: MASK, result)
    result = sub(_BEARER, lambda m: m.group(1) + MASK, result)
    # Plain numbers after e.g. "tokens_in:" are counters, not secrets.
    result = sub(_ASSIGNMENT, lambda m: m.group(0) if (m.group(2) == MASK or re.fullmatch(r"[\d.,:\-]+", m.group(2)))
                 else m.group(1) + MASK, result)
    result = sub(_LONG_HEX, lambda m: MASK, result)
    result = sub(_LONG_B64, lambda m: MASK if _looks_random(m.group(0)) else m.group(0), result)
    return result, count


def redact(text: str, extra_values: Iterable[str] = ()) -> str:
    """Mask sk-or-, sk-, gsk_, AIza, AQ., JWTs, bearer tokens, key=value
    assignments and long hex/base64 secrets. Use on anything that may be
    logged, stored, shown or sent to a model."""
    return redact_count(text, extra_values)[0]


_SECRET_KEY_NAMES = re.compile(r"(?i)(api[_\-]?key|secret|token|password|passwd|authorization|cookie|credential)")


def redact_obj(obj: Any, extra_values: Iterable[str] = (), _depth: int = 0) -> Any:
    """Recursively redact strings in dicts/lists; values under secret-named keys
    are replaced entirely."""
    extra = tuple(extra_values)
    if _depth > 12:
        return obj
    if isinstance(obj, str):
        return redact(obj, extra)
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and _SECRET_KEY_NAMES.search(k) and isinstance(v, str) and v:
                out[k] = MASK
            else:
                out[k] = redact_obj(v, extra, _depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_obj(v, extra, _depth + 1) for v in obj)
    return obj


class _RedactionFilter(logging.Filter):
    """Logging filter masking secrets in every record (message and args)."""

    def __init__(self) -> None:
        super().__init__("sam-redaction")
        self._values: tuple[str, ...] = ()

    def refresh(self, secrets: "Secrets | None") -> None:
        if secrets is not None:
            try:
                self._values = tuple(secrets.known_values())
            except Exception:  # noqa: BLE001 - redaction must never break logging
                pass

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            masked = redact(message, self._values)
            if masked != message or record.args:
                record.msg, record.args = masked, None
            if record.exc_info and record.exc_info[1] is not None:
                import traceback
                record.exc_text = redact("".join(traceback.format_exception(*record.exc_info)), self._values)
                record.exc_info = None
        except Exception:  # noqa: BLE001
            pass
        return True


_REDACTION = _RedactionFilter()


def install_log_redaction(secrets: Secrets | None = None) -> logging.Filter:
    """Attach the redaction filter to the root logger's handlers (idempotent)."""
    _REDACTION.refresh(secrets)
    root = logging.getLogger()
    for handler in root.handlers:
        if _REDACTION not in handler.filters:
            handler.addFilter(_REDACTION)
    return _REDACTION


__all__ = [
    "DPAPI_FORMAT", "SECRETS_FILENAME", "KEY_PATTERNS", "SUPPORTED_KEYS", "ENV_NAMES",
    "SecretStore", "Secrets", "redact", "redact_count", "redact_obj", "install_log_redaction", "MASK",
]
