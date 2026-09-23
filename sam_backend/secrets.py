"""Backend-only credential storage and live provider status.

A key set through the UI is written to a file the browser can never read, is
never echoed back, and is never placed in a response body. Only its presence, a
short fingerprint, and the outcome of a live health check are ever exposed.

At rest the store is encrypted with Windows DPAPI, keyed to the logged-in user
account. File ACLs alone only stop another account reading the file in place;
DPAPI also makes a copy of the file useless on another machine or under another
account. There is no key material in this repository -- the OS holds it.

Where DPAPI is unavailable (non-Windows, or a hardened environment that blocks
it) the store falls back to the previous owner-only plaintext file and says so
through `storage_status()`, rather than silently pretending to be encrypted.

Precedence is process environment, then `.env`, then this store: an operator's
explicit environment always wins over something typed into a form earlier.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import re
import stat
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from .contracts import CapabilityState

SECRETS_FILENAME = "secrets.json"
# Marks a file whose payload is DPAPI ciphertext rather than a plain mapping.
DPAPI_FORMAT = "dpapi-v1"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _dpapi(call: str, payload: bytes) -> bytes | None:
    """Run one DPAPI round trip, or return None when it is unavailable.

    Returning None rather than raising keeps a credential storable on a machine
    where DPAPI is blocked; the caller degrades to the ACL-only file and
    reports that state honestly.
    """
    if os.name != "nt":
        return None
    try:
        crypt = ctypes.windll.crypt32
        source = _Blob(len(payload), ctypes.cast(ctypes.create_string_buffer(payload), ctypes.POINTER(ctypes.c_char)))
        result = _Blob()
        function = crypt.CryptProtectData if call == "protect" else crypt.CryptUnprotectData
        # CRYPTPROTECT_UI_FORBIDDEN (0x1): never prompt; a backend has no desktop.
        ok = function(ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(result))
        if not ok:
            return None
        try:
            return ctypes.string_at(result.pbData, result.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(result.pbData)
    except Exception:  # noqa: BLE001 - any OS-level failure means "not available"
        return None
# Rough shapes, used only to reject obvious paste errors before a network call.
KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    "openrouter_api_key": re.compile(r"^sk-or-[A-Za-z0-9._\-]{20,200}$"),
    "openai_api_key": re.compile(r"^sk-[A-Za-z0-9._\-]{20,200}$"),
    "litellm_api_key": re.compile(r"^[A-Za-z0-9._\-]{8,200}$"),
    # Groq issues gsk_-prefixed keys; Google AI Studio issues AIza-prefixed
    # ones. Shapes only -- enough to catch a paste error before a network call.
    "groq_api_key": re.compile(r"^gsk_[A-Za-z0-9._\-]{20,200}$"),
    "gemini_api_key": re.compile(r"^AIza[A-Za-z0-9._\-]{20,200}$"),
    # n8n issues long JWT-shaped personal API keys.
    "n8n_api_key": re.compile(r"^[A-Za-z0-9._\-]{20,600}$"),
    # KurdishTTS issues separate hex keys for speech-to-text and text-to-speech.
    "kurdishtts_stt_api_key": re.compile(r"^[A-Za-z0-9._\-]{16,200}$"),
    "kurdishtts_tts_api_key": re.compile(r"^[A-Za-z0-9._\-]{16,200}$"),
    "google_stt_credentials_path": re.compile(r"^[^\r\n]{3,400}$"),
}
SUPPORTED_KEYS = tuple(KEY_PATTERNS)


class SecretStore:
    """File-backed secret storage with restrictive permissions."""

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / SECRETS_FILENAME
        self._lock = threading.RLock()
        self._encrypted_at_rest = False
        self._migrate_plaintext()

    def _migrate_plaintext(self) -> None:
        """Encrypt a legacy plaintext store in place, once, at startup.

        Reads without logging, rewrites encrypted, and only then is the
        plaintext gone -- the rewrite is what removes it, so there is no window
        where the credential exists in neither form.
        """
        with self._lock:
            if not self.path.is_file():
                self._encrypted_at_rest = _dpapi("protect", b"probe") is not None
                return
            values = self._read()
            if not values:
                return
            if self._stored_format() == DPAPI_FORMAT:
                self._encrypted_at_rest = True
                return
            self._write(values)

    def _stored_format(self) -> str | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return payload.get("_format") if isinstance(payload, dict) else None

    def _read(self) -> dict[str, str]:
        with self._lock:
            if not self.path.is_file():
                return {}
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
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
                    # Written by another account, or DPAPI is now unavailable.
                    # Refusing beats returning a half-decoded credential.
                    return {}
                try:
                    payload = json.loads(plain.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return {}
                if not isinstance(payload, dict):
                    return {}
            return {key: str(value) for key, value in payload.items()
                    if isinstance(value, str) and not key.startswith("_")}

    def _write(self, values: dict[str, str]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            blob = _dpapi("protect", json.dumps(values).encode("utf-8"))
            self._encrypted_at_rest = blob is not None
            document = (
                {"_format": DPAPI_FORMAT, "data": base64.b64encode(blob).decode("ascii")}
                if blob is not None else values
            )
            self.path.write_text(json.dumps(document, indent=2), encoding="utf-8")
            try:
                # Owner read/write only; the file must not be world readable.
                self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
            self._restrict_windows_acl()

    def _restrict_windows_acl(self) -> None:
        """Make the file owner-only on Windows.

        `chmod` on Windows only toggles the read-only attribute; the POSIX mode
        bits above are ignored, which left the file readable by every account on
        the machine. Access is controlled by ACLs here, so inheritance is
        dropped and the current user is granted sole access.
        """
        if os.name != "nt":
            return
        import subprocess

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
            # Losing the hardening is worth reporting, but not worth refusing to
            # store a credential the user just entered.
            pass

    def storage_status(self) -> dict[str, Any]:
        """How the store is protected at rest. Never any value."""
        return {
            "encrypted_at_rest": self._encrypted_at_rest,
            "mechanism": "windows-dpapi" if self._encrypted_at_rest else "owner-only-file",
            "path": str(self.path),
        }

    def acl_summary(self) -> dict[str, Any]:
        """Who can read the store, for the security surface. No values."""
        if not self.path.is_file():
            return {"exists": False}
        entry: dict[str, Any] = {"exists": True, "path_inside_web_root": "frontend" in self.path.as_posix()}
        if os.name != "nt":
            entry["mode"] = oct(stat.S_IMODE(self.path.stat().st_mode))
            return entry
        import subprocess

        try:
            result = subprocess.run(["icacls", str(self.path)], capture_output=True, text=True,
                                    timeout=15, check=False,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            # Each ACE reads "DOMAIN\Account:(perms)". icacls prefixes the first
            # line with the path it was given, which is stripped rather than
            # pattern-matched around, since an account name may contain spaces.
            principals: list[str] = []
            for index, line in enumerate(result.stdout.splitlines()):
                line = line.strip()
                if ":(" not in line:
                    continue
                if index == 0:
                    line = line.removeprefix(str(self.path)).strip() or line
                principals.append(line.split(":(")[0].strip())
            entry["principals"] = principals[:8]
        except (OSError, subprocess.SubprocessError):
            entry["principals"] = []
        return entry

    @staticmethod
    def fingerprint(value: str | None) -> str | None:
        """A stable, non-reversible identifier so a key can be told apart."""
        if not value:
            return None
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]

    def get(self, name: str) -> str | None:
        return self._read().get(name)

    def set(self, name: str, value: str) -> dict[str, Any]:
        if name not in SUPPORTED_KEYS:
            raise ValueError(f"Unsupported credential: {name}")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("The credential is empty.")
        if not KEY_PATTERNS[name].match(cleaned):
            raise ValueError(
                f"That does not look like a {name.replace('_', ' ')}. Check for a truncated paste or stray whitespace."
            )
        values = self._read()
        values[name] = cleaned
        self._write(values)
        return {"name": name, "stored": True, "fingerprint": self.fingerprint(cleaned)}

    def clear(self, name: str) -> bool:
        values = self._read()
        if name not in values:
            return False
        values.pop(name)
        self._write(values)
        return True

    def public_status(self) -> dict[str, Any]:
        """Presence and fingerprint only. The value itself never leaves this class."""
        values = self._read()
        return {
            name: {"configured": name in values, "fingerprint": self.fingerprint(values.get(name))}
            for name in SUPPORTED_KEYS
        }


def resolve_credential(name: str, store: SecretStore) -> tuple[str | None, str]:
    """Return the effective credential and where it came from."""
    environment = os.getenv(name.upper())
    if environment:
        return environment, "environment"
    stored = store.get(name)
    if stored:
        return stored, "secret_store"
    return None, "unset"


# --- Live provider status -----------------------------------------------------


async def openrouter_status(base_url: str, api_key: str | None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Classify OpenRouter reachability without ever returning the key."""
    if not api_key:
        return {
            "provider": "openrouter", "status": "UNCONFIGURED",
            "state": CapabilityState.UNCONFIGURED.value,
            "detail": "No OPENROUTER_API_KEY in the environment and nothing in the local secret store.",
        }
    request_headers = {"Authorization": f"Bearer {api_key}", **{k: v for k, v in (headers or {}).items() if v}}
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            response = await client.get(f"{base_url.rstrip('/')}/key", headers=request_headers)
    except httpx.HTTPError as exc:
        return {"provider": "openrouter", "status": "ERROR", "state": CapabilityState.UNAVAILABLE.value,
                "detail": f"Could not reach OpenRouter: {exc}"}
    latency = round((time.perf_counter() - started) * 1000, 1)
    if response.status_code in (401, 403):
        return {"provider": "openrouter", "status": "AUTH_FAILED", "state": CapabilityState.UNAVAILABLE.value,
                "detail": "OpenRouter rejected the credential.", "latency_ms": latency}
    if response.status_code == 429:
        return {"provider": "openrouter", "status": "RATE_LIMITED", "state": CapabilityState.PARTIALLY_AVAILABLE.value,
                "detail": "OpenRouter is rate limiting this credential.", "latency_ms": latency}
    if response.status_code >= 400:
        return {"provider": "openrouter", "status": "ERROR", "state": CapabilityState.UNAVAILABLE.value,
                "detail": f"OpenRouter returned HTTP {response.status_code}.", "latency_ms": latency}
    payload: dict[str, Any] = {}
    try:
        payload = (response.json() or {}).get("data") or {}
    except ValueError:
        payload = {}
    return {
        "provider": "openrouter", "status": "CONNECTED", "state": CapabilityState.AVAILABLE.value,
        "latency_ms": latency,
        # Usage and limit are safe to surface; the key itself is not included.
        "usage": payload.get("usage"),
        "limit": payload.get("limit"),
        "limit_remaining": payload.get("limit_remaining"),
        "is_free_tier": payload.get("is_free_tier"),
        "detail": "Authenticated against the OpenRouter key endpoint.",
    }


async def ollama_status(base_url: str) -> dict[str, Any]:
    """Classify the local Ollama daemon and its installed models."""
    try:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            models = (response.json() or {}).get("models") or []
    except httpx.HTTPError as exc:
        return {"provider": "ollama", "status": "DOWN", "state": CapabilityState.UNAVAILABLE.value,
                "models": [], "detail": f"The local Ollama API is not answering: {exc}"}
    except ValueError:
        return {"provider": "ollama", "status": "ERROR", "state": CapabilityState.UNAVAILABLE.value,
                "models": [], "detail": "Ollama returned a response that could not be parsed."}
    names = [item.get("name") or item.get("model") for item in models if item.get("name") or item.get("model")]
    if not names:
        return {"provider": "ollama", "status": "NO_MODELS", "state": CapabilityState.PARTIALLY_AVAILABLE.value,
                "models": [], "detail": "Ollama is running but no model is installed. Pull one, e.g. `ollama pull qwen3.5:4b`."}
    return {"provider": "ollama", "status": "CONNECTED", "state": CapabilityState.AVAILABLE.value,
            "models": names, "detail": f"{len(names)} local model(s) available."}


def find_ollama_executable(project_root: Path) -> str | None:
    """The system Ollama, or the portable copy shipped under tools/."""
    import shutil

    system = shutil.which("ollama")
    if system:
        return system
    candidates = sorted(
        (path for path in (project_root / "tools").glob("ollama-v*") if path.is_dir()),
        key=lambda path: path.name, reverse=True,
    )
    for directory in candidates:
        executable = directory / "ollama.exe"
        if executable.is_file():
            return str(executable)
    return None


def start_ollama(project_root: Path) -> dict[str, Any]:
    """Start the local daemon, keeping its data inside the project."""
    import subprocess

    executable = find_ollama_executable(project_root)
    if not executable:
        return {"started": False, "reason": "No Ollama executable was found on PATH or under tools/."}
    profile = project_root / "data" / "ollama-user"
    models = project_root / "data" / "ollama-models"
    profile.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "USERPROFILE": str(profile), "OLLAMA_MODELS": str(models)}
    try:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            [executable, "serve"], env=environment, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=creation_flags,
        )
    except OSError as exc:
        return {"started": False, "reason": f"Could not launch Ollama: {exc}"}
    return {"started": True, "executable": executable, "detail": "Ollama was started; it may take a few seconds to answer."}


async def openai_compatible_status(provider: str, base_url: str, api_key: str | None) -> dict[str, Any]:
    """Classify a keyed OpenAI-compatible provider without spending a completion.

    Listing models is the cheapest call that still proves the credential works,
    so a status panel costs a catalogue request rather than tokens. The key is
    used to authenticate and never appears in what is returned.
    """
    label = provider.upper()
    if not api_key:
        return {
            "provider": provider, "status": "UNCONFIGURED",
            "state": CapabilityState.UNCONFIGURED.value,
            "detail": f"No {label}_API_KEY in the environment and nothing in the local secret store.",
        }
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            response = await client.get(
                f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError as exc:
        return {"provider": provider, "status": "ERROR", "state": CapabilityState.UNAVAILABLE.value,
                "detail": f"Could not reach {label}: {type(exc).__name__}"}
    latency = round((time.perf_counter() - started) * 1000, 1)
    common = {"provider": provider, "latency_ms": latency}
    if response.status_code in (401, 403):
        return {**common, "status": "AUTH_FAILED", "state": CapabilityState.UNAVAILABLE.value,
                "detail": f"{label} rejected the credential."}
    if response.status_code == 429:
        return {**common, "status": "RATE_LIMITED", "state": CapabilityState.PARTIALLY_AVAILABLE.value,
                "detail": f"{label} is rate limiting this credential."}
    if response.status_code == 402:
        return {**common, "status": "QUOTA", "state": CapabilityState.UNAVAILABLE.value,
                "detail": f"{label} reports this credential is out of quota."}
    if response.status_code >= 400:
        return {**common, "status": "ERROR", "state": CapabilityState.UNAVAILABLE.value,
                "detail": f"{label} returned HTTP {response.status_code}."}
    try:
        catalogue = (response.json() or {}).get("data") or []
    except ValueError:
        catalogue = []
    return {**common, "status": "CONNECTED", "state": CapabilityState.AVAILABLE.value,
            "models_listed": len(catalogue),
            "detail": f"Authenticated against the {label} model catalogue."}
