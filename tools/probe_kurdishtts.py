"""Discover the live KurdishTTS contract. Prints schema only, never a credential."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from sam_backend.config import Settings  # noqa: E402
from sam_backend.secrets import SecretStore, resolve_credential  # noqa: E402

BASE = "https://www.kurdishtts.com/api"
PHRASE = "سڵاو، من سامم."


def main() -> None:
    settings = Settings.from_env()
    settings.prepare()
    store = SecretStore(settings.data_dir)
    tts_key, tts_source = resolve_credential("kurdishtts_tts_api_key", store)
    stt_key, stt_source = resolve_credential("kurdishtts_stt_api_key", store)
    print(f"tts key present: {bool(tts_key)} (from {tts_source})")
    print(f"stt key present: {bool(stt_key)} (from {stt_source})")
    if not tts_key:
        return

    # Look for a speaker/voice listing endpoint under several plausible names.
    print("\n--- speaker discovery ---")
    for path in ("/speakers", "/tts-speakers", "/voices", "/tts-proxy/speakers", "/models"):
        try:
            with httpx.Client(timeout=25, trust_env=False) as client:
                response = client.get(f"{BASE}{path}", headers={"x-api-key": tts_key})
            body = response.text[:240].replace("\n", " ")
            print(f"  GET {path:22s} -> {response.status_code} {response.headers.get('content-type','')[:30]} | {body}")
        except Exception as exc:
            print(f"  GET {path:22s} -> error {str(exc)[:70]}")

    print("\n--- TTS with no speaker_id (to surface the expected shape) ---")
    try:
        with httpx.Client(timeout=60, trust_env=False) as client:
            response = client.post(
                f"{BASE}/tts-proxy",
                headers={"x-api-key": tts_key, "Content-Type": "application/json"},
                json={"text": PHRASE, "model_version": "v4"},
            )
        print(f"  status={response.status_code} type={response.headers.get('content-type')}")
        print(f"  body={response.text[:400]}")
    except Exception as exc:
        print(f"  error: {str(exc)[:120]}")

    print("\n--- STT with no file (to surface required fields) ---")
    if stt_key:
        try:
            with httpx.Client(timeout=30, trust_env=False) as client:
                response = client.post(f"{BASE}/stt-proxy", headers={"x-api-key": stt_key},
                                       data={"dialect": "sorani"})
            print(f"  status={response.status_code} type={response.headers.get('content-type')}")
            print(f"  body={response.text[:400]}")
        except Exception as exc:
            print(f"  error: {str(exc)[:120]}")

    print("\n--- auth rejection shape (invalid key) ---")
    try:
        with httpx.Client(timeout=25, trust_env=False) as client:
            response = client.post(f"{BASE}/tts-proxy", headers={"x-api-key": "definitely-not-valid"},
                                   json={"text": PHRASE, "model_version": "v4", "speaker_id": 1})
        print(f"  status={response.status_code} body={response.text[:200]}")
    except Exception as exc:
        print(f"  error: {str(exc)[:120]}")


if __name__ == "__main__":
    main()
