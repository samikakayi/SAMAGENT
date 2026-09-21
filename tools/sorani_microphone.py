"""Live microphone check for Sorani speech (the missing room-and-headset leg).

sorani_acceptance.py proves the provider round trip with synthetic audio.
This script asks a person to speak one command, transcribes it through
KurdishTTS, maps it to a SAM intent, and never prints a credential.

Run:  .venv\\Scripts\\python.exe tools\\sorani_microphone.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_backend.config import Settings  # noqa: E402
from sam_backend.db import Database  # noqa: E402
from sam_backend.secrets import SecretStore, resolve_credential  # noqa: E402
from sam_backend.sorani_intent import parse as parse_intent  # noqa: E402
from sam_backend.voice import VoiceService  # noqa: E402


def main() -> int:
    settings = Settings.from_env()
    settings.prepare()
    store = SecretStore(settings.data_dir)
    stt_key, source = resolve_credential("kurdishtts_stt_api_key", store)
    print("KurdishTTS STT configured:", bool(stt_key), f"(source={source})")
    if not stt_key:
        print("Configure the STT key in Settings before running the microphone check.")
        return 2

    voice = VoiceService(settings, Database(settings.database_path))
    print("Speak one Sorani command after the beep-less wait. Example: بچۆ بۆ پێنج خولەکی")
    print("Listening up to 12 seconds...")
    result = voice.listen_once(max_seconds=12.0, language="ckb-IQ")
    text = (result.get("text") or "").strip()
    engine = result.get("engine")
    print(f"engine={engine}  language={result.get('language')}")
    if result.get("error"):
        print("error:", result["error"])
        return 1
    if not text:
        print("No Sorani transcript. Speak a little longer and try again.")
        return 1
    print("heard:", text)
    intent = parse_intent(text)
    if intent is None:
        print("intent: (none — SAM will send this to the chat model)")
        return 0
    print(f"intent: {intent.action}  args={intent.arguments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
