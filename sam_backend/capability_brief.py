"""What SAM can do right now, told to the chat model in a few lines.

The chat model only ever receives text, so unless it is told otherwise it
assumes that is all SAM is. That is what happened: gpt-oss-20b told a Sorani
speaker it could not hear them and only worked through text, although two turns
of that very conversation had arrived through the microphone and KurdishTTS
(the audit log shows each transcription 50-64 ms before the chat turn). The
system prompt had never mentioned voice, and the tool list -- the only other
self-description -- is left out of most chat turns by the router's keyword gate
(on purpose: ~6.2k tokens for 57 tools, none of them about voice).

So every turn carries a short brief built from live state rather than a fixed
claim. A fixed "say Hey SAM" line would have been false on the very same
install: hands-free was switched on, but its wake listener had failed to load
its speech model. Rendered, the brief is about 75-150 o200k tokens.

Only cheap checks are used. The credential probes read the local secret store
(file + DPAPI, no network), so their answer is kept for BRIEF_TTL_SECONDS; the
switches and the wake listener's state are attribute reads and are taken fresh
every turn, so turning something on in Settings shows up on the next message.
Nothing here calls a provider: a health check per chat turn would cost a
network round trip before every answer.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .sorani import is_sorani

BRIEF_TTL_SECONDS = 30.0

# How a spoken turn is shown to the model. Only the model sees it: the stored
# message and the transcript on screen stay exactly what was said.
VOICE_TAG = "[voice] "


@dataclass(frozen=True)
class Brief:
    text: str
    # Whether the user can talk to SAM right now, so the Sorani style anchor
    # (which weak models copy word for word) can say so -- or not.
    voice_input: bool


class CapabilityBrief:
    """Builds the capability section of the chat system prompt from live state."""

    def __init__(
        self,
        settings: Any,
        *,
        voice: Any = None,
        voice_session: Any = None,
        trading: Any = None,
        autonomy: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.voice = voice
        self.voice_session = voice_session
        self.trading = trading
        self.autonomy = autonomy
        self._clock = clock
        self._keys: tuple[float, bool | None, bool | None] | None = None

    # -- probes -------------------------------------------------------------
    def _sorani_keys(self) -> tuple[bool | None, bool | None]:
        """(speech-to-text configured, text-to-speech configured); None = unknown."""
        now = self._clock()
        cached = self._keys
        if cached is not None and now - cached[0] < BRIEF_TTL_SECONDS:
            return cached[1], cached[2]
        try:
            stt: bool | None = bool(self.voice.sorani_input_configured())
            tts: bool | None = bool(self.voice.sorani_output_configured())
        except Exception:  # noqa: BLE001 - an unreadable store claims nothing
            stt = tts = None
        self._keys = (now, stt, tts)
        return stt, tts

    def _wake(self) -> dict[str, Any] | None:
        """The wake listener's own report, or None when there is no listener."""
        wake = getattr(self.voice_session, "wake", None)
        if wake is None:
            return None
        try:
            described = wake.describe()
        except Exception as exc:  # noqa: BLE001
            return {"running": False, "error": type(exc).__name__, "phrase": "Hey SAM"}
        return described if isinstance(described, dict) else None

    # -- rendering ----------------------------------------------------------
    def current(self) -> Brief:
        lines = ["What SAM can do right now (live state):"]
        voice_input = False
        if self.voice is not None:
            voice_input = self._voice_lines(lines)
        control = "on" if getattr(self.settings, "computer_control_enabled", False) else "off"
        screen = "on" if getattr(self.settings, "screen_access_enabled", False) else "off"
        lines.append(f"- Desktop control: {control}; screen access: {screen} (switched in Settings).")
        extras = []
        if self.trading is not None:
            extras.append("TradingView chart and market analysis (no order placement)")
        if self.autonomy:
            # No chat tool starts a run, so the model must not offer to start
            # one itself -- the user does, from the Autopilot tab.
            extras.append("an Autopilot tab the user can start for multi-step tasks")
        if extras:
            lines.append("- Also: " + "; ".join(extras) + ".")
        return Brief("\n".join(lines), voice_input)

    def text(self) -> str:
        return self.current().text

    def _voice_lines(self, lines: list[str]) -> bool:
        """Append what is true about voice; return whether the user can speak to SAM."""
        sorani = is_sorani(str(getattr(self.settings, "voice_language", "") or ""))
        if sorani:
            stt, tts = self._sorani_keys()
            if stt is None:
                # Not knowing is not the same as not having; claim nothing.
                return False
            if not stt:
                lines.append("- Voice input is not set up (no Sorani speech-recognition key); "
                             "if asked, say so and point to Settings.")
                return False
            recogniser = "Sorani speech recognition"
        else:
            # Other languages are recognised by the browser (Web Speech) for the
            # mic button and by the local model for hands-free; neither needs a
            # key, and the browser's own voice reads the reply.
            tts = True
            recogniser = "speech recognition"
        phrase, hands_free_note = self._hands_free()
        sources = f'the mic button or by saying "{phrase}"' if phrase else "the mic button"
        lines.append(
            f"- You get text; SAM's own pipeline does the audio. Speech via {sources} is turned into text by "
            f'{recogniser} and reaches you as "{VOICE_TAG}<transcript>", which may contain recognition errors, '
            "so ask if a word seems off. Never say you cannot hear the user or suggest another speech-to-text app."
        )
        if tts:
            lines.append("- SAM can read replies aloud" + (" in Sorani." if sorani else "."))
        else:
            lines.append("- Spoken Sorani replies are not set up (no text-to-speech key in Settings).")
        if hands_free_note:
            lines.append(hands_free_note)
        return True

    def _hands_free(self) -> tuple[str, str]:
        """(the wake phrase when hands-free works, else ""; a line saying why it does not).

        Switched off says nothing: the user chose that, and it is not a fault.
        """
        if not bool(getattr(self.settings, "hands_free_enabled", False)):
            return "", ""
        wake = self._wake()
        if wake is None:
            return "", ""
        phrase = _clean_phrase(wake.get("phrase") or getattr(self.settings, "voice_wake_word", "") or "Hey SAM")
        detector = wake.get("detector") if isinstance(wake.get("detector"), dict) else {}
        # The detector's own error counts too: on the install this was written
        # for, the listener's model failed to load (a numpy import race), and a
        # thread that cannot recognise the phrase is not a working wake word.
        failed = bool(wake.get("error") or detector.get("error"))
        if wake.get("running") and not failed:
            if wake.get("ready", True):
                return phrase, ""
            return "", f'- Hands-free "{phrase}" is starting (its speech model is loading).'
        state = "has an error" if failed else "is not running"
        return "", f'- Hands-free "{phrase}" is on but its wake listener {state}; the mic button still works (see Settings).'


def _clean_phrase(value: Any) -> str:
    """The wake phrase as one short line, so a setting cannot reshape the prompt."""
    return re.sub(r"[\s\"]+", " ", str(value)).strip()[:40] or "Hey SAM"
