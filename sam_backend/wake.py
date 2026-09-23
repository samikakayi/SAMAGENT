"""Hands-free voice: hearing "Hey SAM" without sending the room to anybody.

Two objects, deliberately small, because the hard parts already exist in
`voice.py` and the agent. `WakeWordService` owns the microphone and decides
when the phrase was spoken. `VoiceConversationController` runs the loop that
follows: capture, transcribe, ask SAM the ordinary way, speak the answer,
listen again.

The privacy rule shapes the design more than anything else. Before the wake
phrase, audio exists only as a bounded in-memory ring of recent frames. It is
never written to disk and never leaves the machine. An energy gate means the
recogniser is not even consulted until somebody actually speaks, so idle
silence costs nothing and reaches nowhere.

The detector is a protocol on purpose. Today it recognises the phrase with the
local speech model SAM already ships, which is the only option that both hears
"Hey SAM" specifically and needs no account: openWakeWord publishes no model
for this phrase and licenses its pretrained ones NonCommercial, and Porcupine
wants an AccessKey to hear a wake word at all. If a better offline engine
appears, it implements `detect` and nothing else changes.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# How much recent audio the detector may look at. Bounded on purpose: this is
# the only place microphone audio lives before a wake phrase, and it is memory,
# never a file. It holds a whole sentence because people say "Hey SAM, what is
# gold doing?" in one breath -- with a two-second window the phrase has already
# scrolled out by the time the sentence ends, and SAM never wakes.
WAKE_WINDOW_SECONDS = 6.0
WAKE_FRAMES = int(WAKE_WINDOW_SECONDS * 1000 / FRAME_MS)

# After a detection, ignore the phrase for this long. Stops one utterance
# starting several sessions, and stops a repeat being heard as a second wake.
WAKE_DEBOUNCE_SECONDS = 3.0

# After SAM finishes speaking, wait before listening for the phrase again, so
# the tail of its own voice cannot wake it.
ECHO_COOLDOWN_SECONDS = 1.0

SENSITIVITY_ENERGY = {"LOW": 0.020, "NORMAL": 0.010, "HIGH": 0.005}

# How far above the room's own noise a frame must be before it counts as
# somebody talking. The sensitivity above is the floor under this, not a
# replacement for it.
NOISE_MARGIN = 2.0
NOISE_FALL = 0.90   # how fast the estimate follows a room going quiet
NOISE_RISE = 0.998  # how slowly it follows one getting louder

# A phrase is examined when it has *finished*, not when it starts. Looking at
# the buffer the moment energy rises shows the two seconds before the phrase
# -- mostly the silence leading up to it -- which is how "Hey SAM" came back
# from the recogniser as "" and "Thank you." Enough speech, then a short
# pause, and the buffer holds the whole thing.
MIN_SPEECH_FRAMES = 8          # ~240 ms of voice before it is worth a look
# 600 ms, not 300: people pause mid-sentence. At 300 ms "OK, hey SAM, hello
# there" was cut at the comma, and the fragment holding the phrase was too
# short to recognise -- it came back as "Peace out". Waiting a little longer
# costs a little latency and buys the phrase staying in one piece.
TRAILING_SILENCE_FRAMES = 20   # ~600 ms of quiet means they have finished
PRE_ROLL_FRAMES = 10           # a little air before the first syllable


class VoiceState(StrEnum):
    """What the hands-free loop is doing, as the panel shows it."""

    OFF = "OFF"
    WAKE_LISTENING = "WAKE_LISTENING"
    WAKE_DETECTED = "WAKE_DETECTED"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


@dataclass(slots=True)
class VoiceStatus:
    state: VoiceState = VoiceState.OFF
    detail: str = ""
    wake_phrase: str = "Hey SAM"
    sensitivity: str = "NORMAL"
    continuation_seconds: float = 0.0
    continuation_active: bool = False
    last_transcript: str = ""
    last_reply: str = ""
    last_error: str = ""
    device: int | None = None
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value, "detail": self.detail,
            "wake_phrase": self.wake_phrase, "sensitivity": self.sensitivity,
            "continuation_seconds": self.continuation_seconds,
            "continuation_active": self.continuation_active,
            # Transcripts are what the user just said out loud to SAM, so the
            # panel may echo them back. Nothing here is audio.
            "last_transcript": self.last_transcript[:400],
            "last_reply": self.last_reply[:400],
            "last_error": self.last_error[:200],
            "device": self.device,
            "updated_at": self.updated_at,
        }


def normalise(text: str) -> str:
    """Lower-case letters and digits only, so punctuation cannot hide a match."""
    lowered = str(text or "").lower()
    # A speech model that reads the name as an initialism writes it with stops:
    # a real microphone returned "Hey, S.A.M. What is Gold doing right now?" and
    # the match failed on the full stops alone. Those are one word to a
    # listener, so join them before the rest of the punctuation goes.
    lowered = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), lowered)
    return re.sub(r"[^a-z0-9؀-ۿ]+", " ", lowered).strip()


def phrase_heard(text: str, phrase: str) -> bool:
    """Whether a transcript contains the wake phrase, allowing for mishearing.

    A small model writes "hey Sam", "hey, Sam!", "Hey sam." and sometimes
    "hey some". The first three must all count, so the comparison happens on
    normalised words rather than the raw string. The last must not: a wake word
    that fires on a near-rhyme is worse than one that occasionally misses.
    """
    spoken = normalise(text)
    wanted = normalise(phrase)
    if not spoken or not wanted:
        return False
    if wanted in spoken:
        return True
    # "Hey SAM" also arrives as "hey sam" split across segments, and Sorani
    # speakers say the name with the English greeting; both reduce to the same
    # word sequence once punctuation is gone.
    words = wanted.split()
    if len(words) > 1:
        return re.search(r"\b" + r"\W*".join(re.escape(word) for word in words) + r"\b", spoken) is not None
    return re.search(r"\b" + re.escape(wanted) + r"\b", spoken) is not None


class WakeDetector:
    """Anything that can say whether a window of audio held the phrase."""

    def detect(self, audio: Any, phrase: str) -> bool:  # pragma: no cover - protocol
        raise NotImplementedError

    @property
    def available(self) -> bool:  # pragma: no cover - protocol
        return False

    def describe(self) -> dict[str, Any]:  # pragma: no cover - protocol
        return {}


class LocalPhraseDetector(WakeDetector):
    """Recognise the phrase with the local speech model SAM already has.

    It shares the dictation model rather than loading its own. Measured over a
    real microphone, the smaller models do not hear this phrase reliably --
    `tiny` returned "They said" and `base` returned "Hey sir" for a clear "Hey
    SAM", while the `small` SAM already loads for dictation returned "Hey,
    Sam." A wake word that only sometimes works is worse than none, and a
    second copy of the same weights in memory would buy nothing.

    It only ever sees one utterance, which the energy gate has already judged
    to be speech and the ring buffer bounds to a few seconds, so a silent room
    costs nothing at all.
    """

    def __init__(self, model_size: str = "small", *, shared: Any = None) -> None:
        self.model_size = model_size
        self.shared = shared
        self._model: Any = None
        self._error = ""
        self._warmed = threading.Event()

    @property
    def ready(self) -> bool:
        """True once the weights are loaded, so nobody waits on the lock.

        Deliberately not "is it currently loading": the listener starts warming
        on another thread, and asking whether loading has *begun* answers True
        in the moment before it does.
        """
        return self._warmed.is_set()

    @property
    def available(self) -> bool:
        if self.shared is not None:
            probe = getattr(self.shared, "available", False)
            # The dictation service exposes this as a method, and a bound
            # method is always truthy -- asking without calling it answers
            # "yes" even when the package is missing and nothing could ever
            # be heard. Hands-free would then start and stay silent forever
            # instead of saying the microphone button still works.
            return bool(probe() if callable(probe) else probe)
        try:
            import faster_whisper  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def warm(self) -> None:
        """Load the weights before anybody speaks.

        The first transcription of a cold model takes seconds, and it happens
        on the thread that is draining the microphone, so every frame arriving
        during the load is dropped -- which is exactly the first "Hey SAM"
        somebody says after switching hands-free on. Paying that cost at start
        means the listener is never the thing that is busy.
        """
        try:
            if self.shared is not None:
                loader = getattr(self.shared, "load", None)
                if callable(loader):
                    loader()
                return
            self.load()
        except Exception as exc:  # noqa: BLE001 - a cold start is not a crash
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._warmed.set()

    def load(self) -> Any:
        if self._model is None and not self._error:
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            except Exception as exc:  # noqa: BLE001 - a missing model is a state, not a crash
                self._error = f"{type(exc).__name__}: {exc}"
        return self._model

    def transcribe(self, audio: Any) -> str:
        if self.shared is not None:
            # The dictation model, already in memory. The wake phrase is
            # English even when the command that follows is not, so this asks
            # for English regardless of the configured command language.
            #
            # That is a limit, and a deliberate one. Sorani has no local
            # recogniser -- SAM sends it to a provider -- and the wake phrase
            # is the one thing that may never leave this machine. So the
            # phrase is English and the command is whatever the user speaks.
            try:
                return str(self.shared.transcribe(audio, language="en").get("text") or "")
            except Exception as exc:  # noqa: BLE001 - a bad frame must not end the loop
                self._error = f"{type(exc).__name__}: {exc}"
                return ""
        model = self.load()
        if model is None:
            return ""
        try:
            segments, _info = model.transcribe(audio, language="en", beam_size=1,
                                               condition_on_previous_text=False)
            return " ".join(segment.text for segment in segments)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not end the loop
            self._error = f"{type(exc).__name__}: {exc}"
            return ""

    def detect(self, audio: Any, phrase: str) -> bool:
        return phrase_heard(self.transcribe(audio), phrase)

    def describe(self) -> dict[str, Any]:
        return {"engine": "local-speech-model", "model": self.model_size,
                "shared_with_dictation": self.shared is not None,
                "cloud": False, "credential_required": False, "error": self._error}


class WakeWordService:
    """Owns the microphone and decides when the phrase was spoken.

    One thread, one bounded buffer, and a clear way to be told to be quiet
    while SAM is talking. Nothing here reaches the network.
    """

    def __init__(self, settings: Any, *, detector: WakeDetector | None = None,
                 stream_factory: Callable[..., Any] | None = None) -> None:
        self.settings = settings
        self.detector = detector or LocalPhraseDetector()
        self._stream_factory = stream_factory
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._suppressed = threading.Event()
        self._lock = threading.Lock()
        self._buffer: deque = deque(maxlen=WAKE_FRAMES)
        self._last_detection = 0.0
        self._noise_floor = 0.0
        self._on_wake: Callable[[], None] | None = None
        self.error = ""

    # -- configuration ------------------------------------------------------
    @property
    def phrase(self) -> str:
        return str(getattr(self.settings, "voice_wake_word", "Hey SAM") or "Hey SAM")

    @property
    def sensitivity(self) -> str:
        value = str(getattr(self.settings, "hands_free_sensitivity", "NORMAL") or "NORMAL").upper()
        return value if value in SENSITIVITY_ENERGY else "NORMAL"

    @property
    def energy_gate(self) -> float:
        return SENSITIVITY_ENERGY[self.sensitivity]

    @property
    def threshold(self) -> float:
        """What counts as speech in *this* room, not in a quiet one.

        A fixed number cannot do this job. Measured on a desk that sounded
        silent, 54% of frames were already above the NORMAL gate, so every
        frame looked like speech, the utterance never appeared to end, and the
        detector was handed six seconds of fan noise with a phrase buried in
        it. The bar therefore sits a few times above the room's own noise, and
        never below what the chosen sensitivity asks for.
        """
        return max(self.energy_gate, self._noise_floor * NOISE_MARGIN)

    def _track_noise(self, energy: float, speech: bool) -> None:
        """Follow the quiet quickly; rise slowly.

        Asymmetric on purpose. Dropping fast means a room that goes silent is
        recognised as silent within a breath. Rising slowly means an utterance
        cannot lift the bar above itself and cut itself off -- while still
        letting a room that has genuinely become noisier raise it over a few
        seconds, instead of deadlocking with everything classified as speech.
        """
        if speech:
            self._noise_floor = NOISE_RISE * self._noise_floor + (1 - NOISE_RISE) * energy
        else:
            self._noise_floor = NOISE_FALL * self._noise_floor + (1 - NOISE_FALL) * energy

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- lifecycle ----------------------------------------------------------
    def start(self, on_wake: Callable[[], None]) -> bool:
        with self._lock:
            if self.running:
                return True
            self._on_wake = on_wake
            self._stop.clear()
            self.error = ""
            warm = getattr(self.detector, "warm", None)
            if callable(warm):
                threading.Thread(target=warm, name="sam-wake-warm", daemon=True).start()
            self._thread = threading.Thread(target=self._listen, name="sam-wake", daemon=True)
            self._thread.start()
            return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            self._thread = None
            self._buffer.clear()

    def suppress(self) -> None:
        """Stop listening for the phrase -- somebody else needs the room.

        Used both when SAM is about to speak and while it is capturing a
        command. During a command there is nothing to wake up: SAM is already
        awake, and a second listener transcribing the same words only competes
        for the microphone and the CPU.
        """
        self._suppressed.set()
        self._buffer.clear()

    def resume(self, cooldown: float = ECHO_COOLDOWN_SECONDS) -> None:
        """Listen again, after a pause long enough to miss its own echo.

        The pause is only owed to SAM's own voice reaching a live microphone.
        After a command the user has stopped talking, and when hands-free is
        off nothing is listening at all -- waiting in either case would just be
        latency charged to every spoken reply.
        """
        if cooldown > 0 and self.running:
            time.sleep(cooldown)
        self._buffer.clear()
        self._last_detection = time.monotonic()
        self._suppressed.clear()

    @property
    def suppressed(self) -> bool:
        return self._suppressed.is_set()

    # -- the loop -----------------------------------------------------------
    def _open_stream(self) -> Any:
        if self._stream_factory is not None:
            return self._stream_factory()
        from .voice import MicrophoneStream

        device = getattr(self.settings, "voice_input_device", None)
        return MicrophoneStream(device=device if isinstance(device, int) else None)

    def _listen(self) -> None:
        from .voice import rms_energy

        voiced = 0
        silence = 0
        spoken = 0
        self._noise_floor = 0.0
        try:
            with self._open_stream() as microphone:
                # `frames()` ends when the queue stays empty, which happens
                # whenever transcription briefly outruns the audio callback. That
                # is a pause in the room, not a lost device: without this outer
                # loop the listener exits silently after the first examination
                # and SAM never hears anything again.
                while not self._stop.is_set():
                    for frame in microphone.frames(timeout=0.4):
                        if self._stop.is_set():
                            return
                        if self._suppressed.is_set():
                            # Deliberately drop the frame rather than buffer it:
                            # what SAM is saying must not survive to be examined.
                            continue
                        self._buffer.append(frame)
                        energy = rms_energy(frame)
                        speech = energy >= self.threshold
                        self._track_noise(energy, speech)
                        if speech:
                            if voiced == 0:
                                spoken = 0  # first syllable of a new utterance
                            voiced += 1
                            spoken += 1
                            silence = 0
                            continue
                        if voiced < MIN_SPEECH_FRAMES:
                            # Not speech, or not enough of it to be a phrase.
                            voiced = 0
                            spoken = 0
                            continue
                        silence += 1
                        spoken += 1
                        if silence < TRAILING_SILENCE_FRAMES:
                            continue
                        # Somebody spoke and stopped. Look at exactly what they
                        # said -- from the first syllable, not the last two
                        # seconds -- so the phrase is still in view at the end of
                        # a long sentence.
                        span = min(len(self._buffer), spoken + PRE_ROLL_FRAMES)
                        voiced = 0
                        silence = 0
                        spoken = 0
                        if time.monotonic() - self._last_detection < WAKE_DEBOUNCE_SECONDS:
                            continue
                        if not getattr(self.detector, "ready", True):
                            # Loading. Examining now would block this thread on the
                            # model lock, and the microphone would go unread for as
                            # long as that takes -- losing real speech, not just
                            # this utterance.
                            continue
                        if self._examine(span):
                            self._last_detection = time.monotonic()
                            self._buffer.clear()
                            if self._on_wake is not None:
                                self._on_wake()
        except Exception as exc:  # noqa: BLE001 - a lost device ends the loop, not SAM
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._buffer.clear()

    def _examine(self, span: int | None = None) -> bool:
        import numpy

        frames = list(self._buffer)
        if span is not None:
            frames = frames[-span:]
        if not frames:
            return False
        try:
            window = numpy.concatenate(frames).astype("float32")
        except Exception:  # noqa: BLE001
            return False
        return bool(self.detector.detect(window, self.phrase))

    def describe(self) -> dict[str, Any]:
        return {
            "running": self.running, "suppressed": self.suppressed,
            "ready": bool(getattr(self.detector, "ready", True)),
            "phrase": self.phrase, "sensitivity": self.sensitivity,
            "energy_gate": self.energy_gate,
            "threshold": round(self.threshold, 5),
            "error": self.error,
            "buffer_seconds": WAKE_WINDOW_SECONDS,
            "detector": self.detector.describe(),
        }
