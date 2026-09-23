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
appears, it implements `locate` and nothing else changes.

The phrase and the command are one utterance to the person saying them, so
they are one utterance here too. The listener finds where the phrase ends and
keeps capturing on the same stream; what follows is the command, handed to the
recogniser for the user's language. Only that -- never the phrase, never what
came before it -- is ever sent anywhere.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
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

# A long sentence is examined before it ends, while the phrase that opens it
# is still inside the buffer. Waiting for the end of "Hey SAM, open Gold and
# analyse the fifteen-minute chart" would find the phrase already scrolled out.
EARLY_EXAMINE_FRAMES = int(4.0 * 1000 / FRAME_MS)

# After the phrase, on the same stream: what counts as a command, and how long
# to keep listening for one. "Gold" and "stop" are commands, so the bar is a
# syllable, not a sentence.
POST_WAKE_MIN_SPEECH_FRAMES = 5                           # ~150 ms of voice in one run
# Once the listener has caught up with the room, how long it waits for a
# command to begin on the same stream. Never counted while somebody is
# talking. Long enough that nobody replying to "Listening..." is cut off by a
# hand-off to a second capture; after it, SAM asks the ordinary way.
POST_WAKE_ONSET_SECONDS = 5.0
POST_WAKE_ONSET_CAP_FRAMES = int(10.0 * 1000 / FRAME_MS)  # never wait longer than this
POST_WAKE_MAX_FRAMES = int(20.0 * 1000 / FRAME_MS)        # the same bound as any command
# When the microphone cannot say how much is queued, a frame that takes this
# long to arrive was not queued. Queued frames arrive in microseconds.
REALTIME_GAP_SECONDS = 0.004
# The quietest a command may be, before the room's own noise raises the bar.
# The same floor push-to-talk has always used.
COMMAND_GATE = 0.006


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


def _phrase_pattern(phrase: str) -> re.Pattern[str] | None:
    """The wake phrase as a pattern over raw recogniser text.

    Every word of the phrase must appear whole. Inside a word, letters may be
    separated by the stops or hyphens a recogniser writes into an initialism --
    "S.A.M.", "S-A-M" and "SA-M" all came back from real rooms -- and between
    words anything that is not a letter may appear. Whole words is the point:
    a plain substring test woke on "hey same here" and "they sampled".

    One pattern serves both questions -- was it said, and where does it end --
    so the two answers cannot disagree.
    """
    words = normalise(phrase).split()
    if not words:
        return None

    def word(letters: str) -> str:
        return r"[.\-]?".join(re.escape(letter) for letter in letters)

    return re.compile(r"(?<!\w)" + r"\W*".join(word(w) for w in words) + r"(?!\w)")


def phrase_heard(text: str, phrase: str) -> bool:
    """Whether a transcript contains the wake phrase, allowing for mishearing.

    A small model writes "hey Sam", "hey, Sam!", "Hey sam." and sometimes
    "hey some". The first three must all count; the last must not: a wake word
    that fires on a near-rhyme is worse than one that occasionally misses.
    Sorani speakers say the name with the English greeting, which matches the
    same way.
    """
    pattern = _phrase_pattern(phrase)
    return bool(pattern and text and pattern.search(str(text).lower()))


def _phrase_span(words: list[tuple[str, float, float]], phrase: str) -> tuple[float, bool] | None:
    """(where the phrase ends, whether any words follow it), or None if absent.

    The cut is the latest end among the phrase and every word before it, never
    just the phrase's own last timestamp: if the recogniser ever returns
    collapsed or out-of-order times, the cut moves later, never earlier, so
    nothing said before the phrase can slip into the command.
    """
    pattern = _phrase_pattern(phrase)
    if pattern is None or not words:
        return None
    text = ""
    owner: list[int] = []
    for index, (piece, _start, _end) in enumerate(words):
        lowered = str(piece or "").lower()
        text += lowered
        owner.extend([index] * len(lowered))
    match = pattern.search(text)
    if match is None:
        return None
    last = owner[match.end() - 1]
    cut = max(float(end) for _piece, _start, end in words[:last + 1])
    followed = any(re.search(r"\w", str(piece or "")) for piece, _start, _end in words[last + 1:])
    return cut, followed


def phrase_end(words: list[tuple[str, float, float]], phrase: str) -> float | None:
    """Where the wake phrase ends, in seconds, from a recogniser's timed words.

    `words` are (text, start, end) pieces as faster-whisper returns them;
    a word may arrive in several pieces (" S", ".A", ".M"). None when the
    phrase is not there.

    Measured on real microphone audio, the end of the phrase is a clean cut:
    "Okay hey SAM tell me the current status" cut there left exactly "tell me
    the current status." for the command.
    """
    span = _phrase_span(words, phrase)
    return None if span is None else span[0]


class WakeDetection:
    """What was said after the phrase, handed from the listener to the loop.

    The listener fills it on the stream that heard the phrase and marks it
    done; the conversation loop waits on it. `audio` is only ever what came
    after the phrase -- never the phrase, never anything before it -- and is
    dropped as soon as it has been transcribed.
    """

    def __init__(self, phrase_end: float | None = None) -> None:
        self.phrase_end = phrase_end
        self.audio: Any = None
        self.speech = False
        # Stopped or silenced part-way: whatever was heard is incomplete, and
        # half a command -- "close all positions" without "except gold" -- must
        # not be acted on.
        self.interrupted = False
        self._done = threading.Event()

    def finish(self, audio: Any = None, *, speech: bool = False, interrupted: bool = False) -> None:
        self.interrupted = bool(interrupted)
        self.speech = bool(speech) and not self.interrupted
        self.audio = audio if self.speech else None
        self._done.set()

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def release(self) -> None:
        self.audio = None


class WakeDetector:
    """Anything that can say whether a window of audio held the phrase.

    `locate` is the better answer: where the phrase ends, so the words after
    it can become the command. A detector with only `detect` still works; the
    command is then whatever is said next.
    """

    def detect(self, audio: Any, phrase: str) -> bool:  # pragma: no cover - protocol
        raise NotImplementedError

    def locate(self, audio: Any, phrase: str) -> float | None:  # pragma: no cover - protocol
        return len(audio) / SAMPLE_RATE if self.detect(audio, phrase) else None

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

    def locate(self, audio: Any, phrase: str) -> float | None:
        """Where the phrase ends in `audio`, in seconds, or None if it is not there.

        One local pass with word timestamps answers both questions at once:
        whether the phrase was said, and where the words after it begin. The
        words after it are not read here -- they may be Sorani, which this
        English pass cannot understand -- only located, so they can be handed
        to the recogniser the user configured.
        """
        words = self._timed_words(audio)
        span = _phrase_span(words, phrase) if words else None
        if span is None:
            return None
        cut, followed = span
        window_end = len(audio) / SAMPLE_RATE
        if cut <= 0 or not followed:
            # Nothing recognised after the phrase -- only the tail of "SAM", a
            # breath, the room -- or times too broken to cut by. Either way
            # nothing in this window is carried forward: the command is
            # whatever is said next, and nothing from before the phrase can
            # ride along on a bad timestamp.
            return window_end
        return min(cut, window_end)

    def _timed_words(self, audio: Any) -> list[tuple[str, float, float]]:
        try:
            if self.shared is not None:
                loader = getattr(self.shared, "load", None)
                model = loader() if callable(loader) else None
            else:
                model = self.load()
        except Exception as exc:  # noqa: BLE001 - a missing model is a state, not a crash
            self._error = f"{type(exc).__name__}: {exc}"
            return []
        if model is None:
            return []
        try:
            segments, _info = model.transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                               condition_on_previous_text=False, word_timestamps=True)
            return [(str(word.word), float(word.start), float(word.end))
                    for segment in segments for word in (segment.words or [])]
        except Exception as exc:  # noqa: BLE001 - a bad frame must not end the loop
            self._error = f"{type(exc).__name__}: {exc}"
            return []

    def detect(self, audio: Any, phrase: str) -> bool:
        return self.locate(audio, phrase) is not None

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
        self._pending: WakeDetection | None = None
        # How many frames the microphone is holding, when it can say.
        self._queued: Callable[[], int] | None = None
        self.onset_seconds = POST_WAKE_ONSET_SECONDS
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
    def command_threshold(self) -> float:
        """The bar for a command: push-to-talk's floor, raised by a noisy room."""
        return max(COMMAND_GATE, self._noise_floor * NOISE_MARGIN)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stopped(self) -> bool:
        """Switched off, as opposed to merely never started."""
        return self._stop.is_set()

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

    def take_detection(self) -> WakeDetection | None:
        """Claim the wake that is being reported, and with it what follows.

        Called from `on_wake`. A wake nobody claims -- a session is already
        running, or the caller only wanted to know -- carries nothing forward:
        the listener does not capture a command for no one.
        """
        with self._lock:
            detection, self._pending = self._pending, None
        return detection

    # -- the loop -----------------------------------------------------------
    def _open_stream(self) -> Any:
        if self._stream_factory is not None:
            return self._stream_factory()
        from .voice import MicrophoneStream

        device = getattr(self.settings, "voice_input_device", None)
        return MicrophoneStream(device=device if isinstance(device, int) else None)

    def _frames(self, microphone: Any) -> Iterator[Any]:
        """One endless iterator over the microphone, shared by everything here.

        `frames()` ends whenever the queue stays empty for a moment -- a pause
        in the room, not a lost device -- so it is simply asked again. The wake
        loop and the capture after the phrase pull from this same iterator,
        which is what keeps a one-breath command on the stream that heard it.

        Between those runs it yields None: a stalled or unplugged microphone
        must still let its readers notice being stopped or silenced, rather
        than blocking inside `next()` until frames return.
        """
        while not self._stop.is_set():
            yield from microphone.frames(timeout=0.4)
            yield None

    def _listen(self) -> None:
        from .voice import rms_energy

        voiced = 0
        silence = 0
        spoken = 0
        examined_early = False
        self._noise_floor = 0.0
        try:
            with self._open_stream() as microphone:
                self._queued = getattr(microphone, "pending", None)
                frames = self._frames(microphone)
                for frame in frames:
                    if self._stop.is_set():
                        return
                    if frame is None:
                        continue
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
                            examined_early = False
                        voiced += 1
                        spoken += 1
                        silence = 0
                        if spoken >= EARLY_EXAMINE_FRAMES and not examined_early:
                            # Still talking, and the buffer is filling. Look now,
                            # while the start of the sentence is still in it.
                            examined_early = True
                            span = min(len(self._buffer), spoken + PRE_ROLL_FRAMES)
                            if self._consider(span, frames):
                                voiced = silence = spoken = 0
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
                    voiced = silence = spoken = 0
                    self._consider(span, frames)
        except Exception as exc:  # noqa: BLE001 - a lost device ends the loop, not SAM
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._buffer.clear()

    def _consider(self, span: int, frames: Iterator[Any]) -> bool:
        """Examine one utterance; on the phrase, report it and keep what follows."""
        if time.monotonic() - self._last_detection < WAKE_DEBOUNCE_SECONDS:
            return False
        if not getattr(self.detector, "ready", True):
            # Loading. Examining now would block this thread on the model
            # lock, and the microphone would go unread for as long as that
            # takes -- losing real speech, not just this utterance.
            return False
        window = self._window(span)
        if window is None:
            return False
        end = self._locate(window)
        if end is None:
            return False
        self._last_detection = time.monotonic()
        self._buffer.clear()
        detection = WakeDetection(phrase_end=end)
        with self._lock:
            self._pending = detection
        if self._on_wake is not None:
            self._on_wake()
        with self._lock:
            unclaimed = self._pending is detection
            if unclaimed:
                self._pending = None
        if unclaimed:
            return True
        # Everything after the phrase, and nothing before it.
        cut = min(len(window), max(0, int(round(end * SAMPLE_RATE))))
        # A copy, and the window let go: what came before the phrase is not
        # kept alive for the length of the command.
        remainder = window[cut:].copy()
        window = None
        self._follow_phrase(remainder, frames, detection)
        self._last_detection = time.monotonic()
        self._buffer.clear()
        return True

    def _window(self, span: int | None = None) -> Any:
        import numpy

        frames = list(self._buffer)
        if span is not None:
            frames = frames[-span:]
        if not frames:
            return None
        try:
            return numpy.concatenate(frames).astype("float32")
        except Exception:  # noqa: BLE001
            return None

    def _locate(self, window: Any) -> float | None:
        """Seconds into `window` where the phrase ends; None if it was not said.

        A detector that can only say yes or no is treated as if the phrase
        filled the window: nothing in it is carried forward, and the command is
        whatever is said next.
        """
        locate = getattr(self.detector, "locate", None)
        if callable(locate):
            end = locate(window, self.phrase)
            return None if end is None else float(end)
        return len(window) / SAMPLE_RATE if self.detector.detect(window, self.phrase) else None

    def _examine(self, span: int | None = None) -> bool:
        window = self._window(span)
        return window is not None and self._locate(window) is not None

    def _command_silence_frames(self) -> int:
        silence_ms = int(getattr(self.settings, "voice_silence_ms", 800) or 800)
        return max(TRAILING_SILENCE_FRAMES, silence_ms // FRAME_MS)

    def _follow_phrase(self, remainder: Any, frames: Iterator[Any], detection: WakeDetection) -> None:
        """Capture what is said after the phrase, on the stream that heard it.

        `remainder` is the part of the examined utterance after the phrase;
        the frames queued while it was being examined come next; then the room
        in real time. A command already spoken is kept whole, one still being
        spoken is followed to its end, and one that starts after a pause is
        caught too -- all without a second capture, so nothing falls into a
        gap between two streams.

        Speech is judged a run at a time, the way push-to-talk judges it: a
        run is voice up to a silence long enough to end a sentence, and one
        with less than a syllable in it -- a click, a breath, the tail of
        "SAM" -- is dropped rather than added to the next. Waiting for a
        command to begin is measured in real time once the listener has caught
        up with the room, and never while somebody is talking. If nothing
        comes, nothing is kept, and the loop asks the ordinary way.
        """
        import numpy

        from .voice import rms_energy

        preroll: deque = deque(maxlen=PRE_ROLL_FRAMES)
        command: list[Any] = []
        run_voiced = 0
        silence = 0
        waited = 0
        slow_reads = 0
        onset_deadline: float | None = None
        enough_silence = self._command_silence_frames()
        deadline = time.monotonic() + (POST_WAKE_ONSET_CAP_FRAMES + POST_WAKE_MAX_FRAMES) * FRAME_MS / 1000
        queued = self._queued

        def take(frame: Any) -> bool | None:
            """True: a command is complete. False: none is coming. None: go on."""
            nonlocal run_voiced, silence, waited
            energy = rms_energy(frame)
            voiced = energy >= self.threshold
            self._track_noise(energy, voiced)
            if command:
                command.append(frame)
                if voiced:
                    run_voiced += 1
                    silence = 0
                else:
                    silence += 1
                if silence >= enough_silence:
                    if run_voiced >= POST_WAKE_MIN_SPEECH_FRAMES:
                        return True
                    # A click, a breath, the tail of the phrase: not a command.
                    preroll.clear()
                    preroll.extend(command[-PRE_ROLL_FRAMES:])
                    command.clear()
                    run_voiced = silence = 0
                    return None
                if len(command) >= POST_WAKE_MAX_FRAMES:
                    return run_voiced >= POST_WAKE_MIN_SPEECH_FRAMES
                return None
            if voiced:
                command.extend(preroll)
                command.append(frame)
                run_voiced, silence = 1, 0
                return None
            preroll.append(frame)
            waited += 1
            if waited >= POST_WAKE_ONSET_CAP_FRAMES:
                return False
            if onset_deadline is not None and time.monotonic() >= onset_deadline:
                return False
            return None

        outcome: bool | None = None
        interrupted = False
        try:
            for offset in range(0, len(remainder), FRAME_SAMPLES):
                outcome = take(remainder[offset:offset + FRAME_SAMPLES])
                if outcome is not None:
                    break
            remainder = None
            while outcome is None:
                if self._stop.is_set() or self._suppressed.is_set():
                    interrupted = True
                    break
                if time.monotonic() > deadline:
                    break
                started = time.perf_counter()
                frame = next(frames, None)
                caught_up = frame is None
                if frame is not None:
                    if callable(queued):
                        caught_up = queued() == 0
                    else:
                        slow_reads = slow_reads + 1 if time.perf_counter() - started >= REALTIME_GAP_SECONDS else 0
                        caught_up = slow_reads >= 2
                if caught_up and onset_deadline is None:
                    onset_deadline = time.monotonic() + self.onset_seconds
                if frame is None:
                    if not command and onset_deadline is not None and time.monotonic() >= onset_deadline:
                        outcome = False
                    continue
                outcome = take(frame)
        finally:
            speech = outcome is True and not interrupted
            audio = numpy.concatenate(command).astype("float32") if speech and command else None
            command.clear()
            preroll.clear()
            detection.finish(audio, speech=speech and audio is not None, interrupted=interrupted)

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
