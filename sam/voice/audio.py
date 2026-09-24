"""Microphone and speaker I/O (sounddevice / PortAudio), plus PCM helpers.

Formats (Live API docs, reports/realtime-voice.json): the mic sends 16 kHz
16-bit mono PCM in 20-40 ms chunks (30 ms here: also a valid webrtcvad frame);
every voice path speaks 24 kHz 16-bit mono PCM (Gemini Live, Gemini TTS
streaming "audio/l16", KurdishTTS ``/api/tts-stream`` ``stream_format=pcm``).

The speaker is ONE continuous output stream fed from a buffer, never a
play()+wait() per chunk. v1 measured on this PC (sam_backend/voice.py): 200 ms
slices with ``sounddevice.play()+wait()`` cost 466 ms of wall time each (162 ms
to open a stream, 309 ms to wait it out), so 5.0 s of speech took 11.8 s;
one stream played the same 5.0 s in 5.4 s. v1 also measured ``latency='high'``
as 183 ms of buffering vs 91 ms for 'low' -- kept as the margin for this busy
machine; barge-in does not pay it because ``flush()`` empties OUR buffer at
once and only the device's own buffer drains.

Nothing here imports sounddevice or numpy at module level (import cost on this
PC: sounddevice 0.6 s, numpy ~2 s -- measured 2026-09-24).
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import threading
import time
import wave
from array import array
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

log = logging.getLogger("sam.voice.audio")

MIC_RATE = 16_000
OUT_RATE = 24_000
SAMPLE_BYTES = 2
HEADSET_WORDS = ("headset", "headphone", "a50", "earbud", "buds", "airpods", "hands-free", "handsfree")


# -- PCM helpers -------------------------------------------------------------------------

def pcm_rms(pcm: bytes) -> float:
    """RMS of 16-bit mono PCM, 0..1 (pure Python: ~25 us for a 30 ms frame)."""
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[:usable])
    total = 0
    for value in samples:
        total += value * value
    return math.sqrt(total / len(samples)) / 32768.0


def level_from_rms(rms: float) -> float:
    """Perceptual 0..1 level for the island orb: -60 dBFS -> 0, 0 dBFS -> 1."""
    if rms <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, (20.0 * math.log10(rms) + 60.0) / 60.0))


def scale_pcm16(pcm: bytes, gain: float) -> bytes:
    """Scale 16-bit samples by ``gain`` (0..1): the barge-in duck. One 20 ms
    block is 480 samples, cheap enough for the PortAudio callback."""
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    factor = max(0.0, min(1.0, float(gain)))
    return array("h", (int(x * factor) for x in samples)).tobytes()


def silence(ms: float, rate: int = MIC_RATE) -> bytes:
    return b"\x00\x00" * int(rate * ms / 1000.0)


def pcm_seconds(pcm: bytes, rate: int) -> float:
    return len(pcm) / (SAMPLE_BYTES * rate) if rate else 0.0


def resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Linear resample of 16-bit mono PCM (numpy, imported lazily: only used
    by the self-test and by devices that refuse 16/24 kHz)."""
    if from_rate == to_rate or not pcm:
        return pcm
    import numpy

    data = numpy.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=numpy.int16).astype(numpy.float32)
    if data.size == 0:
        return b""
    target = max(1, int(round(data.size * to_rate / from_rate)))
    src_x = numpy.linspace(0.0, 1.0, num=data.size, endpoint=False)
    dst_x = numpy.linspace(0.0, 1.0, num=target, endpoint=False)
    out = numpy.interp(dst_x, src_x, data)
    return numpy.clip(out, -32768, 32767).astype(numpy.int16).tobytes()


def pcm16_to_wav(pcm: bytes, rate: int = MIC_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def wav_to_pcm16(payload: bytes) -> tuple[bytes, int]:
    """Decode a 16-bit WAV (any channel count -> mono). Returns (pcm, rate)."""
    with wave.open(io.BytesIO(payload), "rb") as handle:
        channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"unsupported WAV sample width {width}")
    if channels == 1:
        return frames, rate
    samples = array("h")
    samples.frombytes(frames)
    mono = array("h", (sum(samples[i:i + channels]) // channels for i in range(0, len(samples), channels)))
    return mono.tobytes(), rate


# -- devices -------------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    inputs: int
    outputs: int
    default_rate: int
    is_default: bool = False

    def public(self) -> dict[str, Any]:
        return {"index": self.index, "name": self.name, "hostapi": self.hostapi, "inputs": self.inputs,
                "outputs": self.outputs, "default_rate": self.default_rate, "default": self.is_default}


def _sd() -> Any:
    import sounddevice  # lazy: 0.6 s import measured on this PC

    return sounddevice


def list_devices(sd: Any = None) -> dict[str, Any]:
    """Input/output devices of the DEFAULT host API only (Windows lists each
    device once per host API: MME, DirectSound, WASAPI, WDM-KS). MME is the
    default here and resamples whatever rate it is given (v1 comment,
    sam_backend/voice.py _open_output), so 16/24 kHz streams just work."""
    try:
        sd = sd or _sd()
        devices = sd.query_devices()
        default_in, default_out = sd.default.device
        hostapi_index = sd.default.hostapi
        hostapis = sd.query_hostapis()
        hostapi_name = hostapis[hostapi_index]["name"] if 0 <= hostapi_index < len(hostapis) else ""
    except Exception as exc:  # noqa: BLE001 - no audio stack is a status, not a crash
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200], "inputs": [], "outputs": []}
    inputs, outputs = [], []
    for index, dev in enumerate(devices):
        if dev.get("hostapi") != hostapi_index:
            continue
        info = DeviceInfo(index=index, name=str(dev["name"]), hostapi=hostapi_name,
                          inputs=int(dev["max_input_channels"]), outputs=int(dev["max_output_channels"]),
                          default_rate=int(dev["default_samplerate"]),
                          is_default=index in (default_in, default_out))
        if info.inputs > 0 and "mapper" not in info.name.lower():
            inputs.append(info)
        if info.outputs > 0 and "mapper" not in info.name.lower():
            outputs.append(info)
    return {"ok": True, "hostapi": hostapi_name, "inputs": inputs, "outputs": outputs,
            "default_input": default_in, "default_output": default_out}


def pick_device(kind: str, configured: Any = None, keywords: tuple[str, ...] | list[str] = ("a50",),
                sd: Any = None) -> DeviceInfo | None:
    """Device to open: the setting (index or name part) -> a preferred device
    (the user's A50 X headset) -> None (= the Windows default device)."""
    listing = list_devices(sd)
    pool: list[DeviceInfo] = listing.get("inputs" if kind == "input" else "outputs") or []
    if configured not in (None, ""):
        for dev in pool:
            if (isinstance(configured, int) and dev.index == configured) or \
               (isinstance(configured, str) and configured.strip().lower() in dev.name.lower()):
                return dev
    for word in keywords or ():
        for dev in pool:
            if word.lower() in dev.name.lower():
                return dev
    return None


def refresh_devices(sd: Any = None) -> None:
    """Re-scan devices (PortAudio caches the list at init; the headset may have
    been plugged in since). Only call while no stream is open."""
    sd = sd or _sd()
    try:
        sd._terminate()  # noqa: SLF001 - documented way to re-enumerate in python-sounddevice
        sd._initialize()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        log.debug("device refresh failed", exc_info=True)


def looks_like_headset(name: str | None) -> bool:
    lowered = (name or "").lower()
    return any(word in lowered for word in HEADSET_WORDS)


def default_device_name(kind: str, sd: Any = None) -> str:
    try:
        sd = sd or _sd()
        index = sd.default.device[0 if kind == "input" else 1]
        return str(sd.query_devices(index)["name"]) if index is not None and index >= 0 else ""
    except Exception:  # noqa: BLE001
        return ""


# -- microphone ---------------------------------------------------------------------------------

StreamFactory = Callable[..., Any]


class MicStream:
    """16 kHz int16 mono capture delivering fixed ``block_ms`` frames to an
    async iterator. The PortAudio callback only copies bytes and hands them
    to the core loop (``call_soon_threadsafe``); nothing else runs there.
    Nothing is ever written to disk."""

    def __init__(self, *, device: int | None = None, rate: int = MIC_RATE, block_ms: int = 30,
                 max_queue_s: float = 6.0, stream_factory: StreamFactory | None = None) -> None:
        self.device = device
        self.rate = rate
        self.block = int(rate * block_ms / 1000)
        self.block_ms = block_ms
        self._factory = stream_factory
        self._stream: Any = None
        self._queue: asyncio.Queue[bytes | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._max_items = max(10, int(max_queue_s * 1000 / block_ms))
        self._device_rate = rate
        self.overflows = 0
        self.dropped = 0

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def _deliver(self, data: bytes) -> None:
        queue = self._queue
        if queue is None:
            return
        if queue.qsize() >= self._max_items:
            try:
                queue.get_nowait()  # keep the newest audio: a stalled consumer must not grow memory
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(data)

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status and getattr(status, "input_overflow", False):
            self.overflows += 1
        data = bytes(indata)
        if self._device_rate != self.rate:
            data = resample_pcm16(data, self._device_rate, self.rate)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._deliver, data)

    def _open_sync(self) -> None:
        factory = self._factory or _sd().RawInputStream
        kwargs = dict(samplerate=self.rate, blocksize=self.block, channels=1, dtype="int16",
                      device=self.device, latency="low", callback=self._callback)
        try:
            stream = factory(**kwargs)
        except Exception as refused:  # noqa: BLE001 - retry once at the device's own rate
            if self._factory is not None:
                raise
            try:
                info = _sd().query_devices(self.device, "input")
                device_rate = int(info["default_samplerate"])
            except Exception:  # noqa: BLE001
                raise refused from None
            if device_rate == self.rate:
                raise
            self._device_rate = device_rate
            kwargs.update(samplerate=device_rate, blocksize=int(device_rate * self.block_ms / 1000))
            stream = factory(**kwargs)
        stream.start()
        self._stream = stream

    async def start(self) -> None:
        if self._stream is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        await asyncio.to_thread(self._open_sync)

    async def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            def _close() -> None:
                try:
                    stream.stop()
                finally:
                    stream.close()
            try:
                await asyncio.to_thread(_close)
            except Exception:  # noqa: BLE001
                log.debug("mic close failed", exc_info=True)
        if self._queue is not None:
            self._queue.put_nowait(None)

    async def frames(self) -> AsyncIterator[bytes]:
        """Yield frames until ``stop()``."""
        queue = self._queue
        if queue is None:
            return
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item


# -- speaker -------------------------------------------------------------------------------------

class Speaker:
    """One continuous 24 kHz output stream fed from a byte buffer.

    ``write()`` never blocks (it only appends; the stream is opened in a
    worker thread on first use: ~160 ms, v1 measured). ``flush()`` drops
    everything queued at once (barge-in) and bumps ``epoch`` so late chunks
    of a cancelled producer are ignored. Thread-safe: the PortAudio callback
    and the core loop share the buffer under a lock.
    """

    def __init__(self, *, device: int | None = None, rate: int = OUT_RATE, block_ms: int = 20,
                 latency: str = "high", stream_factory: StreamFactory | None = None,
                 on_level: Callable[[float], None] | None = None, idle_close_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.device = device
        self.rate = rate
        self.block = int(rate * block_ms / 1000)
        self.latency = latency
        self._factory = stream_factory
        self.on_level = on_level
        self.idle_close_s = idle_close_s
        self._clock = clock
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._stream: Any = None
        self._stream_rate = rate
        self._opening = False
        self._open_error: str | None = None
        self.epoch = 0
        self.opens = 0
        self.underflows = 0
        self.bytes_played = 0
        self._last_audio_at = 0.0
        self._last_level_at = 0.0
        self._tail_s = 0.25  # device buffer still sounding after our buffer empties
        self.gain = 1.0      # < 1 ducks the voice while a possible barge-in is checked (engine.py)

    # -- producer side (core loop) -------------------------------------------------------
    def write(self, pcm: bytes, *, epoch: int | None = None, rate: int = OUT_RATE) -> bool:
        """Queue audio. False when dropped (stale epoch, empty, or no device)."""
        if not pcm or (epoch is not None and epoch != self.epoch):
            return False
        if rate != self._stream_rate:
            pcm = resample_pcm16(pcm, rate, self._stream_rate)
        with self._lock:
            self._buf.extend(pcm[: len(pcm) - len(pcm) % 2])
            self._last_audio_at = self._clock()
        if self._stream is None and not self._opening:
            self._schedule_open()
        return True

    def flush(self) -> int:
        """Drop queued audio now; returns the number of bytes dropped."""
        with self._lock:
            dropped = len(self._buf)
            self._buf.clear()
            self.epoch += 1
        return dropped

    def buffered_ms(self) -> float:
        with self._lock:
            return len(self._buf) / (SAMPLE_BYTES * self._stream_rate) * 1000.0

    @property
    def playing(self) -> bool:
        with self._lock:
            if self._buf:
                return True
            return self._stream is not None and (self._clock() - self._last_audio_at) < self._tail_s

    async def wait_idle(self, timeout: float | None = None, poll_s: float = 0.04) -> bool:
        """Wait until everything queued has been played. False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.playing:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            await asyncio.sleep(poll_s)
        return True

    # -- stream lifecycle -----------------------------------------------------------------------
    def _schedule_open(self) -> None:
        self._opening = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            self._open_sync_safe()
        else:
            loop.run_in_executor(None, self._open_sync_safe)

    def _open_sync_safe(self) -> None:
        try:
            self._open_sync()
            self._open_error = None
        except Exception as exc:  # noqa: BLE001 - a missing device is reported, not raised
            self._open_error = f"{type(exc).__name__}: {exc}"[:200]
            log.warning("speaker open failed: %s", self._open_error)
            with self._lock:
                self._buf.clear()
        finally:
            self._opening = False

    def _open_sync(self) -> None:
        if self._stream is not None:
            return
        factory = self._factory or _sd().RawOutputStream
        kwargs = dict(samplerate=self.rate, blocksize=self.block, channels=1, dtype="int16",
                      device=self.device, latency=self.latency, callback=self._callback)
        try:
            stream = factory(**kwargs)
            self._stream_rate = self.rate
        except Exception as refused:  # noqa: BLE001 - a strict host API may refuse 24 kHz
            if self._factory is not None:
                raise
            try:
                device_rate = int(_sd().query_devices(self.device, "output")["default_samplerate"])
            except Exception:  # noqa: BLE001
                raise refused from None
            if device_rate == self.rate:
                raise
            kwargs.update(samplerate=device_rate, blocksize=int(device_rate * self.block / self.rate))
            stream = factory(**kwargs)
            with self._lock:  # queued audio is at 24 kHz: convert it once
                pending = bytes(self._buf)
                self._buf[:] = resample_pcm16(pending, self.rate, device_rate)
            self._stream_rate = device_rate
        stream.start()
        self._stream = stream
        self.opens += 1

    async def open(self) -> bool:
        """Open ahead of the first reply (saves ~160 ms on the first sentence)."""
        if self._stream is None:
            self._opening = True
            await asyncio.to_thread(self._open_sync_safe)
        return self._stream is not None

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    @property
    def last_error(self) -> str | None:
        return self._open_error

    def close(self) -> None:
        stream, self._stream = self._stream, None
        with self._lock:
            self._buf.clear()
        if stream is not None:
            try:
                stream.abort()
            except Exception:  # noqa: BLE001
                pass
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    async def close_if_idle(self) -> bool:
        """Release the device after ``idle_close_s`` of silence."""
        if self._stream is None or self.playing:
            return False
        if self._clock() - self._last_audio_at < self.idle_close_s:
            return False
        await asyncio.to_thread(self.close)
        return True

    # -- consumer side (PortAudio thread) ---------------------------------------------------------
    def _callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status and getattr(status, "output_underflow", False):
            self.underflows += 1
        need = frames * SAMPLE_BYTES
        with self._lock:
            chunk = bytes(self._buf[:need])
            del self._buf[:need]
        if len(chunk) < need:
            chunk = chunk + b"\x00" * (need - len(chunk))
        else:
            self.bytes_played += need
        gain = self.gain
        if gain < 0.999:
            chunk = scale_pcm16(chunk, gain)
        outdata[:] = chunk
        if self.on_level is not None:
            now = time.monotonic()
            if now - self._last_level_at >= 0.066:  # ~15 Hz is plenty for the orb
                self._last_level_at = now
                try:
                    self.on_level(level_from_rms(pcm_rms(chunk)))
                except Exception:  # noqa: BLE001
                    pass


__all__ = ["MicStream", "Speaker", "DeviceInfo", "list_devices", "pick_device", "refresh_devices",
           "looks_like_headset", "default_device_name", "pcm_rms", "level_from_rms", "silence", "pcm_seconds",
           "resample_pcm16", "pcm16_to_wav", "wav_to_pcm16", "MIC_RATE", "OUT_RATE"]
