"""Speaker / MicStream with fake PortAudio streams (no sound, no recording) and
the PCM + device helpers."""

from __future__ import annotations

import asyncio
import math
from array import array

from voice_helpers import FakeOutputStream, settle

from sam.voice.audio import (MicStream, Speaker, level_from_rms, looks_like_headset, pcm16_to_wav, pcm_rms,
                             pick_device, resample_pcm16, wav_to_pcm16)


def pcm_of(value: int, samples: int) -> bytes:
    return array("h", [value] * samples).tobytes()


async def test_speaker_is_one_continuous_stream_for_many_chunks():
    FakeOutputStream.instances.clear()
    speaker = Speaker(stream_factory=FakeOutputStream)
    chunks = [pcm_of(i + 1, 480) for i in range(5)]
    for chunk in chunks:
        assert speaker.write(chunk)
    assert await settle(lambda: speaker.is_open)
    stream = FakeOutputStream.instances[-1]
    assert speaker.opens == 1 and stream.started
    assert stream.kwargs["samplerate"] == 24000 and stream.kwargs["dtype"] == "int16"
    assert stream.kwargs["channels"] == 1 and stream.kwargs["latency"] == "high"
    for chunk in chunks:           # the device pulls exactly what was queued, in order
        assert stream.pull(480) == chunk
    assert stream.pull(480) == bytes(960)  # then silence, stream still running (no per-chunk play/wait)
    speaker.write(pcm_of(9, 480))
    assert stream.pull(480) == pcm_of(9, 480)
    assert speaker.opens == 1


async def test_flush_drops_queued_audio_and_stale_epoch_writes():
    FakeOutputStream.instances.clear()
    speaker = Speaker(stream_factory=FakeOutputStream)
    await speaker.open()
    stream = FakeOutputStream.instances[-1]
    epoch = speaker.epoch
    speaker.write(pcm_of(5, 4800), epoch=epoch)           # 200 ms queued
    assert 190 < speaker.buffered_ms() < 210
    dropped = speaker.flush()                              # barge-in
    assert dropped == 9600 and speaker.buffered_ms() == 0
    assert not speaker.write(pcm_of(7, 480), epoch=epoch)  # late chunk of the cancelled producer
    assert stream.pull(480) == bytes(960)
    assert speaker.write(pcm_of(8, 480), epoch=speaker.epoch)
    assert stream.pull(480) == pcm_of(8, 480)


async def test_wait_idle_and_idle_close_with_fake_clock():
    now = [100.0]
    speaker = Speaker(stream_factory=FakeOutputStream, clock=lambda: now[0], idle_close_s=30)
    await speaker.open()
    stream = FakeOutputStream.instances[-1]
    speaker.write(pcm_of(1, 480))
    assert speaker.playing
    waiter = asyncio.ensure_future(speaker.wait_idle(timeout=2))
    stream.pull(480)
    now[0] += 1.0                 # past the device-buffer tail
    assert await waiter
    assert not await speaker.close_if_idle()     # only 1 s of silence
    now[0] += 40.0
    assert await speaker.close_if_idle() and stream.closed and not speaker.is_open


async def test_speaker_level_callback_throttled():
    levels = []
    speaker = Speaker(stream_factory=FakeOutputStream, on_level=levels.append)
    await speaker.open()
    stream = FakeOutputStream.instances[-1]
    speaker.write(pcm_of(16000, 4800))
    for _ in range(10):
        stream.pull(480)   # 10 callbacks in a burst -> one level event (<= 15 Hz)
    assert len(levels) == 1 and 0.8 < levels[0] <= 1.0


async def test_speaker_open_failure_is_reported_not_raised():
    def broken(**kwargs):
        raise OSError("no output device")
    speaker = Speaker(stream_factory=broken)
    assert not await speaker.open()
    assert "no output device" in (speaker.last_error or "")
    speaker.write(pcm_of(1, 480))
    assert await settle(lambda: not speaker.playing)


class FakeInputStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.closed = False

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        self.closed = True


async def test_mic_stream_delivers_frames_from_the_callback_thread():
    import threading

    created = []

    def factory(**kwargs):
        created.append(FakeInputStream(**kwargs))
        return created[-1]

    mic = MicStream(stream_factory=factory, block_ms=30)
    await mic.start()
    stream = created[0]
    assert stream.kwargs["samplerate"] == 16000 and stream.kwargs["blocksize"] == 480
    assert stream.kwargs["dtype"] == "int16" and stream.kwargs["channels"] == 1
    frames = [pcm_of(i, 480) for i in range(3)]
    thread = threading.Thread(target=lambda: [stream.callback(f, 480, None, None) for f in frames])
    thread.start()
    thread.join()
    got = []

    async def read():
        async for frame in mic.frames():
            got.append(frame)
            if len(got) == 3:
                await mic.stop()
    await asyncio.wait_for(read(), 2)
    assert got == frames and stream.closed


async def test_mic_queue_is_bounded_and_keeps_newest_audio():
    created = []
    mic = MicStream(stream_factory=lambda **kw: created.append(FakeInputStream(**kw)) or created[-1],
                    block_ms=30, max_queue_s=0.3)
    await mic.start()
    for i in range(30):
        mic._deliver(pcm_of(i, 480))  # noqa: SLF001 - simulate a stalled consumer
    assert mic.dropped > 0 and mic._queue.qsize() == 10  # noqa: SLF001


def test_pcm_helpers():
    assert pcm_rms(b"") == 0.0
    assert math.isclose(pcm_rms(pcm_of(16384, 100)), 0.5, rel_tol=1e-3)
    assert level_from_rms(0.0) == 0.0 and level_from_rms(1.0) == 1.0 and 0.4 < level_from_rms(0.03) < 0.6
    pcm = pcm_of(1234, 1600)
    back, rate = wav_to_pcm16(pcm16_to_wav(pcm, 16000))
    assert back == pcm and rate == 16000
    down = resample_pcm16(pcm_of(1000, 2400), 24000, 16000)
    assert len(down) == 1600 * 2 and set(array("h", down)) == {1000}


class FakeSd:
    class default:
        device = (1, 3)
        hostapi = 0

    @staticmethod
    def query_hostapis():
        return [{"name": "MME"}, {"name": "WASAPI"}]

    @staticmethod
    def query_devices():
        return [
            {"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100},
            {"name": "Microphone Array (Realtek(R) Au", "hostapi": 0, "max_input_channels": 4, "max_output_channels": 0, "default_samplerate": 44100},
            {"name": "Headset Microphone (A50 X)", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000},
            {"name": "Speakers (Realtek(R) Audio)", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100},
            {"name": "Headset Earphone (A50 X)", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
            {"name": "Speakers (Realtek(R) Audio)", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
        ]


def test_pick_device_prefers_setting_then_headset_then_default():
    assert pick_device("input", None, ("A50",), sd=FakeSd).index == 2
    assert pick_device("output", None, ("A50",), sd=FakeSd).index == 4
    assert pick_device("input", "realtek", ("A50",), sd=FakeSd).index == 1
    assert pick_device("output", 3, ("A50",), sd=FakeSd).index == 3
    assert pick_device("input", None, (), sd=FakeSd) is None          # -> Windows default
    assert looks_like_headset("Headset Earphone (A50 X)") and not looks_like_headset("Speakers (Realtek(R) Audio)")
