"""Voice live checks on this PC (run by hand; not part of pytest).

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe acceptance\\voice_live.py                 # devices, mic, hotkey, STT, cascade
    .venv\\Scripts\\python.exe acceptance\\voice_live.py --no-cascade    # skip the STT -> brain -> TTS run
    .venv\\Scripts\\python.exe acceptance\\voice_live.py --selftest      # also the Live self-test (Gemini key)

What it does (nothing is played on the speakers, nothing is recorded or saved):
1. Lists the audio devices of the default host API and whether the A50 X
   headset is present.
2. Opens the input device for ~0.3 s to prove it opens (frames are dropped
   at once) and reports the open time.
3. Registers and releases the global hotkey (default Ctrl+Alt+Space; when
   another program owns it, the engine's fallbacks in order) and reports the
   chord SAM would use.
4. If ``kurdishtts_stt_api_key`` is configured: transcribes a Sorani clip
   (KurdishTTS-synthesised "هێی سام، نرخی زێڕ چەندە؟", 16 kHz, from the lead's
   scratchpad ``clips/``) and reports latency and CER.
5. Cascade end to end on the same clip: STT -> the brain's respond_stream
   (memory + persona + conversation only, so no desktop/trading tool can run)
   -> TTS -> a silent sink that simulates real-time playback; reports the
   per-stage timings (target TTFA <= 4.5 s). The test conversation and its
   turns are deleted afterwards.
6. With a Gemini key and ``--selftest``: the Live self-test (skipped otherwise,
   with the reason).

Keys are read at runtime by SAM 2's own Secrets (DPAPI store / .env of
SAM_HOME) and only sent to their providers; nothing prints them.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _common import Acceptance  # noqa: E402

from sam.app import App  # noqa: E402
from sam.voice.audio import MicStream, list_devices, looks_like_headset, pick_device  # noqa: E402
from sam.voice.cascade import CascadeVoice  # noqa: E402
from sam.voice.hooks import RecordingHooks  # noqa: E402
from sam.voice.hotkey import GlobalHotkey  # noqa: E402
from sam.voice.speech_text import cer  # noqa: E402
from sam.voice.stt import KurdishTtsStt, SttRouter  # noqa: E402
from sam.voice.tts import TtsRouter  # noqa: E402

DEFAULT_HOME = r"C:\Users\samit\Desktop\SAM-Agent"
CLIP_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / "claude" / "c--Users-samit-Desktop-SAM-Agent" / \
    "27d2b6d0-734c-45ad-aed6-5f6894a3ced1" / "scratchpad" / "clips"
CLIP_NAME = "d_ckb_hey_sam_gold"
CLIP_TEXT = "هێی سام، نرخی زێڕ چەندە؟"
GREETING_CLIP = ("c_ckb_hey_sam", "هێی سام")


class SilentSink:
    """Speaker stand-in: keeps a real-time playback clock, plays nothing."""

    def __init__(self) -> None:
        self.epoch = 0
        self.bytes = 0
        self.first_write: float | None = None
        self._ends_at = 0.0

    def write(self, pcm: bytes, *, epoch: int | None = None, rate: int = 24000) -> bool:
        if not pcm or (epoch is not None and epoch != self.epoch):
            return False
        now = time.monotonic()
        if self.first_write is None:
            self.first_write = time.perf_counter()
        self._ends_at = max(self._ends_at, now) + len(pcm) / (2 * rate)
        self.bytes += len(pcm)
        return True

    def flush(self) -> int:
        self.epoch += 1
        self._ends_at = 0.0
        return 0

    def buffered_ms(self) -> float:
        return max(0.0, self._ends_at - time.monotonic()) * 1000.0

    @property
    def playing(self) -> bool:
        return self.buffered_ms() > 0

    async def wait_idle(self, timeout: float | None = None, poll_s: float = 0.05) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.playing:
            if deadline is not None and time.monotonic() > deadline:
                return False
            await asyncio.sleep(poll_s)
        return True


def load_clip(name: str = CLIP_NAME) -> bytes | None:
    """KurdishTTS-synthesised Sorani (float32, 16 kHz, lead's scratchpad) -> int16 PCM."""
    path = CLIP_DIR / f"{name}.npy"
    if not path.is_file():
        return None
    import numpy
    audio = numpy.load(path)
    return (numpy.clip(audio, -1.0, 1.0) * 32767.0).astype(numpy.int16).tobytes()


async def run(args: argparse.Namespace) -> int:
    acc = Acceptance("voice_live")
    app = App(args.home)
    app.bus.bind_loop(asyncio.get_running_loop())
    app.loop = asyncio.get_running_loop()
    try:
        with acc.check("audio devices") as c:
            listing = await asyncio.to_thread(list_devices)
            assert listing.get("ok"), listing.get("error")
            inputs = [d.name for d in listing["inputs"]]
            outputs = [d.name for d in listing["outputs"]]
            c.data = {"hostapi": listing.get("hostapi"), "inputs": inputs, "outputs": outputs,
                      "default_input": listing.get("default_input"), "default_output": listing.get("default_output"),
                      "a50x_present": any("a50" in n.lower() for n in inputs + outputs),
                      "headset_like_output": any(looks_like_headset(n) for n in outputs)}
            c.detail = f"{len(inputs)} inputs, {len(outputs)} outputs, A50 X present: {c.data['a50x_present']}"
            assert inputs and outputs

        with acc.check("microphone opens (frames dropped, nothing saved)") as c:
            device = await asyncio.to_thread(pick_device, "input", app.config.get("voice.input_device"),
                                             tuple(app.config.get("voice.preferred_devices", ["A50"]) or ()))
            mic = MicStream(device=device.index if device else None)
            began = time.perf_counter()
            await mic.start()
            open_ms = (time.perf_counter() - began) * 1000.0
            frames = 0

            async def count() -> None:
                nonlocal frames
                async for _frame in mic.frames():  # dropped immediately
                    frames += 1

            counter = asyncio.ensure_future(count())
            await asyncio.sleep(0.3)
            await mic.stop()
            await asyncio.wait_for(counter, 2)
            c.data = {"device": device.name if device else "Windows default", "open_ms": round(open_ms, 1),
                      "frames_in_300ms": frames, "block_ms": mic.block_ms}
            c.detail = f"opened in {open_ms:.0f} ms, {frames} frames of {mic.block_ms} ms"
            assert frames > 0

        with acc.check("global hotkey (configured chord, else the engine's first free fallback)") as c:
            from sam.voice.engine import VOICE_DEFAULTS
            keys = str(app.config.get("voice.hotkey", "ctrl+alt+space"))
            fallbacks = list(app.config.get("voice.hotkey_fallbacks", VOICE_DEFAULTS["voice.hotkey_fallbacks"]))
            tried: dict[str, str] = {}
            usable = None
            for chord in [keys] + [f for f in fallbacks if f != keys]:
                hotkey = GlobalHotkey(chord, lambda: None)
                ok = await asyncio.to_thread(hotkey.start)   # registered and released at once
                await asyncio.to_thread(hotkey.stop)
                tried[chord] = "free" if ok else str(hotkey.error)
                if ok:
                    usable = chord
                    break
            c.data = {"configured": keys, "tried": tried, "sam_would_use": usable}
            c.detail = ", ".join(f"{k}: {v}" for k, v in tried.items())
            assert usable, "no chord could be registered"

        clip = load_clip()
        with acc.check("KurdishTTS STT on a Sorani clip") as c:
            if clip is None:
                c.skip(f"no Sorani clip found in {CLIP_DIR}")
            if not app.secrets.has("kurdishtts_stt_api_key"):
                c.skip("kurdishtts_stt_api_key is not configured")
            stt = KurdishTtsStt(app)
            result = await stt.transcribe(clip)
            await stt.aclose()
            error_rate = cer(CLIP_TEXT, result.text)
            c.data = {"expected": CLIP_TEXT, "heard": result.text, "cer": round(error_rate, 3),
                      "latency_ms": round(result.ms, 1), "audio_s": round(result.audio_s, 2),
                      "dialect": result.detected_dialect}
            c.detail = f"{result.ms:.0f} ms, CER {error_rate:.2f}: {result.text}"
            assert result.text

        if not args.no_cascade:
            with acc.check("cascade end to end: STT -> brain -> TTS (silent sink)") as c:
                if clip is None:
                    c.skip("no Sorani clip")
                clips = [(CLIP_NAME, clip, CLIP_TEXT)]
                greeting = load_clip(GREETING_CLIP[0])
                if greeting is not None:
                    clips.insert(0, (GREETING_CLIP[0], greeting, GREETING_CLIP[1]))
                await cascade_run(app, clips, c)

        with acc.check("Gemini Live self-test (CER, script, TTFA)") as c:
            if not app.secrets.has("gemini_api_key"):
                c.skip("no Gemini key in the store: Live, Gemini TTS/STT and the self-test cannot be proven yet")
            if not args.selftest:
                c.skip("pass --selftest to spend 3 Gemini TTS requests and one short Live session")
            from sam.voice.selftest import run_selftest
            result = await run_selftest(app, store=True)   # nothing is played; the result drives "Automatic"
            c.data = {k: result.get(k) for k in ("ok", "model", "cer", "script_ok", "ttfa_ms", "reply_sample",
                                                 "error")}
            c.detail = f"model {result.get('model')}, CER {result.get('cer')}, script {result.get('script_ok')}, " \
                       f"TTFA {result.get('ttfa_ms')} ms, error {result.get('error')}"
            assert result.get("error") is None, result.get("error")
    finally:
        await app.llm.aclose()
        app.close()
    return acc.finish()


TURN_WAIT_S = 75.0


async def cascade_run(app: App, clips: list[tuple[str, bytes, str]], c: Any) -> None:
    """One cascade, two spoken turns in ONE conversation: a greeting (no
    tool) and the price question (``get_price`` over MT5, read-only). The
    brain's acknowledgements are pre-cached first, as the engine does on the
    first listening window."""
    status = app.load_packages(["sam.brain.memory", "sam.brain.persona", "sam.brain.conversation",
                                "sam.trading.tools"])
    for name, module in app.loaded.items():
        start = getattr(module, "start", None)
        if start is not None and name != "sam.trading.tools":   # no monitor: only the price tool is needed
            await start(app)
    if app.conversation is None:
        raise RuntimeError(f"brain not loaded: {status}")
    mt5 = getattr(app.trading, "mt5", None)
    mt5_ok = bool(await mt5.connect()) if mt5 is not None else False
    conversation_id = app.conversation.new_conversation("voice")
    sink, hooks = SilentSink(), RecordingHooks()
    stt, tts = SttRouter.default(app), TtsRouter.default(app)
    cascade = CascadeVoice(app, sink, stt, tts, hooks)
    results: list[dict[str, Any]] = []
    try:
        from sam.brain.conversation import ACKS_DO, ACKS_LOOK
        began = time.perf_counter()
        added = await tts.prewarm([*ACKS_DO, *ACKS_LOOK])
        prewarm_ms = (time.perf_counter() - began) * 1000.0
        for name, pcm, expected in clips:
            cascade.last_ttfa_ms = None
            sink.first_write = None
            since = time.time()
            eos = time.perf_counter()
            cascade.submit_utterance(pcm, eos)
            await asyncio.sleep(0.2)
            deadline = time.monotonic() + TURN_WAIT_S
            while cascade.busy and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
            finished = not cascade.busy
            await asyncio.sleep(0.2)
            rows = app.db.query("SELECT turn_id, stage, ms, extra FROM timings WHERE kind='cascade' AND at >= ? "
                                "ORDER BY id", (since - 1,))
            turn_ids = [r["turn_id"] for r in rows if r["stage"] == "end_of_speech"]
            stages: dict[str, Any] = {}
            for r in rows:
                if turn_ids and r["turn_id"] == turn_ids[-1]:
                    label = r["stage"]
                    if label in stages:                     # llm stages repeat per round
                        label = f"{label}#{sum(1 for k in stages if k.split('#')[0] == r['stage']) + 1}"
                    stages[label] = round(r["ms"], 1)
            results.append({"clip": name, "expected": expected, "finished": finished,
                            "ttfa_ms": cascade.last_ttfa_ms, "stages_ms": stages,
                            "tts_cached_first": tts.last_cached})
        turns = app.db.query("SELECT role, text, source FROM turns WHERE conversation_id=? ORDER BY id",
                             (conversation_id,))
        c.data = {"turns": turns, "results": results, "prewarm_added": added, "prewarm_ms": round(prewarm_ms),
                  "mt5_connected": mt5_ok, "tts_provider": tts.last_provider, "stt_provider": stt.last_provider,
                  "audio_s": round(sink.bytes / 48000, 2), "states": [s for s, _ in hooks.states]}
        c.detail = "; ".join(
            f"{r['clip']}: TTFA {r['ttfa_ms']:.0f} ms" if r["ttfa_ms"] else f"{r['clip']}: no audio"
            for r in results) + " (target <= 4500)"
        assert all(r["ttfa_ms"] for r in results), "a turn produced no audio"
    finally:
        await cascade.close()
        await stt.aclose()
        await tts.aclose()
        if mt5 is not None:
            await mt5.close()
        for name, module in reversed(list(app.loaded.items())):
            stop = getattr(module, "stop", None)
            if stop is not None and name != "sam.trading.tools":
                await stop(app)
        with app.db.transaction():  # leave no test conversation behind
            app.db.execute("DELETE FROM turns WHERE conversation_id=?", (conversation_id,))
            app.db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            app.db.execute("DELETE FROM brain_conversation_state WHERE conversation_id=?", (conversation_id,))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", default=os.environ.get("SAM_HOME") or DEFAULT_HOME)
    parser.add_argument("--no-cascade", action="store_true")
    parser.add_argument("--selftest", action="store_true", help="with a Gemini key: run the Live self-test")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
