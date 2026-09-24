"""Live voice self-test (design 2.1): is Gemini Live good enough in Sorani?

The Live API's language table lists only "Kurdish (ku)" -- usually Kurmanji --
so Sorani understanding and speech are unproven (reports/realtime-voice.json).
This test, run from Settings and once after a Gemini key appears:

1. Gemini TTS (``voice.tts_model``, lists "Central Kurdish") speaks three
   fixed Sorani sentences (strings.SELFTEST_SENTENCES) -> 24 kHz PCM,
   resampled to the 16 kHz the Live API takes.
2. Each clip is streamed to a Live session in 40 ms chunks at real-time pace,
   followed by silence so the server's VAD ends the turn.
3. Measured per sentence: normalised CER of Live's input transcription vs the
   sentence, whether the spoken reply's transcript is Arabic-script Sorani
   (has ە/ێ/ۆ/ڕ/ڵ, not Latin Kurmanji or Persian), and time to first audio
   from the end of the clip.

Pass = mean CER <= ``voice.selftest_max_cer`` AND at least 2 of 3 replies in
Sorani script AND median TTFA <= ``voice.selftest_max_ttfa_ms`` (the cascade's
estimated TTFA is 2.3-4.5 s, so a slower Live buys nothing). The result is
stored in setting ``voice.selftest`` and drives the "Automatic" engine choice.
Nothing is played on the speakers; no audio is saved. Cost: 3 TTS requests and
one short Live session.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from typing import Any, Callable

from . import strings
from .audio import MIC_RATE, OUT_RATE, resample_pcm16, silence
from .live_config import build_live_config, classify_error, join_parts, turn_is_idle
from .speech_text import cer, sorani_script_ok

log = logging.getLogger("sam.voice.selftest")

SELFTEST_INSTRUCTION = (
    "You are SAM, a voice assistant for a Kurdish user. RESPOND IN CENTRAL KURDISH (SORANI), ARABIC SCRIPT. "
    "YOU MUST RESPOND UNMISTAKABLY IN SORANI. Answer every message with ONE short, natural Sorani sentence. "
    "You have no tools in this conversation; if asked to do something, say briefly that you will."
)
CHUNK_MS = 40


async def _one_turn(session: Any, pcm16: bytes, *, pace: float, turn_timeout_s: float) -> dict[str, Any]:
    """Send one clip, collect what Live heard/said and when audio started."""
    from google.genai import types

    heard: list[str] = []
    said: list[str] = []
    first_audio: list[float] = []
    eos: list[float] = []

    async def sender() -> None:
        step = int(MIC_RATE * CHUNK_MS / 1000) * 2
        for offset in range(0, len(pcm16), step):
            await session.send_realtime_input(audio=types.Blob(data=pcm16[offset:offset + step],
                                                               mime_type="audio/pcm;rate=16000"))
            if pace:
                await asyncio.sleep(CHUNK_MS / 1000.0 * pace)
        eos.append(time.perf_counter())
        tail = silence(1200)  # let the server's VAD (~600 ms silence) close the turn
        for offset in range(0, len(tail), step):
            await session.send_realtime_input(audio=types.Blob(data=tail[offset:offset + step],
                                                               mime_type="audio/pcm;rate=16000"))
            if pace:
                await asyncio.sleep(CHUNK_MS / 1000.0 * pace)

    async def receiver() -> None:
        while True:
            async for msg in session.receive():
                sc = getattr(msg, "server_content", None)
                if sc is None:
                    continue
                if sc.input_transcription is not None and sc.input_transcription.text:
                    heard.append(sc.input_transcription.text)
                if sc.output_transcription is not None and sc.output_transcription.text:
                    said.append(sc.output_transcription.text)
                parts = getattr(getattr(sc, "model_turn", None), "parts", None) or []
                if not first_audio and any(getattr(getattr(p, "inline_data", None), "data", None) for p in parts):
                    first_audio.append(time.perf_counter())
                if turn_is_idle(sc) and (said or first_audio):
                    return

    send_task = asyncio.ensure_future(sender())
    try:
        await asyncio.wait_for(receiver(), turn_timeout_s)
        timed_out = False
    except asyncio.TimeoutError:
        timed_out = True
    finally:
        if not send_task.done():
            send_task.cancel()
        await asyncio.gather(send_task, return_exceptions=True)
    ttfa = (first_audio[0] - eos[0]) * 1000.0 if first_audio and eos else None
    return {"heard": join_parts(heard), "said": join_parts(said), "ttfa_ms": None if ttfa is None else round(ttfa, 1),
            "timed_out": timed_out}


async def run_selftest(app: Any, *, tts: Any = None, client_factory: Callable[[str], Any] | None = None,
                       sentences: tuple[str, ...] = strings.SELFTEST_SENTENCES, pace: float = 1.0,
                       turn_timeout_s: float = 20.0, store: bool = True) -> dict[str, Any]:
    """Run the self-test; returns {"ok","cer","script_ok","ttfa_ms","reply_sample","at",...}."""
    result: dict[str, Any] = {"ok": False, "cer": None, "script_ok": False, "ttfa_ms": None, "reply_sample": "",
                              "at": time.time(), "model": "", "details": [], "error": None}
    key = app.secrets.get("gemini_api_key")
    if not key:
        result["error"] = "no_gemini_key"
        result["message_ckb"] = strings.SELFTEST_NO_KEY
        return _store(app, result, store)
    if tts is None:
        from .tts import GeminiTts
        tts = GeminiTts(app)
    try:
        clips = [resample_pcm16(await tts.synthesize(text), OUT_RATE, MIC_RATE) for text in sentences]
    except Exception as exc:  # noqa: BLE001
        result["error"] = "tts_failed: " + app.redact(f"{type(exc).__name__}: {exc}")[:200]
        return _store(app, result, store)
    if client_factory is not None:
        client = client_factory(key)
    else:
        from google import genai
        client = genai.Client(api_key=key)
    models = [str(app.config.get("voice.live_model", "gemini-3.8-live"))]
    fallback = str(app.config.get("voice.live_fallback_model", "") or "")
    if fallback and fallback not in models:
        models.append(fallback)
    details: list[dict[str, Any]] = []
    for model in models:
        try:
            config = build_live_config(app, model, instruction=SELFTEST_INSTRUCTION, with_tools=False)
            async with client.aio.live.connect(model=model, config=config) as session:
                result["model"] = model
                for sentence, clip in zip(sentences, clips):
                    turn = await _one_turn(session, clip, pace=pace, turn_timeout_s=turn_timeout_s)
                    turn["sentence"] = sentence
                    turn["cer"] = round(cer(sentence, turn["heard"]), 3)
                    turn["script_ok"] = sorani_script_ok(turn["said"])
                    details.append(turn)
            break
        except Exception as exc:  # noqa: BLE001
            kind = classify_error(exc)
            result["error"] = f"live_{kind}: " + app.redact(f"{type(exc).__name__}: {exc}")[:200]
            if kind in ("model", "rejected") and model != models[-1]:
                continue
            return _store(app, result, store)
    result["error"] = None
    result["details"] = details
    cers = [d["cer"] for d in details]
    ttfas = [d["ttfa_ms"] for d in details if d["ttfa_ms"] is not None]
    script_hits = sum(1 for d in details if d["script_ok"])
    result["cer"] = round(sum(cers) / len(cers), 3) if cers else None
    result["script_ok"] = bool(details) and script_hits >= max(1, (2 * len(details) + 2) // 3)
    result["ttfa_ms"] = round(statistics.median(ttfas), 1) if ttfas else None
    result["reply_sample"] = next((d["said"] for d in details if d["said"]), "")[:160]
    max_cer = float(app.config.get("voice.selftest_max_cer", 0.35))
    max_ttfa = float(app.config.get("voice.selftest_max_ttfa_ms", 4000))
    result["ok"] = (result["cer"] is not None and result["cer"] <= max_cer and result["script_ok"]
                    and result["ttfa_ms"] is not None and result["ttfa_ms"] <= max_ttfa)
    return _store(app, result, store)


#: Failures that say nothing about Live's Sorani quality (network, quota, TTS
#: hiccup): the "Automatic" choice must not lock the user into the cascade
#: because of them, and the automatic self-test runs again later.
INCONCLUSIVE_PREFIXES = ("tts_failed", "live_transient", "live_quota")


def selftest_verdict(result: Any) -> str:
    """none | pass | fail | inconclusive for a stored ``voice.selftest``.

    ``fail`` = measured and not good enough (CER / script / TTFA), or Live
    cannot work with this key (auth, model or setup rejected)."""
    if not isinstance(result, dict) or result.get("error") == "no_gemini_key":
        return "none"
    if result.get("ok"):
        return "pass"
    error = str(result.get("error") or "")
    if error.startswith(INCONCLUSIVE_PREFIXES):
        return "inconclusive"
    return "fail"


def _store(app: Any, result: dict[str, Any], store: bool) -> dict[str, Any]:
    if store:
        try:
            app.config.set("voice.selftest", app.redact_obj(result))
            app.db.log_activity("voice", "selftest", ok=bool(result.get("ok")),
                                summary=f"cer={result.get('cer')} script_ok={result.get('script_ok')} "
                                        f"ttfa_ms={result.get('ttfa_ms')} error={result.get('error')}"[:300])
        except Exception:  # noqa: BLE001
            log.exception("could not store the self-test result")
    return result


__all__ = ["run_selftest", "selftest_verdict", "SELFTEST_INSTRUCTION", "INCONCLUSIVE_PREFIXES"]
