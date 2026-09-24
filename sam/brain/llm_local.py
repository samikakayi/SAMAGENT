"""The local brain as the LAST rung of every ladder (``LLMClient`` mixin).

The user's free cloud quotas ran out on the evening of 2026-09-24 (Gemini
resets at 10:00 Iraq time) and every typed or spoken turn then ended in
«ببورە، ئێستا ناتوانم پەیوەندی بە مۆدێلەکانەوە بکەم». The local model on this
PC (Ollama, qwen3:8b on CPU; numbers in llm_ollama.py) has no quota, so:

- ``chat``/``stream`` try the cloud ladder exactly as before; only when it is
  exhausted (every rung resting, failing, over its daily cap, unconfigured or
  offline) does the local rung answer -- with its own timeout, because the
  cloud round's deadline was already spent and a cold local answer takes
  ~1.5 minutes on this busy PC (a warm one 3-9 s);
- callers opt out with ``local=False`` where a local answer is worse than none
  (a reword of words the user already has, the slow head of a round whose
  fast rungs have not been asked yet, background summaries);
- images never go to it (qwen3:8b has no vision);
- when SAM switches to or from the local brain it says so once: a
  ``ComponentStatus("brain", ...)`` for the panel and a ``VoiceNotice(kind=
  "local")`` whose text the island shows («مێشکی ناوخۆیی»).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:  # llm.py imports this module: runtime imports of it stay inside functions
    from .llm import LLMChunk, LLMError, LLMRequest, LLMResponse

log = logging.getLogger("sam.local_brain")

LOCAL_PROVIDER = "ollama"
# Added to the per-turn part of every local request. Live run 2026-09-25 (after ten
# fast-path turns in the same conversation): qwen3:8b answered «پێم بڵێ زێڕ ئێستا بە
# چەند مامەڵە دەکرێت» with a made-up price and «ئەو پەنجەرانەی ... بژمێرە» with made-up
# counts -- no tool call. The history showed only plain question -> answer pairs
# (tool turns are not replayed), and a small model imitates that pattern.
LOCAL_RULES = ("SAM's local brain is answering now (the online models are resting). Tools are the ONLY source "
               "of prices, the chart, windows, alerts and files: call the tool now, even when an earlier message "
               "mentions a value (it is outdated). For an action call its tool; never say it is done without the "
               "tool's result. Answer in one or two short sentences.")
LOCAL_NOTICE_CKB = "مۆدێلە ئۆنلاینەکان ئێستا بەردەست نین؛ بە مێشکی ناوخۆیی وەڵام دەدەمەوە، کەمێک هێواشترە."
CLOUD_BACK_CKB = "مۆدێلە ئۆنلاینەکان گەڕانەوە."
MODEL_CHECK_S = 600.0


class LocalRung:
    """Mixin for ``LLMClient`` (needs ``backends``, ``_setting``, ``_count``,
    ``_record``, ``_cooling``, ``_cool``, ``_over_cap``, ``bus``)."""

    brain_mode: str = "cloud"          # "cloud" | "local": who answered last
    _local_models: tuple[float, list[str]] | None = None
    _local_warming: Any = None
    _local_inflight: int = 0            # local answers being generated now

    # -- availability ------------------------------------------------------------------------------
    def local_backend(self) -> Any:
        backends = getattr(self, "_backends", None)
        if backends is None:
            backends = self.backends  # type: ignore[attr-defined]
        return backends.get(LOCAL_PROVIDER)

    def local_ready(self) -> bool:
        """The local rung may answer now (enabled, a server answers or can be
        started, not resting after a start failure)."""
        backend = self.local_backend()
        if backend is None:
            return False
        try:
            if not backend.configured():
                return False
        except Exception:  # noqa: BLE001
            return False
        return not self._cooling(LOCAL_PROVIDER, f"{LOCAL_PROVIDER}:*")  # type: ignore[attr-defined]

    def _local_allowed(self, local: bool | None, req: LLMRequest) -> bool:
        if local is False:
            return False
        if req.has_images() and not self._setting("llm.local.vision", False):  # type: ignore[attr-defined]
            return False
        return self.local_ready()

    def cloud_usable(self, refs: list[str]) -> bool:
        """Some cloud rung of ``refs`` could be asked now (configured, not
        resting, under its daily cap). False = the local brain will answer."""
        from .llm import split_ref

        for ref in refs:
            try:
                provider, model = split_ref(ref)
            except ValueError:
                continue
            if provider == LOCAL_PROVIDER:
                continue
            backend = self.backends.get(provider)  # type: ignore[attr-defined]
            try:
                if backend is None or not backend.configured():
                    continue
            except Exception:  # noqa: BLE001
                continue
            if self._cooling(provider, ref) or self._over_cap(provider, model):  # type: ignore[attr-defined]
                continue
            return True
        return False

    async def local_model(self) -> str:
        """``llm.local.model`` when installed, else the first installed
        ``llm.local.fallback_models`` (listing cached 10 minutes)."""
        backend = self.local_backend()
        wanted = str(self._setting("llm.local.model", "qwen3:8b") or "qwen3:8b")  # type: ignore[attr-defined]
        fallbacks = [str(m) for m in (self._setting("llm.local.fallback_models", []) or [])]  # type: ignore[attr-defined]
        cached = self._local_models
        if cached is None or time.monotonic() - cached[0] > MODEL_CHECK_S:
            try:
                names = await backend.list_models()
            except Exception:  # noqa: BLE001 - the request itself will say what is wrong
                names = []
            self._local_models = cached = (time.monotonic(), names)
        installed = set(cached[1])
        for name in [wanted, *fallbacks]:
            if not installed or name in installed or f"{name}:latest" in installed:
                return name
        return wanted

    async def local_cold(self) -> bool:
        """The local model is not in memory (a load of ~8 s and an uncached
        prompt of up to ~86 s wait for the next local answer on this PC)."""
        backend = self.local_backend()
        loaded = getattr(backend, "loaded", None)
        if loaded is None:
            return False
        try:
            return not await loaded()
        except Exception:  # noqa: BLE001 - unknown: say nothing rather than a wrong «one moment»
            return False

    async def announce_local(self, on_local: Any) -> None:
        """Call ``on_local(cold)`` before a local answer (``chat(on_local=)``)."""
        try:
            result = on_local(await self.local_cold())
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001 - an announcement must never cost the answer
            log.debug("on_local callback failed", exc_info=True)

    # -- the rung --------------------------------------------------------------------------------------
    def _local_request(self, req: LLMRequest) -> LLMRequest:
        """The request as the local model gets it: its own timeout, a short
        history (``llm.local.history_messages``, default 0, before the current
        request) and LOCAL_RULES. Measured live 2026-09-25: with the last
        exchange kept, qwen3:8b copied it -- after «نرخی زێڕ و زیو» it answered
        an alert question and a window count with the gold price and no tool
        (3 runs); the A/B without that pattern was equal (2/3 either way).
        Fewer uncached tokens also help: ~38 tok/s on this PC under load."""
        timeout = float(self._setting("llm.local.timeout_s", 150) or 150)  # type: ignore[attr-defined]
        keep = int(self._setting("llm.local.history_messages", 0) or 0)  # type: ignore[attr-defined]
        messages = with_local_rules(trim_history(req.messages, keep))
        return replace(req, timeout_s=timeout, messages=messages)

    async def _local_chat(self, req: LLMRequest, turn: Any, cloud_err: LLMError) -> LLMResponse:
        from .llm import LLMError

        backend = self.local_backend()
        model = await self.local_model()
        local_req = self._local_request(req)
        self._note_brain("local", f"{LOCAL_PROVIDER}:{model}")
        started = time.perf_counter()
        self._local_inflight += 1
        try:
            response = await asyncio.wait_for(backend.complete(model, local_req), local_req.timeout_s + 5)
        except asyncio.TimeoutError:
            err: LLMError = LLMError("timeout", f"no local reply in {local_req.timeout_s:.0f} s",
                                     provider=LOCAL_PROVIDER, model=model)
        except LLMError as exc:
            err = exc
        else:
            response.total_ms = response.total_ms or (time.perf_counter() - started) * 1000.0
            self._count(LOCAL_PROVIDER, model, local_req, usage=response.usage)  # type: ignore[attr-defined]
            self._record(turn, response)  # type: ignore[attr-defined]
            self._note_brain("local", response.model_ref)
            return response
        finally:
            self._local_inflight -= 1         # also when the turn is cancelled (stop_all)
        raise self._local_failed(model, local_req, cloud_err, err)

    async def _local_stream(self, req: LLMRequest, turn: Any, cloud_err: LLMError) -> AsyncIterator[LLMChunk]:
        from .llm import LLMError

        backend = self.local_backend()
        model = await self.local_model()
        local_req = self._local_request(req)
        self._note_brain("local", f"{LOCAL_PROVIDER}:{model}")
        yielded = False
        iterator = backend.stream(model, local_req).__aiter__()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(iterator.__anext__(), local_req.timeout_s)
                except StopAsyncIteration:
                    break
                if chunk.kind == "done" and chunk.response is not None:
                    self._count(LOCAL_PROVIDER, model, local_req, usage=chunk.response.usage)  # type: ignore[attr-defined]
                    self._record(turn, chunk.response)  # type: ignore[attr-defined]
                    self._note_brain("local", chunk.response.model_ref)
                yielded = True
                yield chunk
            return
        except asyncio.TimeoutError:
            err: LLMError = LLMError("timeout", "local stream stalled", provider=LOCAL_PROVIDER, model=model)
        except LLMError as exc:
            err = exc
        finally:
            closer = getattr(iterator, "aclose", None)
            if closer is not None:
                try:
                    await closer()
                except Exception:  # noqa: BLE001
                    pass
        if yielded:
            raise err
        raise self._local_failed(model, local_req, cloud_err, err)

    def _local_failed(self, model: str, req: LLMRequest, cloud_err: LLMError, err: LLMError) -> LLMError:
        from .llm import LLMError

        self._count(LOCAL_PROVIDER, model, req, error=err)  # type: ignore[attr-defined]
        if err.kind in ("network", "unconfigured", "not_found"):
            # No server / no model: do not make every turn wait for the same failure.
            self._cool(f"{LOCAL_PROVIDER}:*", 120.0)  # type: ignore[attr-defined]
        log.info("local brain failed (%s): %s", err.kind, err.message[:120])
        attempts = [*cloud_err.attempts, f"{LOCAL_PROVIDER}:{model}: {err.kind}"]
        return LLMError("exhausted", "; ".join(attempts), attempts=attempts)

    # -- who is answering ------------------------------------------------------------------------------
    def _note_brain(self, mode: str, ref: str) -> None:
        """Publish a switch between the cloud and the local brain (once per
        switch; a cloud answer only matters after a local one)."""
        if mode == self.brain_mode:
            return
        self.brain_mode = mode
        bus = getattr(self, "bus", None)
        if bus is None:
            return
        from ..events import ComponentStatus

        try:
            if mode == "local":
                bus.publish(ComponentStatus(component="brain", state="degraded", detail=f"local: {ref}"))
                notice = _voice_notice("local", LOCAL_NOTICE_CKB, ref)
            else:
                bus.publish(ComponentStatus(component="brain", state="ok", detail=f"cloud: {ref}"))
                notice = _voice_notice("cloud", CLOUD_BACK_CKB, ref)
            if notice is not None:
                bus.publish(notice)
        except Exception:  # noqa: BLE001 - a UI hint must never break a reply
            log.debug("brain notice failed", exc_info=True)

    def note_cloud_answer(self, response: LLMResponse) -> None:
        if response.provider != LOCAL_PROVIDER:
            self._note_brain("cloud", response.model_ref)

    # -- warm-up ---------------------------------------------------------------------------------------
    def prewarm_local(self, messages: list[dict[str, Any]] | None = None,
                      tools: list[dict[str, Any]] | None = None) -> Any:
        """Load the local model (and read the stable prompt into its cache) in
        the background; at most one warm-up at a time. Returns the task or None."""
        if not self._setting("llm.local.prewarm", True) or not self.local_ready():  # type: ignore[attr-defined]
            return None
        if self._local_inflight:
            # Ollama has one slot: a warm-up next to a real answer made that answer wait
            # (live run 2026-09-25: a 98 s voice-prompt warm-up pushed a library answer
            # past the 150 s local timeout).
            return None
        running = self._local_warming
        if running is not None and not running.done():
            return running

        async def warm() -> bool:
            model = await self.local_model()
            ok = await self.local_backend().warm(model, messages, tools)
            log.info("local brain %s warm-up %s", model, "done" if ok else "failed")
            return ok

        try:
            self._local_warming = asyncio.get_running_loop().create_task(warm())
        except RuntimeError:
            return None
        return self._local_warming

    def local_status(self) -> dict[str, Any]:
        backend = self.local_backend()
        out: dict[str, Any] = {"mode": self.brain_mode, "configured": False}
        if backend is None:
            return out
        try:
            out["configured"] = bool(backend.configured())
            out["model"] = str(self._setting("llm.local.model", ""))  # type: ignore[attr-defined]
            server = getattr(backend, "server", None)
            if server is not None and hasattr(server, "status"):
                out["server"] = server.status()
        except Exception:  # noqa: BLE001
            pass
        return out


def trim_history(messages: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """System messages, the last ``keep`` messages before the LAST user message,
    and everything from that user message on (the current tool loop). A worker
    task (one user goal, then tool rounds) is never shortened."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    if last_user < 0:
        return list(messages)
    head = [m for m in messages[:last_user] if m.get("role") == "system"]
    history = [m for m in messages[:last_user] if m.get("role") != "system"]
    kept = history[-keep:] if keep > 0 else []
    while kept and kept[0].get("role") == "tool":      # never start with an orphan tool result
        kept = kept[1:]
    return [*head, *kept, *messages[last_user:]]


def with_local_rules(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """LOCAL_RULES at the end of the system message (its per-turn part, after the
    persona's CONTEXT_HEADING, so Ollama's cached stable prefix is unchanged)."""
    out = list(messages)
    for index, message in enumerate(out):
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            out[index] = {**message, "content": f"{message['content']}\n\n{LOCAL_RULES}"}
            return out
    return [{"role": "system", "content": LOCAL_RULES}, *out]


def _voice_notice(kind: str, text: str, detail: str) -> Any:
    try:
        from ..voice.notices import VoiceNotice
    except Exception:  # noqa: BLE001 - voice package missing: the ComponentStatus still tells the UI
        return None
    return VoiceNotice(kind=kind, text_ckb=text, detail=detail)


__all__ = ["LocalRung", "LOCAL_PROVIDER", "LOCAL_NOTICE_CKB", "CLOUD_BACK_CKB", "LOCAL_RULES", "trim_history",
           "with_local_rules"]
