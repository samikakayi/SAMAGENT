"""Text LLM client: one call API over a ladder of free models.

Canonical message format is OpenAI chat (``{"role", "content", "tool_calls",
"tool_call_id"}``; images as ``{"type": "image_url", "image_url": {"url":
"data:image/jpeg;base64,..."}}`` parts). Backends convert as needed; keys that
start with ``_`` (e.g. ``_gemini_content``, which keeps Gemini 3 thought
signatures for multi-turn tool use) are provider-private and stripped for the
others.

Why a ladder: free quotas are small and unpublished (Google shows them only
in AI Studio; ~20 RPD for gemini-3.8-flash, ~500 RPD for 3.5-flash-lite,
reports/computer-control.json). On 429 a rung cools down and the next one is
tried at once; transient errors (5xx, timeout, network) are retried ONCE on the
same rung -- v1's 1.5 s + 3 s retry waits were a measured part of its latency
(reports/audit-latency.json) -- but only when the failure came back fast: a
SLOW failure moves on and cools the rung (integration smoke 2026-09-24:
OmniRoute's gemini-3.1-flash-lite answered 503 after 39 s and again after 34 s
on the retry, so one typed turn took 118 s). Reasoning effort is always sent (Gemini 3 cannot
switch thinking off and thinking tokens count against max tokens, which cut a
v1 reply to "ئێ"), and max tokens is at least 4096.

Rests grow with repeated failures (60 -> 180 -> 600 s for a 429, 180 -> 600 ->
1200 s for a slow failure) and survive a restart (table ``llm_health``): the
repair review (2026-09-24) measured sam-fast failing 15 of 38 requests (14 of
them 429) and gemini-3-flash-preview 10 of 18, and with a flat 60/180 s rest
every typed turn paid the same 15 s timeouts again (24-39 s to the first
reply). ``chat`` also takes per-rung timeouts and a deadline for the whole
ladder, so a caller (the conversation) can bound a spoken turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Callable, Iterator, Literal, Protocol

log = logging.getLogger("sam.llm")

ErrorKind = Literal["rate_limit", "auth", "quota", "bad_request", "not_found", "server", "network",
                    "timeout", "unconfigured", "exhausted", "cancelled"]
# "ollama" is the local brain (llm_ollama.py): never in a ladder setting, it is
# the implicit last rung of every ladder (llm_local.py).
PROVIDERS = ("omniroute", "groq", "openrouter", "gemini", "ollama")
MIN_MAX_TOKENS = 4096


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def as_openai(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)}}


@dataclass
class LLMRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None       # OpenAI format (ToolRegistry.openai_tools())
    tool_choice: str | None = None                  # auto|none|required
    max_tokens: int = MIN_MAX_TOKENS
    temperature: float | None = None
    reasoning: str | None = "low"                   # minimal|low|medium|high|None
    json_schema: dict[str, Any] | None = None       # structured output
    timeout_s: float = 40.0

    def has_images(self) -> bool:
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in content):
                return True
        return False


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)   # tokens_in, tokens_out
    ttft_ms: float | None = None
    total_ms: float = 0.0
    raw_message: dict[str, Any] = field(default_factory=dict)  # assistant msg to append to history

    @property
    def model_ref(self) -> str:
        return f"{self.provider}:{self.model}"

    def assistant_message(self) -> dict[str, Any]:
        """The assistant turn to append to ``messages`` before tool results."""
        if self.raw_message:
            return dict(self.raw_message)
        message: dict[str, Any] = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            message["tool_calls"] = [c.as_openai() for c in self.tool_calls]
        return message

    def json(self) -> Any:
        """Parse the reply as JSON (tolerates ```json fences / leading prose)."""
        return parse_json_text(self.text)


@dataclass
class LLMChunk:
    kind: Literal["text", "tool_call", "done"]
    text: str = ""
    tool_call: ToolCall | None = None
    response: LLMResponse | None = None


def _empty_stream(response: "LLMResponse") -> bool:
    """A streamed reply with no text, no tool calls and no finish reason: the
    stream ended without the model ever answering (a legitimately empty reply
    still carries finish_reason "stop")."""
    return not (response.text or "").strip() and not response.tool_calls and not response.finish_reason


class LLMError(Exception):
    """A provider failure. ``message`` is already redacted/short."""

    def __init__(self, kind: ErrorKind, message: str = "", *, provider: str = "", model: str = "",
                 status: int | None = None, retry_after: float | None = None,
                 attempts: list[str] | None = None) -> None:
        super().__init__(f"{provider}:{model} {kind}{f' ({status})' if status else ''}: {message}".strip())
        self.kind = kind
        self.message = message
        self.provider = provider
        self.model = model
        self.status = status
        self.retry_after = retry_after
        self.attempts = attempts or []


class Backend(Protocol):
    provider: str

    def configured(self) -> bool: ...
    async def complete(self, model: str, req: LLMRequest) -> LLMResponse: ...
    def stream(self, model: str, req: LLMRequest) -> AsyncIterator[LLMChunk]: ...
    async def list_models(self) -> list[str]: ...
    async def aclose(self) -> None: ...


def parse_json_text(text: str) -> Any:
    value = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", value, re.S)
    if fence:
        value = fence.group(1).strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start = min([i for i in (value.find("{"), value.find("[")) if i >= 0], default=-1)
        if start < 0:
            raise
        end = max(value.rfind("}"), value.rfind("]"))
        return json.loads(value[start:end + 1])


def split_ref(ref: str) -> tuple[str, str]:
    """'groq:openai/gpt-oss-20b' -> ('groq', 'openai/gpt-oss-20b')."""
    provider, sep, model = ref.partition(":")
    if not sep or provider not in PROVIDERS or not model:
        raise ValueError(f"bad model ref {ref!r}; use provider:model with provider in {PROVIDERS}")
    return provider, model


from .llm_local import LocalRung  # noqa: E402 - imports nothing from here at import time


class LLMClient(LocalRung):
    """Ladder-based client. ``backends`` may be injected (tests: fakes).
    After the ladder, the local brain answers when allowed (``local``; llm_local.py)."""

    def __init__(self, config: Any, secrets: Any, *, db: Any = None, timing: Any = None, bus: Any = None,
                 backends: dict[str, Backend] | None = None) -> None:
        self.config = config
        self.secrets = secrets
        self.db = db
        self.timing = timing
        self.bus = bus
        self._backends = backends
        self._cooldown: dict[str, float] = {}      # ref or "provider:*" -> monotonic deadline
        self._available: dict[str, set[str]] = {}  # provider -> listed model ids
        self._strikes: dict[str, int] = {}         # ref -> consecutive rate-limit/slow failures
        self._strike_at: dict[str, float] = {}     # ref -> wall time of the last counted failure
        self._health_loaded = False

    # -- setup -----------------------------------------------------------------
    @property
    def backends(self) -> dict[str, Backend]:
        if self._backends is None:
            from .llm_backends import default_backends
            self._backends = default_backends(self.config, self.secrets)
        return self._backends

    def _setting(self, key: str, default: Any) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:  # noqa: BLE001
            value = default
        return default if value is None else value

    def ladder(self, name_or_refs: str | list[str]) -> list[str]:
        """Resolve a ladder name ('chat', 'strong', 'vision', 'extract',
        'hard') or a single ref / explicit list into model refs."""
        if isinstance(name_or_refs, list):
            return list(name_or_refs)
        if ":" in name_or_refs:
            return [name_or_refs]
        refs = self._setting(f"llm.ladder.{name_or_refs}", None)
        if not refs:
            raise ValueError(f"unknown ladder {name_or_refs!r}")
        return list(refs)

    def _request(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
                 tool_choice: str | None, max_tokens: int | None, temperature: float | None,
                 reasoning: str | None, json_schema: dict[str, Any] | None, timeout_s: float | None) -> LLMRequest:
        return LLMRequest(
            messages=messages, tools=tools or None, tool_choice=tool_choice,
            max_tokens=max(int(max_tokens or self._setting("llm.max_tokens", MIN_MAX_TOKENS)), MIN_MAX_TOKENS),
            temperature=temperature,
            reasoning=reasoning if reasoning is not None else self._setting("llm.reasoning", "low"),
            json_schema=json_schema,
            timeout_s=float(timeout_s or self._setting("llm.timeout_s", 40)))

    # -- ladder bookkeeping ------------------------------------------------------
    def _cooling(self, provider: str, ref: str) -> bool:
        self._load_health()
        now = time.monotonic()
        return self._cooldown.get(ref, 0) > now or self._cooldown.get(f"{provider}:*", 0) > now

    def cooling(self, ref: str) -> bool:
        """True while ``ref`` (or its whole provider) rests after failures."""
        provider = ref.split(":", 1)[0]
        return self._cooling(provider, ref)

    def strikes(self, ref: str) -> int:
        """Consecutive rate-limit/slow failures of ``ref`` (0 after a success,
        or when the last failure is older than ``llm.strike_decay_s``: a demoted
        rung must get its place back once its trouble is over)."""
        self._load_health()
        if time.time() - self._strike_at.get(ref, 0.0) > float(self._setting("llm.strike_decay_s", 1800)):
            return 0
        return int(self._strikes.get(ref, 0))

    def healthy_order(self, refs: list[str]) -> list[str]:
        """``refs`` with resting rungs last and rungs with recent failures after
        clean ones (stable otherwise): the order the caller chose stays the
        preference, live health only demotes."""
        indexed = list(enumerate(refs))
        indexed.sort(key=lambda item: (self.cooling(item[1]), min(self.strikes(item[1]), 3), item[0]))
        return [ref for _, ref in indexed]

    def _cool(self, key: str, seconds: float) -> None:
        self._cooldown[key] = time.monotonic() + max(1.0, float(seconds))
        self._save_health(key)

    def _strike(self, ref: str, base_s: float, ladder: tuple[float, ...], retry_after: float | None = None) -> None:
        """Rest ``ref`` longer on every consecutive failure (``ladder`` are
        multipliers of ``base_s``); a server's Retry-After is a minimum."""
        count = self.strikes(ref) + 1
        self._strikes[ref] = count
        self._strike_at[ref] = time.time()
        seconds = float(base_s) * ladder[min(count, len(ladder)) - 1]
        if retry_after:
            seconds = max(seconds, float(retry_after))
        self._cool(ref, seconds)

    def _succeeded(self, ref: str) -> None:
        if self._strikes.pop(ref, None) is not None:
            self._save_health(ref)

    def reset_provider(self, provider: str, *, auth_only: bool = False) -> list[str]:
        """Forget the rests (and strikes) of ``provider``: the user saved a new
        key, or its Settings test passed (``auth_only``: only the provider-wide
        rest a 401/403 set). Verify review 2026-09-24: a wrong-key rest is
        persisted, so after pasting the corrected key Gemini stayed skipped for
        up to 10 minutes and across restarts while 'Test' said connected."""
        self._load_health()
        wide = f"{provider}:*"
        if auth_only:
            keys = [wide] if wide in self._cooldown else []
        else:
            keys = sorted({k for k in [*self._cooldown, *self._strikes] if k.split(":", 1)[0] == provider})
        for key in keys:
            self._cooldown.pop(key, None)
            self._strikes.pop(key, None)
            self._strike_at.pop(key, None)
            self._save_health(key)
        if keys:
            log.info("llm rests of %s cleared (%d)", provider, len(keys))
        return keys

    # Persistence: a restart must not forget that a rung keeps failing.
    _HEALTH_SQL = ("CREATE TABLE IF NOT EXISTS llm_health (ref TEXT PRIMARY KEY, strikes INTEGER NOT NULL DEFAULT 0, "
                   "until REAL NOT NULL DEFAULT 0, updated_at REAL NOT NULL)")

    def _load_health(self) -> None:
        if self._health_loaded:
            return
        self._health_loaded = True
        if self.db is None:
            return
        try:
            self.db.ensure_schema("llm", [(1, self._HEALTH_SQL)])
            now_wall, now_mono = time.time(), time.monotonic()
            for row in self.db.query("SELECT ref, strikes, until, updated_at FROM llm_health"):
                if row["strikes"]:
                    self._strikes[str(row["ref"])] = int(row["strikes"])
                    self._strike_at[str(row["ref"])] = float(row["updated_at"] or 0.0)
                left = float(row["until"] or 0) - now_wall
                if left > 0:
                    self._cooldown[str(row["ref"])] = max(self._cooldown.get(str(row["ref"]), 0), now_mono + left)
        except Exception:  # noqa: BLE001 - health memory is an optimisation
            log.debug("llm health load failed", exc_info=True)

    def _save_health(self, key: str) -> None:
        if self.db is None or not self._health_loaded:
            return
        left = self._cooldown.get(key, 0) - time.monotonic()
        try:
            self.db.execute("INSERT INTO llm_health(ref, strikes, until, updated_at) VALUES (?,?,?,?) "
                            "ON CONFLICT(ref) DO UPDATE SET strikes=excluded.strikes, until=excluded.until, "
                            "updated_at=excluded.updated_at",
                            (key, int(self._strikes.get(key, 0)), time.time() + max(0.0, left), time.time()))
        except Exception:  # noqa: BLE001
            log.debug("llm health save failed", exc_info=True)

    def _over_cap(self, provider: str, model: str) -> bool:
        caps = self._setting("llm.daily_caps", {}) or {}
        cap = caps.get(f"{provider}:{model}")
        if not cap or self.db is None:
            return False
        try:
            return int(self.db.usage_for(provider, model).get("requests", 0)) >= int(cap)
        except Exception:  # noqa: BLE001
            return False

    def _count(self, provider: str, model: str, req: LLMRequest, *, error: LLMError | None = None,
               usage: dict[str, int] | None = None) -> None:
        if self.db is None:
            return
        try:
            self.db.bump_usage(provider, model, "vision" if req.has_images() else "text",
                               errors=1 if error else 0,
                               rate_limited=1 if error is not None and error.kind == "rate_limit" else 0,
                               tokens_in=int((usage or {}).get("tokens_in", 0)),
                               tokens_out=int((usage or {}).get("tokens_out", 0)))
        except Exception:  # noqa: BLE001
            log.exception("usage counter failed")

    def _on_error(self, provider: str, ref: str, err: LLMError, elapsed_s: float = 0.0) -> None:
        if err.kind == "rate_limit":
            self._strike(ref, self._setting("llm.cooldown_429_s", 60), (1, 3, 10), err.retry_after)
        elif err.kind in ("server", "timeout") and self._slow(elapsed_s):
            # A rung that needs 30+ s to fail would cost the next turn the same wait.
            self._strike(ref, self._setting("llm.cooldown_slow_s", 180), (1, 10 / 3, 20 / 3))
        elif err.kind == "server":
            # A fast 5xx that also failed its retry: Gemini direct answered "503
            # Service Unavailable" in 1.2-4.1 s to all 9 requests on 2026-09-24
            # (sam2.log); without a rest every turn paid both attempts again.
            self._strike(ref, self._setting("llm.cooldown_429_s", 60), (1, 3, 10))
        elif err.kind in ("auth", "quota"):
            self._cool(f"{provider}:*", self._setting("llm.cooldown_auth_s", 600))
        elif err.kind == "not_found":
            self._cool(ref, 3600)
        elif err.kind in ("network", "unconfigured"):
            self._cool(f"{provider}:*", self._setting("llm.cooldown_down_s", 30))
        if err.kind in ("auth", "quota", "network") and self.bus is not None:
            from ..events import ComponentStatus
            try:
                self.bus.publish(ComponentStatus(component=provider, state="down" if err.kind != "quota" else "degraded",
                                                 detail=err.kind))
            except Exception:  # noqa: BLE001
                pass

    def _candidates(self, refs: list[str], attempts: list[str]) -> Iterator[tuple[str, str, str, Backend]]:
        """Lazily yield usable rungs: cooldowns set by an earlier rung's
        failure (e.g. a provider-wide auth failure) apply to later rungs."""
        for ref in refs:
            try:
                provider, model = split_ref(ref)
            except ValueError:
                attempts.append(f"{ref}: invalid ref")
                continue
            backend = self.backends.get(provider)
            if backend is None or not backend.configured():
                attempts.append(f"{ref}: unconfigured")
                continue
            if self._cooling(provider, ref):
                attempts.append(f"{ref}: cooling down")
                continue
            if self._over_cap(provider, model):
                attempts.append(f"{ref}: daily cap reached")
                continue
            yield ref, provider, model, backend

    @staticmethod
    def _relaxed(req: LLMRequest, err: LLMError) -> LLMRequest | None:
        """A 400 caused by an optional parameter: retry once without it."""
        text = (err.message or "").lower()
        if req.reasoning and ("reason" in text or "thinking" in text):
            return replace(req, reasoning=None)
        if req.json_schema is not None and ("schema" in text or "response_format" in text or "json" in text):
            return replace(req, json_schema=None)
        return None

    # -- public API --------------------------------------------------------------
    async def chat(self, messages: list[dict[str, Any]], *, ladder: str | list[str] = "chat",
                   tools: list[dict[str, Any]] | None = None, tool_choice: str | None = None,
                   max_tokens: int | None = None, temperature: float | None = None, reasoning: str | None = None,
                   json_schema: dict[str, Any] | None = None, timeout_s: float | None = None,
                   turn: Any = None, rung_timeouts: dict[str, float] | None = None,
                   deadline_s: float | None = None, retry_transient: bool = True,
                   local: bool | None = None, on_local: Callable[[bool], Any] | None = None) -> LLMResponse:
        """One completion through the ladder. Raises LLMError('exhausted').

        ``rung_timeouts`` ({ref or provider: seconds}) caps single rungs below
        ``timeout_s`` (e.g. OmniRoute's first reply: good answers came in 1-5 s,
        busy ones failed after 15-40 s); a rung that misses its cap is not
        retried and rests like a slow failure. ``deadline_s`` bounds the whole
        ladder: a rung cut short by it is NOT blamed (no rest) unless it had
        already used 3/4 of its own cap.
        ``retry_transient=False`` moves on after a 5xx/network error instead of
        asking the same rung again (a spoken turn has other rungs; Gemini
        direct answered 503 twice in a row, 3.9 s, in the repair probe).
        ``local``: after an exhausted ladder the local brain answers (its own
        timeout, not the deadline); False skips it (llm_local.py).
        ``on_local(cold)`` is called right before the local brain runs (``cold``
        = its model is not in memory: ~1.5 min on this PC), so a voice turn can
        say «one moment» instead of staying silent."""
        req = self._request(messages, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens,
                            temperature=temperature, reasoning=reasoning, json_schema=json_schema, timeout_s=timeout_s)
        try:
            response = await self._ladder_chat(req, ladder, turn, rung_timeouts, deadline_s, retry_transient)
        except LLMError as err:
            if err.kind == "exhausted" and self._local_allowed(local, req):
                if on_local is not None:
                    await self.announce_local(on_local)
                return await self._local_chat(req, turn, err)
            raise
        self.note_cloud_answer(response)
        return response

    async def _ladder_chat(self, req: LLMRequest, ladder: str | list[str], turn: Any,
                           rung_timeouts: dict[str, float] | None, deadline_s: float | None,
                           retry_transient: bool) -> LLMResponse:
        """The cloud ladder of ``chat`` (every rung except the local brain)."""
        attempts: list[str] = []
        ends = time.monotonic() + float(deadline_s) if deadline_s else None
        for ref, provider, model, backend in self._candidates(self.ladder(ladder), attempts):
            current = req
            for attempt in (1, 2):
                cap = self._rung_cap(ref, provider, rung_timeouts)
                limit = min(current.timeout_s, cap) if cap else current.timeout_s
                cut_by_deadline = False
                if ends is not None:
                    left = ends - time.monotonic()
                    if left < 0.5:
                        attempts.append(f"{ref}: deadline")
                        raise LLMError("exhausted", "; ".join(attempts), attempts=attempts)
                    cut_by_deadline = left < limit
                    limit = min(limit, left)
                started = time.perf_counter()
                try:
                    response = await asyncio.wait_for(backend.complete(model, current), limit)
                except asyncio.TimeoutError:
                    err: LLMError = LLMError("timeout", f"no reply in {limit:.0f} s", provider=provider, model=model)
                except LLMError as exc:
                    err = exc
                else:
                    response.total_ms = response.total_ms or (time.perf_counter() - started) * 1000.0
                    self._count(provider, model, current, usage=response.usage)
                    self._record(turn, response)
                    self._succeeded(ref)
                    return response
                self._count(provider, model, current, error=err)
                elapsed_s = time.perf_counter() - started
                if err.kind == "timeout" and cut_by_deadline:
                    if cap is not None and elapsed_s >= 0.75 * cap:
                        # It had most of its own cap and still said nothing: blame it like a
                        # missed cap, or the next turn pays the same wait again (acceptance
                        # review 2026-09-24: an unblamed hanging Gemini rung cost every turn 7 s).
                        self._on_error(provider, ref, err, max(elapsed_s, self._slow_s()))
                    attempts.append(f"{ref}: deadline")
                    raise LLMError("exhausted", "; ".join(attempts), attempts=attempts)
                attempts.append(f"{ref}: {err.kind}{f' {err.status}' if err.status else ''}")
                log.info("llm %s failed (%s) attempt %d after %.1f s", ref, err.kind, attempt, elapsed_s)
                capped = err.kind == "timeout" and cap is not None and cap < current.timeout_s
                no_retry = capped or (not retry_transient and err.kind in ("server", "timeout", "network"))
                retry = None if no_retry else self._retry_plan(provider, current, err, attempt, elapsed_s)
                if retry is not None:
                    current = retry
                    continue
                # A missed cap counts as slow: the rung would make the next turn wait too.
                self._on_error(provider, ref, err, max(elapsed_s, self._slow_s()) if capped else elapsed_s)
                break
        raise LLMError("exhausted", "; ".join(attempts) or "no model configured", attempts=attempts)

    @staticmethod
    def _rung_cap(ref: str, provider: str, caps: dict[str, float] | None) -> float | None:
        if not caps:
            return None
        value = caps.get(ref, caps.get(provider))
        return float(value) if value else None

    def _slow_s(self) -> float:
        return float(self._setting("llm.slow_failure_s", 8))

    def _slow(self, elapsed_s: float) -> bool:
        return elapsed_s >= self._slow_s()

    def _retry_plan(self, provider: str, req: LLMRequest, err: LLMError, attempt: int,
                    elapsed_s: float = 0.0) -> LLMRequest | None:
        """The request for the single same-rung retry, or None to move on."""
        if attempt != 1:
            return None
        if err.kind == "bad_request":
            return self._relaxed(req, err)
        if err.kind in ("server", "timeout"):
            return None if self._slow(elapsed_s) else req
        # A local gateway that refuses connections is down: do not wait twice.
        if err.kind == "network" and provider != "omniroute":
            return req
        return None

    async def stream(self, messages: list[dict[str, Any]], *, ladder: str | list[str] = "chat",
                     tools: list[dict[str, Any]] | None = None, tool_choice: str | None = None,
                     max_tokens: int | None = None, temperature: float | None = None, reasoning: str | None = None,
                     timeout_s: float | None = None, turn: Any = None,
                     local: bool | None = None) -> AsyncIterator[LLMChunk]:
        """Stream text deltas (kind='text'), then tool calls, then 'done' with
        the full LLMResponse. Falls back to the next rung only if the failure
        happens before the first chunk was yielded; after the ladder, to the
        local brain (``local`` as in ``chat``)."""
        req = self._request(messages, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens,
                            temperature=temperature, reasoning=reasoning, json_schema=None, timeout_s=timeout_s)
        attempts: list[str] = []
        for ref, provider, model, backend in self._candidates(self.ladder(ladder), attempts):
            current = req
            for attempt in (1, 2):
                yielded = False
                started = time.perf_counter()
                iterator = backend.stream(model, current).__aiter__()
                try:
                    while True:
                        try:
                            # Per-chunk stall timeout (first chunk included).
                            chunk = await asyncio.wait_for(iterator.__anext__(), current.timeout_s)
                        except StopAsyncIteration:
                            break
                        if chunk.kind == "done" and chunk.response is not None:
                            response = chunk.response
                            if not yielded and _empty_stream(response):
                                # Measured 2026-09-24 (brain builder): OmniRoute hides an upstream 429
                                # while streaming -- HTTP 200, keepalives, then an empty stream with no
                                # finish reason (7.0 s). Treat it as a rate limit so the rung cools
                                # and the next one answers, instead of returning "" as a reply.
                                raise LLMError("rate_limit", "empty stream (upstream limit hidden by the gateway)",
                                               provider=provider, model=model, retry_after=30.0)
                            response.total_ms = response.total_ms or (time.perf_counter() - started) * 1000.0
                            self._count(provider, model, current, usage=response.usage)
                            self._record(turn, response)
                            self._succeeded(ref)
                            self.note_cloud_answer(response)
                        yielded = True
                        yield chunk
                    return
                except asyncio.TimeoutError:
                    err: LLMError = LLMError("timeout", "stream stalled", provider=provider, model=model)
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
                self._count(provider, model, current, error=err)
                attempts.append(f"{ref}: {err.kind}{f' {err.status}' if err.status else ''}")
                elapsed_s = time.perf_counter() - started
                retry = self._retry_plan(provider, current, err, attempt, elapsed_s)
                if retry is not None:
                    current = retry
                    continue
                self._on_error(provider, ref, err, elapsed_s)
                break
        exhausted = LLMError("exhausted", "; ".join(attempts) or "no model configured", attempts=attempts)
        if not self._local_allowed(local, req):
            raise exhausted
        async for chunk in self._local_stream(req, turn, exhausted):
            yield chunk

    def _record(self, turn: Any, response: LLMResponse) -> None:
        extra = {"model": response.model_ref}
        try:
            if turn is not None:
                if response.ttft_ms is not None:
                    turn.add("llm_first_token", response.ttft_ms, **extra)
                turn.add("llm_total", response.total_ms, **extra)
            elif self.timing is not None:
                self.timing.record("llm_total", response.total_ms, kind="llm", **extra)
        except Exception:  # noqa: BLE001
            log.exception("llm timing failed")

    # -- catalogue / health -------------------------------------------------------
    async def list_models(self, provider: str) -> list[str]:
        backend = self.backends.get(provider)
        if backend is None or not backend.configured():
            raise LLMError("unconfigured", provider=provider)
        models = await backend.list_models()
        self._available[provider] = set(models)
        return models

    async def verify_models(self) -> dict[str, bool | None]:
        """For every ladder ref: True listed, False missing, None unknown.
        Missing refs cool down for an hour so the ladder skips them."""
        refs: list[str] = []
        for key in self.config.all():
            if key.startswith("llm.ladder."):
                refs.extend(r for r in self._setting(key, []) if r not in refs)
        result: dict[str, bool | None] = {}
        for provider in {r.split(":", 1)[0] for r in refs}:
            try:
                await self.list_models(provider)
            except Exception as exc:  # noqa: BLE001
                log.info("model listing for %s failed: %s", provider, type(exc).__name__)
        for ref in refs:
            provider, model = ref.split(":", 1)
            listed = self._available.get(provider)
            result[ref] = None if listed is None else model in listed
            if result[ref] is False:
                self._cool(ref, 3600)
        return result

    async def test_provider(self, provider: str) -> dict[str, Any]:
        """Settings 'Test' button: list models (no quota spent)."""
        started = time.perf_counter()
        try:
            models = await self.list_models(provider)
        except LLMError as err:
            status = {"unconfigured": "unconfigured", "auth": "auth_failed", "rate_limit": "rate_limited",
                      "network": "unreachable"}.get(err.kind, "error")
            return {"ok": False, "provider": provider, "status": status, "detail": err.kind,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "provider": provider, "status": "error", "detail": type(exc).__name__,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        self.reset_provider(provider, auth_only=True)   # the key works: a wrong-key rest is over
        return {"ok": True, "provider": provider, "status": "connected", "models": len(models),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "configured": {p: bool(b.configured()) for p, b in self.backends.items()},
            "cooling": {k: round(v - now) for k, v in self._cooldown.items() if v > now},
            "strikes": dict(self._strikes),
        }

    async def aclose(self) -> None:
        if self._backends:
            for backend in self._backends.values():
                try:
                    await backend.aclose()
                except Exception:  # noqa: BLE001
                    pass


__all__ = ["LLMClient", "LLMRequest", "LLMResponse", "LLMChunk", "LLMError", "ToolCall", "Backend",
           "split_ref", "parse_json_text", "MIN_MAX_TOKENS", "PROVIDERS"]
