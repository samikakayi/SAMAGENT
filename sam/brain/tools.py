"""ToolRegistry: one definition -> Gemini FunctionDeclarations + OpenAI tools.

v1 sent Sorani commands no tools at all (an English keyword gate) and the
model typed fake calls instead (reports/audit-latency.json). SAM 2 sends every
tool on every turn (~25, no gating) and lets the model choose.

Defining a tool (in any package)::

    from sam.brain.tools import tool, ToolContext, ok, fail

    @tool("open_app",
          description="Open or focus a Windows app by English or Sorani name.",
          description_ckb="کردنەوەی بەرنامە",
          params={"type": "object",
                  "properties": {"name": {"type": "string", "description": "App name"}},
                  "required": ["name"]},
          risk="safe", blocking=True, examples_ckb=("کرۆم بکەرەوە",))
    async def open_app(ctx: ToolContext, name: str) -> dict:
        ...
        return ok("Chrome is open.", window="Google Chrome")

    def register(app):
        app.tools.add(open_app, owner="hands")

Risk is decided by CODE (``risk`` or ``classify(args)``), never by the model:
``safe`` runs at once; ``confirm`` asks the user through the ConfirmBroker
(voice "بەڵێ" or a click, 20 s -> NO); ``blocked`` never runs.

Results are compact JSON-able dicts ``{"ok": bool, "summary": str, "data": ...}``,
redacted and size-capped before any model, log or UI sees them.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterable, Literal

from ..events import EventBus, ToolFinished, ToolStarted, WorkerProgress, new_id
from . import taint

log = logging.getLogger("sam.tools")

Risk = Literal["safe", "confirm", "blocked"]
RISKS: tuple[str, ...] = ("safe", "confirm", "blocked")
Classifier = Callable[[dict[str, Any]], "Risk | tuple[Risk, str | None]"]

MAX_SUMMARY_CHARS = 600
MAX_RESULT_CHARS = 6000


def ok(summary: str, **data: Any) -> dict[str, Any]:
    """Successful tool result."""
    return {"ok": True, "summary": summary, "data": data or None}


def fail(summary: str, **data: Any) -> dict[str, Any]:
    """Failed tool result (the model must report it honestly)."""
    return {"ok": False, "summary": summary, "data": data or None}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str                      # English, for the model
    handler: Callable[..., Any]
    params: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    risk: Risk = "safe"
    description_ckb: str = ""             # Sorani, for the UI
    confirm_text_ckb: Any = None          # str | callable(args)->str | None
    blocking: bool = True                 # Live: BLOCKING vs NON_BLOCKING
    timeout_s: float = 60.0
    classify: Classifier | None = None    # dynamic risk from the arguments
    examples_ckb: tuple[str, ...] = ()    # a few Sorani phrasings (sent to the model)
    owner: str = ""
    private_args: tuple[str, ...] = ()    # logged as their length only (typed text, file content)

    def model_description(self, *, compact: bool = False) -> str:
        """Full: the description + up to 3 Sorani examples. Compact (the
        conversation's core tier): the first sentence + 2 examples -- the
        review measured 33 full schemas at 18.8k characters, 73% of every
        request, which alone used most of Groq's 8k tokens a minute."""
        text = self.description.strip()
        if compact:
            text = text.split(". ")[0].rstrip(".") + "."
        if self.examples_ckb:
            text += " Sorani examples: " + " | ".join(self.examples_ckb[:2 if compact else 3])
        return text


def tool(name: str, *, description: str, params: dict[str, Any] | None = None, risk: Risk = "safe",
         description_ckb: str = "", confirm_text_ckb: Any = None, blocking: bool = True,
         timeout_s: float = 60.0, classify: Classifier | None = None,
         examples_ckb: Iterable[str] = (),
         private_args: Iterable[str] = ()) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator attaching a ``ToolSpec`` as ``fn.tool_spec`` (no global state)."""
    if risk not in RISKS:
        raise ValueError(f"risk must be one of {RISKS}")
    if not name.replace("_", "").isalnum() or not name[0].isalpha():
        raise ValueError(f"bad tool name {name!r}")
    schema = params or {"type": "object", "properties": {}}
    if schema.get("type") != "object":
        raise ValueError("tool params must be a JSON schema of type 'object'")

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.tool_spec = ToolSpec(  # type: ignore[attr-defined]
            name=name, description=description, handler=fn, params=schema, risk=risk,
            description_ckb=description_ckb, confirm_text_ckb=confirm_text_ckb, blocking=blocking,
            timeout_s=timeout_s, classify=classify, examples_ckb=tuple(examples_ckb),
            private_args=tuple(private_args))
        return fn
    return decorate


@dataclass
class ToolContext:
    """Passed as the first argument of every handler."""

    app: Any
    registry: "ToolRegistry"
    name: str
    call_id: str
    source: str                           # live|cascade|text|worker|ui
    cancel: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()

    async def confirm(self, question_ckb: str, detail: str = "") -> bool:
        """Mid-tool confirmation (e.g. a dangerous click inside screen_act)."""
        return await self.registry.ask_confirmation(question_ckb, detail, self.name)

    def progress(self, step: int, max_steps: int, text_ckb: str, *, done: bool = False,
                 ok_: bool | None = None) -> None:
        """Progress line for long tools (island progress bar)."""
        if self.registry.bus is not None:
            self.registry.bus.publish(WorkerProgress(task_id=self.call_id, step=step, max_steps=max_steps,
                                                     text_ckb=text_ckb, done=done, ok=ok_))


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    """Light coercion for what models commonly send (numbers as strings...)."""
    kind = schema.get("type")
    try:
        if kind == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        if kind == "integer" and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind == "number" and isinstance(value, str):
            return float(value.strip().replace(",", ""))
        if kind == "boolean" and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "1"):
                return True
            if lowered in ("false", "no", "0"):
                return False
        if kind == "array" and isinstance(value, str):
            parsed = json.loads(value) if value.strip().startswith("[") else [v.strip() for v in value.split(",") if v.strip()]
            return parsed
        if kind == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
            return [_coerce(v, schema["items"]) for v in value]
        if kind == "object" and isinstance(value, dict) and isinstance(schema.get("properties"), dict):
            props = schema["properties"]
            return {k: (_coerce(v, props[k]) if k in props else v) for k, v in value.items()}
    except (ValueError, TypeError, json.JSONDecodeError):
        return value
    return value


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Coerce + check required/enum/basic types. Returns (args, error).

    Arguments the schema does not declare are DROPPED unless the schema sets
    ``additionalProperties`` (true or a sub-schema). Measured 2026-09-24:
    Gemini called the parameterless ``tv_open`` with ``{"reason": ...}``, and a
    handler without ``**kwargs`` then failed with TypeError. Dropping here
    covers every caller (Live, cascade, text, worker, UI) in one place."""
    props: dict[str, Any] = schema.get("properties") or {}
    extra_ok = schema.get("additionalProperties", False)
    clean: dict[str, Any] = {}
    for key, value in args.items():
        if key in props:
            clean[key] = _coerce(value, props[key])
        elif extra_ok is True or isinstance(extra_ok, dict):
            clean[key] = value
    for key in schema.get("required") or []:
        if key not in clean or clean[key] is None or clean[key] == "":
            return clean, f"missing required argument '{key}'"
    type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
                "array": list, "object": dict}
    for key, value in clean.items():
        spec = props.get(key) or {}
        expected = type_map.get(spec.get("type", ""))
        if expected is not None and value is not None and not isinstance(value, expected):
            return clean, f"argument '{key}' must be {spec.get('type')}"
        if expected is int and isinstance(value, bool):
            return clean, f"argument '{key}' must be integer"
        if "enum" in spec and value is not None and value not in spec["enum"]:
            return clean, f"argument '{key}' must be one of {spec['enum']}"
    return clean, None


class ToolRegistry:
    """Holds ToolSpecs; generates schemas; dispatches with safety + timing."""

    def __init__(self, *, app: Any = None, bus: EventBus | None = None, confirm: Any = None,
                 timing: Any = None, db: Any = None,
                 redact_obj: Callable[[Any], Any] | None = None) -> None:
        self.app = app
        self.bus = bus
        self.confirm_broker = confirm
        self.timing = timing
        self.db = db
        self._redact_obj = redact_obj or (lambda obj: obj)
        self._specs: dict[str, ToolSpec] = {}
        self._running: dict[str, tuple[asyncio.Task[Any] | None, ToolContext]] = {}

    # -- registration --------------------------------------------------------
    def add(self, fn_or_spec: Callable[..., Any] | ToolSpec, *, owner: str = "", replace: bool = False) -> ToolSpec:
        spec = fn_or_spec if isinstance(fn_or_spec, ToolSpec) else getattr(fn_or_spec, "tool_spec", None)
        if not isinstance(spec, ToolSpec):
            raise TypeError("add() needs a @tool-decorated function or a ToolSpec")
        if owner and not spec.owner:
            spec = ToolSpec(**{**spec.__dict__, "owner": owner})
        if spec.name in self._specs and not replace:
            raise ValueError(f"tool {spec.name!r} is already registered by {self._specs[spec.name].owner!r}")
        self._specs[spec.name] = spec
        return spec

    def add_from(self, source: ModuleType | Iterable[Any], *, owner: str = "") -> list[str]:
        """Register every @tool function of a module (or iterable)."""
        items = vars(source).values() if isinstance(source, ModuleType) else source
        added = []
        for item in list(items):
            if isinstance(getattr(item, "tool_spec", None), ToolSpec):
                added.append(self.add(item, owner=owner).name)
        return added

    def remove(self, name: str) -> None:
        self._specs.pop(name, None)

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self, names: Iterable[str] | None = None) -> list[ToolSpec]:
        if names is None:
            return [self._specs[n] for n in sorted(self._specs)]
        return [self._specs[n] for n in names if n in self._specs]

    # -- schema generation -----------------------------------------------------
    def gemini_declarations(self, names: Iterable[str] | None = None, *, live: bool = True) -> list[Any]:
        """``google.genai.types.FunctionDeclaration`` list. ``behavior`` is only
        accepted by the Live API (BidiGenerateContent), so it is set only when
        ``live``; 3.8 Live defaults to NON_BLOCKING, so quick tools whose answer
        the model needs are marked BLOCKING explicitly."""
        from google.genai import types  # lazy: the SDK import costs ~2 s here

        out = []
        for spec in self.specs(names):
            kwargs: dict[str, Any] = {"name": spec.name, "description": spec.model_description()}
            if spec.params.get("properties"):
                kwargs["parameters_json_schema"] = spec.params
            if live:
                kwargs["behavior"] = types.Behavior.BLOCKING if spec.blocking else types.Behavior.NON_BLOCKING
            out.append(types.FunctionDeclaration(**kwargs))
        return out

    def gemini_tools(self, names: Iterable[str] | None = None, *, live: bool = True) -> list[Any]:
        from google.genai import types

        declarations = self.gemini_declarations(names, live=live)
        return [types.Tool(function_declarations=declarations)] if declarations else []

    def openai_tools(self, names: Iterable[str] | None = None, *, compact: bool = False) -> list[dict[str, Any]]:
        """OpenAI-compatible ``tools`` (Groq, OmniRoute, OpenRouter).
        ``compact`` shortens descriptions (see ``ToolSpec.model_description``)."""
        return [{"type": "function", "function": {
            "name": s.name, "description": s.model_description(compact=compact),
            "parameters": s.params if s.params.get("properties") else {"type": "object", "properties": {}}}}
            for s in self.specs(names)]

    def describe_for_prompt(self, names: Iterable[str] | None = None) -> str:
        """One line per tool, for the persona's 'what I can do' section."""
        return "\n".join(f"- {s.name}: {s.description.strip().split('. ')[0]}" for s in self.specs(names))

    # -- safety ----------------------------------------------------------------
    def risk_of(self, name: str, args: dict[str, Any]) -> tuple[Risk, str | None]:
        """(risk, confirm_text) decided by code from tool + arguments."""
        spec = self._specs[name]
        risk: Risk = spec.risk
        text: str | None = None
        if spec.classify is not None:
            try:
                verdict = spec.classify(args)
            except Exception:  # noqa: BLE001 - a broken classifier must fail safe
                log.exception("risk classifier of %s failed", name)
                verdict = "confirm"
            if isinstance(verdict, tuple):
                risk, text = verdict[0], verdict[1]
            else:
                risk = verdict
            # The classifier's verdict replaces the static risk (e.g. the
            # run_powershell allowlist returns "safe" for read-only commands,
            # "blocked" for credential dumping). Unknown verdicts fail safe.
            if risk not in RISKS:
                risk = "confirm"
        if risk == "confirm" and not text:
            template = spec.confirm_text_ckb
            if callable(template):
                try:
                    text = template(args)
                except Exception:  # noqa: BLE001
                    text = None
            elif isinstance(template, str):
                try:
                    text = template.format(**args)
                except (KeyError, IndexError, ValueError):
                    text = template
            if not text:
                text = f"دڵنیایت کە ئەمە بکەم؟ ({spec.description_ckb or spec.name})"
        return risk, text

    async def ask_confirmation(self, question_ckb: str, detail: str, tool_name: str) -> bool:
        if self.confirm_broker is None:
            return False  # no way to ask = NO
        started = time.perf_counter()
        approved = bool(await self.confirm_broker.confirm(question_ckb, detail, tool_name=tool_name))
        if self.timing is not None:
            self.timing.record("confirm_wait", (time.perf_counter() - started) * 1000.0, kind="tool",
                               tool=tool_name, approved=approved)
        return approved

    # -- dispatch ----------------------------------------------------------------
    async def dispatch(self, name: str, args: dict[str, Any] | str | None = None, *, source: str = "text",
                       call_id: str | None = None) -> dict[str, Any]:
        """Validate, gate, run and normalise one tool call. Never raises."""
        call_id = call_id or new_id()
        spec = self._specs.get(name)
        if spec is None:
            return fail(f"Unknown tool '{name}'. Available: {', '.join(self.names())}")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return fail(f"Arguments for {name} were not valid JSON.")
        args = dict(args or {})
        args, error = validate_args(spec.params, args)
        if error:
            return fail(f"{name}: {error}.")
        redacted_args = self._redact_obj(args)
        safe_args = self._public_args(spec, redacted_args)

        risk, question = self.risk_of(name, args)
        scope = taint.current()
        if risk == "safe" and scope is not None:
            gated = taint.check(name, args, scope)
            if gated is not None:
                risk, question = gated  # type: ignore[assignment]
        if risk == "blocked":
            result = fail(question or f"'{name}' with these arguments is blocked by SAM's safety rules.",
                          blocked=True)
            self._log(spec, source, result, 0.0, safe_args)
            return result
        if risk == "confirm":
            # The card shows the real (redacted) arguments -- e.g. the message about
            # to be sent; only the question itself is read aloud.
            detail = json.dumps(redacted_args, ensure_ascii=False)[:400]
            if not await self.ask_confirmation(question or name, detail, name):
                result = fail("The user did not approve this action, so it was not done.", declined=True)
                self._log(spec, source, result, 0.0, safe_args)
                return result

        ctx = ToolContext(app=self.app, registry=self, name=name, call_id=call_id, source=source)
        if self.bus is not None:
            self.bus.publish(ToolStarted(call_id=call_id, name=name, args=safe_args, source=source))
        started = time.perf_counter()
        task: asyncio.Task[Any] = asyncio.ensure_future(self._invoke(spec, ctx, args))
        self._running[call_id] = (task, ctx)
        try:
            raw = await asyncio.wait_for(asyncio.shield(task), timeout=spec.timeout_s)
            result = self._normalise(raw)
        except asyncio.TimeoutError:
            ctx.cancel.set()
            task.cancel()
            result = fail(f"{name} timed out after {spec.timeout_s:.0f} s.", timeout=True)
        except asyncio.CancelledError:
            me = asyncio.current_task()
            if me is not None and me.cancelling():
                # The caller itself is being cancelled: propagate after
                # stopping the handler.
                ctx.cancel.set()
                task.cancel()
                raise
            # Only the handler task was cancelled (stop_all -> cancel_all).
            result = fail("Stopped by the user.", cancelled=True)
        except Exception as exc:  # noqa: BLE001 - errors become honest results
            log.warning("tool %s failed: %s", name, exc, exc_info=True)
            result = fail(f"{name} failed: {type(exc).__name__}: {exc}")
        finally:
            self._running.pop(call_id, None)
        duration = (time.perf_counter() - started) * 1000.0
        taint.note(scope, name, result)
        result = self._cap(self._redact_obj(result))
        if self.timing is not None:
            self.timing.record(f"tool:{name}", duration, kind="tool", turn_id=call_id, ok=result["ok"], source=source)
        if self.bus is not None:
            self.bus.publish(ToolFinished(call_id=call_id, name=name, ok=bool(result["ok"]),
                                          summary=str(result.get("summary", ""))[:300],
                                          duration_ms=round(duration, 1), source=source))
        self._log(spec, source, result, duration, safe_args)
        return result

    async def _invoke(self, spec: ToolSpec, ctx: ToolContext, args: dict[str, Any]) -> Any:
        handler = spec.handler
        if inspect.iscoroutinefunction(handler):
            return await handler(ctx, **args)
        result = await asyncio.to_thread(handler, ctx, **args)
        if inspect.isawaitable(result):
            result = await result
        return result

    @staticmethod
    def _normalise(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict) and "ok" in raw:
            out = {"ok": bool(raw.get("ok")), "summary": str(raw.get("summary") or raw.get("summary_ckb") or ""),
                   "data": raw.get("data")}
            if out["data"] is None:
                extra = {k: v for k, v in raw.items() if k not in ("ok", "summary", "summary_ckb", "data")}
                out["data"] = extra or None
            return out
        if raw is None:
            return ok("Done.")
        if isinstance(raw, str):
            return ok(raw)
        return ok("Done.", result=raw)

    @staticmethod
    def _cap(result: dict[str, Any]) -> dict[str, Any]:
        summary = str(result.get("summary", ""))
        if len(summary) > MAX_SUMMARY_CHARS:
            summary = summary[:MAX_SUMMARY_CHARS - 1] + "…"
        out = {"ok": bool(result.get("ok")), "summary": summary, "data": result.get("data")}
        try:
            encoded = json.dumps(out["data"], ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            encoded = json.dumps(str(out["data"]), ensure_ascii=False)
            out["data"] = str(out["data"])
        if len(encoded) > MAX_RESULT_CHARS:
            out["data"] = {"truncated": True, "preview": encoded[:MAX_RESULT_CHARS]}
        return out

    @staticmethod
    def _public_args(spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
        """Arguments for events, the activity table and the conversation's
        tool turns: private ones (text typed into apps, file content -- could be
        a dictated password) become their length only (review 2026-09-24:
        they were stored verbatim in sam2.sqlite3 with no retention limit)."""
        if not spec.private_args:
            return args
        return {k: (f"<{len(str(v))} chars>" if k in spec.private_args and v not in (None, "") else v)
                for k, v in args.items()}

    def _log(self, spec: ToolSpec, source: str, result: dict[str, Any], duration: float, args: Any) -> None:
        if self.db is None:
            return
        try:
            self.db.log_activity("tool", spec.name, ok=bool(result.get("ok")), summary=str(result.get("summary", "")),
                                 detail={"args": args}, duration_ms=round(duration, 1), source=source)
        except Exception:  # noqa: BLE001
            log.exception("activity log failed")

    # -- cancellation ------------------------------------------------------------
    def running(self) -> list[dict[str, Any]]:
        return [{"call_id": cid, "name": ctx.name, "source": ctx.source} for cid, (_, ctx) in self._running.items()]

    def cancel_all(self, *, except_call: str | None = None) -> int:
        """Cancel every in-flight tool (stop_all). Returns how many."""
        count = 0
        for call_id, (task, ctx) in list(self._running.items()):
            if call_id == except_call:
                continue
            ctx.cancel.set()
            if task is not None and not task.done():
                task.cancel()
            count += 1
        return count


__all__ = ["tool", "ToolSpec", "ToolContext", "ToolRegistry", "ok", "fail", "validate_args", "Risk"]
