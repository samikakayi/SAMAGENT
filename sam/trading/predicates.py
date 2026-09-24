"""Deterministic rule checks for strategy cards.

A card rule is either tied to one of these predicates (checked from the
engine's numbers, never guessed) or marked for visual/LLM judgement. v1 had 7
predicates (``analyst._execute_custom_theory``); the research plan asked for
~30 built on outputs the engine already computes
(reports/trading-intelligence.json). Each predicate returns
``(passed, detail)`` where ``passed`` is True/False, or None when it cannot be
decided (timeframe not analysed, data missing) -- "unknown" must never count
as a pass.

Timeframe params accept a timeframe (``"M15"``, ``"١٥ خولەک"``) or a card role
(``"bias" | "setup" | "entry"``). Direction params accept up/down/long/short/
bullish/bearish/range, Sorani (سەرەوە/خوارەوە) or ``"setup"`` (= the
direction being evaluated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..textnorm import normalize_ckb
from .engine.types import Candle, Direction, normalize_timeframe

Result = tuple[bool | None, str]


@dataclass
class PredicateContext:
    analyses: dict[str, dict[str, Any]]
    candles: dict[str, list[Candle]]
    price: float
    direction: str | None = None                # "long" | "short" | None
    now: float = 0.0                            # UTC unix seconds
    plan: dict[str, Any] | None = None          # {"rr", "stop", "targets"}
    extra: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)  # {"USDX": {tf: analysis}}
    spread: float | None = None
    roles: dict[str, str] = field(default_factory=dict)  # {"bias": "H1", "setup": "M15", "entry": "M5"}
    settings: dict[str, Any] = field(default_factory=dict)
    cache: dict[Any, Any] = field(default_factory=dict)  # per-evaluation memo (structure breaks)


@dataclass(frozen=True)
class PredicateSpec:
    name: str
    fn: Callable[[PredicateContext, dict[str, Any]], Result]
    params: dict[str, dict[str, Any]]
    description: str
    description_ckb: str


PREDICATES: dict[str, PredicateSpec] = {}


def predicate(pred_name: str, description: str, description_ckb: str, /, **params: dict[str, Any]):
    """Register a predicate; keyword arguments describe its params (JSON-schema-ish)."""
    def register(fn: Callable[[PredicateContext, dict[str, Any]], Result]) -> Callable[..., Result]:
        PREDICATES[pred_name] = PredicateSpec(pred_name, fn, params, description, description_ckb)
        return fn
    return register


# --- helpers -------------------------------------------------------------------------

_UP = {"up", "bull", "bullish", "long", "buy", "higher", "سەرەوە", "بەرەوسەرەوە", "بەرز", "کڕین"}
_DOWN = {"down", "bear", "bearish", "short", "sell", "lower", "خوارەوە", "بەرەوخوارەوە", "نزم", "فرۆشتن"}
_RANGE = {"range", "neutral", "sideways", "flat", "بێئاراستە", "ڕەینج"}


def to_direction(value: Any, ctx: PredicateContext | None = None) -> Direction | None:
    """Normalise a direction word; 'setup'/'any'/empty -> the context's direction."""
    text = normalize_ckb(str(value or ""), strip_punct=True).replace(" ", "")
    if text in _UP:
        return Direction.BULLISH
    if text in _DOWN:
        return Direction.BEARISH
    if text in _RANGE:
        return Direction.NEUTRAL
    if ctx is not None and ctx.direction in ("long", "short"):
        return Direction.BULLISH if ctx.direction == "long" else Direction.BEARISH
    return None


def _tf(ctx: PredicateContext, params: dict[str, Any], default_role: str = "entry") -> str:
    raw = str(params.get("tf") or "").strip()
    if raw.lower() in ("bias", "setup", "entry"):
        return ctx.roles.get(raw.lower(), "")
    if not raw:
        return ctx.roles.get(default_role) or next(iter(ctx.analyses), "")
    return normalize_timeframe(raw)


def _analysis(ctx: PredicateContext, params: dict[str, Any], role: str = "entry") -> tuple[str, dict[str, Any] | None]:
    tf = _tf(ctx, params, role)
    return tf, ctx.analyses.get(tf)


def _missing(tf: str) -> Result:
    return None, f"{tf or 'timeframe'} was not analysed"


def _atr(analysis: dict[str, Any]) -> float:
    return float((analysis.get("indicators") or {}).get("atr14") or 0.0)


def _num(params: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def _int(params: dict[str, Any], key: str, default: int) -> int:
    return int(_num(params, key, default))


# --- card evaluation ------------------------------------------------------------------

KIND_ROLE = {"bias": "bias", "setup": "setup", "trigger": "entry", "entry": "entry", "stop": "entry",
             "target": "entry", "risk": "entry", "filter": "entry", "manage": "entry"}


def coerce_params(spec: PredicateSpec, raw: dict[str, Any]) -> dict[str, Any]:
    """Model-written params -> typed values (unknown names are kept)."""
    out: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        kind = (spec.params.get(key) or {}).get("type")
        try:
            if kind == "number" and value is not None:
                value = float(value)
            elif kind == "integer" and value is not None:
                value = int(float(value))
            elif kind == "boolean" and isinstance(value, str):
                value = value.strip().lower() in ("true", "yes", "1")
        except (TypeError, ValueError):
            pass
        out[key] = value
    for key, meta in spec.params.items():
        if key not in out and "default" in meta:
            out[key] = meta["default"]
    return out


def evaluate_rule(rule: dict[str, Any], ctx: PredicateContext) -> dict[str, Any]:
    base = {"id": rule.get("id"), "kind": rule.get("kind"), "text_ckb": rule.get("text_ckb", ""),
            "text_en": rule.get("text_en", "")}
    check = rule.get("check") or {}
    name = check.get("predicate")
    if not name:
        return {**base, "passed": None, "how": "llm", "detail": "judged from the chart"}
    spec = PREDICATES.get(name)
    if spec is None:
        return {**base, "passed": None, "how": "unsupported", "detail": f"unknown predicate {name}"}
    params = coerce_params(spec, check.get("params") or {})
    if "tf" in spec.params and not params.get("tf"):
        params["tf"] = KIND_ROLE.get(str(rule.get("kind")), "entry")
    try:
        passed, detail = spec.fn(ctx, params)
    except Exception as exc:  # noqa: BLE001 - one bad rule must not sink the analysis
        passed, detail = None, f"{name} failed: {type(exc).__name__}"
    return {**base, "passed": passed, "how": "predicate", "predicate": name, "detail": detail}


def evaluate_card(card: dict[str, Any], ctx: PredicateContext) -> dict[str, Any]:
    """Every rule of a card -> pass/fail/unknown with how it was decided."""
    rules = [evaluate_rule(rule, ctx) for rule in card.get("rules") or []]
    return {"id": card.get("id"), "title_ckb": card.get("title_ckb"), "title_en": card.get("title_en"),
            "version": card.get("version"), "direction": ctx.direction, "rules": rules,
            "all_passed": bool(rules) and all(r["passed"] is True for r in rules),
            "pending": [r["id"] for r in rules if r["how"] == "llm" and r["passed"] is None],
            "failed": [r["id"] for r in rules if r["passed"] is False]}


def _param_hint(name: str, meta: dict[str, Any]) -> str:
    if meta.get("enum"):
        return f"{name}={'|'.join(str(v) for v in meta['enum'])}"
    if "default" in meta:
        return f"{name}={meta['default']}"
    return name


def catalogue_for_prompt() -> str:
    """One line per predicate for the ingest prompt, with allowed values."""
    lines = []
    for spec in PREDICATES.values():
        params = ", ".join(_param_hint(k, v) for k, v in spec.params.items())
        lines.append(f"- {spec.name}({params}): {spec.description}")
    return "\n".join(lines)


def parse_predicate_call(text: str) -> tuple[str, dict[str, Any]]:
    """'swept(tf=M15, level=asian_low)' or 'swept(M15, asian_low)' -> (name, params).

    Models sometimes write the call inline instead of the name + params list;
    positional arguments follow the predicate's declared parameter order.
    """
    raw = (text or "").strip()
    if "(" not in raw:
        return raw.strip().lower(), {}
    name, _, rest = raw.partition("(")
    name = name.strip().lower()
    spec = PREDICATES.get(name)
    order = list(spec.params) if spec else []
    params: dict[str, Any] = {}
    for index, part in enumerate(p.strip() for p in rest.rstrip(") ").split(",") if p.strip()):
        key, sep, value = part.partition("=")
        if sep:
            params[key.strip()] = value.strip().strip("'\"")
        elif index < len(order):
            params[order[index]] = key.strip().strip("'\"")
    return name, params


# Importing the check library registers every predicate in PREDICATES.
from .predicate_checks import KILLZONES, trading_day_start  # noqa: E402

__all__ = ["PREDICATES", "PredicateContext", "PredicateSpec", "evaluate_card", "evaluate_rule", "coerce_params",
           "catalogue_for_prompt", "parse_predicate_call", "to_direction", "trading_day_start", "KILLZONES"]
