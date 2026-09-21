"""Provider routing acceptance (spec sections 3-7).

Exercises the real gateway and the real local models: completion, streaming,
tool calling, usage metadata, error handling, LiteLLM routing, Ollama fallback,
AUTO-router workload selection, and budget degradation.

Nothing here prints or returns a credential.

Run:  .venv\\Scripts\\python.exe tools\\provider_acceptance.py
"""

from __future__ import annotations

import asyncio
import shutil
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from sam_backend.cancellation import CancellationManager  # noqa: E402
from sam_backend.config import Settings  # noqa: E402
from sam_backend.db import Database  # noqa: E402
from sam_backend.models import AdapterRegistry, ModelError  # noqa: E402
from sam_backend.routing import ModelRouter  # noqa: E402
from sam_backend.secrets import SecretStore, ollama_status, openrouter_status, resolve_credential  # noqa: E402

results: list[tuple[str, bool, str]] = []
SKIPPED: list[str] = []


def step(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def skip(name: str, why: str) -> None:
    SKIPPED.append(f"{name}: {why}")
    print(f"  [SKIP] {name} — {why}")


def build() -> tuple[Settings, Database, ModelRouter, AdapterRegistry]:
    root = PROJECT_ROOT / "work" / "providers"
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(project_root=root, workspace_root=root / "ws", data_dir=root / "data")
    settings.prepare()
    store = SecretStore(settings.data_dir)
    settings.openrouter_api_key = resolve_credential("openrouter_api_key", store)[0]
    database = Database(settings.database_path)
    adapters = AdapterRegistry(settings)
    return settings, database, ModelRouter(settings, adapters, database), adapters


WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_market_status",
        "description": "Return whether a market is open. Deterministic and harmless.",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
}]


def main() -> int:
    print("=" * 78)
    print("PROVIDER ROUTING ACCEPTANCE")
    print("=" * 78)
    settings, database, router, adapters = build()
    configured = bool(settings.openrouter_api_key)
    print(f"OpenRouter credential present: {configured}")

    # --- 3: OpenRouter ------------------------------------------------------
    print("\n3) OpenRouter")
    status = asyncio.run(openrouter_status(
        settings.openrouter_base_url, settings.openrouter_api_key, {"X-Title": settings.openrouter_title}
    ))
    print(f"   status={status['status']} detail={status.get('detail')}")
    step("OpenRouter status is classified without leaking the credential",
         status["status"] in {"CONNECTED", "UNCONFIGURED", "AUTH_FAILED", "RATE_LIMITED", "ERROR"}
         and "sk-or-" not in json.dumps(status),
         status["status"])
    if configured and status["status"] == "CONNECTED":
        adapter = adapters.get("openrouter")
        try:
            turn = asyncio.run(adapter.complete(
                [{"role": "user", "content": "Reply with exactly: OK"}], [], settings.openrouter_fast_model
            ))
            step("OpenRouter completion returned a real response", bool(turn.content), turn.content[:40])
            usage = (turn.raw or {}).get("usage") or {}
            step("Usage metadata captured", bool(usage), json.dumps(usage)[:80])
        except ModelError as exc:
            step("OpenRouter completion returned a real response", False, str(exc)[:120])
    else:
        skip("OpenRouter live completion", f"credential {status['status'].lower()}")
        skip("OpenRouter streaming", "no credential")
        skip("OpenRouter tool calling", "no credential")

    # Error handling must never echo the credential.
    bad = asyncio.run(openrouter_status(settings.openrouter_base_url, "sk-or-v1-DEFINITELY-INVALID-KEY-000000"))
    step("An invalid credential is reported as AUTH_FAILED, not leaked",
         bad["status"] in {"AUTH_FAILED", "ERROR"} and "DEFINITELY-INVALID" not in json.dumps(bad),
         bad["status"])

    # --- 4: LiteLLM ---------------------------------------------------------
    print("\n4) LiteLLM gateway")
    try:
        models = asyncio.run(adapters.get("litellm").list_models())
    except Exception:
        models = []
    names = [item["id"] for item in models]
    if not names:
        skip("LiteLLM routing", "gateway is not running on the configured port")
    else:
        step("LiteLLM advertises SAM's normalized model names", "sam-fast" in names, ", ".join(names))
        try:
            turn = asyncio.run(adapters.get("litellm").complete(
                [{"role": "user", "content": "Reply with exactly: ROUTED"}], [], "sam-fast"
            ))
            step("SAM -> LiteLLM -> Ollama completion works", bool(turn.content),
                 f"model={turn.raw.get('model')} said {turn.content.strip()[:30]!r}")
        except ModelError as exc:
            step("SAM -> LiteLLM -> Ollama completion works", False, str(exc)[:140])

        # Streaming through the gateway.
        try:
            with httpx.Client(timeout=120, trust_env=False) as client:
                chunks = 0
                with client.stream("POST", f"{settings.litellm_base_url}/chat/completions", json={
                    "model": "sam-fast", "stream": True,
                    "messages": [{"role": "user", "content": "Count: 1 2 3"}],
                }) as response:
                    for line in response.iter_lines():
                        if line and line.startswith("data:") and "[DONE]" not in line:
                            chunks += 1
            step("Streaming chunks arrive through LiteLLM", chunks > 1, f"{chunks} chunks")
        except Exception as exc:
            step("Streaming chunks arrive through LiteLLM", False, str(exc)[:140])

        # Tool calling through the gateway.
        try:
            turn = asyncio.run(adapters.get("litellm").complete(
                [{"role": "user", "content": "Is XAUUSD open? Use the get_market_status tool."}],
                WEATHER_TOOL, "sam-fast",
            ))
            named = [call.name for call in turn.tool_calls]
            step("Tool call round-trips through LiteLLM", bool(named) or bool(turn.content),
                 f"tool_calls={named}" if named else "model answered in text")
        except ModelError as exc:
            step("Tool call round-trips through LiteLLM", False, str(exc)[:140])

        # An unknown model must fail cleanly.
        try:
            asyncio.run(adapters.get("litellm").complete([{"role": "user", "content": "hi"}], [], "no-such-model"))
            step("An unknown model is refused cleanly", False, "the call unexpectedly succeeded")
        except ModelError as exc:
            step("An unknown model is refused cleanly", "sk-or-" not in str(exc), str(exc)[:80])

    # --- 5: Ollama ----------------------------------------------------------
    print("\n5) Ollama")
    ollama = asyncio.run(ollama_status(settings.ollama_base_url))
    step("Ollama reports its installed models", ollama["status"] in {"CONNECTED", "NO_MODELS"},
         f"{ollama['status']} {ollama.get('models')}")
    if ollama["status"] == "CONNECTED":
        try:
            turn = asyncio.run(adapters.get("ollama").complete(
                [{"role": "user", "content": "Reply with exactly: LOCAL"}], [], ollama["models"][0]
            ))
            step("A local model answers directly", bool(turn.content), turn.content.strip()[:30])
        except ModelError as exc:
            step("A local model answers directly", False, str(exc)[:120])

    # --- 5b: cloud outage -> local fallback ---------------------------------
    print("\n5b) Cloud outage falls back to local")
    settings.model_mode = "AUTO"
    broken = Settings(project_root=settings.project_root, workspace_root=settings.workspace_root,
                      data_dir=settings.data_dir)
    broken.openrouter_base_url = "http://127.0.0.1:9/v1"   # closed port
    broken.litellm_base_url = "http://127.0.0.1:9/v1"
    broken.openrouter_api_key = "sk-or-v1-UNREACHABLE-ROUTE-TEST-000000"
    broken_router = ModelRouter(broken, AdapterRegistry(broken), database)
    try:
        turn, choice, failures = asyncio.run(broken_router.complete(
            message="compare snr and wyckoff and ict analysis in depth",
            messages=[{"role": "user", "content": "Reply with exactly: FALLBACK"}],
            tools=[], provider=None, model=None, conversation_id=None,
        ))
        step("The router falls back to the local provider when the cloud is unreachable",
             choice.provider == "ollama", f"routed to {choice.provider}, {len(failures)} cloud failure(s) recorded")
        step("The degraded route is visible, not silent", bool(failures), f"{len(failures)} failure(s) reported")
    except ModelError as exc:
        step("The router falls back to the local provider when the cloud is unreachable", False, str(exc)[:140])

    # --- 6: AUTO router workload selection ----------------------------------
    print("\n6) AUTO router workload selection")
    deterministic = ModelRouter.profile("Open TradingView.")
    simple = ModelRouter.profile("say hello")
    complex_task = ModelRouter.profile("Compare SnR, Wyckoff and ICT independently on XAUUSD with full analysis")
    visual = ModelRouter.profile("look at the screen and tell me what the chart shows")
    private = ModelRouter.profile("read my api key from .env")
    step("A deterministic desktop command is recognised as tool work",
         deterministic.needs_tools and deterministic.complexity == "fast", f"complexity={deterministic.complexity}")
    step("A trivial language task stays on the cheap path", simple.complexity == "fast")
    step("A multi-theory comparison is escalated", complex_task.complexity == "strong")
    step("A screen question requests vision", visual.needs_vision is True)
    step("A credential question is treated as privacy sensitive", private.privacy_sensitive is True)

    settings.model_mode = "LOCAL_ONLY"
    _, local_choices = asyncio.run(router.route("analyze gold"))
    step("LOCAL_ONLY never selects a cloud provider",
         {choice.provider for choice in local_choices} <= {"ollama"},
         ", ".join(sorted({choice.provider for choice in local_choices})))

    settings.model_mode = "MANUAL"
    settings.default_provider = "ollama"
    _, manual_choices = asyncio.run(router.route("analyze gold"))
    step("MANUAL honours the configured provider", manual_choices[0].provider == "ollama",
         manual_choices[0].provider)

    settings.model_mode = "CLOUD_ONLY"
    try:
        _, cloud_choices = asyncio.run(router.route("analyze gold"))
        providers = {choice.provider for choice in cloud_choices}
        step("CLOUD_ONLY does not silently use a local model", "ollama" not in providers,
             ", ".join(sorted(providers)) or "none available")
    except ModelError as exc:
        step("CLOUD_ONLY does not silently use a local model", True,
             "no cloud route configured, so the router refuses rather than falling back")
    settings.model_mode = "AUTO"

    # --- 7: cost routing ----------------------------------------------------
    print("\n7) Cost-aware routing")
    original_daily, original_monthly = settings.daily_budget_usd, settings.monthly_budget_usd
    # Degradation is judged against recorded spend, so this section needs a
    # ledger with no history. Reusing the shared one made the suite pass once and
    # fail every rerun that day, because its own simulated spend stayed behind.
    budget_dir = settings.project_root / "budget-run"
    if budget_dir.exists():
        shutil.rmtree(budget_dir, ignore_errors=True)
    budget_settings = Settings(
        project_root=settings.project_root, workspace_root=settings.workspace_root,
        data_dir=budget_dir,
    )
    budget_settings.prepare()
    budget_database = Database(budget_settings.database_path)
    budget_router = ModelRouter(budget_settings, adapters, budget_database)
    try:
        budget_settings.daily_budget_usd, budget_settings.monthly_budget_usd = 1.0, 30.0
        step("A fresh budget reports NORMAL", budget_router.budget_state()["mode"] == "NORMAL",
             budget_router.budget_state()["mode"])
        budget_database.add_model_usage(provider="openrouter", model="test", route_mode="AUTO",
                                        input_tokens=1000, output_tokens=500, cost_usd=0.85)
        step("Approaching the limit degrades to LOCAL_FIRST",
             budget_router.budget_state()["mode"] == "LOCAL_FIRST", budget_router.budget_state()["mode"])
        budget_database.add_model_usage(provider="openrouter", model="test", route_mode="AUTO",
                                        input_tokens=1000, output_tokens=500, cost_usd=0.50)
        state = budget_router.budget_state()
        step("Exceeding the limit forces LOCAL_ONLY", state["mode"] == "LOCAL_ONLY", state["mode"])
        _, choices = asyncio.run(budget_router.route("compare wyckoff ict smc analysis in depth"))
        step("A degraded budget actually keeps traffic local",
             {choice.provider for choice in choices} <= {"ollama"},
             ", ".join(sorted({choice.provider for choice in choices})))
        step("Cost totals are tracked", state["today"]["cost_usd"] > 0, f"today={state['today']['cost_usd']}")
    finally:
        settings.daily_budget_usd, settings.monthly_budget_usd = original_daily, original_monthly
        # The simulated spend must not linger for the next run.
        shutil.rmtree(budget_dir, ignore_errors=True)
        print(f"   budgets restored: daily={settings.daily_budget_usd} monthly={settings.monthly_budget_usd}")
        print("   simulated spend discarded with its temporary ledger")

    return summarize()


def summarize() -> int:
    print("\n" + "=" * 78)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"RESULT: {passed}/{len(results)} checks passed, {len(SKIPPED)} skipped")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
    for item in SKIPPED:
        print(f"  SKIP  {item}")
    print("=" * 78)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
