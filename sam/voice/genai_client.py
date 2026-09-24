"""google-genai clients for the voice package with the SDK's own retries OFF.

Measured tonight (sam2.log 2026-09-24 20:51:31 -> 20:51:58): a Gemini TTS
request answered 429 and the SDK waited ~27 s before asking again -- the
island sat on «بیردەکەمەوە» for ~40 s. In google-genai 2.25.0 the
Interactions API (Gemini TTS) runs on a separate generated client
(``_gaos``) whose retry config comes from ``HttpOptions.retry_options``:
unset -> 3 retries on 408/409/429/5XX, sleeping for the server's
``Retry-After`` without a cap (``_gaos/utils/retries.py``); ``attempts=1``
still means ONE retry there (``_translate_retry_config`` turns ``attempts``
into the retry COUNT, and the main client rewrites 0 to 1). So:

- ``HttpOptions(retry_options=HttpRetryOptions(attempts=1))``: the main
  client (``models.generate_content``: Gemini STT) makes exactly one attempt;
- the Interactions resource gets ``sdk_configuration.retry_config = None``:
  the generated ``create`` then passes no retry config at all.

``tests/test_voice_quota.py`` proves both with an httpx mock transport that
answers 429: exactly one HTTP request each. SAM decides itself what a 429
means (quota.py) and switches to KurdishTTS at once.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger("sam.voice.genai")


def make_client(key: str, *, factory: Callable[[str], Any] | None = None, httpx_async_client: Any = None) -> Any:
    """A ``genai.Client`` that never retries by itself (see module docstring)."""
    if factory is not None:
        return factory(key)
    from google import genai
    from google.genai import types

    options: dict[str, Any] = {"retry_options": types.HttpRetryOptions(attempts=1)}
    if httpx_async_client is not None:  # tests: a mock transport
        options["httpx_async_client"] = httpx_async_client
    client = genai.Client(api_key=key, http_options=types.HttpOptions(**options))
    no_interaction_retries(client)
    return client


def no_interaction_retries(client: Any) -> bool:
    """Turn off the generated Interactions client's retries. True when done."""
    try:
        resource = client.aio.interactions
        config = getattr(resource, "sdk_configuration", None)
        if config is not None and hasattr(config, "retry_config"):
            config.retry_config = None
            return True
    except Exception:  # noqa: BLE001 - an SDK without this resource: nothing to change
        log.debug("could not reach the interactions client", exc_info=True)
    return False


__all__ = ["make_client", "no_interaction_retries"]
