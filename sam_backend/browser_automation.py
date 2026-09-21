from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class BrowserAutomationError(RuntimeError):
    pass


def _web_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BrowserAutomationError("Browser automation accepts only absolute http(s) URLs.")
    if parsed.username or parsed.password:
        raise BrowserAutomationError("Credentials must not be embedded in a URL.")
    return value


def run_browser_workflow(
    arguments: dict[str, Any],
    *,
    workspace: Path,
    max_output_chars: int,
) -> dict[str, Any]:
    """Run one bounded workflow in a fresh, credential-free browser profile.

    The policy layer decides whether the workflow needs approval. This executor
    intentionally offers no JavaScript evaluation, downloads, persistent browser
    profile, cookies import, extension loading, or access to SAM's approval UI.
    """

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise BrowserAutomationError(
            "Browser automation is not installed. Run setup.ps1 again to install the Playwright package."
        ) from exc

    start_url = _web_url(str(arguments.get("url", "")))
    raw_actions = arguments.get("actions") or [{"type": "extract_text", "selector": "body"}]
    if not isinstance(raw_actions, list) or not raw_actions:
        raise BrowserAutomationError("actions must be a non-empty list")
    if len(raw_actions) > 20:
        raise BrowserAutomationError("A browser workflow is limited to 20 actions")

    timeout_ms = max(1_000, min(120_000, int(arguments.get("timeout_ms", 30_000))))
    headless = bool(arguments.get("headless", False))
    results: list[dict[str, Any]] = []

    with sync_playwright() as runtime:
        launch_options = {"headless": headless}
        try:
            browser = runtime.chromium.launch(channel="msedge" if os.name == "nt" else "chrome", **launch_options)
        except PlaywrightError:
            try:
                browser = runtime.chromium.launch(**launch_options)
            except PlaywrightError as exc:
                raise BrowserAutomationError(
                    "No supported isolated browser is available. On Windows, install Microsoft Edge; "
                    "otherwise run 'python -m playwright install chromium'."
                ) from exc

        context = browser.new_context(
            accept_downloads=False,
            ignore_https_errors=False,
            java_script_enabled=True,
            service_workers="block",
        )
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        try:
            page.goto(start_url, wait_until="domcontentloaded", timeout=timeout_ms)
            for index, raw_action in enumerate(raw_actions, 1):
                if not isinstance(raw_action, dict):
                    raise BrowserAutomationError(f"Action {index} must be an object")
                action_type = str(raw_action.get("type", "")).strip().lower()
                selector = str(raw_action.get("selector", "body")).strip() or "body"

                if action_type == "goto":
                    target = _web_url(str(raw_action.get("url", "")))
                    page.goto(target, wait_until="domcontentloaded", timeout=timeout_ms)
                    results.append({"index": index, "type": action_type, "url": page.url})
                elif action_type == "click":
                    page.locator(selector).first.click(timeout=timeout_ms)
                    results.append({"index": index, "type": action_type, "selector": selector, "url": page.url})
                elif action_type == "fill":
                    value = str(raw_action.get("value", ""))
                    if len(value) > 10_000:
                        raise BrowserAutomationError("A fill value is limited to 10,000 characters")
                    page.locator(selector).first.fill(value, timeout=timeout_ms)
                    results.append({"index": index, "type": action_type, "selector": selector, "characters": len(value)})
                elif action_type == "press":
                    key = str(raw_action.get("key", "")).strip()
                    if not key:
                        raise BrowserAutomationError("press requires a key")
                    page.locator(selector).first.press(key, timeout=timeout_ms)
                    results.append({"index": index, "type": action_type, "selector": selector, "key": key, "url": page.url})
                elif action_type == "wait_for":
                    page.locator(selector).first.wait_for(state="visible", timeout=timeout_ms)
                    results.append({"index": index, "type": action_type, "selector": selector})
                elif action_type == "extract_text":
                    text = page.locator(selector).first.inner_text(timeout=timeout_ms)
                    truncated = len(text) > max_output_chars
                    results.append({
                        "index": index,
                        "type": action_type,
                        "selector": selector,
                        "text": text[:max_output_chars],
                        "truncated": truncated,
                    })
                elif action_type == "screenshot":
                    raw_path = str(raw_action.get("path", "browser-screenshot.png"))
                    requested = Path(raw_path).expanduser()
                    screenshot_path = requested.resolve(strict=False) if requested.is_absolute() else (workspace / requested).resolve(strict=False)
                    if not screenshot_path.is_relative_to(workspace):
                        raise BrowserAutomationError("Screenshots must stay inside the configured workspace")
                    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(screenshot_path), full_page=bool(raw_action.get("full_page", True)))
                    results.append({"index": index, "type": action_type, "path": str(screenshot_path)})
                else:
                    raise BrowserAutomationError(
                        f"Unsupported browser action '{action_type}'. Use goto, click, fill, press, wait_for, extract_text, or screenshot."
                    )

            return {
                "ok": True,
                "isolated_profile": True,
                "title": page.title()[:500],
                "url": page.url,
                "actions": results,
            }
        except PlaywrightError as exc:
            raise BrowserAutomationError(f"Browser action failed: {exc}") from exc
        finally:
            context.close()
            browser.close()
