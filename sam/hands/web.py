"""Web: open a URL in the default browser, search, read a page's text.

Search: Gemini with Google Search grounding when the user has pasted a
Gemini key (fresh, cited answers; free tier), otherwise DuckDuckGo's keyless
HTML endpoint (ported from v1 ``search.py``: bounded on every axis, parsed
without executing anything). Everything read from the web is untrusted DATA
and is returned under ``untrusted``; it never becomes an instruction.

``fetch_page`` refuses loopback/private/link-local addresses (DNS checked
before connecting and after every redirect): pages must not be able to make
SAM read OmniRoute (127.0.0.1:20128) or TradingView's DevTools port.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import logging
import os
import re
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

log = logging.getLogger("sam.hands.web")

HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140 Safari/537.36"
TIMEOUT_S = 12.0
MAX_RESULTS = 8
MAX_PAGE_BYTES = 1_500_000
PAGE_TEXT_CHARS = 4500
_RESULT_BLOCK = re.compile(r'<a[^>]+class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
                           r'(?P<rest>.*?)(?=<a[^>]+class="result__a"|\Z)', re.DOTALL | re.IGNORECASE)
_SNIPPET = re.compile(r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.DOTALL | re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_DROP = re.compile(r"<(script|style|noscript|svg|template|iframe|head|title)\b.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
_BLOCK_TAGS = re.compile(r"</?(p|div|br|li|h[1-6]|tr|section|article|header|footer|ul|ol|table)\b[^>]*>", re.I)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def _text(raw: str, limit: int) -> str:
    stripped = html.unescape(_TAGS.sub(" ", raw or ""))
    return re.sub(r"\s+([,.;:!?)\]])", r"\1", " ".join(stripped.split()))[:limit]


def _direct_url(href: str) -> str:
    """Unwrap DuckDuckGo's redirect so the real destination is returned."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(html.unescape(href))
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            parsed = urlparse(unquote(target[0]))
    return parsed.geturl() if parsed.scheme in ("http", "https") else ""


def parse_ddg_html(body: str, limit: int = MAX_RESULTS) -> list[SearchResult]:
    results: list[SearchResult] = []
    for match in _RESULT_BLOCK.finditer(body or ""):
        url = _direct_url(match.group("href"))
        title = _text(match.group("title"), 200)
        if not url or not title or "duckduckgo.com/y.js" in url:
            continue
        snippet_match = _SNIPPET.search(match.group("rest") or "")
        results.append(SearchResult(title, url, _text(snippet_match.group("snippet"), 400) if snippet_match else ""))
        if len(results) >= limit:
            break
    return results


def page_text(body: str) -> tuple[str, str]:
    """(title, readable text) of an HTML page, scripts and styles removed."""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", body or "", re.DOTALL | re.IGNORECASE)
    title = _text(title_match.group(1), 200) if title_match else ""
    cleaned = _BLOCK_TAGS.sub("\n", _DROP.sub(" ", body or ""))
    lines = [" ".join(html.unescape(_TAGS.sub(" ", line)).split()) for line in cleaned.split("\n")]
    return title, "\n".join(line for line in lines if len(line) > 1)


def is_public_host(host: str, resolver: Callable[..., Any] = socket.getaddrinfo) -> bool:
    """True only if every address of ``host`` is a public internet address."""
    if not host or host.lower() in ("localhost", "localhost.localdomain") or host.endswith(".local"):
        return False
    try:
        infos = resolver(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if (address.is_private or address.is_loopback or address.is_link_local or address.is_multicast
                or address.is_reserved or address.is_unspecified):
            return False
    return bool(infos)


GEMINI_SEARCH_REST_S = 300.0


class Web:
    def __init__(self, app: Any, *, client_factory: Callable[[], Any] | None = None,
                 genai_factory: Callable[[str], Any] | None = None, startfile_fn: Any = None,
                 resolver: Callable[..., Any] = socket.getaddrinfo) -> None:
        self.app = app
        self._client_factory = client_factory
        self._genai_factory = genai_factory
        self._genai: Any = None
        self._genai_fp: str | None = None
        self._startfile = startfile_fn or (lambda url: os.startfile(url))  # type: ignore[attr-defined]
        self._resolver = resolver
        self._gemini_rest_until = 0.0   # monotonic: grounded search failed lately, use DuckDuckGo meanwhile

    def _client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        import httpx

        return httpx.AsyncClient(timeout=TIMEOUT_S, trust_env=False, follow_redirects=False,
                                 headers={"User-Agent": USER_AGENT, "Accept-Language": "en,ckb;q=0.8,ar;q=0.6"})

    async def open_url(self, url: str) -> dict[str, Any]:
        await asyncio.to_thread(self._startfile, url)
        return {"ok": True, "url": url, "summary": f"Opened {url} in the default browser."}

    # -- search ----------------------------------------------------------------------
    async def search(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        cleaned = " ".join(str(query or "").split())[:400]
        if not cleaned:
            return {"ok": False, "summary": "A search needs a query."}
        if self.app.secrets.has("gemini_api_key") and self._gemini_usable():
            try:
                return await self._search_gemini(cleaned)
            except Exception as exc:  # noqa: BLE001 - fall back to DuckDuckGo
                self._gemini_rest_until = time.monotonic() + GEMINI_SEARCH_REST_S
                log.info("gemini grounded search failed: %s", type(exc).__name__)
        return await self._search_ddg(cleaned, limit)

    def _gemini_usable(self) -> bool:
        """Grounded search uses the conversation's own Gemini model and quota:
        skip it while it rests (a 429 cost every search a request on the user's
        evening test, 2026-09-24 20:55-20:56, before DuckDuckGo answered)."""
        if time.monotonic() < self._gemini_rest_until:
            return False
        model = str(self.app.config.get("hands.search_model", "gemini-3.5-flash-lite"))
        llm = getattr(self.app, "llm", None)
        try:
            return not (llm is not None and llm.cooling(f"gemini:{model}"))
        except Exception:  # noqa: BLE001
            return True

    async def _search_ddg(self, query: str, limit: int) -> dict[str, Any]:
        try:
            async with self._client() as client:
                response = await client.post(HTML_ENDPOINT, data={"q": query})
                response.raise_for_status()
                results = parse_ddg_html(response.text, max(1, min(limit, MAX_RESULTS)))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "summary": f"Web search failed: {type(exc).__name__}."}
        if not results:
            return {"ok": True, "engine": "duckduckgo", "results": [], "summary": f"No results for '{query}'."}
        return {"ok": True, "engine": "duckduckgo", "query": query,
                "untrusted": [r.as_dict() for r in results],
                "summary": f"{len(results)} web results for '{query}' (DuckDuckGo)."}

    async def _search_gemini(self, query: str) -> dict[str, Any]:
        """Gemini answer grounded with Google Search (free tier, counted)."""
        import hashlib

        from google.genai import types

        key = self.app.secrets.get("gemini_api_key")
        if not key:
            raise RuntimeError("no key")
        fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]
        if self._genai is None or fingerprint != self._genai_fp:
            if self._genai_factory is not None:
                self._genai = self._genai_factory(key)
            else:
                from google import genai

                self._genai = genai.Client(api_key=key)
            self._genai_fp = fingerprint
        model = str(self.app.config.get("hands.search_model", "gemini-3.5-flash-lite"))
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            system_instruction="Answer the search briefly with facts and dates. Reply in the language of the query.")
        try:
            response = await asyncio.wait_for(
                self._genai.aio.models.generate_content(model=model, contents=query, config=config), 30)
        except Exception:
            self.app.db.bump_usage("gemini", model, "text", errors=1)
            raise
        self.app.db.bump_usage("gemini", model, "text")
        sources: list[dict[str, str]] = []
        candidates = getattr(response, "candidates", None) or []
        metadata = getattr(candidates[0], "grounding_metadata", None) if candidates else None
        for chunk in (getattr(metadata, "grounding_chunks", None) or [])[:6]:
            web = getattr(chunk, "web", None)
            if web is not None and getattr(web, "uri", None):
                sources.append({"title": str(getattr(web, "title", "") or "")[:200], "url": str(web.uri)})
        answer = str(getattr(response, "text", "") or "").strip()[:2500]
        return {"ok": bool(answer), "engine": "gemini-google-search", "query": query,
                "untrusted": {"answer": answer, "sources": sources},
                "summary": f"Google-grounded answer for '{query}' with {len(sources)} sources."}

    # -- page text ----------------------------------------------------------------------
    async def fetch_page(self, url: str) -> dict[str, Any]:
        current = url
        async with self._client() as client:
            for _ in range(4):
                parsed = urlparse(current)
                if parsed.scheme not in ("http", "https") or not parsed.hostname:
                    return {"ok": False, "summary": "Only http(s) pages can be read."}
                if not await asyncio.to_thread(is_public_host, parsed.hostname, self._resolver):
                    return {"ok": False, "summary": "That address is on this PC or a private network; SAM does not read it."}
                try:
                    response = await client.get(current)
                except Exception as exc:  # noqa: BLE001
                    return {"ok": False, "summary": f"Could not load the page: {type(exc).__name__}."}
                if response.status_code in (301, 302, 303, 307, 308) and response.headers.get("location"):
                    current = urljoin(current, response.headers["location"])
                    continue
                break
            else:
                return {"ok": False, "summary": "Too many redirects."}
        if response.status_code >= 400:
            return {"ok": False, "summary": f"The page answered HTTP {response.status_code}."}
        kind = response.headers.get("content-type", "")
        if "html" not in kind and "text" not in kind:
            return {"ok": False, "summary": f"That is not a text page ({kind or 'unknown type'})."}
        body = response.content[:MAX_PAGE_BYTES].decode(response.encoding or "utf-8", errors="replace")
        title, text = page_text(body) if "html" in kind else ("", body)
        return {"ok": True, "url": current, "title": title, "chars": len(text),
                "untrusted": text[:PAGE_TEXT_CHARS], "truncated": len(text) > PAGE_TEXT_CHARS,
                "summary": f"Read '{title or current}' ({len(text)} characters of text)."}

    @staticmethod
    def search_url(query: str) -> str:
        return "https://www.google.com/search?q=" + quote_plus(query)


__all__ = ["SearchResult", "Web", "is_public_host", "page_text", "parse_ddg_html"]
