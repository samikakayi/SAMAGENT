"""Free web search, with no key and no new dependency.

DuckDuckGo needs no account, so SAM can search in FREE mode without a paid
provider. Two endpoints are tried in order: the HTML result page, which
carries ordinary web results, and then the Instant Answer JSON API, which is
narrower but structured and stable. Both are read-only GETs over the httpx
client the project already depends on.

Everything returned here is untrusted text from the open internet. It is
bounded on every axis -- request timeout, result count, and the length of each
field -- and handed back as data. It never becomes an instruction: the caller
puts it in front of the model as a tool result like any other, and it cannot
reach task state, approvals or verification.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
JSON_ENDPOINT = "https://api.duckduckgo.com/"
# A search is a side quest inside a longer run, so it fails fast rather than
# holding a turn open.
TIMEOUT_SECONDS = 12.0
MAX_RESULTS = 10
DEFAULT_RESULTS = 5
# Long enough to judge relevance, short enough that one search cannot crowd
# the next prompt.
MAX_TITLE_CHARS = 200
MAX_SNIPPET_CHARS = 400
# A browser-shaped agent: the HTML endpoint returns an empty page without one.
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SAM/2.0 (+local agent)"

_RESULT_BLOCK = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'(?P<rest>.*?)(?=<a[^>]+class="result__a"|\Z)',
    re.DOTALL | re.IGNORECASE,
)
_SNIPPET = re.compile(r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.DOTALL | re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?)\]])")


class SearchUnavailable(RuntimeError):
    """No backend answered. Said plainly rather than returned as no results."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def _text(raw: str, limit: int) -> str:
    """Strip markup and collapse whitespace, then bound the length.

    Tags become a space so `a<b>b</b>c` does not run together, which leaves a
    gap before punctuation that was only ever inside a tag -- closed up again
    here so a highlighted word does not arrive as "laboratory ."
    """
    stripped = html.unescape(_TAGS.sub(" ", raw or ""))
    collapsed = " ".join(stripped.split())
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", collapsed)[:limit]


def _direct_url(href: str) -> str:
    """Unwrap DuckDuckGo's redirect so the model sees the real destination."""
    if not href:
        return ""
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            href = unquote(target[0])
            parsed = urlparse(href)
    return href if parsed.scheme in {"http", "https"} else ""


def _from_html(body: str, limit: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    for match in _RESULT_BLOCK.finditer(body):
        url = _direct_url(match.group("href"))
        title = _text(match.group("title"), MAX_TITLE_CHARS)
        if not url or not title:
            continue
        snippet_match = _SNIPPET.search(match.group("rest") or "")
        snippet = _text(snippet_match.group("snippet"), MAX_SNIPPET_CHARS) if snippet_match else ""
        results.append(SearchResult(title, url, snippet))
        if len(results) >= limit:
            break
    return results


def _from_instant_answer(payload: dict[str, Any], limit: int) -> list[SearchResult]:
    """The JSON fallback: an abstract, then whatever topics it lists."""
    results: list[SearchResult] = []
    abstract = _text(str(payload.get("AbstractText") or ""), MAX_SNIPPET_CHARS)
    abstract_url = _direct_url(str(payload.get("AbstractURL") or ""))
    if abstract and abstract_url:
        heading = _text(str(payload.get("Heading") or abstract_url), MAX_TITLE_CHARS)
        results.append(SearchResult(heading, abstract_url, abstract))

    def walk(items: Any) -> None:
        for item in items if isinstance(items, list) else []:
            if len(results) >= limit:
                return
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("Topics"), list):
                walk(item["Topics"])
                continue
            url = _direct_url(str(item.get("FirstURL") or ""))
            text = _text(str(item.get("Text") or ""), MAX_SNIPPET_CHARS)
            if url and text:
                results.append(SearchResult(text[:MAX_TITLE_CHARS], url, text))

    walk(payload.get("RelatedTopics"))
    return results[:limit]


class WebSearch:
    """Read-only search over DuckDuckGo's keyless endpoints."""

    def __init__(self, *, client_factory: Any = None) -> None:
        # Injected in tests so the suite never reaches the network.
        self._client_factory = client_factory or (
            lambda: httpx.Client(timeout=TIMEOUT_SECONDS, trust_env=False, follow_redirects=True)
        )

    def search(self, query: str, limit: int = DEFAULT_RESULTS) -> list[SearchResult]:
        cleaned = " ".join(str(query or "").split())[:400]
        if not cleaned:
            raise ValueError("A search needs a query")
        bounded = max(1, min(int(limit or DEFAULT_RESULTS), MAX_RESULTS))
        failures: list[str] = []
        with self._client_factory() as client:
            for attempt in (self._html, self._instant_answer):
                try:
                    found = attempt(client, cleaned, bounded)
                except (httpx.HTTPError, ValueError) as exc:
                    failures.append(type(exc).__name__)
                    continue
                if found:
                    return found
        if failures:
            raise SearchUnavailable(f"No search backend answered ({', '.join(sorted(set(failures)))})")
        return []

    @staticmethod
    def _html(client: Any, query: str, limit: int) -> list[SearchResult]:
        response = client.post(
            HTML_ENDPOINT, data={"q": query}, headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        return _from_html(response.text, limit)

    @staticmethod
    def _instant_answer(client: Any, query: str, limit: int) -> list[SearchResult]:
        response = client.get(
            JSON_ENDPOINT,
            params={"q": query, "format": "json", "no_html": 1, "no_redirect": 1},
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        return _from_instant_answer(response.json(), limit)
