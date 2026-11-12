"""Fetch real competitor pages so claims can be grounded in retrieved text.

Replaces asking a model with no browsing to "gather information" about a
competitor from memory, and replaces ScrapeWebsiteTool being handed no URL
(so the agent invented which pages to fetch). Search finds candidate URLs;
fetching and stripping to text happens here in plain code, never inside an
LLM tool call, so it is deterministic and testable with fixtures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

MAX_PAGE_BYTES = 500_000
FETCH_TIMEOUT_SECONDS = 10
DEFAULT_MAX_PAGES = 3
_PREFERRED_PATH_HINTS = ("pricing", "price", "docs", "documentation", "feature")


@dataclass(frozen=True)
class Page:
    url: str
    fetched_at: str
    text: str


class SearchBackend(Protocol):
    def search(self, query: str) -> list[str]: ...


class FixtureSearch:
    """Deterministic search backend for tests and demos.

    Maps a competitor/query string to a fixed list of URLs instead of
    calling a real search API, so tests never touch the network.
    """

    def __init__(self, fixtures: dict[str, list[str]]) -> None:
        self._fixtures = fixtures

    def search(self, query: str) -> list[str]:
        return list(self._fixtures.get(query, []))


class SerperSearch:
    """Real web search via the Serper API. Only constructed when SERPER_API_KEY is set."""

    ENDPOINT = "https://google.serper.dev/search"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def search(self, query: str) -> list[str]:
        response = requests.post(
            self.ENDPOINT,
            headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
            json={"q": query},
            timeout=FETCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return [item["link"] for item in data.get("organic", []) if "link" in item]


def get_search_backend(fixtures: dict[str, list[str]] | None = None) -> SearchBackend:
    api_key = os.getenv("SERPER_API_KEY")
    if api_key:
        return SerperSearch(api_key)
    return FixtureSearch(fixtures or {})


def _rank_urls(urls: list[str], max_pages: int) -> list[str]:
    def score(url: str) -> tuple[int, int]:
        path = urlparse(url).path.lower()
        hinted = any(hint in path for hint in _PREFERRED_PATH_HINTS)
        return (0 if hinted else 1, len(path))

    ranked = sorted(dict.fromkeys(urls), key=score)
    return ranked[:max_pages]


def _strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def fetch_page(url: str, use_cache: bool = True) -> Page | None:
    """Fetch one https URL, strip to plain text, with a size cap and timeout.

    Returns None rather than raising on any network/parse failure, since a
    single dead competitor page should not fail the whole run. Cached on
    disk keyed by URL with a 24h TTL (see market_research.cache) so a
    second run of the same competitor makes zero fetches.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return None

    cache = None
    if use_cache:
        from market_research.cache import pages_cache

        cache = pages_cache()
        cached = cache.get(url)
        if cached is not None:
            return Page(**cached)

    try:
        response = requests.get(
            url,
            timeout=FETCH_TIMEOUT_SECONDS,
            stream=True,
            headers={"User-Agent": "market-research-assistant/1.0"},
        )
        response.raise_for_status()
        raw = response.raw.read(MAX_PAGE_BYTES + 1, decode_content=True)
        if len(raw) > MAX_PAGE_BYTES:
            raw = raw[:MAX_PAGE_BYTES]
        html = raw.decode(response.encoding or "utf-8", errors="replace")
    except (requests.RequestException, UnicodeDecodeError, ValueError):
        return None
    text = _strip_html(html)
    if not text:
        return None
    page = Page(url=url, fetched_at=datetime.now(UTC).isoformat(), text=text)
    if cache is not None:
        cache.set(url, {"url": page.url, "fetched_at": page.fetched_at, "text": page.text})
    return page


def fetch_page_for_test(url: str, html: str) -> Page:
    """Build a Page directly from already-known html, bypassing the network.

    Used by tests that serve fixture HTML over a local HTTP server or hold
    it in memory, so the grounding/stripping logic is still exercised.
    """
    return Page(url=url, fetched_at=datetime.now(UTC).isoformat(), text=_strip_html(html))


@dataclass
class CompetitorPages:
    competitor: str
    pages: list[Page] = field(default_factory=list)


def gather_pages(
    competitor: str,
    search: SearchBackend,
    max_pages: int = DEFAULT_MAX_PAGES,
    query_suffix: str = "pricing features documentation",
) -> CompetitorPages:
    urls = search.search(f"{competitor} {query_suffix}")
    ranked = _rank_urls(urls, max_pages)
    pages: list[Page] = []
    for url in ranked:
        page = fetch_page(url)
        if page is not None:
            pages.append(page)
    return CompetitorPages(competitor=competitor, pages=pages)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def quote_is_grounded(quote: str, page_text: str) -> bool:
    """A fact is only kept if its quote is a substring of the fetched page
    it names, after whitespace normalization (deterministic, no LLM call).
    """
    if not quote.strip():
        return False
    return normalize_whitespace(quote).lower() in normalize_whitespace(page_text).lower()
