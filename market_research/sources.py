"""Fetch real competitor pages so claims can be grounded in retrieved text.

Replaces asking a model with no browsing to "gather information" about a
competitor from memory, and replaces ScrapeWebsiteTool being handed no URL
(so the agent invented which pages to fetch). Search finds candidate URLs;
fetching and stripping to text happens here in plain code, never inside an
LLM tool call, so it is deterministic and testable with fixtures.

SSRF hardening: every URL fetched (the original, and any redirect target) is
checked for scheme AND resolved host before a single byte is requested, and
redirects are followed manually (one hop at a time, re-checked each time)
rather than left to requests' default `allow_redirects=True`, which would
otherwise happily follow an https:// URL straight to
`http://169.254.169.254/...` or any other internal/loopback/link-local/
private address.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    import charset_normalizer
except ImportError:  # pragma: no cover - charset_normalizer ships with requests
    charset_normalizer = None

MAX_PAGE_BYTES = 500_000
FETCH_TIMEOUT_SECONDS = 10
MAX_REDIRECTS = 5
DEFAULT_MAX_PAGES = 3
_PREFERRED_PATH_HINTS = ("pricing", "price", "docs", "documentation", "feature")
_REDIRECT_STATUS_CODES = (301, 302, 303, 307, 308)

# Minimum shape a grounded quote must have before it is even compared
# against the page or the claim it supports. Chosen as "a short sentence
# fragment, not a single word or a bare connector": 20 characters and 3
# words is enough to rule out `quote="e"` or `quote="Enterprise"` (both seen
# in the wild accepted by the pre-fix version) while still allowing a short,
# real factual quote ("Free tier available" is 20 chars / 3 words exactly).
MIN_QUOTE_CHARS = 20
MIN_QUOTE_WORDS = 3

# Generic connector words excluded when checking that a claim and its quote
# actually share real content, not just articles/prepositions. Deliberately
# small: this is a cheap plausibility check, not a full grounding model.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "for", "and", "or", "in", "on", "at", "with", "by", "as",
        "from", "that", "this", "it", "its", "has", "have", "had", "will",
        "per", "per_month",
    }
)
MIN_SUBSTANTIVE_TOKEN_LEN = 3


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


def _is_unsafe_ip(ip_str: str) -> bool:
    """True for loopback/link-local/private/reserved/multicast/unspecified,
    i.e. anything that is not a routable public address. Also true (fail
    safe) if the string cannot be parsed as an IP at all.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_safe_fetch_url(url: str) -> bool:
    """True iff `url` is https AND every address its host resolves to is a
    routable, non-internal address.

    Checked for the original URL AND for every redirect hop's target before
    it is followed, so neither the entry point nor a redirect can reach
    127.0.0.1, 169.254.169.254 (cloud metadata endpoints), RFC1918 ranges,
    etc. Fails closed: an unresolvable host or any parse failure is unsafe.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme != "https":
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    try:
        infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, UnicodeError, OSError):
        return False
    if not infos:
        return False
    return all(not _is_unsafe_ip(info[4][0]) for info in infos)


def _charset_from_content_type(content_type: str) -> str | None:
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1].strip().strip("\"'") or None
    return None


def decode_page_bytes(raw: bytes, content_type: str = "") -> str:
    """Decode fetched page bytes to text, without trusting requests'
    `response.encoding` default of ISO-8859-1 for a `text/html` response
    that never declared an explicit charset (the mojibake bug this
    replaces): prefer an explicit Content-Type charset, then real utf-8,
    then sniff with charset_normalizer, and only fall back to utf-8 with
    errors='replace' as a last resort.
    """
    declared = _charset_from_content_type(content_type)
    if declared:
        try:
            return raw.decode(declared, errors="strict")
        except (LookupError, UnicodeDecodeError):
            pass

    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        pass

    if charset_normalizer is not None:
        best = charset_normalizer.from_bytes(raw).best()
        if best is not None:
            return str(best)

    return raw.decode("utf-8", errors="replace")


def fetch_page(url: str, use_cache: bool = True) -> Page | None:
    """Fetch one https URL, strip to plain text, with a size cap, an overall
    time budget, and SSRF-safe scheme/host checks on the original URL and
    every redirect hop.

    Returns None rather than raising on any network/parse/safety failure,
    since a single dead or unsafe competitor page should not fail the whole
    run. Cached on disk keyed by the original URL with a 24h TTL (see
    market_research.cache) so a second run of the same competitor makes
    zero fetches.
    """
    if not is_safe_fetch_url(url):
        return None

    cache = None
    if use_cache:
        from market_research.cache import pages_cache

        cache = pages_cache()
        cached = cache.get(url)
        if cached is not None:
            return Page(**cached)

    deadline = time.monotonic() + FETCH_TIMEOUT_SECONDS
    current_url = url
    response: requests.Response | None = None
    try:
        for _ in range(MAX_REDIRECTS + 1):
            response = requests.get(
                current_url,
                timeout=(FETCH_TIMEOUT_SECONDS, FETCH_TIMEOUT_SECONDS),
                stream=True,
                allow_redirects=False,
                headers={"User-Agent": "market-research-assistant/1.0"},
            )
            if response.status_code in _REDIRECT_STATUS_CODES:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    return None
                next_url = urljoin(current_url, location)
                if not is_safe_fetch_url(next_url):
                    return None
                current_url = next_url
                continue
            break
        else:
            # Exhausted MAX_REDIRECTS + 1 attempts without a non-redirect response.
            return None

        response.raise_for_status()

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=8192):
            if time.monotonic() > deadline:
                return None
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= MAX_PAGE_BYTES:
                break
        raw = b"".join(chunks)[:MAX_PAGE_BYTES]
        html = decode_page_bytes(raw, response.headers.get("Content-Type", ""))
    except (requests.RequestException, UnicodeDecodeError, ValueError, OSError):
        return None
    finally:
        if response is not None:
            response.close()

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
    """A fact's quote is a substring of the fetched page it names, after
    whitespace normalization (deterministic, no LLM call).

    This is only the substring half of grounding: it says nothing about
    whether the quote is long enough to be meaningful, or whether it has
    anything to do with the claim it is supposed to support. See
    `is_claim_grounded` for the full check used by the real pipeline.
    """
    if not quote.strip():
        return False
    return normalize_whitespace(quote).lower() in normalize_whitespace(page_text).lower()


def _quote_long_enough(quote: str) -> bool:
    normalized = normalize_whitespace(quote).strip()
    if not normalized:
        return False
    words = normalized.split()
    return len(normalized) >= MIN_QUOTE_CHARS and len(words) >= MIN_QUOTE_WORDS


def _substantive_tokens(text: str) -> set[str]:
    words = normalize_whitespace(text).lower().replace("-", " ").split()
    cleaned = ("".join(ch for ch in w if ch.isalnum()) for w in words)
    return {w for w in cleaned if w and w not in _STOPWORDS and len(w) >= MIN_SUBSTANTIVE_TOKEN_LEN}


def _claim_and_quote_overlap(claim: str, quote: str) -> bool:
    """True if `claim` and `quote` share enough real content that the quote
    plausibly supports the claim, rather than merely existing somewhere on
    the same page. Requires at least one shared substantive token, and at
    least a third of the claim's substantive tokens to appear in the quote
    (so "Acme Enterprise is $5 per seat" is not considered supported by an
    unrelated quote that happens to also contain the word "Acme").
    """
    claim_tokens = _substantive_tokens(claim)
    if not claim_tokens:
        return False
    quote_tokens = _substantive_tokens(quote)
    overlap = claim_tokens & quote_tokens
    if not overlap:
        return False
    required = max(1, len(claim_tokens) // 3)
    return len(overlap) >= required


def is_claim_grounded(claim: str, quote: str, page_text: str) -> bool:
    """The full grounding check for one competitor claim: the quote must be
    long enough to be a real sentence fragment (not "e" or a single bare
    word), it must actually be a substring of the fetched page, and it must
    share real content with the claim it is supposed to support (not just
    happen to exist somewhere on the same page).

    A false "grounded" on a fabricated claim is worse than a false
    rejection of a true one, so every one of these three checks must pass;
    any single failure drops the claim.
    """
    if not claim.strip():
        return False
    if not _quote_long_enough(quote):
        return False
    if not quote_is_grounded(quote, page_text):
        return False
    return _claim_and_quote_overlap(claim, quote)
