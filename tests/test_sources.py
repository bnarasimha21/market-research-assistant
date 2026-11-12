"""Unit tests for market_research.sources: grounding, URL ranking, https-only.

fetch_page itself (real requests.get) is exercised by the local-HTTP-server
integration tests, not here.
"""

from __future__ import annotations

from market_research.sources import (
    FixtureSearch,
    _rank_urls,
    fetch_page,
    fetch_page_for_test,
    normalize_whitespace,
    quote_is_grounded,
)


def test_fixture_search_returns_mapped_urls():
    search = FixtureSearch({"Acme pricing features documentation": ["https://acme.com/pricing"]})
    assert search.search("Acme pricing features documentation") == ["https://acme.com/pricing"]
    assert search.search("unknown query") == []


def test_rank_urls_prefers_pricing_and_docs_paths():
    urls = [
        "https://acme.com/about-us",
        "https://acme.com/pricing",
        "https://acme.com/blog/2024/launch",
    ]
    ranked = _rank_urls(urls, max_pages=3)
    assert ranked[0] == "https://acme.com/pricing"


def test_rank_urls_dedupes_and_caps():
    urls = ["https://a.com/pricing"] * 3 + ["https://a.com/docs", "https://a.com/other"]
    ranked = _rank_urls(urls, max_pages=2)
    assert len(ranked) == 2
    assert "https://a.com/pricing" in ranked


def test_quote_is_grounded_matches_substring_ignoring_whitespace_and_case():
    page_text = "  We offer   Fast Onboarding  in minutes.  "
    assert quote_is_grounded("fast onboarding in minutes", page_text)


def test_quote_is_grounded_rejects_fabricated_quote():
    page_text = "We offer fast onboarding in minutes."
    assert not quote_is_grounded("this text is not on the page", page_text)


def test_quote_is_grounded_rejects_empty_quote():
    assert not quote_is_grounded("   ", "any text")


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("a   b\n\tc") == "a b c"


def test_fetch_page_rejects_non_https_url():
    assert fetch_page("http://example.com/insecure") is None
    assert fetch_page("ftp://example.com/file") is None


def test_fetch_page_for_test_strips_html_without_network():
    page = fetch_page_for_test(
        "https://example.com/pricing",
        "<html><body><script>evil()</script><h1>Pricing</h1><p>Get started in minutes.</p></body></html>",
    )
    assert "Get started in minutes." in page.text
    assert "evil" not in page.text
    assert page.url == "https://example.com/pricing"
