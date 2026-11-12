"""Unit tests for market_research.sources: grounding, URL ranking, https-only.

fetch_page itself (real requests.get) is exercised by the local-HTTP-server
integration tests, not here.
"""

from __future__ import annotations

from market_research.sources import (
    FixtureSearch,
    _rank_urls,
    decode_page_bytes,
    fetch_page,
    fetch_page_for_test,
    is_claim_grounded,
    is_safe_fetch_url,
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


# -- is_claim_grounded / _claim_and_quote_overlap --
#
# Round-1 cases ("e" quote, "Enterprise" quote) plus the two round-2
# gameable cases found by the reviewer: a quote that only generically
# mentions the competitor, and a quote that talks about pricing in the
# abstract with no matching number. All four must be dropped. A genuinely
# well-grounded claim, including one with a matching number, must pass.


def test_single_character_quote_is_rejected():
    # Round-1 case: claim "Acme is free forever and SOC2 certified", quote "e".
    page_text = "Acme is free forever and SOC2 certified for every customer."
    assert not is_claim_grounded(
        "Acme is free forever and SOC2 certified", "e", page_text, "Acme"
    )


def test_single_word_quote_is_rejected():
    # Round-1 case: claim "Acme Enterprise is $5 per seat", quote "Enterprise".
    page_text = "Acme Enterprise is $5 per seat, billed monthly."
    assert not is_claim_grounded("Acme Enterprise is $5 per seat", "Enterprise", page_text, "Acme")


def test_generic_company_description_does_not_support_a_specific_claim():
    # Round-2 gameable case: the quote only generically describes the
    # competitor; sharing the competitor's name is not evidence.
    claim = "Acme is free forever and SOC2 certified"
    quote = "Acme is a developer platform for teams"
    page_text = f"{quote}. Learn more about what we offer."
    assert quote_is_grounded(quote, page_text)  # sanity: the quote really is on the page
    assert not is_claim_grounded(claim, quote, page_text, "Acme")


def test_abstract_pricing_mention_without_a_matching_number_does_not_support_a_price_claim():
    # Round-2 gameable case: no digits anywhere in the quote, so it cannot
    # support a claim that names a specific number.
    claim = "Acme Enterprise is $5 per seat"
    quote = "Enterprise plans are available on request"
    page_text = f"{quote} for teams of any size."
    assert quote_is_grounded(quote, page_text)
    assert not is_claim_grounded(claim, quote, page_text, "Acme")


def test_well_grounded_claim_with_matching_number_is_accepted():
    claim = "Acme charges $5 per seat per month"
    quote = "Acme Enterprise plans start at $5 per seat per month for teams of any size"
    page_text = f"{quote}."
    assert is_claim_grounded(claim, quote, page_text, "Acme")


def test_well_grounded_claim_without_a_number_is_still_accepted():
    claim = "Acme offers a free tier with unlimited projects"
    quote = "Acme includes a free tier with unlimited projects for every account"
    page_text = f"{quote}."
    assert is_claim_grounded(claim, quote, page_text, "Acme")


def test_single_substantive_token_claim_is_not_rejected_by_the_overlap_floor():
    """Adversarial-review regression: the overlap bar must never exceed
    what a claim's own token count can satisfy. A claim with only one
    substantive token left after removing the entity name and stopwords
    (here, just "profitable") has claim_tokens of size 1, so a fixed
    floor like max(2, ...) would reject it outright no matter how well
    the quote supports it. The real fix is max(1, ceil(len/2)).
    """
    claim = "Acme is profitable"
    quote = "Acme has been profitable every quarter since launch"
    page_text = f"{quote}."
    assert is_claim_grounded(claim, quote, page_text, "Acme")


# -- is_safe_fetch_url (SSRF) --


def test_is_safe_fetch_url_rejects_cloud_metadata_ip_directly():
    assert not is_safe_fetch_url("https://169.254.169.254/latest/meta-data/")


def test_is_safe_fetch_url_rejects_loopback_ip_directly():
    assert not is_safe_fetch_url("https://127.0.0.1/")


def test_is_safe_fetch_url_rejects_redirect_to_metadata_ip(monkeypatch):
    """fetch_page follows redirects manually, re-checking is_safe_fetch_url
    on every hop's target before following it. Mocks requests.get so no
    real network call happens; asserts the metadata-IP redirect target is
    never fetched.
    """
    import socket as socket_mod

    import market_research.sources as sources_mod

    class _RedirectResponse:
        status_code = 302
        headers = {"Location": "https://169.254.169.254/latest/meta-data/"}

        def close(self):
            pass

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _RedirectResponse()

    real_getaddrinfo = socket_mod.getaddrinfo

    def fake_getaddrinfo(host, port):
        # Fake a public-IP answer only for the entry hostname, which isn't a
        # real domain; a literal IP target (like the 169.254.169.254
        # redirect below) needs no DNS lookup at all, so defer to the real
        # resolver for it (ipaddress.ip_address parses it directly either
        # way) rather than mask it behind a fake "safe" answer.
        if host == "example.com":
            return [(socket_mod.AF_INET, None, None, "", ("93.184.216.34", 0))]
        return real_getaddrinfo(host, port)

    monkeypatch.setattr(sources_mod.requests, "get", fake_get)
    monkeypatch.setattr(socket_mod, "getaddrinfo", fake_getaddrinfo)
    result = sources_mod.fetch_page("https://example.com/redirect-to-metadata", use_cache=False)
    assert result is None
    # Only the original safe-looking URL was ever requested; the unsafe
    # redirect target must never reach requests.get.
    assert calls == ["https://example.com/redirect-to-metadata"]


def test_is_safe_fetch_url_accepts_a_normal_public_https_url(monkeypatch):
    import socket as socket_mod

    def fake_getaddrinfo(host, port):
        return [(socket_mod.AF_INET, None, None, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket_mod, "getaddrinfo", fake_getaddrinfo)
    assert is_safe_fetch_url("https://example.com/pricing")


# -- decode_page_bytes --


def test_decode_page_bytes_handles_utf8_non_ascii_without_mojibake():
    text = "Pricing: €10 per seat per month"  # euro sign
    raw = text.encode("utf-8")
    decoded = decode_page_bytes(raw, "text/html; charset=utf-8")
    assert decoded == text
    assert "€" in decoded
    assert "Ã" not in decoded  # the classic mojibake tell for utf-8 misread as latin-1
