"""Unit tests for market_research.facts.LocalMarkdownSource.

Entirely offline: reads product_narsi.md (or a fixture markdown file) from
disk, ranks with BM25, never calls out.
"""

from __future__ import annotations

import pytest

from market_research.errors import ConfigError
from market_research.facts import LocalMarkdownSource, _parse_sections

FIXTURE_MARKDOWN = """\
# Pricing and Licensing
Narsi costs $10 per seat per month, with a free tier for up to 3 seats.

# Security and Privacy
All data is encrypted at rest and in transit using AES-256.

# Natural Language Interface
Narsi understands plain English commands and converts them to actions.
"""


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "product.md"
    path.write_text(FIXTURE_MARKDOWN, encoding="utf-8")
    return LocalMarkdownSource(markdown_path=path, top_k=2)


def test_missing_file_raises_configerror(tmp_path):
    with pytest.raises(ConfigError):
        LocalMarkdownSource(markdown_path=tmp_path / "does-not-exist.md")


def test_parse_sections_splits_on_headings():
    sections = _parse_sections(FIXTURE_MARKDOWN)
    headings = [s.heading for s in sections]
    assert "Pricing and Licensing" in headings
    assert "Security and Privacy" in headings


def test_pricing_query_ranks_pricing_section_above_unrelated(source):
    ranked = source.rank_sections("pricing cost per seat")
    assert ranked[0] == "Pricing and Licensing"


def test_every_fact_carries_a_source_heading(source):
    facts = source.get_facts("Narsi")
    assert facts.items, "expected at least one fact"
    for fact in facts.items:
        assert fact.source.startswith("product.md#")
        assert "#" in fact.source
        heading = fact.source.split("#", 1)[1]
        assert heading  # non-empty heading name


def test_get_facts_sets_product_name(source):
    facts = source.get_facts("Narsi")
    assert facts.product == "Narsi"


def test_heading_only_match_still_ranks_above_unrelated_body_text(source):
    """Before heading text was included in the BM25 index, a query that
    matched a section's HEADING but shared no words with that section's
    BODY could score 0 against the exact-topic section while still scoring
    nonzero against a completely unrelated section whose body happened to
    share some other word with the query. "Natural Language Interface" is
    exactly this case against FIXTURE_MARKDOWN: the heading matches
    verbatim, but the body ("Narsi understands plain English commands...")
    shares none of those three words.
    """
    ranked = source.rank_sections("Natural Language Interface")
    assert ranked and ranked[0] == "Natural Language Interface"


def test_category_with_zero_lexical_overlap_attaches_no_section(tmp_path):
    """A category that shares no real term with any section (not even via
    its heading) must not get a fact attached at all. Previously
    `_rank_for` always returned the top_k sections regardless of score, so
    a category like "Future Development and Roadmap" against a one-section
    file about an unrelated topic still got that section attached as if it
    were a relevant fact.
    """
    path = tmp_path / "product.md"
    path.write_text("# Unrelated Topic\nSomething about widgets and gadgets only.\n", encoding="utf-8")
    source = LocalMarkdownSource(markdown_path=path, top_k=2)
    assert source.rank_sections("Future Development and Roadmap") == []
