"""Integration test: the real Flow with a scripted FakeLLM whose first
comparison reply is malformed JSON and whose second is valid. Asserts the
returned table is valid, the fake LLM saw two comparison prompts, and the
second prompt contains the guardrail's error text (proving CrewAI really
folds the guardrail error into the retry prompt, not just that our
guardrail function returns the right tuple in isolation).
"""

from __future__ import annotations

import json

import pytest

from market_research.categories import CATEGORIES
from market_research.flow import MarketResearchFlow
from market_research.models import Fact, ProductFacts
from market_research.sources import FixtureSearch
from tests.conftest import FakeLLM

pytestmark = pytest.mark.integration


class _StubProductFactsSource:
    def get_facts(self, product):
        return ProductFacts(product=product, items=[Fact(text="stub fact", source="stub#x")])


def _full_comparison_reply(product_names):
    comparisons = []
    for category in CATEGORIES:
        cells = {
            name: {"text": f"{name} on {category}", "sources": [f"src:{name}"], "status": "supported"}
            for name in product_names
        }
        comparisons.append({"category": category, "cells": cells})
    return json.dumps({"comparisons": comparisons})


def test_malformed_first_reply_then_valid_second_reply_succeeds_with_one_retry(monkeypatch):
    import market_research.sources as sources_mod

    monkeypatch.setattr(
        sources_mod,
        "fetch_page",
        lambda url, use_cache=True: sources_mod.Page(url=url, fetched_at="now", text="no matching text here"),
    )

    valid_reply = _full_comparison_reply(["Narsi", "Acme"])
    fake = FakeLLM(
        replies={
            "Product Researcher": "report",
            "Competitor Analyst": '{"items": []}',
            "Comparison Report Generator": ["this is not valid json", valid_reply],
        }
    )

    flow = MarketResearchFlow(
        llm=fake,
        facts_source=_StubProductFactsSource(),
        search_backend=FixtureSearch({}),
    )
    flow.state.my_product = "Narsi"
    flow.state.competitors = ["Acme"]
    flow.kickoff()

    comparison_list = flow.state.comparison
    assert comparison_list is not None
    categories_seen = {c.category for c in comparison_list.comparisons}
    assert categories_seen == set(CATEGORIES)
    assert flow.state.guardrail_retries == 1
    assert not flow.state.guardrail_partial

    comparison_calls = [p["prompt"] for p in fake.recorded_prompts if p["role"] == "Comparison Report Generator"]
    assert len(comparison_calls) == 2
    # CrewAI's retry re-runs the agent with the guardrail's error folded
    # into the next prompt; the second call must literally contain it.
    assert "not valid JSON" in comparison_calls[1]
