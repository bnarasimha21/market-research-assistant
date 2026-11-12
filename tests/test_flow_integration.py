"""Integration test: the real MarketResearchFlow end-to-end with FakeLLM and
a stub product-facts source, no network, no real LLM.

Marked `integration` (not `slow`): it drives the real Crew/Agent/Task/Flow
machinery, as opposed to the unit tests that call pure functions directly.
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

PRODUCT_STAGE_MARKER = "STAGE1_MARKER_narsi_free_tier"


class StubProductFactsSource:
    def get_facts(self, product: str) -> ProductFacts:
        return ProductFacts(
            product=product,
            items=[
                Fact(
                    text=f"{PRODUCT_STAGE_MARKER}: {product} has a free tier for up to 3 seats.",
                    source="product_narsi.md#pricing",
                )
            ],
        )


def _full_comparison_reply(product_names: list[str]) -> str:
    comparisons = []
    for category in CATEGORIES:
        cells = {
            name: {"text": f"{name} on {category}", "sources": [f"src:{name}"], "status": "supported"}
            for name in product_names
        }
        comparisons.append({"category": category, "cells": cells})
    return json.dumps({"comparisons": comparisons})


def test_full_flow_runs_end_to_end_and_every_category_is_populated(monkeypatch):
    import market_research.sources as sources_mod

    monkeypatch.setattr(
        sources_mod,
        "fetch_page",
        lambda url, use_cache=True: sources_mod.Page(
            url=url, fetched_at="now", text="Get started in minutes with our onboarding flow."
        ),
    )

    recorded_prompts: list[dict[str, str]] = []

    competitor_reply = json.dumps(
        {
            "items": [
                {
                    "claim": "Fast onboarding",
                    "source_url": "https://acme.com/pricing",
                    "quote": "Get started in minutes",
                }
            ]
        }
    )
    comparison_reply = _full_comparison_reply(["Narsi", "Acme"])

    fake = FakeLLM(
        replies={
            "Product Researcher": f"Report mentioning {PRODUCT_STAGE_MARKER} for Narsi.",
            "Competitor Analyst": competitor_reply,
            "Comparison Report Generator": comparison_reply,
        }
    )

    flow = MarketResearchFlow(
        llm=fake,
        facts_source=StubProductFactsSource(),
        search_backend=FixtureSearch({"Acme pricing features documentation": ["https://acme.com/pricing"]}),
    )
    flow.state.my_product = "Narsi"
    flow.state.competitors = ["Acme"]
    flow.kickoff()

    recorded_prompts.extend(fake.recorded_prompts)

    comparison_list = flow.state.comparison
    assert comparison_list is not None
    categories_seen = {c.category for c in comparison_list.comparisons}
    assert categories_seen == set(CATEGORIES)
    for comparison in comparison_list.comparisons:
        for cell in comparison.cells.values():
            assert cell.text.strip(), f"empty cell in {comparison.category}"

    # The product stage's own report text flows into the comparison stage's
    # prompt, proving real hand-off through typed Flow state rather than a
    # shared Task.context pointing at a stale module-level Task.
    comparison_prompts = [p["prompt"] for p in recorded_prompts if p["role"] == "Comparison Report Generator"]
    assert comparison_prompts, "comparison stage was never called"
    assert any(PRODUCT_STAGE_MARKER in prompt for prompt in comparison_prompts)
