"""Integration test: the real Flow with FakeLLM, asserting trace.py's
per-stage recorder actually sees real LLMCallCompletedEvents from real
Crew/Agent/Task execution (not just the unit-level RunRecorder calls in
test_trace.py)."""

from __future__ import annotations

import json

import pytest

from market_research.categories import CATEGORIES
from market_research.flow import MarketResearchFlow
from market_research.models import Fact, ProductFacts
from market_research.sources import FixtureSearch
from market_research.trace import RunTrace
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


def test_trace_has_three_stages_with_nonnegative_durations_and_positive_tokens(monkeypatch):
    import market_research.sources as sources_mod

    monkeypatch.setattr(
        sources_mod,
        "fetch_page",
        lambda url, use_cache=True: sources_mod.Page(url=url, fetched_at="now", text="irrelevant text"),
    )

    fake = FakeLLM(
        replies={
            "Product Researcher": "report",
            "Competitor Analyst": '{"items": []}',
            "Comparison Report Generator": _full_comparison_reply(["Narsi", "Acme"]),
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

    trace = flow.recorder.build_trace()
    assert {s.name for s in trace.stages} == {"product", "competitor", "comparison"}
    for stage in trace.stages:
        assert stage.duration_seconds >= 0.0
    assert trace.total_duration_seconds >= sum(s.duration_seconds for s in trace.stages) - 1e-6
    assert trace.total_tokens > 0

    # JSON round trip, same as what gets written to runs/<timestamp>.json
    restored = RunTrace.model_validate_json(trace.model_dump_json())
    assert restored == trace
