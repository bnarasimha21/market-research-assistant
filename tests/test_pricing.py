"""Unit tests for market_research.pricing: loading pricing.yaml and
estimating a cost from token counts."""

from __future__ import annotations

from market_research.pricing import estimate_cost_usd, load_pricing


def test_load_pricing_reads_the_real_committed_file():
    pricing = load_pricing()
    assert "fake-llm" in pricing
    assert "llama-3.1-70b-instruct" in pricing
    for entry in pricing.values():
        assert "source_url" in entry
        assert "as_of" in entry


def test_estimate_cost_unknown_model_returns_none():
    assert estimate_cost_usd("not-a-real-model", 1000, 1000, pricing={}) is None


def test_estimate_cost_known_model_computes_expected_value():
    pricing = {
        "test-model": {
            "input_per_million_tokens_usd": 1.0,
            "output_per_million_tokens_usd": 2.0,
        }
    }
    cost = estimate_cost_usd("test-model", 1_000_000, 500_000, pricing=pricing)
    assert cost == 1.0 + 1.0


def test_estimate_cost_fake_llm_is_free():
    pricing = load_pricing()
    cost = estimate_cost_usd("fake-llm", 10_000, 10_000, pricing=pricing)
    assert cost == 0.0
