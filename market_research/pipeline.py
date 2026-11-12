"""The single entry point the UI and CLI call.

This module imports no streamlit and makes no network call just by being
imported (crewai.LLM objects are only constructed inside run_comparison,
never at module import time). That is what issue 1's acceptance test
`python -c "import market_research.pipeline"` checks for.
"""

from __future__ import annotations

import os

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel, Field, ValidationError

from market_research.cache import result_cache_key, results_cache
from market_research.errors import ConfigError
from market_research.facts import ProductFactsSource
from market_research.flow import MarketResearchFlow
from market_research.llm import get_llm
from market_research.models import ComparisonList, ResearchInput
from market_research.sources import SearchBackend
from market_research.trace import RunTrace


class ComparisonResult(BaseModel):
    my_product: str
    competitors: list[str]
    comparisons: ComparisonList
    dropped_facts: int = 0
    guardrail_retries: int = 0
    guardrail_partial: bool = False
    missing_categories: list[str] = Field(default_factory=list)
    trace: RunTrace
    from_cache: bool = False


_CONFIG_ENV_VARS = ("LLM_PROVIDER", "DO_MODEL_ID", "FACTS_SOURCE")


def _config_hash() -> str:
    return "|".join(f"{key}={os.getenv(key, '')}" for key in _CONFIG_ENV_VARS)


def run_comparison(
    inputs: ResearchInput | dict,
    llm: BaseLLM | None = None,
    use_cache: bool = True,
    facts_source: ProductFactsSource | None = None,
    search_backend: SearchBackend | None = None,
) -> ComparisonResult:
    """Run the full product-vs-competitors pipeline.

    inputs: a ResearchInput, or a dict with "my_product" and "competitors"
        that will be validated into one. Invalid input raises ConfigError
        here, before any LLM call or network access happens.
    llm: an explicit LLM/BaseLLM to use for every stage (tests pass a
        FakeLLM). When None, a real LLM is built from LLM_PROVIDER.
    use_cache: disk-cache the final result keyed by normalized names plus a
        hash of the provider config; a repeat call with the same inputs and
        config returns the cached result with zero LLM calls and zero
        fetches.
    facts_source: override for the own-product facts source (tests pass a
        fixture-backed source instead of reading product_narsi.md). When
        None, MarketResearchFlow builds the default from FACTS_SOURCE.
    search_backend: override for the competitor web-search backend (tests
        pass a FixtureSearch instead of hitting Serper). When None,
        MarketResearchFlow builds the default from SERPER_API_KEY.
    """
    if not isinstance(inputs, ResearchInput):
        try:
            inputs = ResearchInput.model_validate(inputs)
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc

    cache = results_cache()
    cache_key = result_cache_key(inputs.my_product, inputs.competitors, _config_hash())
    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            result = ComparisonResult.model_validate(cached)
            result.from_cache = True
            return result

    resolved_llm = llm if llm is not None else get_llm("pipeline")

    flow = MarketResearchFlow(
        llm=resolved_llm,
        facts_source=facts_source,
        search_backend=search_backend,
    )
    flow.state.my_product = inputs.my_product
    flow.state.competitors = list(inputs.competitors)
    flow.kickoff()

    trace = flow.recorder.build_trace()
    assert flow.state.comparison is not None  # guaranteed by build_comparison or StageFailed
    result = ComparisonResult(
        my_product=inputs.my_product,
        competitors=list(inputs.competitors),
        comparisons=flow.state.comparison,
        dropped_facts=flow.state.dropped_facts,
        guardrail_retries=flow.state.guardrail_retries,
        guardrail_partial=flow.state.guardrail_partial,
        missing_categories=flow.state.missing_categories,
        trace=trace,
    )
    trace.write()
    if use_cache:
        cache.set(cache_key, result.model_dump(mode="json"))
    return result
