"""The three-stage pipeline as a typed, testable CrewAI Flow.

Replaces crew_flow.py's MarketResearchAssistant, whose get_competitor_info
and get_comparison_info took no parameters and read product/competitor
names from a shared inputs.INPUTS dict instead of from Flow state, and
whose comparison stage pulled data through `context=[...]` pointing at
module-level Task singletons.

Here each @listen method takes the previous stage's actual return value as
its argument (crewai passes it positionally when the method declares a
parameter) and also writes its result onto self.state, which is a typed
Pydantic model, not a dict. The comparison stage reads both prior stages'
output directly from self.state and puts it in its own task description;
nothing is shared through Task.context.
"""

from __future__ import annotations

import json

from crewai import Crew
from crewai.flow.flow import Flow, listen, start
from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel, Field

from market_research.agents import (
    build_comparison_agent,
    build_comparison_task,
    build_competitor_agent,
    build_competitor_task,
    build_product_agent,
    build_product_task,
)
from market_research.errors import StageFailed
from market_research.facts import ProductFactsSource, get_product_facts_source
from market_research.models import ComparisonList, CompetitorFact, CompetitorFacts
from market_research.sources import (
    SearchBackend,
    gather_pages,
    get_search_backend,
    quote_is_grounded,
)
from market_research.trace import RunRecorder, recording_stage


class MarketState(BaseModel):
    my_product: str = ""
    competitors: list[str] = Field(default_factory=list)
    product_report: str = ""
    competitor_facts: dict[str, CompetitorFacts] = Field(default_factory=dict)
    dropped_facts: int = 0
    comparison: ComparisonList | None = None
    guardrail_retries: int = 0
    guardrail_partial: bool = False
    missing_categories: list[str] = Field(default_factory=list)


class MarketResearchFlow(Flow[MarketState]):
    """Dependencies (llm, facts source, search backend, recorder) are plain
    instance attributes set after construction, not Flow state: state is
    the serializable research data, these are runtime collaborators.
    """

    def __init__(
        self,
        llm: BaseLLM,
        facts_source: ProductFactsSource | None = None,
        search_backend: SearchBackend | None = None,
        recorder: RunRecorder | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._facts_source = facts_source or get_product_facts_source()
        self._search_backend = search_backend or get_search_backend()
        self.recorder = recorder or RunRecorder()

    @start()
    def gather_product_facts(self) -> str:
        with recording_stage(self.recorder, "product"):
            try:
                facts = self._facts_source.get_facts(self.state.my_product)
                agent = build_product_agent(self._llm)
                task = build_product_task(agent, self.state.my_product, facts)
                crew = Crew(agents=[agent], tasks=[task])
                result = crew.kickoff()
            except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised typed
                raise StageFailed("product", str(exc)) from exc
        report = result.raw if hasattr(result, "raw") else str(result)
        self.state.product_report = report
        return report

    @listen(gather_product_facts)
    def gather_competitor_facts(self, product_report: str) -> dict[str, CompetitorFacts]:
        results: dict[str, CompetitorFacts] = {}
        with recording_stage(self.recorder, "competitor"):
            try:
                for competitor in self.state.competitors:
                    pages = gather_pages(competitor, self._search_backend)
                    self.recorder.record_pages_fetched("competitor", len(pages.pages))
                    agent = build_competitor_agent(self._llm)
                    task = build_competitor_task(agent, competitor, pages)
                    crew = Crew(agents=[agent], tasks=[task])
                    crew.kickoff()
                    raw = task.output.raw if task.output else "{}"
                    facts, dropped = _parse_and_ground_competitor_facts(raw, pages)
                    self.state.dropped_facts += dropped
                    results[competitor] = CompetitorFacts(competitor=competitor, items=facts)
            except StageFailed:
                raise
            except Exception as exc:  # noqa: BLE001
                raise StageFailed("competitor", str(exc)) from exc
        self.state.competitor_facts = results
        return results

    @listen(gather_competitor_facts)
    def build_comparison(self, competitor_facts: dict[str, CompetitorFacts]) -> ComparisonList:
        with recording_stage(self.recorder, "comparison"):
            try:
                agent = build_comparison_agent(self._llm)
                task, tracker = build_comparison_task(
                    agent,
                    self.state.my_product,
                    self.state.competitors,
                    self.state.product_report,
                    competitor_facts,
                )
                crew = Crew(agents=[agent], tasks=[task])
                crew.kickoff()
                comparison_list = ComparisonList.model_validate_json(task.output.raw)
            except StageFailed:
                raise
            except Exception as exc:  # noqa: BLE001
                raise StageFailed("comparison", str(exc)) from exc
            self.recorder.record_guardrail_retries("comparison", tracker.state.retries)
        self.state.comparison = comparison_list
        self.state.guardrail_retries = tracker.state.retries
        self.state.guardrail_partial = tracker.state.partial
        self.state.missing_categories = tracker.state.missing_categories
        return comparison_list


def _parse_and_ground_competitor_facts(raw: str, pages) -> tuple[list[CompetitorFact], int]:
    """Deterministic grounding check: keep a fact only if its quote is a
    substring (after whitespace normalization) of the fetched page it
    names. Everything else is dropped and counted, never shown as if it
    were real. Malformed JSON from the agent yields zero facts, not a crash.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], 0

    items = data.get("items", []) if isinstance(data, dict) else []
    pages_by_url = {page.url: page for page in pages.pages}

    kept: list[CompetitorFact] = []
    dropped = 0
    for item in items:
        if not isinstance(item, dict):
            dropped += 1
            continue
        claim = str(item.get("claim", "")).strip()
        source_url = str(item.get("source_url", "")).strip()
        quote = str(item.get("quote", "")).strip()
        page = pages_by_url.get(source_url)
        if not claim or not quote or page is None or not quote_is_grounded(quote, page.text):
            dropped += 1
            continue
        kept.append(CompetitorFact(claim=claim, source_url=source_url, quote=quote))
    return kept, dropped
