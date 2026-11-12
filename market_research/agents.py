"""Factory functions building fresh Agent/Task objects per run.

Replaces module-level Agent/Task singletons (product_researcher,
competitor_analyst, comparison_analyst and their *_task counterparts) that
were shared across every Streamlit session and every Flow run, and whose
Task.context pointed at each other as module globals. Every call here
builds new objects, and the actual data for each stage is put directly into
the task description instead of being pulled in through a shared context=.
"""

from __future__ import annotations

from collections.abc import Callable

from crewai import Agent, Task
from crewai.llms.base_llm import BaseLLM

from market_research.categories import CATEGORIES
from market_research.guardrails import GUARDRAIL_MAX_RETRIES, make_comparison_guardrail
from market_research.models import CompetitorFacts, ProductFacts
from market_research.sources import CompetitorPages

_CATEGORY_LIST_TEXT = "\n".join(f"- {c}" for c in CATEGORIES)


def build_product_agent(llm: BaseLLM) -> Agent:
    return Agent(
        role="Product Researcher",
        goal="Write a clear, well-organized research report about the product using only the supplied facts.",
        backstory=(
            "A product analyst who never invents information: every claim in the report "
            "traces back to one of the supplied, cited facts."
        ),
        llm=llm,
        verbose=False,
    )


def build_product_task(agent: Agent, product: str, facts: ProductFacts) -> Task:
    return Task(
        description=(
            f"Write a research report about the product \"{product}\" covering these "
            f"categories where the facts below give you information:\n{_CATEGORY_LIST_TEXT}\n\n"
            f"Use only the following sourced facts, do not invent anything beyond them:\n\n"
            f"{facts.as_prompt_block()}"
        ),
        expected_output=(
            f"A readable plain-text research report about {product}, organized by category, "
            "citing the source given for each fact used."
        ),
        agent=agent,
    )


def build_competitor_agent(llm: BaseLLM) -> Agent:
    return Agent(
        role="Competitor Analyst",
        goal=(
            "Extract factual claims about a competitor strictly from the numbered page "
            "excerpts given, each backed by a verbatim quote from that excerpt."
        ),
        backstory=(
            "A careful analyst who never makes a claim without a verbatim quote from the "
            "source material to back it, because unsupported claims get discarded "
            "downstream."
        ),
        llm=llm,
        verbose=False,
    )


def _format_excerpts(pages: CompetitorPages) -> str:
    if not pages.pages:
        return "No pages could be fetched for this competitor."
    blocks = []
    for idx, page in enumerate(pages.pages, start=1):
        excerpt = page.text[:4000]
        blocks.append(f"[{idx}] URL: {page.url}\n{excerpt}")
    return "\n\n".join(blocks)


def build_competitor_task(agent: Agent, competitor: str, pages: CompetitorPages) -> Task:
    return Task(
        description=(
            f"Below are numbered excerpts fetched from {competitor}'s own pages. Using "
            f"ONLY the text in these excerpts, list factual claims about {competitor} "
            "relevant to these categories where the excerpts give you information:\n"
            f"{_CATEGORY_LIST_TEXT}\n\n"
            f"Excerpts:\n{_format_excerpts(pages)}"
        ),
        expected_output=(
            "A JSON object with one key, \"items\", a list of objects each with exactly "
            "these keys: \"claim\" (the factual claim), \"source_url\" (the URL of the "
            "excerpt it came from), \"quote\" (a verbatim substring copied exactly from "
            "that excerpt supporting the claim). Return JSON only, no other text. If no "
            "excerpts are available, return {\"items\": []}."
        ),
        agent=agent,
    )


def build_comparison_agent(llm: BaseLLM) -> Agent:
    return Agent(
        role="Comparison Report Generator",
        goal="Build a complete, evidence-based comparison table across the configured categories.",
        backstory=(
            "An analyst building a side-by-side comparison who only writes 'supported' for "
            "a cell when it cites at least one real source."
        ),
        llm=llm,
        verbose=False,
    )


def build_comparison_task(
    agent: Agent,
    my_product: str,
    competitor_names: list[str],
    product_block: str,
    competitor_facts: dict[str, CompetitorFacts],
) -> tuple[Task, Callable]:
    competitor_blocks = "\n\n".join(
        competitor_facts[name].as_prompt_block() for name in competitor_names
    )
    product_names = [my_product, *competitor_names]
    guardrail_fn = make_comparison_guardrail(product_names)

    task = Task(
        description=(
            f"Compare \"{my_product}\" against {', '.join(competitor_names)} using the "
            "facts below. Do not use any information other than what is given here.\n\n"
            f"{product_block}\n\n{competitor_blocks}\n\n"
            "Return a JSON object with one key \"comparisons\", a list with EXACTLY one "
            "entry per category below, each entry an object with:\n"
            "- \"category\": the category name, copied exactly from this list:\n"
            f"{_CATEGORY_LIST_TEXT}\n"
            f"- \"cells\": an object keyed by product name (exactly {product_names}), each "
            "value an object with \"text\" (a short comparison for that product on that "
            "category), \"sources\" (a list of source identifiers backing the text, taken "
            "from the facts above), and \"status\" (\"supported\" if at least one source "
            "backs the text, else \"no_evidence\"). Cells for different products must not "
            "repeat the exact same text."
        ),
        expected_output=(
            "A single JSON object matching the schema described above. Return JSON only, "
            "no markdown fence, no commentary."
        ),
        agent=agent,
        guardrail=guardrail_fn,
        guardrail_max_retries=GUARDRAIL_MAX_RETRIES,
    )
    return task, guardrail_fn
