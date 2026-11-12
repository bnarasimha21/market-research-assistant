"""Typed data contracts used across the pipeline.

ResearchInput validates user-supplied names before any LLM call is made.
ProductFacts / CompetitorFacts carry cited facts between stages. Comparison /
ComparisonList (moved here from the old comparison_report.py) are the final
output, with cells generalized to carry sources and a support status instead
of bare strings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_COMPETITORS = 5


class ResearchInput(BaseModel):
    """Validated user input: the product being evaluated and its competitors.

    Raising here (via pydantic ValidationError, wrapped by callers into
    ConfigError) is what gives issue 1's "invalid input raises ConfigError
    before any LLM call" guarantee.
    """

    my_product: str
    competitors: list[str] = Field(..., min_length=1, max_length=MAX_COMPETITORS)

    @field_validator("my_product")
    @classmethod
    def _non_empty_product(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("my_product must not be empty")
        return value

    @field_validator("competitors")
    @classmethod
    def _clean_competitors(cls, value: list[str]) -> list[str]:
        cleaned = [c.strip() for c in value]
        if any(not c for c in cleaned):
            raise ValueError("competitor names must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _distinct_names(self) -> ResearchInput:
        names = [self.my_product, *self.competitors]
        lowered = [n.lower() for n in names]
        if len(set(lowered)) != len(lowered):
            raise ValueError("my_product and competitor names must all be distinct")
        return self


class Fact(BaseModel):
    """A single cited fact about the product or a competitor."""

    text: str
    source: str
    snippet: str = ""


class ProductFacts(BaseModel):
    product: str
    items: list[Fact] = Field(default_factory=list)

    def as_prompt_block(self) -> str:
        if not self.items:
            return f"No sourced facts were found for {self.product}."
        lines = [f"- {item.text} (source: {item.source})" for item in self.items]
        return f"Facts about {self.product}:\n" + "\n".join(lines)


class CompetitorFact(BaseModel):
    """A claim about a competitor, grounded in a verbatim quote from a fetched page."""

    claim: str
    source_url: str
    quote: str


class CompetitorFacts(BaseModel):
    competitor: str
    items: list[CompetitorFact] = Field(default_factory=list)

    def as_prompt_block(self) -> str:
        if not self.items:
            return f"No grounded facts were found for {self.competitor}."
        lines = [
            f'- {item.claim} (source: {item.source_url}, quote: "{item.quote}")'
            for item in self.items
        ]
        return f"Facts about {self.competitor}:\n" + "\n".join(lines)


class Cell(BaseModel):
    """One table cell: text plus the sources backing it, or none."""

    text: str
    sources: list[str] = Field(default_factory=list)
    status: Literal["supported", "no_evidence"] = "no_evidence"


class Comparison(BaseModel):
    """One row of the comparison table.

    `cells` is keyed by product name (the user's product name, then each
    competitor name) rather than fixed My_Product/Competitor1/Competitor2
    fields, so the table supports any number of competitors, matching
    ResearchInput.competitors being a list rather than two fixed slots.
    """

    category: str
    cells: dict[str, Cell]


class ComparisonList(BaseModel):
    comparisons: list[Comparison] = Field(default_factory=list)
