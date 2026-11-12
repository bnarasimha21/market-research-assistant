"""Unit tests for market_research.models, mainly ResearchInput validation.

These are the "invalid input raises ConfigError before any LLM call"
building blocks for issue 1's acceptance criteria; pipeline.py wraps
pydantic's ValidationError into ConfigError, tested directly in
test_pipeline.py.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_research.models import Cell, Comparison, ComparisonList, ResearchInput


def test_valid_input_is_accepted():
    inputs = ResearchInput(my_product="Narsi", competitors=["Acme", "Globex"])
    assert inputs.my_product == "Narsi"
    assert inputs.competitors == ["Acme", "Globex"]


def test_empty_competitor_list_rejected():
    with pytest.raises(ValidationError):
        ResearchInput(my_product="Narsi", competitors=[])


def test_empty_competitor_name_rejected():
    with pytest.raises(ValidationError):
        ResearchInput(my_product="Narsi", competitors=["Acme", "  "])


def test_empty_my_product_rejected():
    with pytest.raises(ValidationError):
        ResearchInput(my_product="   ", competitors=["Acme"])


def test_duplicate_names_rejected():
    with pytest.raises(ValidationError):
        ResearchInput(my_product="Narsi", competitors=["narsi"])


def test_duplicate_competitors_rejected():
    with pytest.raises(ValidationError):
        ResearchInput(my_product="Narsi", competitors=["Acme", "ACME"])


def test_too_many_competitors_rejected():
    with pytest.raises(ValidationError):
        ResearchInput(my_product="Narsi", competitors=[f"C{i}" for i in range(6)])


def test_names_are_stripped():
    inputs = ResearchInput(my_product="  Narsi  ", competitors=["  Acme  "])
    assert inputs.my_product == "Narsi"
    assert inputs.competitors == ["Acme"]


def test_comparison_cells_keyed_by_product_name():
    comparison = Comparison(
        category="Pricing and Licensing",
        cells={
            "Narsi": Cell(text="Free tier available", sources=["product_narsi.md#pricing"], status="supported"),
            "Acme": Cell(text="No public pricing", sources=[], status="no_evidence"),
        },
    )
    assert comparison.cells["Narsi"].status == "supported"
    assert comparison.cells["Acme"].status == "no_evidence"


def test_comparison_list_round_trips_through_json():
    original = ComparisonList(
        comparisons=[
            Comparison(
                category="Pricing and Licensing",
                cells={"Narsi": Cell(text="x", sources=["s"], status="supported")},
            )
        ]
    )
    restored = ComparisonList.model_validate_json(original.model_dump_json())
    assert restored == original
