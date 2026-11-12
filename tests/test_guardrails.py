"""Unit tests for market_research.guardrails: one passing and one failing
input per output-contract rule, plus the stateful retry/patch behaviour of
make_comparison_guardrail.
"""

from __future__ import annotations

import json

from market_research.categories import CATEGORIES, MAX_CELL_LENGTH
from market_research.guardrails import make_comparison_guardrail, validate_comparison_payload
from market_research.models import Cell, Comparison, ComparisonList


def _full_valid_list(product_names=("Narsi", "Acme")):
    comparisons = []
    for category in CATEGORIES:
        cells = {
            name: Cell(text=f"{name} on {category}", sources=[f"src:{name}"], status="supported")
            for name in product_names
        }
        comparisons.append(Comparison(category=category, cells=cells))
    return ComparisonList(comparisons=comparisons)


# -- category completeness --


def test_complete_categories_pass():
    ok, error = validate_comparison_payload(_full_valid_list())
    assert ok
    assert error is None


def test_missing_category_fails():
    full = _full_valid_list()
    truncated = ComparisonList(comparisons=full.comparisons[:-1])
    ok, error = validate_comparison_payload(truncated)
    assert not ok
    assert "missing categories" in error


def test_duplicated_category_fails():
    full = _full_valid_list()
    duplicated = ComparisonList(comparisons=[*full.comparisons, full.comparisons[0]])
    ok, error = validate_comparison_payload(duplicated)
    assert not ok
    assert "duplicated categories" in error


def test_unknown_category_fails():
    full = _full_valid_list()
    extra = Comparison(
        category="Not A Real Category",
        cells={name: Cell(text="x", sources=["s"], status="supported") for name in ("Narsi", "Acme")},
    )
    ok, error = validate_comparison_payload(ComparisonList(comparisons=[*full.comparisons, extra]))
    assert not ok
    assert "unknown categories" in error


# -- empty cell --


def test_nonempty_cell_passes():
    ok, _ = validate_comparison_payload(_full_valid_list())
    assert ok


def test_empty_cell_fails():
    full = _full_valid_list()
    comparisons = list(full.comparisons)
    comparisons[0] = Comparison(
        category=comparisons[0].category,
        cells={"Narsi": Cell(text="", sources=[], status="no_evidence"), "Acme": Cell(text="y", sources=["s"], status="supported")},
    )
    ok, error = validate_comparison_payload(ComparisonList(comparisons=comparisons))
    assert not ok
    assert "empty cell" in error


# -- length cap --


def test_cell_under_length_cap_passes():
    ok, _ = validate_comparison_payload(_full_valid_list())
    assert ok


def test_cell_over_length_cap_fails():
    full = _full_valid_list()
    comparisons = list(full.comparisons)
    too_long = "x" * (MAX_CELL_LENGTH + 1)
    comparisons[0] = Comparison(
        category=comparisons[0].category,
        cells={
            "Narsi": Cell(text=too_long, sources=["s"], status="supported"),
            "Acme": Cell(text="y", sources=["s"], status="supported"),
        },
    )
    ok, error = validate_comparison_payload(ComparisonList(comparisons=comparisons))
    assert not ok
    assert "exceeds" in error


# -- duplicate text across products in a row --


def test_distinct_cell_text_passes():
    ok, _ = validate_comparison_payload(_full_valid_list())
    assert ok


def test_identical_cell_text_across_products_fails():
    full = _full_valid_list()
    comparisons = list(full.comparisons)
    comparisons[0] = Comparison(
        category=comparisons[0].category,
        cells={
            "Narsi": Cell(text="same text", sources=["s"], status="supported"),
            "Acme": Cell(text="same text", sources=["s"], status="supported"),
        },
    )
    ok, error = validate_comparison_payload(ComparisonList(comparisons=comparisons))
    assert not ok
    assert "identical cell text" in error


# -- supported needs sources --


def test_supported_with_sources_passes():
    ok, _ = validate_comparison_payload(_full_valid_list())
    assert ok


def test_supported_without_sources_fails():
    full = _full_valid_list()
    comparisons = list(full.comparisons)
    comparisons[0] = Comparison(
        category=comparisons[0].category,
        cells={
            "Narsi": Cell(text="x", sources=[], status="supported"),
            "Acme": Cell(text="y", sources=["s"], status="supported"),
        },
    )
    ok, error = validate_comparison_payload(ComparisonList(comparisons=comparisons))
    assert not ok
    assert "no sources" in error


# -- make_comparison_guardrail: stateful retry/patch behaviour --


def test_guardrail_accepts_a_valid_reply_on_first_attempt():
    guardrail = make_comparison_guardrail(["Narsi", "Acme"])
    ok, payload = guardrail(_full_valid_list().model_dump_json())
    assert ok
    assert guardrail.state.attempts == 1
    assert guardrail.state.retries == 0
    assert not guardrail.state.partial


def test_guardrail_rejects_malformed_json_with_retries_left():
    guardrail = make_comparison_guardrail(["Narsi", "Acme"])
    ok, error = guardrail("not json at all")
    assert not ok
    assert "not valid JSON" in error
    assert guardrail.state.attempts == 1


def test_guardrail_forces_partial_success_after_retries_exhausted():
    guardrail = make_comparison_guardrail(["Narsi", "Acme"])
    # Every attempt returns malformed output; guardrail_max_retries=2 means
    # 3 total attempts (MAX_GUARDRAIL_ATTEMPTS), and the 3rd must succeed
    # with a patched, schema-valid result rather than raise.
    ok1, _ = guardrail("still not json")
    ok2, _ = guardrail("still not json")
    ok3, payload = guardrail("still not json")
    assert not ok1
    assert not ok2
    assert ok3
    parsed = ComparisonList.model_validate_json(payload)
    categories_seen = {c.category for c in parsed.comparisons}
    assert categories_seen == set(CATEGORIES)
    assert guardrail.state.partial
    assert guardrail.state.missing_categories == list(CATEGORIES)
    assert guardrail.state.retries == 2


def test_guardrail_recovers_on_second_attempt():
    guardrail = make_comparison_guardrail(["Narsi", "Acme"])
    ok1, error1 = guardrail("not json")
    assert not ok1
    ok2, payload2 = guardrail(_full_valid_list().model_dump_json())
    assert ok2
    assert guardrail.state.retries == 1
    assert not guardrail.state.partial
    # json.loads should round-trip cleanly
    json.loads(payload2)
