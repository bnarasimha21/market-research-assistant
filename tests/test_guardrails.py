"""Unit tests for market_research.guardrails: one passing and one failing
input per output-contract rule, plus the stateful retry/patch behaviour of
make_comparison_guardrail.
"""

from __future__ import annotations

import json

from market_research.categories import CATEGORIES, MAX_CELL_LENGTH
from market_research.guardrails import (
    MAX_GUARDRAIL_ATTEMPTS,
    make_comparison_guardrail,
    validate_comparison_payload,
)
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


# -- supported sources must match a real fact source, not just be non-trivial --
# Round-2 finding: the previous check was only MIN_SOURCE_LENGTH (3+ chars),
# so a fabricated source like "trust me" passed as "supported". These tests
# pass `valid_sources` (the real fact sources available per product) and
# assert the fabricated case is now rejected while a real one still passes.


def _valid_list_with_sources(cell0_sources_by_product: dict[str, list[str]]):
    full = _full_valid_list()
    comparisons = list(full.comparisons)
    comparisons[0] = Comparison(
        category=comparisons[0].category,
        cells={
            name: Cell(text=f"text for {name}", sources=sources, status="supported")
            for name, sources in cell0_sources_by_product.items()
        },
    )
    return ComparisonList(comparisons=comparisons)


def test_fabricated_source_fails_against_real_fact_sources():
    # The exact ["trust me"] case from the round-2 review.
    comparison_list = _valid_list_with_sources(
        {"Narsi": ["trust me"], "Acme": ["real-source#pricing"]}
    )
    valid_sources = {"Narsi": {"product_narsi.md#pricing"}, "Acme": {"real-source#pricing"}}
    ok, error = validate_comparison_payload(comparison_list, valid_sources=valid_sources)
    assert not ok
    assert "none of which matches a real fact source" in error


def test_abc_and_my_memory_fabricated_sources_also_fail():
    for fabricated in ("abc", "my memory"):
        comparison_list = _valid_list_with_sources(
            {"Narsi": [fabricated], "Acme": ["real-source#pricing"]}
        )
        valid_sources = {"Narsi": {"product_narsi.md#pricing"}, "Acme": {"real-source#pricing"}}
        ok, error = validate_comparison_payload(comparison_list, valid_sources=valid_sources)
        assert not ok, f"expected {fabricated!r} to be rejected"
        assert "none of which matches a real fact source" in error


def test_real_source_string_passes_against_valid_sources():
    # _full_valid_list() fills every category's cells with sources=["src:<name>"],
    # so valid_sources must cover that default too, not just the overridden
    # category's source, or every other category would be (correctly)
    # flagged as citing an unmatched source.
    comparison_list = _valid_list_with_sources(
        {"Narsi": ["product_narsi.md#pricing"], "Acme": ["real-source#pricing"]}
    )
    valid_sources = {
        "Narsi": {"product_narsi.md#pricing", "src:Narsi"},
        "Acme": {"real-source#pricing", "src:Acme"},
    }
    ok, error = validate_comparison_payload(comparison_list, valid_sources=valid_sources)
    assert ok
    assert error is None


def test_guardrail_rejects_fabricated_source_end_to_end():
    """The same ["trust me"] case, but through make_comparison_guardrail
    (the actual Task-attached callable), not just validate_comparison_payload.
    """
    valid_sources = {"Narsi": {"product_narsi.md#pricing"}, "Acme": {"acme.com/pricing"}}
    guardrail = make_comparison_guardrail(["Narsi", "Acme"], valid_sources)
    bad_list = _valid_list_with_sources({"Narsi": ["trust me"], "Acme": ["acme.com/pricing"]})
    ok, error = guardrail(bad_list.model_dump_json())
    assert not ok
    assert "real fact source" in error


def test_guardrail_accepts_real_source_end_to_end():
    valid_sources = {
        "Narsi": {"product_narsi.md#pricing", "src:Narsi"},
        "Acme": {"acme.com/pricing", "src:Acme"},
    }
    guardrail = make_comparison_guardrail(["Narsi", "Acme"], valid_sources)
    good_list = _valid_list_with_sources(
        {"Narsi": ["product_narsi.md#pricing"], "Acme": ["acme.com/pricing"]}
    )
    ok, _ = guardrail(good_list.model_dump_json())
    assert ok


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


# -- round-1 regression coverage (finding 3): these scenarios were fixed in
# round 1 (guardrails.py changed substantially to add them) but never got
# their own tests. Each must be flagged on the FINAL guardrail attempt (the
# forced-success/patch path), proving the bad category is dropped and
# replaced with a "no_evidence" placeholder rather than passed through.


def _force_final_attempt(guardrail, bad_payload: str):
    """Drive a guardrail through every allowed attempt with `bad_payload`
    repeated, returning the final (forced-success) result.
    """
    result = None
    for _ in range(MAX_GUARDRAIL_ATTEMPTS):
        result = guardrail(bad_payload)
    return result


def test_identical_cell_text_is_flagged_on_final_attempt():
    full = _full_valid_list()
    comparisons = list(full.comparisons)
    comparisons[0] = Comparison(
        category=comparisons[0].category,
        cells={
            "Narsi": Cell(text="same text", sources=["s"], status="supported"),
            "Acme": Cell(text="same text", sources=["s"], status="supported"),
        },
    )
    bad = ComparisonList(comparisons=comparisons).model_dump_json()
    guardrail = make_comparison_guardrail(["Narsi", "Acme"])
    ok, payload = _force_final_attempt(guardrail, bad)
    assert ok
    assert comparisons[0].category in guardrail.state.missing_categories
    parsed = ComparisonList.model_validate_json(payload)
    patched_row = next(c for c in parsed.comparisons if c.category == comparisons[0].category)
    assert all(cell.status == "no_evidence" for cell in patched_row.cells.values())


def test_zero_source_supported_cell_structurally_is_flagged_on_final_attempt():
    """Distinct from the string-based source check in finding 1: this is
    `sources=[]` with `status="supported"`, a structurally missing source,
    not a fabricated string.
    """
    full = _full_valid_list()
    comparisons = list(full.comparisons)
    comparisons[0] = Comparison(
        category=comparisons[0].category,
        cells={
            "Narsi": Cell(text="x", sources=[], status="supported"),
            "Acme": Cell(text="y", sources=["s"], status="supported"),
        },
    )
    bad = ComparisonList(comparisons=comparisons).model_dump_json()
    guardrail = make_comparison_guardrail(["Narsi", "Acme"])
    ok, payload = _force_final_attempt(guardrail, bad)
    assert ok
    assert comparisons[0].category in guardrail.state.missing_categories


def test_eleventh_made_up_category_is_flagged_on_final_attempt():
    full = _full_valid_list()
    extra = Comparison(
        category="Not A Real Category",
        cells={name: Cell(text="x", sources=["s"], status="supported") for name in ("Narsi", "Acme")},
    )
    bad = ComparisonList(comparisons=[*full.comparisons, extra]).model_dump_json()
    guardrail = make_comparison_guardrail(["Narsi", "Acme"])
    ok, payload = _force_final_attempt(guardrail, bad)
    assert ok
    parsed = ComparisonList.model_validate_json(payload)
    categories_seen = {c.category for c in parsed.comparisons}
    assert categories_seen == set(CATEGORIES)
    assert "Not A Real Category" not in categories_seen


def test_duplicate_category_is_flagged_on_final_attempt():
    full = _full_valid_list()
    bad = ComparisonList(comparisons=[*full.comparisons, full.comparisons[0]]).model_dump_json()
    guardrail = make_comparison_guardrail(["Narsi", "Acme"])
    ok, payload = _force_final_attempt(guardrail, bad)
    assert ok
    assert full.comparisons[0].category in guardrail.state.missing_categories
    parsed = ComparisonList.model_validate_json(payload)
    categories_seen = [c.category for c in parsed.comparisons]
    assert categories_seen.count(full.comparisons[0].category) == 1


def test_wrong_product_name_keys_is_flagged_on_final_attempt():
    full = _full_valid_list()
    comparisons = list(full.comparisons)
    comparisons[0] = Comparison(
        category=comparisons[0].category,
        cells={
            "WrongName": Cell(text="x", sources=["s"], status="supported"),
            "Acme": Cell(text="y", sources=["s"], status="supported"),
        },
    )
    bad = ComparisonList(comparisons=comparisons).model_dump_json()
    guardrail = make_comparison_guardrail(["Narsi", "Acme"])
    ok, payload = _force_final_attempt(guardrail, bad)
    assert ok
    assert comparisons[0].category in guardrail.state.missing_categories


def test_empty_cells_dict_is_flagged_on_final_attempt():
    full = _full_valid_list()
    comparisons = list(full.comparisons)
    comparisons[0] = Comparison(category=comparisons[0].category, cells={})
    bad = ComparisonList(comparisons=comparisons).model_dump_json()
    guardrail = make_comparison_guardrail(["Narsi", "Acme"])
    ok, payload = _force_final_attempt(guardrail, bad)
    assert ok
    assert comparisons[0].category in guardrail.state.missing_categories
    parsed = ComparisonList.model_validate_json(payload)
    patched_row = next(c for c in parsed.comparisons if c.category == comparisons[0].category)
    assert set(patched_row.cells.keys()) == {"Narsi", "Acme"}


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
