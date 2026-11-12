"""Output-contract guardrails for the comparison stage.

Replaces relying on `output_pydantic=ComparisonList` alone, which checked
shape but nothing about content: category completeness, empty cells,
duplicate text across products, or that "supported" cells actually cite a
source. Attached to the comparison Task as `guardrail=`, which CrewAI
retries (`guardrail_max_retries`) by re-running the agent with the error
message folded into the next prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from market_research.categories import CATEGORIES, MAX_CELL_LENGTH
from market_research.models import Cell, Comparison, ComparisonList

GUARDRAIL_MAX_RETRIES = 2
MAX_GUARDRAIL_ATTEMPTS = GUARDRAIL_MAX_RETRIES + 1

# A "supported" cell's source must be more than an empty or near-empty
# placeholder string (e.g. "" or " ") to count as a real citation.
MIN_SOURCE_LENGTH = 3


def validate_comparison(output: object, product_names: list[str] | None = None) -> tuple[bool, Any]:
    """Validate a task output (or a raw JSON string / ComparisonList) against
    the output contract.

    Returns (True, parsed_json_str) on success, (False, short error message)
    on failure. This is the pure, stateless check: parse, then run every
    rule in `validate_comparison_payload`. It does not retry or patch
    anything itself; see GuardrailTracker for the stateful wrapper attached
    to the real Task.
    """
    raw = getattr(output, "raw", output)
    try:
        comparison_list = _parse(raw)
    except (ValidationError, ValueError, TypeError) as exc:
        return False, f"output is not valid JSON matching the ComparisonList schema: {exc}"

    ok, error = validate_comparison_payload(comparison_list, product_names)
    if not ok:
        return False, error
    return True, comparison_list.model_dump_json()


def _parse(raw: object) -> ComparisonList:
    if isinstance(raw, ComparisonList):
        return raw
    if isinstance(raw, dict):
        return ComparisonList.model_validate(raw)
    if isinstance(raw, str):
        text = raw.strip()
        # Agents sometimes wrap JSON in a markdown fence; strip it.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        return ComparisonList.model_validate(json.loads(text))
    raise TypeError(f"cannot parse comparison output of type {type(raw)!r}")


def _category_violations(comparison: Comparison, product_names: list[str] | None) -> list[str]:
    """Every output-contract violation for a single comparison row, keyed to
    its own category elsewhere. Does NOT cover the list-level rules
    (missing/duplicated/unknown categories); see `validate_comparison_payload`.
    """
    violations: list[str] = []

    if product_names is not None:
        cell_keys = set(comparison.cells.keys())
        expected_keys = set(product_names)
        if cell_keys != expected_keys:
            violations.append(
                f"cells are keyed by {sorted(cell_keys)} but the real product names are "
                f"{sorted(expected_keys)}"
            )

    texts_by_product: dict[str, str] = {}
    for product, cell in comparison.cells.items():
        if not cell.text.strip():
            violations.append(f"empty cell for {product!r}")
            continue
        if len(cell.text) > MAX_CELL_LENGTH:
            violations.append(f"cell for {product!r} exceeds {MAX_CELL_LENGTH} characters")
        if cell.status == "supported":
            if not cell.sources:
                violations.append(f"cell for {product!r} is marked 'supported' but has no sources")
            elif any(not s.strip() or len(s.strip()) < MIN_SOURCE_LENGTH for s in cell.sources):
                violations.append(
                    f"cell for {product!r} is marked 'supported' but has an implausible/trivial "
                    f"source (empty or shorter than {MIN_SOURCE_LENGTH} characters)"
                )
        texts_by_product[product] = cell.text.strip().lower()

    seen_texts: dict[str, str] = {}
    for product, text in texts_by_product.items():
        if text in seen_texts:
            violations.append(f"identical cell text for {seen_texts[text]!r} and {product!r}")
        else:
            seen_texts[text] = product

    return violations


def validate_comparison_payload(
    comparison_list: ComparisonList, product_names: list[str] | None = None
) -> tuple[bool, str | None]:
    """Run every output-contract rule against an already-parsed ComparisonList.

    Returns (True, None) if every rule passes, else (False, "<every
    failing rule found, joined>"). Collects every violation rather than
    stopping at the first, so the error folded back into the retry prompt
    (and the final-attempt diagnosis in `_build_patched_list`) names every
    real problem, not just the first one encountered.

    product_names, when given, is the real list of product names the
    caller actually asked to compare (my_product plus every competitor).
    When omitted, the cells-keys-match-product-names rule is skipped (kept
    optional so existing unit tests that only care about one category/cell
    rule at a time don't need to pass it). Each rule below is covered by
    its own passing/failing unit test in tests/test_guardrails.py.
    """
    errors: list[str] = []
    categories_seen: list[str] = [c.category for c in comparison_list.comparisons]

    missing = [c for c in CATEGORIES if categories_seen.count(c) == 0]
    if missing:
        errors.append(f"missing categories: {', '.join(missing)}")

    duplicated = sorted({c for c in categories_seen if categories_seen.count(c) > 1})
    if duplicated:
        errors.append(f"duplicated categories: {', '.join(duplicated)}")

    extra = sorted(set(categories_seen) - set(CATEGORIES))
    if extra:
        errors.append(f"unknown categories not in the configured list: {', '.join(extra)}")

    for comparison in comparison_list.comparisons:
        for violation in _category_violations(comparison, product_names):
            errors.append(f"category {comparison.category!r}: {violation}")

    if errors:
        return False, "; ".join(errors)
    return True, None


@dataclass
class GuardrailState:
    """Mutable record of what happened across a comparison task's guardrail attempts."""

    attempts: int = 0
    last_error: str | None = None
    partial: bool = False
    missing_categories: list[str] = field(default_factory=list)

    @property
    def retries(self) -> int:
        return max(self.attempts - 1, 0)


def _build_patched_list(
    comparison_list: ComparisonList, product_names: list[str]
) -> tuple[ComparisonList, list[str]]:
    """Build a schema-valid ComparisonList for the final, forced-success
    attempt, and return the full list of categories that failed and were
    therefore replaced with a "no_evidence" placeholder row.

    A category counts as failed here -- and gets patched -- for any of:
    literally missing, duplicated (every occurrence is dropped, not just
    the extras), named outside the configured CATEGORIES list (dropped),
    or present exactly once but failing a cell-level rule (empty cell,
    length cap, an unsupported 'supported' cell, a trivial source, cells
    keyed by the wrong product names, or duplicate text across products).
    This is deliberately broader than "literally absent from the reply":
    a row that is present but wrong must never be passed through
    unflagged just because its category slot was technically filled.
    """
    seen_once: set[str] = set()
    duplicated: set[str] = set()
    good: dict[str, Comparison] = {}

    for comparison in comparison_list.comparisons:
        category = comparison.category
        if category not in CATEGORIES:
            continue  # unknown category: drop entirely, never shown
        if category in seen_once:
            duplicated.add(category)
            good.pop(category, None)
            continue
        seen_once.add(category)
        if _category_violations(comparison, product_names):
            continue  # present but invalid: drop, will be patched below
        good[category] = comparison

    failed = [c for c in CATEGORIES if c not in good]

    comparisons = [good[c] for c in CATEGORIES if c in good]
    for category in failed:
        cells = {
            name: Cell(text="No data available for this category.", sources=[], status="no_evidence")
            for name in product_names
        }
        comparisons.append(Comparison(category=category, cells=cells))
    order = {c: i for i, c in enumerate(CATEGORIES)}
    comparisons.sort(key=lambda c: order.get(c.category, len(CATEGORIES)))
    return ComparisonList(comparisons=comparisons), failed


def make_comparison_guardrail(product_names: list[str]):
    """Build the stateful guardrail callable attached to the comparison Task
    as `guardrail=`.

    This MUST be a plain function (a closure), not a callable class
    instance. CrewAI's guardrail retry path constructs an
    LLMGuardrailStartedEvent that unconditionally calls
    `inspect.getsource(self.guardrail)` on whatever was passed as
    `guardrail=` (crewai/events/types/llm_guardrail_events.py), and
    inspect.getsource only accepts modules, classes, methods, functions,
    tracebacks, frames or code objects -- never an arbitrary callable
    *instance*. Passing a `GuardrailTracker()` instance with `__call__`
    raised `TypeError: module, class, method, function, traceback, frame,
    or code object was expected, got GuardrailTracker`. A closure defined
    with `def` is a real function object with a `__code__` pointing at this
    file, so inspect.getsource can read it back out fine.

    CrewAI's Task retry loop (guardrail_max_retries=GUARDRAIL_MAX_RETRIES)
    raises a bare Exception once attempts are exhausted, with no result
    payload attached, which would reach the UI as a stack trace. To satisfy
    "return a partial result with failing categories marked, not a stack
    trace", this guardrail forces success on the final allowed attempt,
    patching in placeholder rows for every category that is still
    missing, duplicated, unknown, or individually invalid (see
    `_build_patched_list`). The final attempt's `last_error` is NEVER
    cleared on a forced success, and `missing_categories` lists every
    category that was patched -- not just the ones literally absent from
    the reply -- so a caller cannot mistake a forced partial success for a
    real, fully-valid result.

    Returns the guardrail function; its mutable run state is reachable as
    `guardrail_fn.state` (a GuardrailState), since plain functions accept
    arbitrary attribute assignment.
    """

    state = GuardrailState()

    def _fail_or_patch(error: str, comparison_list: ComparisonList):
        state.last_error = error
        if state.attempts < MAX_GUARDRAIL_ATTEMPTS:
            return False, error

        patched, failed = _build_patched_list(comparison_list, product_names)
        state.partial = True
        state.missing_categories = failed
        return True, patched.model_dump_json()

    # No return type annotation here on purpose: this is CrewAI's
    # Task.guardrail callable, and Task.validate_guardrail_function() reads
    # the annotation via plain inspect.signature() (no eval_str=True). With
    # `from __future__ import annotations` active in this module, any
    # annotation would come back as the *string* "tuple[bool, Any]" rather
    # than the real type, so CrewAI's get_origin/get_args check fails and
    # rejects the guardrail at Task construction time. Leaving the
    # annotation off avoids that check entirely (it only runs when an
    # annotation is present).
    def comparison_guardrail(output: object):
        state.attempts += 1
        raw = getattr(output, "raw", output)

        try:
            comparison_list = _parse(raw)
        except (ValidationError, ValueError, TypeError) as exc:
            error = f"output is not valid JSON matching the ComparisonList schema: {exc}"
            return _fail_or_patch(error, ComparisonList(comparisons=[]))

        ok, error = validate_comparison_payload(comparison_list, product_names)
        if ok:
            state.last_error = None
            return True, comparison_list.model_dump_json()

        return _fail_or_patch(error or "validation failed", comparison_list)

    comparison_guardrail.state = state
    return comparison_guardrail
