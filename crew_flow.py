"""CLI entry point. A thin caller of market_research.pipeline.run_comparison,
same as app.py; this module owns no research/grounding/guardrail logic.

Usage: python crew_flow.py <my_product> <competitor1> [<competitor2> ...]
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from market_research.errors import ConfigError, StageFailed
from market_research.models import ResearchInput
from market_research.pipeline import run_comparison

load_dotenv()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python crew_flow.py <my_product> <competitor1> [<competitor2> ...]")
        return 2

    try:
        inputs = ResearchInput(my_product=argv[0], competitors=list(argv[1:]))
    except Exception as exc:
        print(f"Invalid input: {exc}")
        return 2

    try:
        result = run_comparison(inputs)
    except ConfigError as exc:
        print(f"Configuration problem, nothing was run: {exc}")
        return 2
    except StageFailed as exc:
        print(f"The {exc.stage} stage failed: {exc.reason}")
        return 1

    print(result.comparisons.model_dump_json(indent=2))
    if result.guardrail_partial:
        print(f"Warning: partial result, missing categories: {result.missing_categories}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
