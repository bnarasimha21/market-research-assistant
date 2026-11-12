"""Offline evaluation harness for the market-research pipeline (issue 5).

Six product trios live under evals/fixtures/<trio>/: one golden.yaml per
trio plus one saved page-text file per competitor (each file's first two
lines record the source URL and fetch date it was captured from, matching
the real market_research.sources.Page shape). For every competitor and
category, golden.yaml either lists 1-3 short facts the saved page really
supports (each with a verbatim "quote" substring of that page), or lists the
category under no_evidence_categories when the page genuinely says nothing
about it.

For every trio this script:
  1. Builds a FixtureSearch pointed at the trio's saved page URLs and
     monkeypatches market_research.sources.fetch_page to serve the saved
     page text for those URLs instead of making a real HTTP request.
  2. Drives the real MarketResearchFlow (same class run_comparison() uses)
     with an LLM chosen by --llm.
  3. Scores the real output against golden.yaml on four metrics:
     category_coverage, grounding_rate, fact_recall, no_evidence_precision.
  4. Pulls cost/latency/token numbers from the real RunTrace the Flow
     produces (market_research.trace), the same object run_comparison()
     writes to runs/<timestamp>.json.

--llm fake (the default, and the only mode this module is exercised in by
CI and tests/test_evals_integration.py): every stage is answered by a
FakeLLM scripted directly from golden.yaml, so a "perfect" run should score
at or near 1.0 on every metric. This checks the scoring and plumbing are
correct, not a real model's quality -- see the "fake" results file's header.

--llm do_serverless: builds a real LLM via market_research.llm.get_llm(),
exercising the actual DigitalOcean serverless inference provider, and would
produce real per-trio quality numbers plus real cost/latency from the LLM
calls' usage. This flag forces LLM_PROVIDER=do_serverless for the duration
of the run (see run_trio below) rather than trusting whatever LLM_PROVIDER
happens to already be set in the ambient environment, since a mismatch there
would silently run the wrong provider while the report header still claimed
"--llm do_serverless". This script was never actually run in this mode in
this environment: no DO_MODEL_ACCESS_KEY was available, and no API keys were
requested or used. A reviewer or Narsi with DO_MODEL_ACCESS_KEY/DO_MODEL_ID
set can run `python -m evals.run --llm do_serverless` directly to get real
numbers.

A `--llm claude_cli` mode previously existed in this file's CLI choices but
was never actually wired to anything: market_research.llm.get_llm() has no
claude_cli branch, so passing it just called get_llm() with whatever
LLM_PROVIDER happened to already be set in the environment, ignoring the
flag entirely. Rather than fake a provider path that doesn't exist, that
choice has been removed. Wiring a real Anthropic/Claude provider into
market_research/llm.py is a separate, real piece of work (a new LLM_PROVIDER
branch there, a documented env var, a model id default) that this fix does
not invent; it is not done here.

--corrupt-trio <name>: deliberately mangles the first competitor's first
golden quote in that trio's saved page text before running it, so the
deterministic grounding check (quote_is_grounded) can no longer find it.
This is a demonstration flag only -- it is never used for the committed
report -- proving the grounding_rate metric actually detects a broken
fixture rather than always reading 1.0 by construction.
"""

from __future__ import annotations

import os

# Must be set before the first `import crewai` anywhere in the process
# (market_research.flow below pulls it in transitively): crewai's
# Telemetry() singleton is built lazily the first time it's touched, and it
# reads these two env vars at that construction time, not later, so setting
# them after the import would be too late. A reviewer confirmed this harness
# makes outbound telemetry calls on every real run (fake and real --llm
# modes alike) unless these are set; setdefault() so a real override in the
# caller's own environment still wins.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import yaml

import market_research.sources as sources_mod
from market_research.categories import CATEGORIES
from market_research.flow import MarketResearchFlow
from market_research.llm import current_model_id
from market_research.sources import FixtureSearch, Page

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

METRIC_NAMES = ("category_coverage", "grounding_rate", "fact_recall", "no_evidence_precision")


def list_trio_names() -> list[str]:
    return sorted(p.name for p in FIXTURES_DIR.iterdir() if p.is_dir())


def _load_trio(trio_dir: Path) -> dict[str, Any]:
    golden = yaml.safe_load((trio_dir / "golden.yaml").read_text(encoding="utf-8"))
    for competitor in golden["competitors"]:
        raw_file = (trio_dir / competitor["page_file"]).read_text(encoding="utf-8")
        # Fixture files are "URL: ...\nFetched: ...\n\n<body text>".
        _, _, body = raw_file.partition("\n\n")
        competitor["page_text"] = body.strip()
    return golden


def _build_page_map(golden: dict[str, Any], corrupt: bool) -> dict[str, Page]:
    page_map: dict[str, Page] = {}
    for idx, competitor in enumerate(golden["competitors"]):
        text = competitor["page_text"]
        if corrupt and idx == 0 and competitor["facts"]:
            bad_quote = competitor["facts"][0]["quote"]
            text = text.replace(bad_quote, "REDACTED REDACTED REDACTED")
        page_map[competitor["url"]] = Page(url=competitor["url"], fetched_at="2026-01-01", text=text)
    return page_map


def _competitor_analyst_script(golden: dict[str, Any]) -> list[str]:
    """One scripted reply per competitor, in golden.yaml's listed order,
    which is also the order gather_competitor_facts() iterates
    self.state.competitors -- matching FakeLLM's per-role reply-list
    semantics (tests/conftest.py).
    """
    scripts = []
    for competitor in golden["competitors"]:
        items = [
            {"claim": fact["claim"], "source_url": competitor["url"], "quote": fact["quote"]}
            for fact in competitor["facts"]
        ]
        scripts.append(json.dumps({"items": items}))
    return scripts


def _comparison_script(golden: dict[str, Any], my_product: str) -> str:
    """A single, schema-valid comparison reply built directly from
    golden.yaml so the guardrail passes on the first attempt: every
    category is present exactly once, every cell has non-empty, non-
    duplicate text, and every 'supported' cell cites a source.
    """
    competitor_names = [c["name"] for c in golden["competitors"]]
    facts_by_competitor_category = {
        c["name"]: {f["category"]: f for f in c["facts"]} for c in golden["competitors"]
    }
    comparisons = []
    for category in CATEGORIES:
        cells: dict[str, Any] = {
            my_product: {
                "text": f"{my_product} summary for {category}.",
                "sources": ["product_narsi.md#eval"],
                "status": "supported",
            }
        }
        for name in competitor_names:
            fact = facts_by_competitor_category[name].get(category)
            if fact is not None:
                cells[name] = {
                    "text": fact["claim"],
                    "sources": [f"{name}:{category}"],
                    "status": "supported",
                }
            else:
                cells[name] = {
                    "text": "No data available for this category.",
                    "sources": [],
                    "status": "no_evidence",
                }
        comparisons.append({"category": category, "cells": cells})
    return json.dumps({"comparisons": comparisons})


def _score(golden: dict[str, Any], comparison, competitor_facts: dict, dropped_facts: int) -> dict[str, float]:
    categories_seen = {c.category for c in comparison.comparisons} if comparison else set()
    category_coverage = len(categories_seen & set(CATEGORIES)) / len(CATEGORIES)

    kept = sum(len(cf.items) for cf in competitor_facts.values())
    total_attempted = kept + dropped_facts
    grounding_rate = (kept / total_attempted) if total_attempted else 1.0

    golden_facts = [(c["name"], f) for c in golden["competitors"] for f in c["facts"]]
    recalled = 0
    for name, fact in golden_facts:
        golden_tokens = {w for w in fact["claim"].lower().split() if len(w) > 3}
        kept_claims = [item.claim.lower() for item in competitor_facts[name].items] if name in competitor_facts else []
        needed = max(1, len(golden_tokens) // 2)
        hit = any(len(golden_tokens & {w for w in claim.split() if len(w) > 3}) >= needed for claim in kept_claims)
        if hit:
            recalled += 1
    fact_recall = recalled / len(golden_facts) if golden_facts else 1.0

    golden_no_evidence = {c["name"]: set(c["no_evidence_categories"]) for c in golden["competitors"]}
    correct_no_evidence = 0
    total_marked_no_evidence = 0
    if comparison:
        for comp in comparison.comparisons:
            for name, cell in comp.cells.items():
                if name not in golden_no_evidence or cell.status != "no_evidence":
                    continue
                total_marked_no_evidence += 1
                if comp.category in golden_no_evidence[name]:
                    correct_no_evidence += 1
    no_evidence_precision = (
        correct_no_evidence / total_marked_no_evidence if total_marked_no_evidence else 1.0
    )

    return {
        "category_coverage": category_coverage,
        "grounding_rate": grounding_rate,
        "fact_recall": fact_recall,
        "no_evidence_precision": no_evidence_precision,
    }


def run_trio(trio_name: str, llm_mode: str = "fake", corrupt: bool = False) -> dict[str, Any]:
    """Run one trio end-to-end through the real Flow and score it.

    Returns a dict with the four metrics (all floats in [0, 1]) plus
    total_duration_seconds and total_tokens pulled from the real RunTrace.
    """
    trio_dir = FIXTURES_DIR / trio_name
    golden = _load_trio(trio_dir)
    my_product = golden["my_product"]
    competitor_names = [c["name"] for c in golden["competitors"]]

    page_map = _build_page_map(golden, corrupt)
    search = FixtureSearch(
        {
            f"{name} pricing features documentation": [golden["competitors"][i]["url"]]
            for i, name in enumerate(competitor_names)
        }
    )

    if llm_mode == "fake":
        from tests.conftest import FakeLLM

        llm = FakeLLM(
            replies={
                "Product Researcher": f"Report for {my_product}.",
                "Competitor Analyst": _competitor_analyst_script(golden),
                "Comparison Report Generator": _comparison_script(golden, my_product),
            }
        )
    elif llm_mode == "do_serverless":
        from market_research.llm import get_llm

        # Force the provider the flag actually names, rather than trusting
        # whatever LLM_PROVIDER happens to already be set in the ambient
        # environment: without this, `--llm do_serverless` with
        # LLM_PROVIDER=openai still in the shell would silently run openai
        # while the rendered report's header claimed do_serverless.
        os.environ["LLM_PROVIDER"] = "do_serverless"
        llm = get_llm("eval")
    else:
        raise ValueError(f"unknown --llm mode: {llm_mode!r}")

    original_fetch_page = sources_mod.fetch_page

    def fake_fetch_page(url: str, use_cache: bool = True):
        return page_map.get(url)

    sources_mod.fetch_page = fake_fetch_page
    try:
        flow = MarketResearchFlow(llm=llm, facts_source=None, search_backend=search)
        flow.state.my_product = my_product
        flow.state.competitors = competitor_names
        flow.kickoff()
    finally:
        sources_mod.fetch_page = original_fetch_page

    trace = flow.recorder.build_trace()
    metrics = _score(golden, flow.state.comparison, flow.state.competitor_facts, flow.state.dropped_facts)
    metrics["trio"] = trio_name
    metrics["total_duration_seconds"] = trace.total_duration_seconds
    metrics["total_tokens"] = trace.total_tokens
    metrics["corrupted"] = corrupt
    return metrics


def run_eval(
    llm_mode: str = "fake",
    trio_names: list[str] | None = None,
    corrupt_trio: str | None = None,
) -> list[dict[str, Any]]:
    """Run every (or the given subset of) trio(s) and return their metric rows."""
    names = trio_names if trio_names is not None else list_trio_names()
    return [run_trio(name, llm_mode=llm_mode, corrupt=(name == corrupt_trio)) for name in names]


def _render_report(rows: list[dict[str, Any]], llm_mode: str, corrupt_trio: str | None) -> str:
    today = dt.date.today().isoformat()
    lines = [f"# Eval results: {today}, --llm {llm_mode}", ""]
    if llm_mode == "fake":
        lines.append(
            "Fake mode: every stage is answered by a scripted FakeLLM built directly from "
            "each trio's golden.yaml, so this checks the scoring and plumbing are correct, "
            "not a real model's quality. A perfect run scores at or near 1.0 on every metric."
        )
    else:
        # Read back the provider/model that was actually resolved for this
        # run (run_trio sets LLM_PROVIDER itself for do_serverless) rather
        # than just echoing the --llm flag string, so the header can never
        # claim a provider/model different from what really answered the
        # prompts.
        resolved_provider = os.getenv("LLM_PROVIDER", "<unset>")
        resolved_model = current_model_id() or "<unknown>"
        lines.append(
            f"Real-model mode ({llm_mode}): numbers below came from an actual "
            "market_research.llm.get_llm() call against LLM_PROVIDER="
            f"{resolved_provider!r}, model={resolved_model!r}."
        )
    lines.append("")
    header = "| trio | category_coverage | grounding_rate | fact_recall | no_evidence_precision | duration_s | tokens |"
    sep = "|---|---|---|---|---|---|---|"
    lines.extend([header, sep])
    for row in rows:
        marker = " (corrupted)" if row.get("corrupted") else ""
        lines.append(
            f"| {row['trio']}{marker} | {row['category_coverage']:.2f} | {row['grounding_rate']:.2f} | "
            f"{row['fact_recall']:.2f} | {row['no_evidence_precision']:.2f} | "
            f"{row['total_duration_seconds']:.3f} | {row['total_tokens']} |"
        )
    lines.append("")

    if corrupt_trio:
        corrupted_row = next(r for r in rows if r["trio"] == corrupt_trio)
        lines.append("## Corruption demonstration")
        lines.append("")
        lines.append(
            f"`--corrupt-trio {corrupt_trio}` mangled that trio's first competitor's first "
            "golden quote before running it. Its grounding_rate above "
            f"({corrupted_row['grounding_rate']:.2f}) is lower than a normal run of the same "
            "trio, proving the deterministic grounding check (quote_is_grounded) really "
            "detects a broken fixture instead of the metric always reading 1.0 by "
            "construction."
        )
        lines.append("")

    avg = {
        name: sum(r[name] for r in rows) / len(rows) if rows else 0.0
        for name in METRIC_NAMES
    }
    lines.append(
        "Average: category_coverage={category_coverage:.2f}, grounding_rate={grounding_rate:.2f}, "
        "fact_recall={fact_recall:.2f}, no_evidence_precision={no_evidence_precision:.2f}".format(**avg)
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline market-research eval set.")
    parser.add_argument("--llm", choices=["fake", "do_serverless"], default="fake")
    parser.add_argument("--trios", nargs="*", default=None, help="subset of trio names to run (default: all)")
    parser.add_argument(
        "--corrupt-trio",
        default=None,
        help="demonstration only: mangle this trio's first golden quote before running it",
    )
    parser.add_argument("--out", default=None, help="results markdown path (default: evals/results/<date>-<llm>.md)")
    args = parser.parse_args(argv)

    rows = run_eval(llm_mode=args.llm, trio_names=args.trios, corrupt_trio=args.corrupt_trio)
    for row in rows:
        for name in METRIC_NAMES:
            assert 0.0 <= row[name] <= 1.0, f"metric {name} out of range for {row['trio']}: {row[name]}"

    report = _render_report(rows, args.llm, args.corrupt_trio)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{dt.date.today().isoformat()}-{args.llm}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
