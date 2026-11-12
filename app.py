"""The Streamlit UI. A thin caller of market_research.pipeline.run_comparison:
all research/grounding/guardrail logic lives in market_research/, not here.

Known simplification: run_comparison() runs all three stages inside one
blocking call, and crewai's Flow.kickoff() gives no hook for incremental UI
updates without deeper plumbing. So "stage-by-stage progress" here means a
single spinner while the run is in flight, followed by a detailed per-stage
trace table (duration, LLM calls, tokens, pages fetched, guardrail retries)
shown once the run completes, rather than the table filling in live.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from market_research.errors import ConfigError, StageFailed
from market_research.llm import current_model_id
from market_research.models import ResearchInput
from market_research.pipeline import run_comparison
from market_research.pricing import estimate_cost_usd, load_pricing

# The original app.py called this; the typed-pipeline rewrite dropped it,
# which meant following the README's own setup steps (copy .env.example to
# .env, `streamlit run app.py`) silently did nothing: run_comparison() would
# raise ConfigError for a missing OPENAI_API_KEY even though .env had it.
# crew_flow.py (the CLI entry point) still calls this; app.py must too,
# since it no longer meaningfully depends on crew_flow.py for this.
load_dotenv()

DISCLOSURE_BANNER = (
    "\"My product\" in this demo is Narsi, a fictional AI coding assistant "
    "described in product_narsi.md for demonstration purposes. It is not a "
    "real product."
)


def _result_to_rows(comparison_list, product_names: list[str]) -> list[dict]:
    rows = []
    seen_categories: set[str] = set()
    for comparison in comparison_list.comparisons:
        if comparison.category in seen_categories:
            # Defensive: the comparison stage's guardrail should never let a
            # duplicate category through, but if one ever does, keep only
            # the first occurrence. A duplicate "Category" value would
            # otherwise become a duplicate index in the DataFrame built
            # below, and df.loc[category, name] returns a Series instead of
            # a scalar for a duplicate index, which crashes the per-category
            # detail view further down.
            continue
        seen_categories.add(comparison.category)
        row = {"Category": comparison.category}
        for name in product_names:
            cell = comparison.cells.get(name)
            if cell is None:
                row[name] = ""
                row[f"{name} sources"] = ""
                continue
            row[name] = cell.text
            row[f"{name} sources"] = "; ".join(cell.sources)
        rows.append(row)
    return rows


def _markdown_report(result, product_names: list[str]) -> str:
    lines = [f"# Comparison: {' vs '.join(product_names)}", ""]
    if result.guardrail_partial:
        lines.append(
            "> **Note:** the model did not produce a complete, valid table within the "
            f"retry budget. These categories were filled with placeholders: "
            f"{', '.join(result.missing_categories)}."
        )
        lines.append("")
    for comparison in result.comparisons.comparisons:
        lines.append(f"## {comparison.category}")
        for name in product_names:
            cell = comparison.cells.get(name)
            if cell is None:
                continue
            lines.append(f"**{name}:** {cell.text}")
            if cell.sources:
                lines.append(f"Sources: {', '.join(cell.sources)}")
            lines.append("")
    return "\n".join(lines)


def _show_trace(result) -> None:
    pricing = load_pricing()
    model_id = current_model_id()
    with st.expander("Run trace (cost, latency, tokens)"):
        trace_rows = []
        total_cost = 0.0
        any_cost_known = False
        for stage in result.trace.stages:
            cost = (
                estimate_cost_usd(model_id, stage.prompt_tokens, stage.completion_tokens, pricing)
                if model_id
                else None
            )
            if cost is not None:
                total_cost += cost
                any_cost_known = True
            trace_rows.append(
                {
                    "Stage": stage.name,
                    "Duration (s)": round(stage.duration_seconds, 3),
                    "LLM calls": stage.llm_calls,
                    "Prompt tokens": stage.prompt_tokens,
                    "Completion tokens": stage.completion_tokens,
                    "Pages fetched": stage.pages_fetched,
                    "Guardrail retries": stage.guardrail_retries,
                }
            )
        st.dataframe(pd.DataFrame(trace_rows), use_container_width=True)
        st.write(f"Total duration: {result.trace.total_duration_seconds:.3f}s, total tokens: {result.trace.total_tokens}")
        if any_cost_known:
            st.caption(
                f"Estimated cost: ${total_cost:.4f} for model {model_id!r}, from pricing.yaml's "
                "list prices (see that file's source_url/as_of per model). This is an estimate "
                "based on the currently configured LLM_PROVIDER, not a bill, and may not match "
                "the provider a cached result was originally run with."
            )
        if result.dropped_facts:
            st.caption(f"{result.dropped_facts} competitor claim(s) were dropped for lacking a grounded quote.")
        if result.from_cache:
            st.caption("This result was served from the disk cache (no LLM calls were made this run).")


def display_comparison_table(result, product_names: list[str]) -> None:
    rows = _result_to_rows(result.comparisons, product_names)
    df = pd.DataFrame(rows).set_index("Category")

    st.write("")
    st.subheader(" vs ".join(product_names))

    if result.guardrail_partial:
        st.warning(
            "The comparison table is partial: the model could not produce a fully valid "
            f"table within the retry budget. Missing/placeholder categories: "
            f"{', '.join(result.missing_categories)}."
        )

    display_columns = [c for c in df.columns if not c.endswith(" sources")]
    st.dataframe(df[display_columns], use_container_width=True, height=400)

    csv = df.to_csv()
    st.download_button(
        label="Download comparison as CSV (includes a sources column per product)",
        data=csv,
        file_name="comparison.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download report (Markdown with sources)",
        data=_markdown_report(result, product_names),
        file_name="comparison_report.md",
        mime="text/markdown",
    )

    st.subheader("Category details")
    for category in df.index:
        with st.expander(category):
            for name in product_names:
                st.write(f"**{name}:** {df.loc[category, name]}")
                sources = df.loc[category, f"{name} sources"]
                if sources:
                    st.caption(f"Sources: {sources}")

    _show_trace(result)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Product Comparison Analysis")
    st.info(DISCLOSURE_BANNER)

    col1, col2, col3 = st.columns(3)
    with col1:
        my_product = st.text_input("My product name", "Narsi")
    with col2:
        competitor1 = st.text_input("Enter first competitor name", "GitHub Copilot")
    with col3:
        competitor2 = st.text_input("Enter second competitor name", "Cursor")

    force_refresh = st.checkbox(
        "Force a fresh run (ignore the cached result, if any)",
        value=False,
        help="A cached result normally expires after 24h, or when a known-partial "
        "guardrail result would otherwise be served again. Check this to skip the "
        "cache for this run regardless.",
    )

    if st.button("Generate Comparison"):
        competitors = [c.strip() for c in (competitor1, competitor2) if c.strip()]
        try:
            inputs = ResearchInput(my_product=my_product, competitors=competitors)
        except Exception as exc:  # ResearchInput wraps validation into a clear message
            st.error(f"Invalid input: {exc}")
        else:
            with st.spinner("Running product, competitor and comparison research..."):
                try:
                    result = run_comparison(inputs, force_refresh=force_refresh)
                except ConfigError as exc:
                    st.error(f"Configuration problem, nothing was run: {exc}")
                except StageFailed as exc:
                    st.error(f"The {exc.stage} stage failed: {exc.reason}")
                else:
                    display_comparison_table(result, [inputs.my_product, *inputs.competitors])
