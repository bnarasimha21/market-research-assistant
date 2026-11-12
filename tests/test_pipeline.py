"""Unit tests for market_research.pipeline.run_comparison's contract:
import safety, fail-fast validation, caching, and thread isolation.

The full three-stage Flow is exercised by test_flow_integration.py; these
tests use a FakeLLM but monkeypatch fetch_page to avoid any real network
regardless (gather_pages is still called, with an empty FixtureSearch, so
competitor stage legitimately gets zero pages -> no_evidence, which is fine
for what these tests check).
"""

from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from market_research.errors import ConfigError
from market_research.models import ResearchInput
from market_research.pipeline import run_comparison
from market_research.sources import FixtureSearch
from tests.conftest import FakeLLM


def test_importing_pipeline_pulls_in_no_streamlit_and_makes_no_network_call():
    """issue 1 acceptance: `python -c "import market_research.pipeline"` must
    succeed with no streamlit import and no network access, from a clean
    subprocess so we see real top-level import side effects, not whatever
    this test session already imported."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import market_research.pipeline; "
            "assert 'streamlit' not in sys.modules, 'pipeline import pulled in streamlit'",
        ],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_invalid_input_raises_configerror_before_any_llm_call():
    calls = []

    class RecordingLLM(FakeLLM):
        def call(self, *args, **kwargs):
            calls.append(1)
            return super().call(*args, **kwargs)

    llm = RecordingLLM(replies={})
    with pytest.raises(ConfigError):
        run_comparison({"my_product": "Narsi", "competitors": []}, llm=llm, use_cache=False)
    with pytest.raises(ConfigError):
        run_comparison({"my_product": "Narsi", "competitors": ["Narsi"]}, llm=llm, use_cache=False)
    assert calls == []


def _comparison_reply_for(my_product: str, competitors: list[str]) -> str:
    import json

    from market_research.categories import CATEGORIES

    comparisons = []
    for category in CATEGORIES:
        cells = {
            name: {"text": f"{name} - {category}", "sources": [f"src:{name}"], "status": "supported"}
            for name in [my_product, *competitors]
        }
        comparisons.append({"category": category, "cells": cells})
    return json.dumps({"comparisons": comparisons})


def _run(my_product: str, competitors: list[str], use_cache: bool, search=None):
    reply = _comparison_reply_for(my_product, competitors)
    llm = FakeLLM(
        replies={
            "Product Researcher": f"report for {my_product}",
            "Competitor Analyst": '{"items": []}',
            "Comparison Report Generator": reply,
        }
    )
    return run_comparison(
        ResearchInput(my_product=my_product, competitors=competitors),
        llm=llm,
        use_cache=use_cache,
        search_backend=search or FixtureSearch({}),
    )


def test_two_concurrent_runs_with_different_names_stay_isolated():
    results: dict[str, object] = {}
    errors: list[Exception] = []

    def worker(name: str, competitor: str) -> None:
        try:
            results[name] = _run(name, [competitor], use_cache=False)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=("Narsi", "Acme"))
    t2 = threading.Thread(target=worker, args=("Zed", "Globex"))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert not errors, errors
    assert results["Narsi"].my_product == "Narsi"
    assert results["Narsi"].competitors == ["Acme"]
    assert results["Zed"].my_product == "Zed"
    assert results["Zed"].competitors == ["Globex"]
    # each run's own trace only has its own 3 stages, never mixed with the other thread's
    assert len(results["Narsi"].trace.stages) == 3
    assert len(results["Zed"].trace.stages) == 3


def test_repeat_run_hits_the_result_cache(tmp_path, monkeypatch):
    import market_research.pipeline as pipeline_mod
    from market_research.cache import DiskCache

    # pipeline.py did `from market_research.cache import results_cache`, so
    # the name is bound in pipeline_mod's own namespace; patching the
    # original cache module's attribute wouldn't reach this reference.
    monkeypatch.setattr(pipeline_mod, "results_cache", lambda: DiskCache("results", ttl_seconds=None, root=tmp_path))

    first = _run("Narsi", ["Acme"], use_cache=True)
    assert not first.from_cache

    second = _run("Narsi", ["Acme"], use_cache=True)
    assert second.from_cache
    assert second.comparisons == first.comparisons
