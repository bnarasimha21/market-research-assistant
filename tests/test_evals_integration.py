"""Integration test for issue 5's eval runner: runs the real Flow (via
evals.run.run_trio) on two fixture trios in fake mode and asserts every
metric lands in [0, 1] and that main() actually writes a results file.
"""

from __future__ import annotations

import pytest

from evals.run import METRIC_NAMES, list_trio_names, main, run_eval

pytestmark = pytest.mark.integration


def test_two_trios_in_fake_mode_produce_metrics_in_unit_range():
    names = list_trio_names()[:2]
    assert len(names) == 2

    rows = run_eval(llm_mode="fake", trio_names=names)
    assert len(rows) == 2
    for row in rows:
        assert row["trio"] in names
        for metric in METRIC_NAMES:
            assert 0.0 <= row[metric] <= 1.0
        assert row["total_duration_seconds"] >= 0.0
        assert row["total_tokens"] > 0


def test_main_writes_a_results_file(tmp_path):
    names = list_trio_names()[:2]
    out_path = tmp_path / "report.md"
    exit_code = main(["--llm", "fake", "--trios", *names, "--out", str(out_path)])
    assert exit_code == 0
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "category_coverage" in content
    for name in names:
        assert name in content


def test_corrupt_trio_flag_lowers_grounding_rate_for_that_trio():
    names = list_trio_names()
    target = names[0]

    normal = run_eval(llm_mode="fake", trio_names=[target])[0]
    corrupted = run_eval(llm_mode="fake", trio_names=[target], corrupt_trio=target)[0]

    assert corrupted["grounding_rate"] < normal["grounding_rate"]
