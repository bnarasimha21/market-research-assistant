"""Unit tests for market_research.trace.RunRecorder/RunTrace that don't need
the full Flow/Crew machinery (see test_trace_integration.py for that)."""

from __future__ import annotations

import json

from market_research.trace import RunRecorder, RunTrace, recording_stage


def test_stage_duration_is_non_negative():
    recorder = RunRecorder()
    with recording_stage(recorder, "product"):
        pass
    trace = recorder.build_trace()
    assert len(trace.stages) == 1
    assert trace.stages[0].duration_seconds >= 0.0


def test_pages_fetched_and_guardrail_retries_are_recorded():
    recorder = RunRecorder()
    with recording_stage(recorder, "competitor"):
        recorder.record_pages_fetched("competitor", 2)
        recorder.record_pages_fetched("competitor", 1)
    with recording_stage(recorder, "comparison"):
        recorder.record_guardrail_retries("comparison", 2)
    trace = recorder.build_trace()
    by_name = {s.name: s for s in trace.stages}
    assert by_name["competitor"].pages_fetched == 3
    assert by_name["comparison"].guardrail_retries == 2


def test_run_trace_round_trips_through_json():
    recorder = RunRecorder()
    with recording_stage(recorder, "product"):
        pass
    with recording_stage(recorder, "competitor"):
        pass
    trace = recorder.build_trace()
    restored = RunTrace.model_validate_json(trace.model_dump_json())
    assert restored == trace
    # also exercise plain json round trip, since the UI reads runs/*.json directly
    json.loads(trace.model_dump_json())


def test_run_trace_write_creates_file(tmp_path):
    trace = RunTrace(stages=[], total_duration_seconds=0.0, total_tokens=0)
    target = tmp_path / "trace.json"
    written = trace.write(path=target)
    assert written == target
    assert target.exists()
    restored = RunTrace.model_validate_json(target.read_text(encoding="utf-8"))
    assert restored == trace
