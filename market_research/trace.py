"""Per-stage cost, latency and token reporting.

What we found reading the pinned crewai==1.15.40 source (see
crewai/llms/base_llm.py): LLMCallCompletedEvent is NOT emitted
automatically by BaseLLM.call() or by __init_subclass__'s rate-limit
wrapper. Every concrete provider (OpenAICompletion, AnthropicCompletion,
etc.) is responsible for calling the protected helpers
_emit_call_started_event / _emit_call_completed_event itself from inside
its own call() implementation; a custom BaseLLM subclass that skips this
(as a naive FakeLLM would) simply never produces LLM events. tests/conftest.py's
FakeLLM therefore calls _emit_call_completed_event explicitly with a usage
dict ({"prompt_tokens", "completion_tokens", "total_tokens"}), which is
what lets this module's recorder see any token data at all for fake runs.

LLMCallCompletedEvent.usage is `dict[str, Any] | None`: whatever keys the
provider chose to put there. We look for "prompt_tokens"/"completion_tokens"
(OpenAI-style, which FakeLLM and litellm-backed providers both use) and
fall back to flow.usage_metrics (crewai's own cross-call accumulator) for
totals if a given event carries no usage at all.
"""

from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.llm_events import LLMCallCompletedEvent
from pydantic import BaseModel, Field

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


class StageTrace(BaseModel):
    name: str
    duration_seconds: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    guardrail_retries: int = 0
    pages_fetched: int = 0


class RunTrace(BaseModel):
    stages: list[StageTrace] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
    total_tokens: int = 0
    cache_hit: bool = False

    def write(self, path: Path | None = None) -> Path:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        target = path or (RUNS_DIR / f"{int(time.time() * 1000)}.json")
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target


@dataclass
class _StageAccumulator:
    name: str
    started_at: float
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    guardrail_retries: int = 0
    pages_fetched: int = 0
    ended_at: float | None = None

    def to_trace(self) -> StageTrace:
        duration = (self.ended_at or time.monotonic()) - self.started_at
        return StageTrace(
            name=self.name,
            duration_seconds=max(duration, 0.0),
            llm_calls=self.llm_calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            guardrail_retries=self.guardrail_retries,
            pages_fetched=self.pages_fetched,
        )


class RunRecorder:
    """Per-run accumulator. One instance per run_comparison() call.

    Made "current" for the calling thread via a contextvar while the run
    executes, so the process-wide event listener (registered once, see
    get_recorder_listener) can route each event to the right run even when
    several run_comparison() calls are in flight in different threads.
    """

    def __init__(self) -> None:
        self._stages: dict[str, _StageAccumulator] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def start_stage(self, name: str) -> None:
        with self._lock:
            self._stages[name] = _StageAccumulator(name=name, started_at=time.monotonic())
            self._order.append(name)

    def end_stage(self, name: str) -> None:
        with self._lock:
            if name in self._stages:
                self._stages[name].ended_at = time.monotonic()

    def record_guardrail_retries(self, name: str, retries: int) -> None:
        with self._lock:
            if name in self._stages:
                self._stages[name].guardrail_retries = retries

    def record_pages_fetched(self, name: str, pages: int) -> None:
        with self._lock:
            if name in self._stages:
                self._stages[name].pages_fetched += pages

    def _on_llm_call_completed(self, event: LLMCallCompletedEvent) -> None:
        current_stage = _current_stage.get(None)
        if current_stage is None:
            return
        with self._lock:
            acc = self._stages.get(current_stage)
            if acc is None:
                return
            acc.llm_calls += 1
            usage = event.usage or {}
            acc.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            acc.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

    def build_trace(self) -> RunTrace:
        with self._lock:
            stages = [self._stages[name].to_trace() for name in self._order]
        total_duration = sum(s.duration_seconds for s in stages)
        total_tokens = sum(s.prompt_tokens + s.completion_tokens for s in stages)
        return RunTrace(stages=stages, total_duration_seconds=total_duration, total_tokens=total_tokens)


_current_recorder: contextvars.ContextVar[RunRecorder | None] = contextvars.ContextVar(
    "market_research_current_recorder", default=None
)
_current_stage: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "market_research_current_stage", default=None
)


class _TraceEventListener(BaseEventListener):
    """Process-wide listener, instantiated once at import time and kept alive
    by the module-level reference below (crewai's event bus holds a strong
    reference to its bound methods once registered, but we still keep our
    own reference so this object is never a surprise target for GC in a
    future crewai version that switches to weak refs).
    """

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _handle(source: object, event: LLMCallCompletedEvent) -> None:
            recorder = _current_recorder.get(None)
            if recorder is not None:
                recorder._on_llm_call_completed(event)


_trace_listener = _TraceEventListener()


class recording_stage:
    """Context manager: `with recording_stage(recorder, "product"): ...`
    starts/ends stage timing and makes `recorder` the active one for events
    fired on this thread while inside the block.
    """

    def __init__(self, recorder: RunRecorder, stage_name: str) -> None:
        self.recorder = recorder
        self.stage_name = stage_name
        self._recorder_token = None
        self._stage_token = None

    def __enter__(self) -> recording_stage:
        self.recorder.start_stage(self.stage_name)
        self._recorder_token = _current_recorder.set(self.recorder)
        self._stage_token = _current_stage.set(self.stage_name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.recorder.end_stage(self.stage_name)
        if self._stage_token is not None:
            _current_stage.reset(self._stage_token)
        if self._recorder_token is not None:
            _current_recorder.reset(self._recorder_token)
