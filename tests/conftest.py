"""Shared pytest fixtures, including FakeLLM.

FakeLLM is a crewai.llms.base_llm.BaseLLM subclass (the documented
custom-LLM extension point) used to run the real Flow/Crew/Agent/Task
machinery in tests with zero network access.

What we confirmed against the pinned crewai==1.15.40 (read
crewai/llms/base_llm.py and crewai/utilities/converter.py in site-packages,
since nothing in CrewAI's docs pins this down for a from-scratch subclass):

1. BaseLLM.call() is abstract; nothing else is required to subclass it, but
   BaseLLM itself does NOT implement `supports_function_calling`. Pydantic's
   `BaseModel.__getattr__` turns a missing method into an AttributeError, and
   crewai.utilities.converter.Converter.to_pydantic() calls
   `self.llm.supports_function_calling()` unconditionally. Any custom LLM
   used anywhere output_pydantic/output_json conversion can run must define
   this method itself. We return False: none of this project's agents use
   LLM-invoked tools (issue 3 moved page fetching into plain code), so there
   is never a tools= argument to honor, and this pipeline never actually
   routes a task through output_pydantic/Converter (see market_research/
   agents.py and flow.py: the comparison task uses a `guardrail=` instead,
   and Task._export_output only runs the Converter when output_pydantic or
   output_json is set, which none of our tasks set), so this path is not
   exercised by the real pipeline, only kept correct defensively.
2. BaseLLM.call()'s documented contract: when `tools` is provided together
   with `available_functions`, the custom LLM implementation is responsible
   for the whole tool-calling loop itself (deciding to call a tool, invoking
   the callable, feeding the result back) -- crewai's agent executor does
   NOT do this for a custom BaseLLM the way it does for the built-in
   provider implementations. Since no agent in this pipeline is given tools
   (product/competitor/comparison stages all put their data directly in the
   prompt instead of exposing it as a callable tool), FakeLLM.call() raises
   NotImplementedError if it is ever invoked with tools, so a future change
   that adds a tool back would fail loudly in tests rather than silently
   returning nonsense.
3. LLMCallCompletedEvent is never emitted automatically: each concrete
   provider's call() calls the protected `_emit_call_started_event` /
   `_emit_call_completed_event` helpers itself. FakeLLM does the same, with
   a usage dict shaped like OpenAI's ({"prompt_tokens", "completion_tokens",
   "total_tokens"}), which is what market_research.trace's event listener
   reads. Both helpers read call_id via get_current_call_id(), which is a
   plain ContextVar with no default call scope; without wrapping each call
   in crewai.llms.base_llm.llm_call_context() first, CrewAI logs a "LLM
   event emitted outside call context" warning and the event bus reports a
   start/complete pairing mismatch, since the started and completed events
   each get their own random fallback call_id instead of sharing one.
"""

from __future__ import annotations

import os

# Must run before the first `import crewai` anywhere in the process: crewai's
# Telemetry() is constructed lazily via a PrivateAttr default_factory (see
# crewai/events/event_listener.py), but the env vars it reads
# (CREWAI_DISABLE_TELEMETRY, OTEL_SDK_DISABLED) are read at that construction
# time, not at call time. Setting them only inside a test or fixture body is
# too late once some other import has already pulled crewai in and built the
# listener. setdefault() so a developer's own env value (e.g. deliberately
# re-enabling telemetry to debug it) is never overridden.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import threading
from typing import Any

import pytest
from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, llm_call_context
from pydantic import Field, PrivateAttr


def _flatten_messages(messages: str | list[dict[str, Any]]) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(m.get("content", "")) for m in messages)


class FakeLLM(BaseLLM):
    """Scripted LLM test double keyed by agent role.

    replies: {role: str} for a fixed reply every time that role is called,
    or {role: [str, str, ...]} to return successive replies on successive
    calls for that role (e.g. a guardrail retry scenario), clamped to the
    last entry once exhausted.

    Every call is recorded (role, prompt text) in `recorded_prompts`, which
    tests inspect to assert what each stage actually saw.
    """

    replies: dict[str, Any] = Field(default_factory=dict)
    default_reply: str = '{"items": []}'

    _recorded: list[dict[str, str]] = PrivateAttr(default_factory=list)
    _counters: dict[str, int] = PrivateAttr(default_factory=dict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(self, replies: dict[str, Any] | None = None, **data: Any) -> None:
        data.setdefault("model", "fake-llm")
        super().__init__(replies=replies or {}, **data)

    @property
    def recorded_prompts(self) -> list[dict[str, str]]:
        return list(self._recorded)

    def call_count(self, role: str) -> int:
        return self._counters.get(role, 0)

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
    ) -> str:
        if tools:
            raise NotImplementedError(
                "FakeLLM was called with tools, but no agent in this pipeline is expected "
                "to use LLM-invoked tools (see the module docstring, point 2)."
            )

        role = getattr(from_agent, "role", None) or "default"
        prompt_text = _flatten_messages(messages)

        with self._lock:
            idx = self._counters.get(role, 0)
            self._counters[role] = idx + 1
            self._recorded.append({"role": role, "prompt": prompt_text})

        scripted = self.replies.get(role)
        if isinstance(scripted, str):
            reply = scripted
        elif isinstance(scripted, list) and scripted:
            reply = scripted[min(idx, len(scripted) - 1)]
        else:
            reply = self.default_reply

        prompt_tokens = max(len(prompt_text) // 4, 1)
        completion_tokens = max(len(reply) // 4, 1)
        # llm_call_context() establishes the call_id contextvar that
        # _emit_call_started_event/_emit_call_completed_event read via
        # get_current_call_id(); without it CrewAI logs a "generating
        # fallback call_id" warning and the event bus reports a pairing
        # mismatch, since each event gets its own random call_id.
        with llm_call_context():
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )
            self._emit_call_completed_event(
                response=reply,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=messages,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )
        return reply

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False


@pytest.fixture
def fake_llm_factory():
    """Factory so each test builds its own FakeLLM with its own scripted replies."""

    def _make(replies: dict[str, Any] | None = None, **kwargs: Any) -> FakeLLM:
        return FakeLLM(replies=replies, **kwargs)

    return _make
