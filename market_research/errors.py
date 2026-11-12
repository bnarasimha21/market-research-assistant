"""Typed errors for the market research pipeline.

Replaces the bare `except Exception as e: print(e)` pattern in the original
app.py with errors that name what actually went wrong and, for stage
failures, which stage failed, so the UI can show something useful instead of
a raw exception repr.
"""

from __future__ import annotations


class MarketResearchError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(MarketResearchError):
    """Raised for bad input or missing configuration, before any LLM call.

    Examples: an empty competitor name, duplicate product/competitor names,
    too many competitors, a selected provider missing its required
    environment variable.
    """


class StageFailed(MarketResearchError):
    """Raised when one stage of the pipeline (product, competitor, comparison)
    fails after any internal retries. Carries the stage name so the UI can
    say "the competitor stage failed" instead of printing a stack trace.
    """

    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(f"{stage} stage failed: {reason}")
