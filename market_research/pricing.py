"""Turns a run's token counts into an approximate dollar cost for display.

Reads pricing.yaml (model id -> price per million input/output tokens, each
with a source URL and "as of" date). This is a snapshot, not a live lookup:
every number here is a labeled estimate, never presented as billed fact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PRICING_PATH = Path(__file__).resolve().parent.parent / "pricing.yaml"


def load_pricing(path: Path | str = DEFAULT_PRICING_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("models", {})


def estimate_cost_usd(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: dict[str, Any] | None = None,
) -> float | None:
    """Return an estimated USD cost for the given token counts, or None if
    the model id has no entry in pricing.yaml (never silently guesses).
    """
    table = pricing if pricing is not None else load_pricing()
    entry = table.get(model_id)
    if entry is None:
        return None
    input_rate = entry.get("input_per_million_tokens_usd", 0.0)
    output_rate = entry.get("output_per_million_tokens_usd", 0.0)
    return (prompt_tokens / 1_000_000) * input_rate + (completion_tokens / 1_000_000) * output_rate
