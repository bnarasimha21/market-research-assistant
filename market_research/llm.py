"""LLM provider configuration through CrewAI's provider layer.

Replaces the old pattern of each stage calling a separately hosted
DigitalOcean GenAI agent directly through the OpenAI client with one fixed
sentence and no real data. Stages now use ordinary crewai.Agent objects
whose LLM is built here, and the actual product/competitor data is put in
the task prompt (see agents.py), not sent to a black-box hosted agent.
"""

from __future__ import annotations

import os

from crewai import LLM

from market_research.errors import ConfigError

DEFAULT_DO_BASE_URL = "https://inference.do-ai.run/v1"


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def get_llm(role: str = "default", *, base_url_override: str | None = None) -> LLM:
    """Return a crewai.LLM configured from LLM_PROVIDER.

    role is accepted so a future provider could pick a different model per
    stage; today every role uses the same configured model.

    LLM_PROVIDER=do_serverless: DigitalOcean's OpenAI-compatible serverless
        inference endpoint. Needs DO_MODEL_ACCESS_KEY and DO_MODEL_ID. Model
        ids are not hardcoded here: run `GET /v1/models` against the base
        URL with the access key at setup time to find the id to put in
        DO_MODEL_ID (see README).
    LLM_PROVIDER=openai: CrewAI's default behaviour (OPENAI_API_KEY, default
        model unless OPENAI_MODEL_NAME is set).
    LLM_PROVIDER=fake: used by tests; returns a crewai LLM is never actually
        built for this path, pipeline.py / agents.py special-case "fake" and
        inject a FakeLLM test double instead, so this function raises if it
        is reached with provider=fake, which would mean a caller forgot to
        special-case it.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "fake":
        raise ConfigError(
            "get_llm() was called with LLM_PROVIDER=fake; callers must use a "
            "FakeLLM test double instead of building a real LLM for this role"
        )

    if provider == "do_serverless":
        access_key = _env("DO_MODEL_ACCESS_KEY")
        model_id = _env("DO_MODEL_ID")
        if not access_key:
            raise ConfigError(
                "LLM_PROVIDER=do_serverless requires DO_MODEL_ACCESS_KEY to be set"
            )
        if not model_id:
            raise ConfigError(
                "LLM_PROVIDER=do_serverless requires DO_MODEL_ID to be set. "
                "List available model ids with: "
                "curl -s -H \"Authorization: Bearer $DO_MODEL_ACCESS_KEY\" "
                f"{base_url_override or DEFAULT_DO_BASE_URL}/models"
            )
        return LLM(
            model=f"openai/{model_id}",
            base_url=base_url_override or DEFAULT_DO_BASE_URL,
            api_key=access_key,
        )

    if provider == "openai":
        if not _env("OPENAI_API_KEY"):
            raise ConfigError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set")
        model = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")
        return LLM(model=model)

    raise ConfigError(
        f"Unknown LLM_PROVIDER={provider!r}; expected one of: do_serverless, openai, fake"
    )


def current_model_id() -> str | None:
    """Best-effort model id for the currently configured LLM_PROVIDER, used
    only to look up an estimated price in pricing.yaml for display (app.py).
    Returns None rather than raising when nothing is configured yet, since a
    missing price estimate should never block showing the rest of the UI.
    Mirrors get_llm()'s prefix-stripping (crewai.LLM strips a leading
    "openai/" from the model id it is given) without needing a real,
    network-capable LLM object to read it back from.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "do_serverless":
        return _env("DO_MODEL_ID")
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")
        return model.removeprefix("openai/")
    if provider == "fake":
        return "fake-llm"
    return None
