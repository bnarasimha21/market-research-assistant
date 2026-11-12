"""Unit tests for market_research.llm.get_llm's provider switch.

No real network call happens anywhere here: do_serverless and openai both
just construct a crewai.LLM object (which doesn't call out until .call() is
invoked), and we never invoke .call() in this file.
"""

from __future__ import annotations

import pytest

from market_research.errors import ConfigError
from market_research.llm import DEFAULT_DO_BASE_URL, current_model_id, get_llm


def test_fake_provider_raises_configerror(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    with pytest.raises(ConfigError):
        get_llm("pipeline")


def test_unknown_provider_raises_configerror(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ConfigError):
        get_llm("pipeline")


def test_do_serverless_requires_access_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "do_serverless")
    monkeypatch.delenv("DO_MODEL_ACCESS_KEY", raising=False)
    monkeypatch.setenv("DO_MODEL_ID", "some-model")
    with pytest.raises(ConfigError):
        get_llm("pipeline")


def test_do_serverless_requires_model_id(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "do_serverless")
    monkeypatch.setenv("DO_MODEL_ACCESS_KEY", "key123")
    monkeypatch.delenv("DO_MODEL_ID", raising=False)
    with pytest.raises(ConfigError):
        get_llm("pipeline")


def test_do_serverless_builds_llm_with_configured_model_and_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "do_serverless")
    monkeypatch.setenv("DO_MODEL_ACCESS_KEY", "key123")
    monkeypatch.setenv("DO_MODEL_ID", "llama-3.1-70b")
    llm = get_llm("pipeline")
    # crewai.LLM strips the "openai/" provider prefix off .model internally.
    assert llm.model == "llama-3.1-70b"
    assert llm.base_url == DEFAULT_DO_BASE_URL


def test_do_serverless_honors_base_url_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "do_serverless")
    monkeypatch.setenv("DO_MODEL_ACCESS_KEY", "key123")
    monkeypatch.setenv("DO_MODEL_ID", "llama-3.1-70b")
    llm = get_llm("pipeline", base_url_override="http://127.0.0.1:9999/v1")
    assert llm.base_url == "http://127.0.0.1:9999/v1"


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        get_llm("pipeline")


def test_openai_provider_builds_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    llm = get_llm("pipeline")
    assert "gpt" in llm.model


def test_current_model_id_do_serverless(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "do_serverless")
    monkeypatch.setenv("DO_MODEL_ID", "llama-3.1-70b-instruct")
    assert current_model_id() == "llama-3.1-70b-instruct"


def test_current_model_id_openai_strips_prefix(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "openai/gpt-4o-mini")
    assert current_model_id() == "gpt-4o-mini"


def test_current_model_id_fake(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert current_model_id() == "fake-llm"


def test_current_model_id_unknown_provider_returns_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    assert current_model_id() is None
