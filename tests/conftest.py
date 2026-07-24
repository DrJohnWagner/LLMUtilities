from __future__ import annotations

import pytest

from LLMUtilities.config import reload_settings


@pytest.fixture(autouse=True)
def fake_api_keys(monkeypatch):
    """
    Every test runs with fake credentials for every provider so provider
    construction and capability calls never fail on a missing API key -
    tests mock the SDK layer instead of hitting the network.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-moonshot-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    reload_settings()
    yield
    reload_settings()
