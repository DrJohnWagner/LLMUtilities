from __future__ import annotations

import pytest

from LLMUtilities.exceptions import ConfigurationError, MissingDependencyError
from LLMUtilities.providers.registry import get_provider, list_providers


def test_list_providers_returns_known_names():
    assert list_providers() == [
        "anthropic",
        "deepseek",
        "google",
        "moonshot",
        "openai",
    ]


def test_get_provider_lazily_constructs_each_known_provider():
    for name in list_providers():
        provider = get_provider(name)
        assert provider.name == name


def test_get_provider_unknown_name_raises_configuration_error():
    with pytest.raises(ConfigurationError):
        get_provider("not-a-real-provider")


def test_get_provider_is_case_and_whitespace_insensitive():
    assert get_provider(" OpenAI ").name == "openai"


def test_get_provider_raises_missing_dependency_when_sdk_absent(monkeypatch):
    import LLMUtilities.transports.openai_responses as transport

    monkeypatch.setattr(transport, "_OpenAIClient", None)

    with pytest.raises(MissingDependencyError):
        get_provider("openai")


def test_get_provider_does_not_cache_instances():
    first = get_provider("openai")
    second = get_provider("openai")
    assert first is not second
