from __future__ import annotations

import types

import pytest

import LLMUtilities.transports.openai_responses as openai_responses_transport
from LLMUtilities.exceptions import AuthenticationError, RateLimitError, RequestError, ProviderError
from LLMUtilities.providers.registry import get_provider
from LLMUtilities.types import ChatRequest, Message
from tests.openai_sdk_fakes import (
    make_auth_error,
    make_connection_error,
    make_rate_limit_error,
    make_status_error,
)


class _FakeOutputPart:
    def __init__(self, text):
        self.type = "output_text"
        self.text = text


class _FakeOutputItem:
    def __init__(self, text):
        self.content = [_FakeOutputPart(text)]


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens, cached_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.input_tokens_details = types.SimpleNamespace(cached_tokens=cached_tokens)


class _FakeResponse:
    def __init__(self, text, *, model="gpt-5.6-terra-2026-07-18"):
        self.output = [_FakeOutputItem(text)]
        self.usage = _FakeUsage(input_tokens=10, output_tokens=5, cached_tokens=2)
        self.model = model
        self.status = "completed"


def _install_fake_client(monkeypatch, create_fn):
    class _FakeClient:
        def __init__(self, **kwargs):
            self.responses = types.SimpleNamespace(create=create_fn)

    monkeypatch.setattr(openai_responses_transport, "_OpenAIClient", _FakeClient)


def _make_request(text="Hello", model=None):
    return ChatRequest(messages=[Message(role="user", content=text)], model=model)


def test_chat_extracts_text_usage_and_resolved_model(monkeypatch):
    _install_fake_client(monkeypatch, lambda **kwargs: _FakeResponse("Hi there!"))

    provider = get_provider("openai")
    response = provider.chat(_make_request(model="gpt-5.6-terra"))

    assert response.text == "Hi there!"
    assert response.provider == "openai"
    assert response.requested_model == "gpt-5.6-terra"
    assert response.resolved_model == "gpt-5.6-terra-2026-07-18"
    assert response.usage.total_input_tokens == 10
    assert response.usage.total_output_tokens == 5
    assert response.usage.total_tokens == 15


def test_chat_detailed_usage_reports_cached_tokens(monkeypatch):
    _install_fake_client(monkeypatch, lambda **kwargs: _FakeResponse("Hi"))

    provider = get_provider("openai")
    response = provider.chat(_make_request(model="gpt-5.6-terra"))
    details = provider.get_detailed_usage(response)

    assert details.cached_input_tokens == 2


@pytest.mark.parametrize(
    "sdk_error_factory,expected_exception",
    [
        (make_auth_error, AuthenticationError),
        (make_rate_limit_error, RateLimitError),
        (make_connection_error, RequestError),
        (make_status_error, ProviderError),
    ],
)
def test_chat_translates_sdk_exceptions(monkeypatch, sdk_error_factory, expected_exception):
    def raise_error(**kwargs):
        raise sdk_error_factory()

    _install_fake_client(monkeypatch, raise_error)

    provider = get_provider("openai")
    with pytest.raises(expected_exception):
        provider.chat(_make_request(model="gpt-5.6-terra"))
