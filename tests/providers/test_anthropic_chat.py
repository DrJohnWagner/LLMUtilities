from __future__ import annotations

import types

import httpx
import pytest

import LLMUtilities.transports.anthropic_messages as anthropic_transport
from LLMUtilities.exceptions import AuthenticationError, RateLimitError, RequestError, ProviderError
from LLMUtilities.providers.registry import get_provider
from LLMUtilities.types import ChatRequest, Message


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=5, cache_creation=0, cache_read=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation
        self.cache_read_input_tokens = cache_read


class _FakeResponse:
    def __init__(self, text, *, model="claude-sonnet-5-20260101"):
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage()
        self.model = model
        self.stop_reason = "end_turn"


def _install_fake_client(monkeypatch, create_fn):
    class _FakeClient:
        def __init__(self, **kwargs):
            self.messages = types.SimpleNamespace(create=create_fn)

    monkeypatch.setattr(anthropic_transport._anthropic, "Anthropic", _FakeClient)


def _make_request(text="Hello", model=None, system=None):
    messages = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=text))
    return ChatRequest(messages=messages, model=model)


def test_chat_extracts_text_usage_and_resolved_model(monkeypatch):
    _install_fake_client(monkeypatch, lambda **kwargs: _FakeResponse("Hi there!"))

    provider = get_provider("anthropic")
    response = provider.chat(_make_request(model="claude-sonnet-5", system="Be terse."))

    assert response.text == "Hi there!"
    assert response.provider == "anthropic"
    assert response.requested_model == "claude-sonnet-5"
    assert response.resolved_model == "claude-sonnet-5-20260101"
    assert response.stop_reason == "end_turn"
    assert response.usage.total_input_tokens == 10
    assert response.usage.total_output_tokens == 5


def test_detailed_cost_folds_cache_creation_into_cache_write_rate_not_input_rate(monkeypatch):
    def create_with_cache_write(**kwargs):
        response = _FakeResponse("hi")
        response.usage = _FakeUsage(input_tokens=0, output_tokens=0, cache_creation=1_000_000)
        return response

    _install_fake_client(monkeypatch, create_with_cache_write)

    provider = get_provider("anthropic")
    response = provider.chat(_make_request(model="claude-sonnet-5"))
    details = provider.get_detailed_cost(response)
    pricing = provider.get_pricing("claude-sonnet-5")

    # cache_creation_input_tokens must be billed at cached_write_rate, not input_rate
    assert details.cache_write_cost == pytest.approx(pricing.cached_write_rate)
    assert details.ordinary_input_cost == 0.0


@pytest.mark.parametrize(
    "exc_factory,expected_exception",
    [
        (lambda: _auth_error(), AuthenticationError),
        (lambda: _rate_limit_error(), RateLimitError),
        (lambda: _connection_error(), RequestError),
        (lambda: _status_error(), ProviderError),
    ],
)
def test_chat_translates_sdk_exceptions(monkeypatch, exc_factory, expected_exception):
    def raise_error(**kwargs):
        raise exc_factory()

    _install_fake_client(monkeypatch, raise_error)

    provider = get_provider("anthropic")
    with pytest.raises(expected_exception):
        provider.chat(_make_request(model="claude-sonnet-5"))


def _response(status_code):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status_code, request=request, json={"error": {"message": "boom"}})


def _auth_error():
    import anthropic

    return anthropic.AuthenticationError("bad key", response=_response(401), body=None)


def _rate_limit_error():
    import anthropic

    return anthropic.RateLimitError("too many requests", response=_response(429), body=None)


def _status_error():
    import anthropic

    return anthropic.APIStatusError("server error", response=_response(500), body=None)


def _connection_error():
    import anthropic

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(message="connection failed", request=request)
