from __future__ import annotations

import types

import pytest

import LLMUtilities.transports.google_generate_content as google_transport
from LLMUtilities.exceptions import AuthenticationError, RateLimitError, RequestError, ProviderError
from LLMUtilities.providers.registry import get_provider
from LLMUtilities.types import ChatRequest, Message


class _FakePart:
    def __init__(self, text):
        self.text = text


class _FakeContent:
    def __init__(self, text):
        self.parts = [_FakePart(text)]


class _FakeCandidate:
    def __init__(self, text, finish_reason="STOP"):
        self.content = _FakeContent(text)
        self.finish_reason = finish_reason


class _FakeUsageMetadata:
    def __init__(self, prompt=10, candidates=5):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = prompt + candidates


class _FakeResponse:
    def __init__(self, text, *, model_version="gemini-3.5-flash-002"):
        self.candidates = [_FakeCandidate(text)]
        self.usage_metadata = _FakeUsageMetadata()
        self.model_version = model_version


def _install_fake_client(monkeypatch, generate_fn):
    class _FakeClient:
        def __init__(self, **kwargs):
            self.models = types.SimpleNamespace(generate_content=generate_fn)

    monkeypatch.setattr(google_transport, "_genai", types.SimpleNamespace(Client=_FakeClient))


def _make_request(text="Hello", model=None):
    return ChatRequest(messages=[Message(role="user", content=text)], model=model)


def test_chat_extracts_text_usage_and_resolved_model(monkeypatch):
    _install_fake_client(monkeypatch, lambda **kwargs: _FakeResponse("Hi there!"))

    provider = get_provider("google")
    response = provider.chat(_make_request(model="gemini-3.5-flash"))

    assert response.text == "Hi there!"
    assert response.provider == "google"
    assert response.requested_model == "gemini-3.5-flash"
    assert response.resolved_model == "gemini-3.5-flash-002"
    assert response.usage.total_input_tokens == 10
    assert response.usage.total_output_tokens == 5


@pytest.mark.parametrize(
    "code,expected_exception",
    [
        (401, AuthenticationError),
        (429, RateLimitError),
        (500, RequestError),
        (400, ProviderError),
    ],
)
def test_chat_translates_sdk_exceptions(monkeypatch, code, expected_exception):
    from google.genai import errors as genai_errors

    def raise_error(**kwargs):
        raise genai_errors.APIError(code, {"error": {"message": "boom"}})

    class _FakeClient:
        def __init__(self, **kwargs):
            self.models = types.SimpleNamespace(generate_content=raise_error)

    monkeypatch.setattr(
        google_transport, "_genai", types.SimpleNamespace(Client=_FakeClient)
    )

    provider = get_provider("google")
    with pytest.raises(expected_exception):
        provider.chat(_make_request(model="gemini-3.5-flash"))
