from __future__ import annotations

import types

import pytest

import LLMUtilities.transports.openai_chat_completions as chat_completions_transport
from LLMUtilities.providers.registry import get_provider
from LLMUtilities.types import ChatRequest, Message


class _FakeMessage:
    def __init__(self, text):
        self.content = text


class _FakeChoice:
    def __init__(self, text, finish_reason="stop"):
        self.message = _FakeMessage(text)
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5, cached_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.prompt_tokens_details = types.SimpleNamespace(cached_tokens=cached_tokens)


class _FakeResponse:
    def __init__(self, text, *, model):
        self.choices = [_FakeChoice(text)]
        self.usage = _FakeUsage()
        self.model = model


def _install_fake_client(monkeypatch, create_fn):
    class _FakeClient:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create_fn)
            )

    monkeypatch.setattr(chat_completions_transport, "_OpenAIClient", _FakeClient)


def _make_request(text="Hello", model=None):
    return ChatRequest(messages=[Message(role="user", content=text)], model=model)


@pytest.mark.parametrize(
    "provider_name,model",
    [("moonshot", "kimi-k2.6"), ("deepseek", "deepseek-v4-pro")],
)
def test_chat_extracts_text_usage_and_resolved_model(monkeypatch, provider_name, model):
    _install_fake_client(
        monkeypatch, lambda **kwargs: _FakeResponse("Hi there!", model=f"{model}-0101")
    )

    provider = get_provider(provider_name)
    response = provider.chat(_make_request(model=model))

    assert response.text == "Hi there!"
    assert response.provider == provider_name
    assert response.requested_model == model
    assert response.resolved_model == f"{model}-0101"
    assert response.usage.total_input_tokens == 10
    assert response.usage.total_output_tokens == 5


def test_cost_summary_uses_cached_read_rate(monkeypatch):
    _install_fake_client(monkeypatch, lambda **kwargs: _cached_response())

    provider = get_provider("moonshot")
    response = provider.chat(_make_request(model="kimi-k2.6"))
    summary = provider.get_cost_summary(response)

    assert summary.provider == "moonshot"
    assert summary.total_cost > 0


def _cached_response():
    response = _FakeResponse("Hi", model="kimi-k2.6")
    response.usage = _FakeUsage(prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=500_000)
    return response
