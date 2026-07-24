from __future__ import annotations

import pytest

from LLMUtilities.exceptions import PricingUnavailableError
from LLMUtilities.providers.registry import get_provider
from LLMUtilities.types import ChatResponse, CommonUsage


@pytest.mark.parametrize(
    "provider_name,model",
    [
        ("openai", "gpt-5.6-terra"),
        ("anthropic", "claude-sonnet-5"),
        ("google", "gemini-3.5-flash"),
        ("moonshot", "kimi-k2.6"),
        ("deepseek", "deepseek-v4-pro"),
    ],
)
def test_catalogue_loads_and_resolves_known_model(provider_name, model):
    provider = get_provider(provider_name)
    pricings = provider.list_pricings()
    assert pricings
    pricing = provider.get_pricing(model)
    assert pricing.canonical_model_id == model


def test_get_pricing_raises_pricing_unavailable_for_unknown_model():
    provider = get_provider("openai")
    with pytest.raises(PricingUnavailableError):
        provider.get_pricing("not-a-real-model")


def test_openai_cost_summary_plain_input_output():
    provider = get_provider("openai")
    response = ChatResponse(
        text="hi",
        provider="openai",
        requested_model="gpt-5.6-terra",
        resolved_model="gpt-5.6-terra",
        usage=CommonUsage(
            total_input_tokens=1_000_000, total_output_tokens=1_000_000, total_tokens=2_000_000
        ),
        raw=_FakeOpenAIResponse(input_tokens=1_000_000, output_tokens=1_000_000, cached=0),
    )
    summary = provider.get_cost_summary(response)
    # gpt-5.6-terra: input_rate=2.5, output_rate=15.0 (see providers/openai/pricing.json)
    assert summary.input_cost == pytest.approx(2.5)
    assert summary.output_cost == pytest.approx(15.0)
    assert summary.total_cost == pytest.approx(17.5)
    assert summary.provider == "openai"
    assert summary.requested_model == "gpt-5.6-terra"
    assert summary.resolved_model == "gpt-5.6-terra"


def test_openai_cost_summary_with_cached_read_tokens():
    provider = get_provider("openai")
    response = ChatResponse(
        text="hi",
        provider="openai",
        requested_model="gpt-5.6-terra",
        resolved_model="gpt-5.6-terra",
        usage=CommonUsage(total_input_tokens=1_000_000, total_output_tokens=0, total_tokens=1_000_000),
        raw=_FakeOpenAIResponse(input_tokens=1_000_000, output_tokens=0, cached=500_000),
    )
    summary = provider.get_cost_summary(response)
    # 500k tokens at ordinary input_rate=2.5, 500k at cached_read_rate=0.25
    expected = (500_000 / 1_000_000) * 2.5 + (500_000 / 1_000_000) * 0.25
    assert summary.input_cost == pytest.approx(expected)
    assert summary.output_cost == pytest.approx(0.0)


def test_anthropic_cost_summary_folds_cache_reads_and_writes_into_input_cost():
    provider = get_provider("anthropic")
    response = ChatResponse(
        text="hi",
        provider="anthropic",
        requested_model="claude-sonnet-5",
        resolved_model="claude-sonnet-5",
        usage=CommonUsage(total_input_tokens=100, total_output_tokens=100, total_tokens=200),
        raw=_FakeAnthropicResponse(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )
    pricing = provider.get_pricing("claude-sonnet-5")
    summary = provider.get_cost_summary(response)
    assert summary.input_cost == pytest.approx(pricing.input_rate)
    assert summary.other_cost == 0.0


class _FakeUsage:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeOpenAIResponse:
    def __init__(self, *, input_tokens, output_tokens, cached):
        details = _FakeUsage(cached_tokens=cached) if cached else None
        self.usage = _FakeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=details,
        )


class _FakeAnthropicResponse:
    def __init__(
        self, *, input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens
    ):
        self.usage = _FakeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
        self.stop_reason = "end_turn"
