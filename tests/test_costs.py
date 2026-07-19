"""Tests for text-model pricing and cost calculation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import LLMUtilities.costs as costs
from LLMUtilities.costs import (
    PRICING,
    PRICING_CATALOGUE,
    CostEstimate,
    Pricing,
    cost_for_model,
    cost_for_response,
    cost_from_tokens,
    cost_from_usage,
    estimate_cost,
    format_cost,
    get_pricing,
    print_cost_breakdown,
    print_cost_summary,
)
from LLMUtilities.types import ChatUsage


# ---------------------------------------------------------------------------
# Global pricing-state isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_pricing_state() -> Iterator[None]:
    """
    Restore mutable pricing registries after every test.

    Custom pricing and aliases are process-global, so registration tests must
    not affect later tests.
    """
    original_overrides = dict(costs.PRICING_OVERRIDES)
    original_aliases = dict(costs.PRICING_ALIASES)

    try:
        yield
    finally:
        costs.PRICING_OVERRIDES.clear()
        costs.PRICING_OVERRIDES.update(original_overrides)

        costs.PRICING_ALIASES.clear()
        costs.PRICING_ALIASES.update(original_aliases)

        costs._refresh_pricing_index()


# ---------------------------------------------------------------------------
# Pricing catalogue
# ---------------------------------------------------------------------------


class TestPricingCatalogue:
    def test_catalogue_metadata(self) -> None:
        assert PRICING_CATALOGUE.schema_version == 1
        assert PRICING_CATALOGUE.version == "2026-07-18"
        assert PRICING_CATALOGUE.generated_at

    def test_catalogue_contains_expected_providers(self) -> None:
        providers = {
            entry.provider
            for entry in PRICING_CATALOGUE.entries
        }

        assert {
            "openai",
            "anthropic",
            "google",
            "moonshot",
            "deepseek",
        }.issubset(providers)

    def test_openai_pricing_is_loaded(self) -> None:
        pricing = get_pricing("gpt-5.4")

        assert pricing.provider == "openai"
        assert pricing.canonical_model_id == "gpt-5.4"
        assert pricing.input_rate == pytest.approx(2.5)
        assert pricing.output_rate == pytest.approx(15.0)
        assert pricing.cached_read_rate == pytest.approx(0.25)
        assert pricing.cached_write_rate == pytest.approx(2.5)
        assert pricing.batch_input_rate == pytest.approx(1.25)
        assert pricing.batch_output_rate == pytest.approx(7.5)

    def test_anthropic_pricing_is_loaded(self) -> None:
        pricing = get_pricing("claude-sonnet-4.6")

        assert pricing.provider == "anthropic"
        assert pricing.input_rate == pytest.approx(3.0)
        assert pricing.output_rate == pytest.approx(15.0)
        assert pricing.cached_read_rate == pytest.approx(0.3)
        assert pricing.cached_write_rate == pytest.approx(3.75)

    def test_google_long_context_pricing_is_loaded(self) -> None:
        pricing = get_pricing("gemini-2.5-pro")

        assert pricing.provider == "google"
        assert pricing.long_context_threshold == 200_000
        assert pricing.long_context_input_rate == pytest.approx(2.5)
        assert pricing.long_context_output_rate == pytest.approx(15.0)

    def test_moonshot_pricing_is_loaded(self) -> None:
        assert "kimi-k3" in PRICING

        pricing = get_pricing("kimi-k2.7-code-highspeed")

        assert pricing.provider == "moonshot"
        assert pricing.output_rate == pytest.approx(8.0)

    def test_deepseek_pricing_is_loaded(self) -> None:
        assert "deepseek-v4-pro" in PRICING
        assert get_pricing("deepseek-v4-flash").provider == "deepseek"

    @pytest.mark.parametrize(
        (
            "alias",
            "canonical_model",
        ),
        [
            (
                "gpt-5",
                "gpt-5.4",
            ),
            (
                "gpt-5-mini",
                "gpt-5.4-mini",
            ),
            (
                "gpt-5-nano",
                "gpt-5.4-nano",
            ),
            (
                "claude-sonnet-4-0",
                "claude-sonnet-4.6",
            ),
            (
                "claude-sonnet-4",
                "claude-sonnet-4.6",
            ),
            (
                "claude-haiku-4",
                "claude-haiku-4.5",
            ),
            (
                "claude-opus-4",
                "claude-opus-4.6",
            ),
            (
                "gemini-pro",
                "gemini-2.5-pro",
            ),
            (
                "gemini-flash",
                "gemini-2.5-flash",
            ),
        ],
    )
    def test_default_aliases_resolve_to_canonical_models(
        self,
        alias: str,
        canonical_model: str,
    ) -> None:
        pricing = get_pricing(alias)

        assert pricing.canonical_model_id == canonical_model
        assert pricing is get_pricing(canonical_model)

    def test_unknown_model_raises_key_error(self) -> None:
        with pytest.raises(
            KeyError,
            match="No pricing registered",
        ):
            get_pricing("no-such-model-xyz")

    def test_register_pricing_from_mapping(self) -> None:
        register_pricing(
            "custom-model",
            {
                "provider": "custom",
                "canonical_model_id": "custom-model",
                "input_rate": 1,
                "output_rate": 2,
                "cached_read_rate": 0.25,
                "cached_write_rate": 1.5,
                "batch_input_rate": 0.5,
                "batch_output_rate": 1,
            },
        )

        pricing = get_pricing("custom-model")

        assert isinstance(pricing, Pricing)
        assert pricing.provider == "custom"
        assert pricing.input_rate == pytest.approx(1.0)
        assert pricing.output_rate == pytest.approx(2.0)
        assert pricing.cached_read_rate == pytest.approx(0.25)
        assert pricing.cached_write_rate == pytest.approx(1.5)
        assert pricing.batch_input_rate == pytest.approx(0.5)
        assert pricing.batch_output_rate == pytest.approx(1.0)

    def test_register_pricing_object(self) -> None:
        pricing = Pricing(
            provider="custom",
            canonical_model_id="object-model",
            input_rate=1.0,
            output_rate=2.0,
        )

        register_pricing(
            "object-model",
            pricing,
        )

        assert get_pricing("object-model") is pricing
        assert PRICING["object-model"] is pricing

    def test_registered_pricing_overrides_catalogue_entry(self) -> None:
        register_pricing(
            "gpt-5.4",
            Pricing(
                provider="test",
                canonical_model_id="gpt-5.4",
                input_rate=99.0,
                output_rate=199.0,
            ),
        )

        pricing = get_pricing("gpt-5.4")

        assert pricing.provider == "test"
        assert pricing.input_rate == pytest.approx(99.0)
        assert pricing.output_rate == pytest.approx(199.0)

    def test_register_pricing_alias(self) -> None:
        register_pricing(
            "custom-model",
            {
                "provider": "custom",
                "canonical_model_id": "custom-model",
                "input_rate": 1.0,
                "output_rate": 2.0,
            },
        )
        register_pricing_alias(
            "custom-alias",
            "custom-model",
        )

        assert get_pricing("custom-alias") is get_pricing("custom-model")
        assert PRICING["custom-alias"] is get_pricing("custom-model")


# ---------------------------------------------------------------------------
# Direct token costing
# ---------------------------------------------------------------------------


class TestCostFromTokens:
    def test_basic_cost_calculation(self) -> None:
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
        )

        estimate = cost_from_tokens(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            pricing=pricing,
        )

        assert isinstance(estimate, CostEstimate)
        assert estimate.input_tokens == 1_000_000
        assert estimate.output_tokens == 1_000_000

        assert estimate.input_cost_usd == pytest.approx(1.0)
        assert estimate.output_cost_usd == pytest.approx(2.0)
        assert estimate.total_cost_usd == pytest.approx(3.0)

    def test_all_token_categories_are_costed(self) -> None:
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
            cached_read_rate=0.25,
            cached_write_rate=1.5,
        )

        estimate = cost_from_tokens(
            input_tokens=1_000_000,
            output_tokens=500_000,
            cached_input_tokens=250_000,
            cache_creation_input_tokens=100_000,
            cache_write_tokens=200_000,
            pricing=pricing,
        )

        assert estimate.input_cost_usd == pytest.approx(1.0)
        assert estimate.output_cost_usd == pytest.approx(1.0)
        assert estimate.cached_input_cost_usd == pytest.approx(0.0625)
        assert estimate.cache_creation_input_cost_usd == pytest.approx(
            0.1
        )
        assert estimate.cache_write_input_cost_usd == pytest.approx(
            0.3
        )
        assert estimate.total_cost_usd == pytest.approx(2.4625)

    def test_zero_tokens_cost_nothing(self) -> None:
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
        )

        estimate = cost_from_tokens(pricing=pricing)

        assert estimate.input_cost_usd == 0.0
        assert estimate.output_cost_usd == 0.0
        assert estimate.cached_input_cost_usd == 0.0
        assert estimate.cache_creation_input_cost_usd == 0.0
        assert estimate.cache_write_input_cost_usd == 0.0
        assert estimate.total_cost_usd == 0.0

    @pytest.mark.parametrize(
        "field",
        [
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "cache_write_tokens",
        ],
    )
    def test_negative_token_counts_are_rejected(
        self,
        field: str,
    ) -> None:
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
        )
        kwargs = {
            "pricing": pricing,
            field: -1,
        }

        with pytest.raises(
            ValueError,
            match=field,
        ):
            cost_from_tokens(**kwargs)

    def test_missing_cached_rates_fall_back_to_input_rate(
        self,
    ) -> None:
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=3.0,
            output_rate=4.0,
        )

        estimate = cost_from_tokens(
            cached_input_tokens=1_000_000,
            cache_write_tokens=1_000_000,
            pricing=pricing,
        )

        assert estimate.cached_input_cost_usd == pytest.approx(3.0)
        assert estimate.cache_write_input_cost_usd == pytest.approx(
            3.0
        )
        assert estimate.total_cost_usd == pytest.approx(6.0)

    def test_batch_mode_uses_batch_rates(self) -> None:
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=2.0,
            output_rate=8.0,
            batch_input_rate=1.0,
            batch_output_rate=4.0,
        )

        estimate = cost_from_tokens(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            pricing=pricing,
            pricing_mode="batch",
        )

        assert estimate.input_cost_usd == pytest.approx(1.0)
        assert estimate.output_cost_usd == pytest.approx(4.0)
        assert estimate.total_cost_usd == pytest.approx(5.0)

    def test_batch_mode_falls_back_to_standard_rates(self) -> None:
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=2.0,
            output_rate=8.0,
        )

        estimate = cost_from_tokens(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            pricing=pricing,
            pricing_mode="batch",
        )

        assert estimate.input_cost_usd == pytest.approx(2.0)
        assert estimate.output_cost_usd == pytest.approx(8.0)

    def test_long_context_rates_apply_at_threshold(self) -> None:
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
            long_context_threshold=100,
            long_context_input_rate=3.0,
            long_context_output_rate=4.0,
        )

        estimate = cost_from_tokens(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            pricing=pricing,
            context_tokens=100,
        )

        assert estimate.input_cost_usd == pytest.approx(3.0)
        assert estimate.output_cost_usd == pytest.approx(4.0)
        assert estimate.total_cost_usd == pytest.approx(7.0)

    def test_long_context_rates_do_not_apply_below_threshold(
        self,
    ) -> None:
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
            long_context_threshold=100,
            long_context_input_rate=3.0,
            long_context_output_rate=4.0,
        )

        estimate = cost_from_tokens(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            pricing=pricing,
            context_tokens=99,
        )

        assert estimate.input_cost_usd == pytest.approx(1.0)
        assert estimate.output_cost_usd == pytest.approx(2.0)

    def test_long_context_rates_take_precedence_over_batch_rates(
        self,
    ) -> None:
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
            batch_input_rate=0.5,
            batch_output_rate=1.0,
            long_context_threshold=100,
            long_context_input_rate=3.0,
            long_context_output_rate=4.0,
        )

        estimate = cost_from_tokens(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            pricing=pricing,
            pricing_mode="batch",
            context_tokens=100,
        )

        assert estimate.input_cost_usd == pytest.approx(3.0)
        assert estimate.output_cost_usd == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# ChatUsage costing
# ---------------------------------------------------------------------------


class TestCostFromUsage:
    def test_usage_fields_are_copied_to_estimate(self) -> None:
        usage = ChatUsage(
            input_tokens=100,
            output_tokens=50,
            cached_input_tokens=25,
            cache_creation_input_tokens=10,
        )
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
            cached_read_rate=0.25,
        )

        estimate = cost_from_usage(
            usage=usage,
            pricing=pricing,
        )

        assert estimate.input_tokens == 100
        assert estimate.output_tokens == 50
        assert estimate.cached_input_tokens == 25
        assert estimate.cache_creation_input_tokens == 10

    def test_usage_costs_use_normalised_cache_fields(self) -> None:
        usage = ChatUsage(
            input_tokens=1_000_000,
            output_tokens=500_000,
            cached_input_tokens=250_000,
            cache_creation_input_tokens=100_000,
        )
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
            cached_read_rate=0.25,
        )

        estimate = cost_from_usage(
            usage=usage,
            pricing=pricing,
        )

        assert estimate.input_cost_usd == pytest.approx(1.0)
        assert estimate.output_cost_usd == pytest.approx(1.0)
        assert estimate.cached_input_cost_usd == pytest.approx(0.0625)
        assert estimate.cache_creation_input_cost_usd == pytest.approx(
            0.1
        )
        assert estimate.total_cost_usd == pytest.approx(2.1625)

    def test_none_usage_fields_are_treated_as_zero(self) -> None:
        usage = ChatUsage()
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
        )

        estimate = cost_from_usage(
            usage=usage,
            pricing=pricing,
        )

        assert estimate.input_tokens == 0
        assert estimate.output_tokens == 0
        assert estimate.total_cost_usd == 0.0

    def test_explicit_cache_overrides_take_precedence(self) -> None:
        usage = ChatUsage(
            cached_input_tokens=25,
            cache_creation_input_tokens=10,
        )
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
            cached_read_rate=0.25,
        )

        estimate = cost_from_usage(
            usage=usage,
            pricing=pricing,
            cached_input_tokens=3,
            cache_creation_input_tokens=4,
            cache_write_tokens=5,
        )

        assert estimate.cached_input_tokens == 3
        assert estimate.cache_creation_input_tokens == 4
        assert estimate.cache_write_tokens == 5

    def test_explicit_zero_overrides_usage_values(self) -> None:
        usage = ChatUsage(
            cached_input_tokens=25,
            cache_creation_input_tokens=10,
        )
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.0,
            output_rate=2.0,
            cached_read_rate=0.25,
        )

        estimate = cost_from_usage(
            usage=usage,
            pricing=pricing,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
        )

        assert estimate.cached_input_tokens == 0
        assert estimate.cache_creation_input_tokens == 0


# ---------------------------------------------------------------------------
# Public model-based helpers
# ---------------------------------------------------------------------------


class TestModelCostHelpers:
    def test_cost_for_model_uses_registered_pricing(self) -> None:
        estimate = cost_for_model(
            model="gpt-5.4",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

        assert estimate.input_cost_usd == pytest.approx(2.5)
        assert estimate.output_cost_usd == pytest.approx(15.0)
        assert estimate.total_cost_usd == pytest.approx(17.5)

    def test_cost_for_model_accepts_alias(self) -> None:
        alias_estimate = cost_for_model(
            model="gpt-5-mini",
            input_tokens=1_000_000,
        )
        canonical_estimate = cost_for_model(
            model="gpt-5.4-mini",
            input_tokens=1_000_000,
        )

        assert alias_estimate == canonical_estimate

    def test_cost_for_response_uses_model_pricing(self) -> None:
        estimate = cost_for_response(
            model="gemini-2.5-flash",
            usage=ChatUsage(
                input_tokens=1_000_000,
                output_tokens=1_000_000,
            ),
        )

        assert estimate.input_cost_usd == pytest.approx(0.3)
        assert estimate.output_cost_usd == pytest.approx(2.5)
        assert estimate.total_cost_usd == pytest.approx(2.8)

    def test_estimate_cost_is_public_alias(self) -> None:
        usage = ChatUsage(
            input_tokens=500,
            output_tokens=250,
        )

        estimate = estimate_cost(
            model="gemini-2.5-flash",
            usage=usage,
        )

        assert isinstance(estimate, CostEstimate)
        assert estimate.input_tokens == 500
        assert estimate.output_tokens == 250

        assert estimate == cost_for_response(
            model="gemini-2.5-flash",
            usage=usage,
        )

    def test_estimate_cost_batch_mode(self) -> None:
        estimate = estimate_cost(
            model="gpt-5.4",
            usage=ChatUsage(
                input_tokens=1_000_000,
                output_tokens=1_000_000,
            ),
            pricing_mode="batch",
        )

        assert estimate.input_cost_usd == pytest.approx(1.25)
        assert estimate.output_cost_usd == pytest.approx(7.5)
        assert estimate.total_cost_usd == pytest.approx(8.75)

    def test_gemini_long_context_pricing(self) -> None:
        estimate = estimate_cost(
            model="gemini-2.5-pro",
            usage=ChatUsage(
                input_tokens=1_000_000,
                output_tokens=1_000_000,
            ),
            context_tokens=200_000,
        )

        assert estimate.input_cost_usd == pytest.approx(2.5)
        assert estimate.output_cost_usd == pytest.approx(15.0)
        assert estimate.total_cost_usd == pytest.approx(17.5)

    def test_gemini_batch_pricing_below_long_context_threshold(
        self,
    ) -> None:
        estimate = estimate_cost(
            model="gemini-2.5-pro",
            usage=ChatUsage(
                input_tokens=1_000_000,
                output_tokens=1_000_000,
            ),
            pricing_mode="batch",
            context_tokens=199_999,
        )

        assert estimate.input_cost_usd == pytest.approx(0.625)
        assert estimate.output_cost_usd == pytest.approx(5.0)

    def test_long_context_pricing_takes_precedence_over_batch(
        self,
    ) -> None:
        estimate = estimate_cost(
            model="gemini-2.5-pro",
            usage=ChatUsage(
                input_tokens=1_000_000,
                output_tokens=1_000_000,
            ),
            pricing_mode="batch",
            context_tokens=200_000,
        )

        assert estimate.input_cost_usd == pytest.approx(2.5)
        assert estimate.output_cost_usd == pytest.approx(15.0)

    def test_zero_usage_costs_nothing(self) -> None:
        estimate = estimate_cost(
            model="gemini-2.5-flash",
            usage=ChatUsage(
                input_tokens=0,
                output_tokens=0,
            ),
        )

        assert estimate.total_cost_usd == 0.0

    def test_unknown_model_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            estimate_cost(
                model="no-such-model",
                usage=ChatUsage(),
            )


# ---------------------------------------------------------------------------
# Formatting and logging
# ---------------------------------------------------------------------------


class TestCostFormatting:
    def test_format_cost_uses_eight_decimal_places_by_default(
        self,
    ) -> None:
        assert format_cost(1.25) == "$1.25000000"

    def test_format_cost_accepts_custom_precision(self) -> None:
        assert format_cost(1.23456789, decimals=4) == "$1.2346"

    def test_print_cost_breakdown_logs_all_fields(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        estimate = CostEstimate(
            input_tokens=100,
            output_tokens=50,
            cached_input_tokens=25,
            cache_creation_input_tokens=10,
            cache_write_tokens=5,
            input_cost_usd=0.4,
            output_cost_usd=0.2,
            cached_input_cost_usd=0.1,
            cache_creation_input_cost_usd=0.05,
            cache_write_input_cost_usd=0.025,
            total_cost_usd=0.775,
        )

        caplog.set_level(
            "INFO",
            logger="LLMUtilities.costs",
        )

        print_cost_breakdown(
            estimate=estimate,
            model="test-model",
        )

        log_text = "\n".join(
            record.getMessage()
            for record in caplog.records
        )

        assert "Model: test-model" in log_text
        assert "Input tokens: 100" in log_text
        assert "Output tokens: 50" in log_text
        assert "Cached input tokens: 25" in log_text
        assert "Cache creation input tokens: 10" in log_text
        assert "Cache write tokens: 5" in log_text
        assert "Total cost: $0.77500000" in log_text

    def test_print_cost_summary_uses_normalised_fields(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        estimate = CostEstimate(
            input_tokens=100,
            output_tokens=50,
            cached_input_tokens=25,
            cache_creation_input_tokens=10,
            cached_input_cost_usd=0.2,
            cache_creation_input_cost_usd=0.1,
            total_cost_usd=0.9,
        )

        caplog.set_level(
            "INFO",
            logger="LLMUtilities.costs",
        )

        print_cost_summary(
            estimate=estimate,
            model="claude-sonnet-4.6",
        )

        log_text = "\n".join(
            record.getMessage()
            for record in caplog.records
        )

        assert "Model: claude-sonnet-4.6" in log_text
        assert "Cached input tokens: 25" in log_text
        assert "Cache creation input tokens: 10" in log_text
        assert "Cached input cost: $0.20000000" in log_text
        assert "Cache creation input cost: $0.10000000" in log_text
        assert "Total cost: $0.90000000" in log_text
