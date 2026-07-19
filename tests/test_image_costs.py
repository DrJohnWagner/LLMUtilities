"""Tests for image pricing and image cost calculation."""

from __future__ import annotations

import types
from collections.abc import Iterator
from typing import Any

import pytest

import LLMUtilities.costs as costs
from LLMUtilities.costs import (
    IMAGE_PRICING,
    IMAGE_PRICING_CATALOGUE,
    ImageCostEstimate,
    ImagePricing,
    cost_for_image_response,
    cost_for_image_usage,
    estimate_image_cost,
    get_image_pricing,
    normalise_image_usage,
    validate_image_size_for_model,
)
from LLMUtilities.types import (
    ImageArtifact,
    ImageResponse,
    ImageUsage,
)


# ---------------------------------------------------------------------------
# Global pricing-state isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_image_pricing_state() -> Iterator[None]:
    """
    Restore mutable image-pricing registries after every test.

    Registration is intentionally process-global, so tests that add custom
    models or aliases must not leak those changes into later tests.
    """
    original_overrides = dict(costs.IMAGE_PRICING_OVERRIDES)
    original_aliases = dict(costs.IMAGE_PRICING_ALIASES)

    try:
        yield
    finally:
        costs.IMAGE_PRICING_OVERRIDES.clear()
        costs.IMAGE_PRICING_OVERRIDES.update(original_overrides)

        costs.IMAGE_PRICING_ALIASES.clear()
        costs.IMAGE_PRICING_ALIASES.update(original_aliases)

        costs._refresh_pricing_index()


# ---------------------------------------------------------------------------
# Pricing catalogue
# ---------------------------------------------------------------------------


class TestImagePricingCatalogue:
    def test_catalogue_metadata(self) -> None:
        assert IMAGE_PRICING_CATALOGUE.schema_version == 1
        assert IMAGE_PRICING_CATALOGUE.version == "2026-07-18"
        assert IMAGE_PRICING_CATALOGUE.generated_at

    def test_catalogue_contains_current_openai_models(self) -> None:
        canonical_models = {
            entry.canonical_model_id
            for entry in IMAGE_PRICING_CATALOGUE.entries
        }

        assert "gpt-image-1.5" in canonical_models
        assert "gpt-image-2" in canonical_models

    def test_gpt_image_15_rates(self) -> None:
        pricing = get_image_pricing("gpt-image-1.5")

        assert pricing.provider == "openai"
        assert pricing.canonical_model_id == "gpt-image-1.5"

        assert pricing.text_input_rate == pytest.approx(5.0)
        assert pricing.text_cached_read_rate == pytest.approx(1.25)
        assert pricing.image_input_rate == pytest.approx(8.0)
        assert pricing.image_output_rate == pytest.approx(32.0)
        assert pricing.image_cached_read_rate == pytest.approx(2.0)

        assert pricing.batch_text_input_rate == pytest.approx(2.5)
        assert pricing.batch_image_input_rate == pytest.approx(4.0)
        assert pricing.batch_image_output_rate == pytest.approx(16.0)

        assert pricing.partial_image_output_tokens == 100

    def test_gpt_image_2_rates(self) -> None:
        pricing = get_image_pricing("gpt-image-2")

        assert pricing.provider == "openai"
        assert pricing.canonical_model_id == "gpt-image-2"

        assert pricing.text_input_rate == pytest.approx(5.0)
        assert pricing.text_output_rate is None
        assert pricing.image_input_rate == pytest.approx(8.0)
        assert pricing.image_output_rate == pytest.approx(30.0)

        assert pricing.batch_text_input_rate == pytest.approx(2.5)
        assert pricing.batch_text_cached_read_rate == pytest.approx(
            0.625
        )
        assert pricing.batch_image_input_rate == pytest.approx(4.0)
        assert pricing.batch_image_output_rate == pytest.approx(15.0)

    def test_default_alias_resolves_to_canonical_model(self) -> None:
        pricing = get_image_pricing("gpt-image-1")

        assert pricing.canonical_model_id == "gpt-image-1.5"
        assert pricing is get_image_pricing("gpt-image-1.5")

    def test_default_image_alias_is_present_in_index(self) -> None:
        assert "gpt-image-1" in IMAGE_PRICING
        assert "openai-image-default" in IMAGE_PRICING

    def test_unknown_model_raises_key_error(self) -> None:
        with pytest.raises(
            KeyError,
            match="No image pricing registered",
        ):
            get_image_pricing("no-such-image-model")

    def test_register_image_pricing_from_mapping(self) -> None:
        register_image_pricing(
            "custom-image-model",
            {
                "provider": "custom",
                "canonical_model_id": "custom-image-model",
                "text_input_rate": 1,
                "image_output_rate": 10,
                "reference_image_output_costs": {
                    "low": {
                        "512x512": 0.01,
                    }
                },
                "partial_image_output_tokens": 50,
            },
        )

        pricing = get_image_pricing("custom-image-model")

        assert isinstance(pricing, ImagePricing)
        assert pricing.provider == "custom"
        assert pricing.text_input_rate == pytest.approx(1.0)
        assert pricing.image_output_rate == pytest.approx(10.0)
        assert pricing.partial_image_output_tokens == 50
        assert pricing.reference_image_output_costs == {
            "low": {
                "512x512": 0.01,
            }
        }

    def test_register_image_pricing_object(self) -> None:
        pricing = ImagePricing(
            provider="custom",
            canonical_model_id="object-image-model",
            image_output_rate=12.0,
            reference_image_output_costs={
                "medium": {
                    "1024x1024": 0.05,
                }
            },
        )

        register_image_pricing(
            "object-image-model",
            pricing,
        )

        assert get_image_pricing("object-image-model") is pricing

    def test_register_image_pricing_alias(self) -> None:
        register_image_pricing(
            "custom-image-model",
            {
                "provider": "custom",
                "canonical_model_id": "custom-image-model",
                "image_output_rate": 10.0,
                "reference_image_output_costs": {
                    "low": {
                        "512x512": 0.01,
                    }
                },
            },
        )
        register_image_pricing_alias(
            "custom-image-alias",
            "custom-image-model",
        )

        assert (
            get_image_pricing("custom-image-alias")
            is get_image_pricing("custom-image-model")
        )


# ---------------------------------------------------------------------------
# Image-size validation
# ---------------------------------------------------------------------------


class TestImageSizeValidation:
    @pytest.mark.parametrize(
        "size",
        [
            "auto",
            "1024x1024",
            "1024x1536",
            "1536x1024",
            "2880x2880",
            "3072x1024",
        ],
    )
    def test_gpt_image_2_accepts_valid_sizes(
        self,
        size: str,
    ) -> None:
        validate_image_size_for_model(
            "gpt-image-2",
            size,
        )

    @pytest.mark.parametrize(
        (
            "size",
            "match",
        ),
        [
            (
                "1024",
                "WIDTHxHEIGHT",
            ),
            (
                "0x1024",
                "positive integers",
            ),
            (
                "1025x1024",
                "multiples of 16",
            ),
            (
                "3856x1024",
                "maximum edge",
            ),
            (
                "3088x1024",
                "maximum aspect ratio",
            ),
            (
                "800x800",
                "total pixel count",
            ),
            (
                "2896x2896",
                "total pixel count",
            ),
        ],
    )
    def test_gpt_image_2_rejects_invalid_sizes(
        self,
        size: str,
        match: str,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=match,
        ):
            validate_image_size_for_model(
                "gpt-image-2",
                size,
            )

    def test_alias_to_gpt_image_2_uses_gpt_image_2_validation(
        self,
    ) -> None:
        register_image_pricing_alias(
            "latest-image-model",
            "gpt-image-2",
        )

        with pytest.raises(
            ValueError,
            match="multiples of 16",
        ):
            validate_image_size_for_model(
                "latest-image-model",
                "1025x1024",
            )

    def test_other_models_do_not_use_gpt_image_2_validation(
        self,
    ) -> None:
        validate_image_size_for_model(
            "gpt-image-1.5",
            "not-a-dimension",
        )


# ---------------------------------------------------------------------------
# Reference-price image estimates
# ---------------------------------------------------------------------------


class TestEstimateImageCost:
    def test_known_model_size_and_quality(self) -> None:
        estimate = estimate_image_cost(
            model="gpt-image-1.5",
            size="1024x1024",
            quality="medium",
            image_count=2,
        )

        assert isinstance(estimate, ImageCostEstimate)

        assert estimate.model == "gpt-image-1.5"
        assert estimate.size == "1024x1024"
        assert estimate.quality == "medium"
        assert estimate.image_count == 2
        assert estimate.pricing_mode == "standard"

        assert estimate.cost_per_image_usd == pytest.approx(0.034)
        assert estimate.reference_image_output_cost_usd == pytest.approx(
            0.068
        )
        assert estimate.token_based_cost_usd == 0.0
        assert estimate.total_cost_usd == pytest.approx(0.068)

    def test_alias_model_preserves_requested_model_name(self) -> None:
        estimate = estimate_image_cost(
            model="gpt-image-1",
            size="1024x1024",
            quality="medium",
        )

        assert estimate.model == "gpt-image-1"
        assert estimate.cost_per_image_usd == pytest.approx(0.034)
        assert estimate.total_cost_usd == pytest.approx(0.034)

    def test_quality_is_case_insensitive_and_trimmed(self) -> None:
        estimate = estimate_image_cost(
            model="gpt-image-1.5",
            size="1024x1024",
            quality="  MEDIUM  ",
        )

        assert estimate.total_cost_usd == pytest.approx(0.034)

    def test_unknown_size_raises_key_error(self) -> None:
        with pytest.raises(
            KeyError,
            match="Known sizes",
        ):
            estimate_image_cost(
                model="gpt-image-1.5",
                size="2048x2048",
                quality="high",
            )

    def test_unknown_quality_raises_key_error(self) -> None:
        with pytest.raises(
            KeyError,
            match="Known qualities",
        ):
            estimate_image_cost(
                model="gpt-image-1.5",
                size="1024x1024",
                quality="ultra",
            )

    def test_auto_size_is_rejected_for_offline_estimate(self) -> None:
        with pytest.raises(
            ValueError,
            match="explicit listed size",
        ):
            estimate_image_cost(
                model="gpt-image-2",
                size="auto",
                quality="low",
            )

    def test_image_count_must_be_positive(self) -> None:
        with pytest.raises(
            ValueError,
            match="image_count",
        ):
            estimate_image_cost(
                model="gpt-image-1.5",
                size="1024x1024",
                quality="medium",
                image_count=0,
            )

    @pytest.mark.parametrize(
        "field",
        [
            "text_input_tokens",
            "cached_text_input_tokens",
            "text_output_tokens",
            "image_input_tokens",
            "cached_image_input_tokens",
        ],
    )
    def test_negative_token_count_is_rejected(
        self,
        field: str,
    ) -> None:
        kwargs: dict[str, Any] = {
            "model": "gpt-image-1.5",
            "size": "1024x1024",
            "quality": "medium",
            field: -1,
        }

        with pytest.raises(
            ValueError,
            match=field,
        ):
            estimate_image_cost(**kwargs)

    def test_standard_estimate_adds_input_token_costs(self) -> None:
        estimate = estimate_image_cost(
            model="gpt-image-2",
            size="1024x1024",
            quality="low",
            text_input_tokens=1_000_000,
            cached_text_input_tokens=100_000,
            image_input_tokens=200_000,
            cached_image_input_tokens=100_000,
        )

        assert estimate.cost_per_image_usd == pytest.approx(0.006)
        assert estimate.reference_image_output_cost_usd == pytest.approx(
            0.006
        )

        assert estimate.text_input_cost_usd == pytest.approx(5.0)
        assert estimate.cached_text_input_cost_usd == pytest.approx(
            0.125
        )
        assert estimate.image_input_cost_usd == pytest.approx(1.6)
        assert estimate.cached_image_input_cost_usd == pytest.approx(
            0.2
        )

        assert estimate.token_based_cost_usd == pytest.approx(6.925)
        assert estimate.total_cost_usd == pytest.approx(6.931)

    def test_batch_estimate_uses_batch_rates(self) -> None:
        estimate = estimate_image_cost(
            model="gpt-image-2",
            size="1024x1024",
            quality="low",
            image_count=1,
            text_input_tokens=1_000_000,
            cached_text_input_tokens=100_000,
            image_input_tokens=200_000,
            cached_image_input_tokens=100_000,
            pricing_mode="batch",
        )

        assert estimate.pricing_mode == "batch"

        assert estimate.cost_per_image_usd == pytest.approx(0.003)
        assert estimate.reference_image_output_cost_usd == pytest.approx(
            0.003
        )

        assert estimate.text_input_cost_usd == pytest.approx(2.5)
        assert estimate.cached_text_input_cost_usd == pytest.approx(
            0.0625
        )
        assert estimate.image_input_cost_usd == pytest.approx(0.8)
        assert estimate.cached_image_input_cost_usd == pytest.approx(
            0.1
        )

        assert estimate.token_based_cost_usd == pytest.approx(3.4625)
        assert estimate.total_cost_usd == pytest.approx(3.4655)

    def test_batch_reference_price_scales_by_output_rate_ratio(
        self,
    ) -> None:
        estimate = estimate_image_cost(
            model="gpt-image-1.5",
            size="1024x1024",
            quality="medium",
            image_count=2,
            pricing_mode="batch",
        )

        assert estimate.cost_per_image_usd == pytest.approx(0.017)
        assert estimate.reference_image_output_cost_usd == pytest.approx(
            0.034
        )
        assert estimate.total_cost_usd == pytest.approx(0.034)

    def test_batch_reference_estimate_requires_batch_output_rate(
        self,
    ) -> None:
        register_image_pricing(
            "no-batch-image-model",
            {
                "provider": "custom",
                "canonical_model_id": "no-batch-image-model",
                "image_output_rate": 10.0,
                "reference_image_output_costs": {
                    "low": {
                        "512x512": 0.01,
                    }
                },
            },
        )

        with pytest.raises(
            ValueError,
            match="No batch image output rate",
        ):
            estimate_image_cost(
                model="no-batch-image-model",
                size="512x512",
                quality="low",
                pricing_mode="batch",
            )

    def test_custom_registered_model_can_be_estimated(self) -> None:
        register_image_pricing(
            "custom-image-model",
            {
                "provider": "custom",
                "canonical_model_id": "custom-image-model",
                "text_input_rate": 1.0,
                "image_output_rate": 10.0,
                "reference_image_output_costs": {
                    "low": {
                        "512x512": 0.01,
                    }
                },
            },
        )

        estimate = estimate_image_cost(
            model="custom-image-model",
            size="512x512",
            quality="low",
            image_count=3,
        )

        assert estimate.cost_per_image_usd == pytest.approx(0.01)
        assert estimate.total_cost_usd == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Usage normalisation
# ---------------------------------------------------------------------------


class TestNormaliseImageUsage:
    def test_none_returns_empty_usage(self) -> None:
        usage = normalise_image_usage(None)

        assert isinstance(usage, ImageUsage)
        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert usage.total_tokens is None

    def test_existing_image_usage_is_returned_unchanged(self) -> None:
        original = ImageUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )

        assert normalise_image_usage(original) is original

    def test_nested_mapping_is_normalised(self) -> None:
        usage = normalise_image_usage(
            {
                "input_tokens": 250,
                "output_tokens": 80,
                "input_tokens_details": {
                    "text_tokens": 120,
                    "cached_text_tokens": 20,
                    "image_tokens": 80,
                    "cached_image_tokens": 30,
                },
                "output_tokens_details": {
                    "text_tokens": 10,
                    "image_tokens": 70,
                    "partial_image_tokens": 5,
                    "partial_image_count": 1,
                },
            }
        )

        assert usage.text_input_tokens == 120
        assert usage.cached_text_input_tokens == 20
        assert usage.image_input_tokens == 80
        assert usage.cached_image_input_tokens == 30

        assert usage.text_output_tokens == 10
        assert usage.image_output_tokens == 70
        assert usage.partial_image_output_tokens == 5
        assert usage.partial_image_count == 1

        assert usage.input_tokens == 250
        assert usage.cached_input_tokens == 50
        assert usage.output_tokens == 80
        assert usage.total_tokens == 330

    def test_object_and_nested_object_are_normalised(self) -> None:
        usage_object = types.SimpleNamespace(
            input_tokens=100,
            output_tokens=30,
            input_tokens_details=types.SimpleNamespace(
                text_tokens=90,
                cached_text_tokens=10,
            ),
            output_tokens_details=types.SimpleNamespace(
                image_tokens=30,
            ),
        )

        usage = normalise_image_usage(usage_object)

        assert usage.text_input_tokens == 90
        assert usage.cached_text_input_tokens == 10
        assert usage.image_output_tokens == 30

        assert usage.input_tokens == 100
        assert usage.output_tokens == 30
        assert usage.total_tokens == 130

    def test_model_dump_objects_are_supported(self) -> None:
        class _DumpableUsage:
            @staticmethod
            def model_dump(
                *,
                exclude_none: bool,
            ) -> dict[str, Any]:
                assert exclude_none is True

                return {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "output_tokens_details": {
                        "image_tokens": 4,
                    },
                }

        usage = normalise_image_usage(_DumpableUsage())

        assert usage.input_tokens == 10
        assert usage.image_output_tokens == 4
        assert usage.output_tokens == 4
        assert usage.total_tokens == 14

    def test_direct_fields_take_precedence_over_nested_fields(
        self,
    ) -> None:
        usage = normalise_image_usage(
            {
                "text_input_tokens": 7,
                "image_output_tokens": 11,
                "input_tokens_details": {
                    "text_tokens": 700,
                },
                "output_tokens_details": {
                    "image_tokens": 1_100,
                },
            }
        )

        assert usage.text_input_tokens == 7
        assert usage.image_output_tokens == 11

    def test_cached_input_total_is_derived_from_detail_fields(
        self,
    ) -> None:
        usage = normalise_image_usage(
            {
                "cached_text_input_tokens": 4,
                "cached_image_input_tokens": 6,
            }
        )

        assert usage.cached_input_tokens == 10

    def test_output_total_is_derived_from_detail_fields(self) -> None:
        usage = normalise_image_usage(
            {
                "text_output_tokens": 4,
                "image_output_tokens": 6,
            }
        )

        assert usage.output_tokens == 10

    def test_streamed_partial_field_names_are_supported(
        self,
    ) -> None:
        usage = normalise_image_usage(
            {
                "streamed_partial_image_output_tokens": 75,
                "streamed_partial_image_count": 3,
            }
        )

        assert usage.partial_image_output_tokens == 75
        assert usage.partial_image_count == 3

    def test_cache_read_input_tokens_alias_is_supported(
        self,
    ) -> None:
        usage = normalise_image_usage(
            {
                "cache_read_input_tokens": 25,
            }
        )

        assert usage.cached_input_tokens == 25

    def test_unrecognised_object_returns_empty_usage(self) -> None:
        usage = normalise_image_usage(object())

        assert usage == ImageUsage()


# ---------------------------------------------------------------------------
# Exact token-based image costing
# ---------------------------------------------------------------------------


class TestCostForImageUsage:
    def test_exact_costing_from_nested_usage_mapping(self) -> None:
        estimate = cost_for_image_usage(
            model="gpt-image-1.5",
            usage={
                "input_tokens": 250_000,
                "cached_input_tokens": 50_000,
                "output_tokens": 20_000,
                "input_tokens_details": {
                    "text_tokens": 120_000,
                    "cached_text_tokens": 20_000,
                    "image_tokens": 80_000,
                    "cached_image_tokens": 30_000,
                },
                "output_tokens_details": {
                    "image_tokens": 20_000,
                    "partial_image_tokens": 100,
                },
            },
            size="1024x1024",
            quality="low",
            image_count=1,
        )

        assert estimate.model == "gpt-image-1.5"
        assert estimate.image_count == 1
        assert estimate.pricing_mode == "standard"

        assert estimate.text_input_tokens == 120_000
        assert estimate.cached_text_input_tokens == 20_000
        assert estimate.image_input_tokens == 80_000
        assert estimate.cached_image_input_tokens == 30_000

        assert estimate.image_output_tokens == 19_900
        assert estimate.partial_image_output_tokens == 100

        assert estimate.text_input_cost_usd == pytest.approx(0.6)
        assert estimate.cached_text_input_cost_usd == pytest.approx(
            0.025
        )
        assert estimate.image_input_cost_usd == pytest.approx(0.64)
        assert estimate.cached_image_input_cost_usd == pytest.approx(
            0.06
        )
        assert estimate.image_output_cost_usd == pytest.approx(0.6368)
        assert estimate.partial_image_output_cost_usd == pytest.approx(
            0.0032
        )

        assert estimate.reference_image_output_cost_usd == 0.0
        assert estimate.token_based_cost_usd == pytest.approx(1.965)
        assert estimate.total_cost_usd == pytest.approx(1.965)

    def test_output_tokens_are_used_when_image_output_tokens_are_absent(
        self,
    ) -> None:
        estimate = cost_for_image_usage(
            model="gpt-image-2",
            usage=ImageUsage(
                input_tokens=1_000,
                output_tokens=200,
            ),
        )

        assert estimate.image_output_tokens == 200
        assert estimate.image_output_cost_usd == pytest.approx(0.006)
        assert estimate.reference_image_output_cost_usd == 0.0
        assert estimate.total_cost_usd == pytest.approx(0.006)

    def test_exact_output_usage_does_not_require_size_or_quality(
        self,
    ) -> None:
        estimate = cost_for_image_usage(
            model="gpt-image-2",
            usage=ImageUsage(
                image_output_tokens=1_000,
            ),
        )

        assert estimate.size is None
        assert estimate.quality is None
        assert estimate.image_output_tokens == 1_000
        assert estimate.total_cost_usd == pytest.approx(0.03)

    def test_missing_output_usage_requires_size_and_quality(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="size and quality are required",
        ):
            cost_for_image_usage(
                model="gpt-image-2",
                usage=ImageUsage(
                    input_tokens=1_000,
                ),
            )

    def test_missing_output_usage_uses_reference_price(
        self,
    ) -> None:
        estimate = cost_for_image_usage(
            model="gpt-image-2",
            usage=ImageUsage(
                text_input_tokens=1_000,
                input_tokens=1_000,
            ),
            size="1024x1024",
            quality="high",
        )

        assert estimate.text_input_tokens == 1_000
        assert estimate.text_input_cost_usd == pytest.approx(0.005)

        assert estimate.image_output_tokens == 0
        assert estimate.image_output_cost_usd == 0.0
        assert estimate.reference_image_output_cost_usd == pytest.approx(
            0.211
        )

        assert estimate.token_based_cost_usd == pytest.approx(0.005)
        assert estimate.total_cost_usd == pytest.approx(0.216)

    def test_partial_image_count_uses_catalogue_token_value(
        self,
    ) -> None:
        estimate = cost_for_image_usage(
            model="gpt-image-1.5",
            usage=ImageUsage(
                image_output_tokens=500,
                partial_image_count=2,
            ),
        )

        assert estimate.image_output_tokens == 300
        assert estimate.partial_image_output_tokens == 200

    def test_explicit_partial_tokens_override_partial_count(
        self,
    ) -> None:
        estimate = cost_for_image_usage(
            model="gpt-image-1.5",
            usage=ImageUsage(
                image_output_tokens=500,
                partial_image_output_tokens=50,
                partial_image_count=2,
            ),
        )

        assert estimate.image_output_tokens == 450
        assert estimate.partial_image_output_tokens == 50

    def test_exact_batch_costing_uses_batch_rates(self) -> None:
        estimate = cost_for_image_usage(
            model="gpt-image-2",
            usage=ImageUsage(
                text_input_tokens=1_000_000,
                cached_text_input_tokens=100_000,
                image_input_tokens=200_000,
                cached_image_input_tokens=100_000,
                image_output_tokens=100_000,
            ),
            pricing_mode="batch",
        )

        assert estimate.text_input_cost_usd == pytest.approx(2.5)
        assert estimate.cached_text_input_cost_usd == pytest.approx(
            0.0625
        )
        assert estimate.image_input_cost_usd == pytest.approx(0.8)
        assert estimate.cached_image_input_cost_usd == pytest.approx(
            0.1
        )
        assert estimate.image_output_cost_usd == pytest.approx(1.5)
        assert estimate.total_cost_usd == pytest.approx(4.9625)

    def test_missing_rate_raises_when_corresponding_tokens_exist(
        self,
    ) -> None:
        register_image_pricing(
            "no-output-rate-image-model",
            {
                "provider": "custom",
                "canonical_model_id": "no-output-rate-image-model",
                "text_input_rate": 1.0,
                "image_input_rate": 2.0,
                "reference_image_output_costs": {
                    "low": {
                        "1024x1024": 0.01,
                    }
                },
            },
        )

        with pytest.raises(
            ValueError,
            match="No pricing rate is available",
        ):
            cost_for_image_usage(
                model="no-output-rate-image-model",
                usage=ImageUsage(
                    image_output_tokens=10,
                ),
            )

    def test_missing_text_output_rate_raises_for_text_output_tokens(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="text output pricing",
        ):
            cost_for_image_usage(
                model="gpt-image-2",
                usage=ImageUsage(
                    text_output_tokens=10,
                    image_output_tokens=10,
                ),
            )


# ---------------------------------------------------------------------------
# ImageResponse costing
# ---------------------------------------------------------------------------


class TestCostForImageResponse:
    def test_artifact_count_is_used_by_default(self) -> None:
        response = ImageResponse(
            provider="openai",
            model="gpt-image-2",
            artifacts=[
                ImageArtifact(b64_data="a"),
                ImageArtifact(b64_data="b"),
            ],
            usage=ImageUsage(
                image_output_tokens=200,
            ),
        )

        estimate = cost_for_image_response(
            response=response,
            size="1024x1024",
            quality="high",
        )

        assert estimate.image_count == 2
        assert estimate.image_output_tokens == 200
        assert estimate.reference_image_output_cost_usd == 0.0
        assert estimate.total_cost_usd == pytest.approx(0.006)

    def test_explicit_image_count_overrides_artifact_count(
        self,
    ) -> None:
        response = ImageResponse(
            provider="openai",
            model="gpt-image-2",
            artifacts=[
                ImageArtifact(b64_data="a"),
                ImageArtifact(b64_data="b"),
            ],
            usage=ImageUsage(
                image_output_tokens=200,
            ),
        )

        estimate = cost_for_image_response(
            response=response,
            image_count=5,
        )

        assert estimate.image_count == 5

    def test_response_falls_back_from_output_tokens(self) -> None:
        response = ImageResponse(
            provider="openai",
            model="gpt-image-2",
            artifacts=[
                ImageArtifact(b64_data="a"),
            ],
            usage=ImageUsage(
                input_tokens=1_000,
                output_tokens=200,
            ),
        )

        estimate = cost_for_image_response(
            response=response,
        )

        assert estimate.image_output_tokens == 200
        assert estimate.reference_image_output_cost_usd == 0.0
        assert estimate.total_cost_usd == pytest.approx(0.006)

    def test_response_uses_reference_price_when_output_usage_missing(
        self,
    ) -> None:
        response = ImageResponse(
            provider="openai",
            model="gpt-image-2",
            artifacts=[
                ImageArtifact(b64_data="a"),
                ImageArtifact(b64_data="b"),
            ],
            usage=ImageUsage(
                input_tokens=1_000,
            ),
        )

        estimate = cost_for_image_response(
            response=response,
            size="1024x1024",
            quality="high",
        )

        assert estimate.image_count == 2
        assert estimate.image_output_tokens == 0
        assert estimate.reference_image_output_cost_usd == pytest.approx(
            0.422
        )
        assert estimate.total_cost_usd == pytest.approx(0.422)

    def test_response_without_usage_uses_reference_price(
        self,
    ) -> None:
        response = ImageResponse(
            provider="openai",
            model="gpt-image-1.5",
            artifacts=[
                ImageArtifact(b64_data="a"),
            ],
            usage=None,
        )

        estimate = cost_for_image_response(
            response=response,
            size="1024x1024",
            quality="medium",
        )

        assert estimate.reference_image_output_cost_usd == pytest.approx(
            0.034
        )
        assert estimate.total_cost_usd == pytest.approx(0.034)

    def test_response_without_usage_requires_size_and_quality(
        self,
    ) -> None:
        response = ImageResponse(
            provider="openai",
            model="gpt-image-1.5",
            artifacts=[
                ImageArtifact(b64_data="a"),
            ],
            usage=None,
        )

        with pytest.raises(
            ValueError,
            match="size and quality are required",
        ):
            cost_for_image_response(response=response)
