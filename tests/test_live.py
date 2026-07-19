"""Live smoke tests for chat and image providers.

These tests make real, billable API calls.

Ordinary test runs exclude them through the pytest configuration in
``pyproject.toml``.

Run explicitly with:

    pytest -o addopts="" -m live

A provider is skipped unless its corresponding environment variable is set:

    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GOOGLE_API_KEY
    MOONSHOT_API_KEY
    DEEPSEEK_API_KEY
"""

from __future__ import annotations

import base64
import os
from collections import defaultdict
from collections.abc import Iterable

import pytest

from LLMUtilities.chat import chat
from LLMUtilities.costs import (
    IMAGE_PRICING_CATALOGUE,
    PRICING_CATALOGUE,
    ImagePricing,
    Pricing,
)
from LLMUtilities.image import generate_image
from LLMUtilities.types import ChatResponse, ImageResponse


pytestmark = pytest.mark.live


LIVE_TEST_PROMPT = "Reply with exactly: LIVE_TEST_OK"
EXPECTED_INPUT_TOKENS = 12
MAX_OUTPUT_TOKENS = 32

LIVE_IMAGE_PROMPT = (
    "A single solid blue circle centred on a plain white background."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _api_key_name(provider: str) -> str:
    """Return the environment-variable name for a provider API key."""
    return f"{provider.upper()}_API_KEY"


def _skip_without_api_key(provider: str) -> None:
    """Skip a live test when its provider API key is unavailable."""
    api_key_name = _api_key_name(provider)
    api_key = os.getenv(api_key_name)

    if not api_key or not api_key.strip():
        pytest.skip(f"{api_key_name} is not set.")


# ---------------------------------------------------------------------------
# Chat-provider discovery and model selection
# ---------------------------------------------------------------------------


def _chat_entries_by_provider() -> dict[str, list[Pricing]]:
    """Group canonical chat-pricing entries by provider."""
    entries_by_provider: dict[str, list[Pricing]] = defaultdict(list)

    for entry in PRICING_CATALOGUE.entries:
        entries_by_provider[entry.provider].append(entry)

    return dict(entries_by_provider)


def _eligible_chat_models(
    entries: Iterable[Pricing],
) -> list[Pricing]:
    """Return catalogue entries suitable for an ordinary chat request."""
    return [
        entry
        for entry in entries
        if entry.canonical_model_id
        and entry.input_rate is not None
        and entry.output_rate is not None
    ]


def _estimated_chat_cost(entry: Pricing) -> float:
    """
    Estimate the relative cost of the fixed live chat request.

    Rates are expressed per million tokens. Division by one million is
    unnecessary when comparing candidates.
    """
    return (
        EXPECTED_INPUT_TOKENS * entry.input_rate
        + MAX_OUTPUT_TOKENS * entry.output_rate
    )


def _cheapest_chat_model(
    provider: str,
    entries_by_provider: dict[str, list[Pricing]],
) -> Pricing:
    """Select the cheapest eligible chat model for a provider."""
    entries = _eligible_chat_models(entries_by_provider[provider])

    if not entries:
        pytest.fail(
            f"No eligible chat pricing entries found for provider "
            f"{provider!r}."
        )

    return min(
        entries,
        key=_estimated_chat_cost,
    )


CHAT_ENTRIES_BY_PROVIDER = _chat_entries_by_provider()
CHAT_PROVIDERS = sorted(CHAT_ENTRIES_BY_PROVIDER)


# ---------------------------------------------------------------------------
# Image-provider discovery and model selection
# ---------------------------------------------------------------------------


def _image_entries_by_provider() -> dict[str, list[ImagePricing]]:
    """Group canonical image-pricing entries by provider."""
    entries_by_provider: dict[str, list[ImagePricing]] = defaultdict(list)

    for entry in IMAGE_PRICING_CATALOGUE.entries:
        entries_by_provider[entry.provider].append(entry)

    return dict(entries_by_provider)


def _cheapest_image_configuration(
    provider: str,
    entries_by_provider: dict[str, list[ImagePricing]],
) -> tuple[ImagePricing, str, str, float]:
    """
    Select the cheapest model, quality and size combination for a provider.

    Reference image-output prices are used because they represent the expected
    cost of generating one image for a known quality and size.
    """
    candidates: list[tuple[float, ImagePricing, str, str]] = []

    for entry in entries_by_provider[provider]:
        for quality, size_costs in entry.reference_image_output_costs.items():
            for size, cost in size_costs.items():
                candidates.append(
                    (
                        float(cost),
                        entry,
                        quality,
                        size,
                    )
                )

    if not candidates:
        pytest.fail(
            f"No priced image-generation configurations found for provider "
            f"{provider!r}."
        )

    cost, pricing, quality, size = min(
        candidates,
        key=lambda candidate: candidate[0],
    )

    return pricing, quality, size, cost


IMAGE_ENTRIES_BY_PROVIDER = _image_entries_by_provider()
IMAGE_PROVIDERS = sorted(IMAGE_ENTRIES_BY_PROVIDER)


# ---------------------------------------------------------------------------
# Live chat tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    CHAT_PROVIDERS,
)
def test_live_chat_provider(provider: str) -> None:
    """Make one minimal live chat request to each priced provider."""
    _skip_without_api_key(provider)

    pricing = _cheapest_chat_model(
        provider,
        CHAT_ENTRIES_BY_PROVIDER,
    )

    response = chat(
        provider_name=provider,
        model=pricing.canonical_model_id,
        user=LIVE_TEST_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    assert isinstance(response, ChatResponse)
    assert response.provider == provider
    assert response.model
    assert response.text.strip()

    assert response.usage is not None

    assert response.usage.input_tokens is not None
    assert response.usage.input_tokens > 0

    assert response.usage.output_tokens is not None
    assert response.usage.output_tokens > 0

    assert response.usage.total_tokens is not None
    assert response.usage.total_tokens > 0


# ---------------------------------------------------------------------------
# Live image tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    IMAGE_PROVIDERS,
)
def test_live_image_provider(provider: str) -> None:
    """Generate one image using each priced image provider."""
    _skip_without_api_key(provider)

    pricing, quality, size, _ = _cheapest_image_configuration(
        provider,
        IMAGE_ENTRIES_BY_PROVIDER,
    )

    response = generate_image(
        provider_name=provider,
        model=pricing.canonical_model_id,
        prompt=LIVE_IMAGE_PROMPT,
        quality=quality,
        size=size,
        n=1,
    )

    assert isinstance(response, ImageResponse)
    assert response.provider == provider
    assert response.model
    assert len(response.artifacts) == 1

    artifact = response.artifacts[0]

    assert artifact.b64_data or artifact.url

    if artifact.b64_data is not None:
        decoded = base64.b64decode(
            artifact.b64_data,
            validate=True,
        )
        assert decoded

    if artifact.url is not None:
        assert artifact.url.startswith(
            (
                "http://",
                "https://",
            )
        )

    if artifact.mime_type is not None:
        assert artifact.mime_type.startswith("image/")

    if response.usage is not None:
        usage_values = response.usage.model_dump().values()

        for value in usage_values:
            if value is not None:
                assert value >= 0