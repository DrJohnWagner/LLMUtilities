from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from ...types import CostSummary
from ..chat_cost_calculation import calculate_chat_cost
from ..pricing_loading import load_pricing_catalogue, select_pricing

PRICING_PATH = Path(__file__).with_name("pricing.json")
IMAGE_PRICING_PATH = Path(__file__).with_name("image_pricing.json")


class OpenAIChatPricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    canonical_model_id: str
    input_rate: float
    output_rate: float
    cached_read_rate: Optional[float] = None
    cached_write_rate: Optional[float] = None
    batch_input_rate: Optional[float] = None
    batch_output_rate: Optional[float] = None
    long_context_threshold: Optional[int] = None
    long_context_input_rate: Optional[float] = None
    long_context_output_rate: Optional[float] = None
    currency: str = "USD"
    effective_from: Optional[str] = None
    effective_until: Optional[str] = None
    source_url: Optional[str] = None
    last_verified_at: Optional[str] = None


class OpenAIChatUsageDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatCostDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_cost: float
    cached_read_cost: float
    output_cost: float
    total_cost: float
    currency: str
    provider: Literal["openai"] = "openai"
    requested_model: Optional[str] = None
    resolved_model: str


class OpenAIImagePricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    canonical_model_id: str
    text_input_rate: Optional[float] = None
    text_output_rate: Optional[float] = None
    text_cached_read_rate: Optional[float] = None
    text_cached_write_rate: Optional[float] = None
    image_input_rate: Optional[float] = None
    image_output_rate: Optional[float] = None
    image_cached_read_rate: Optional[float] = None
    image_cached_write_rate: Optional[float] = None
    batch_text_input_rate: Optional[float] = None
    batch_text_output_rate: Optional[float] = None
    batch_text_cached_read_rate: Optional[float] = None
    batch_text_cached_write_rate: Optional[float] = None
    batch_image_input_rate: Optional[float] = None
    batch_image_output_rate: Optional[float] = None
    batch_image_cached_read_rate: Optional[float] = None
    batch_image_cached_write_rate: Optional[float] = None
    reference_image_output_costs: Optional[dict[str, dict[str, float]]] = None
    partial_image_output_tokens: Optional[int] = None
    currency: str = "USD"
    effective_from: Optional[str] = None
    effective_until: Optional[str] = None
    source_url: Optional[str] = None
    last_verified_at: Optional[str] = None


class OpenAIImageUsageDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_input_tokens: int = 0
    cached_text_input_tokens: int = 0
    text_output_tokens: int = 0
    image_input_tokens: int = 0
    cached_image_input_tokens: int = 0
    image_output_tokens: int = 0
    partial_image_output_tokens: int = 0


class OpenAIImageCostDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_input_cost: float
    cached_text_input_cost: float
    text_output_cost: float
    image_input_cost: float
    cached_image_input_cost: float
    image_output_cost: float
    partial_image_output_cost: float
    total_cost: float
    currency: str
    provider: Literal["openai"] = "openai"
    requested_model: Optional[str] = None
    resolved_model: str


def list_pricings() -> list[OpenAIChatPricing]:
    return load_pricing_catalogue(PRICING_PATH, OpenAIChatPricing)


def get_pricing(model: str, *, effective_at: Optional[datetime] = None) -> OpenAIChatPricing:
    return select_pricing(
        list_pricings(), model, provider_name="openai", effective_at=effective_at
    )


def list_models() -> list[str]:
    return sorted({entry.canonical_model_id for entry in list_pricings()})


def list_image_pricings() -> list[OpenAIImagePricing]:
    return load_pricing_catalogue(IMAGE_PRICING_PATH, OpenAIImagePricing)


def get_image_pricing(
    model: str, *, effective_at: Optional[datetime] = None
) -> OpenAIImagePricing:
    return select_pricing(
        list_image_pricings(), model, provider_name="openai", effective_at=effective_at
    )


def list_image_models() -> list[str]:
    return sorted({entry.canonical_model_id for entry in list_image_pricings()})


def calculate_cost_summary(
    *,
    usage: OpenAIChatUsageDetails,
    pricing: OpenAIChatPricing,
    requested_model: Optional[str],
    resolved_model: str,
) -> CostSummary:
    breakdown = calculate_chat_cost(
        input_tokens=max(usage.input_tokens - usage.cached_input_tokens, 0),
        output_tokens=usage.output_tokens,
        cached_read_tokens=usage.cached_input_tokens,
        pricing=pricing,
    )
    return CostSummary(
        input_cost=breakdown.input_cost + breakdown.cached_read_cost,
        output_cost=breakdown.output_cost,
        other_cost=0.0,
        total_cost=breakdown.total_cost,
        currency=pricing.currency,
        provider="openai",
        requested_model=requested_model or resolved_model,
        resolved_model=resolved_model,
    )


def _select_image_rates(
    pricing: OpenAIImagePricing, *, batch: bool
) -> tuple[
    Optional[float], Optional[float], Optional[float],
    Optional[float], Optional[float], Optional[float],
]:
    if batch:
        text_input_rate = pricing.batch_text_input_rate or pricing.text_input_rate
        text_output_rate = pricing.batch_text_output_rate or pricing.text_output_rate
        text_cached_read_rate = (
            pricing.batch_text_cached_read_rate or pricing.text_cached_read_rate
        )
        image_input_rate = pricing.batch_image_input_rate or pricing.image_input_rate
        image_output_rate = (
            pricing.batch_image_output_rate or pricing.image_output_rate
        )
        image_cached_read_rate = (
            pricing.batch_image_cached_read_rate or pricing.image_cached_read_rate
        )
    else:
        text_input_rate = pricing.text_input_rate
        text_output_rate = pricing.text_output_rate
        text_cached_read_rate = pricing.text_cached_read_rate
        image_input_rate = pricing.image_input_rate
        image_output_rate = pricing.image_output_rate
        image_cached_read_rate = pricing.image_cached_read_rate

    text_cached_read_rate = text_cached_read_rate or text_input_rate
    image_cached_read_rate = image_cached_read_rate or image_input_rate

    return (
        text_input_rate,
        text_output_rate,
        text_cached_read_rate,
        image_input_rate,
        image_output_rate,
        image_cached_read_rate,
    )


def _rate_cost(tokens: int, rate: Optional[float]) -> float:
    if tokens <= 0 or rate is None:
        return 0.0
    return (tokens / 1_000_000) * rate


def _calculate_image_cost_breakdown(
    *,
    usage: OpenAIImageUsageDetails,
    pricing: OpenAIImagePricing,
    batch: bool,
) -> OpenAIImageCostDetails:
    (
        text_input_rate,
        text_output_rate,
        text_cached_read_rate,
        image_input_rate,
        image_output_rate,
        image_cached_read_rate,
    ) = _select_image_rates(pricing, batch=batch)

    billable_image_output_tokens = max(
        usage.image_output_tokens - usage.partial_image_output_tokens, 0
    )

    text_input_cost = _rate_cost(usage.text_input_tokens, text_input_rate)
    cached_text_input_cost = _rate_cost(
        usage.cached_text_input_tokens, text_cached_read_rate
    )
    text_output_cost = _rate_cost(usage.text_output_tokens, text_output_rate)
    image_input_cost = _rate_cost(usage.image_input_tokens, image_input_rate)
    cached_image_input_cost = _rate_cost(
        usage.cached_image_input_tokens, image_cached_read_rate
    )
    image_output_cost = _rate_cost(billable_image_output_tokens, image_output_rate)
    partial_image_output_cost = _rate_cost(
        usage.partial_image_output_tokens, image_output_rate
    )

    total = (
        text_input_cost
        + cached_text_input_cost
        + text_output_cost
        + image_input_cost
        + cached_image_input_cost
        + image_output_cost
        + partial_image_output_cost
    )

    return OpenAIImageCostDetails(
        text_input_cost=text_input_cost,
        cached_text_input_cost=cached_text_input_cost,
        text_output_cost=text_output_cost,
        image_input_cost=image_input_cost,
        cached_image_input_cost=cached_image_input_cost,
        image_output_cost=image_output_cost,
        partial_image_output_cost=partial_image_output_cost,
        total_cost=total,
        currency=pricing.currency,
        resolved_model=pricing.canonical_model_id,
    )


def calculate_image_cost_details(
    *,
    usage: OpenAIImageUsageDetails,
    pricing: OpenAIImagePricing,
    requested_model: Optional[str],
    resolved_model: str,
    batch: bool = False,
) -> OpenAIImageCostDetails:
    breakdown = _calculate_image_cost_breakdown(usage=usage, pricing=pricing, batch=batch)
    return breakdown.model_copy(
        update={"requested_model": requested_model, "resolved_model": resolved_model}
    )


def calculate_image_cost_summary(
    *,
    usage: OpenAIImageUsageDetails,
    pricing: OpenAIImagePricing,
    requested_model: Optional[str],
    resolved_model: str,
    batch: bool = False,
) -> CostSummary:
    breakdown = _calculate_image_cost_breakdown(usage=usage, pricing=pricing, batch=batch)

    return CostSummary(
        input_cost=breakdown.text_input_cost
        + breakdown.cached_text_input_cost
        + breakdown.image_input_cost
        + breakdown.cached_image_input_cost,
        output_cost=breakdown.text_output_cost
        + breakdown.image_output_cost
        + breakdown.partial_image_output_cost,
        other_cost=0.0,
        total_cost=breakdown.total_cost,
        currency=pricing.currency,
        provider="openai",
        requested_model=requested_model or resolved_model,
        resolved_model=resolved_model,
    )
