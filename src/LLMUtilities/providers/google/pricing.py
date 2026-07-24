from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from ...types import CostSummary
from ..chat_cost_calculation import calculate_chat_cost
from ..pricing_loading import load_pricing_catalogue, select_pricing

PRICING_PATH = Path(__file__).with_name("pricing.json")


class GoogleChatPricing(BaseModel):
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


class GoogleChatUsageDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class GoogleChatCostDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_cost: float
    output_cost: float
    total_cost: float
    currency: str
    provider: Literal["google"] = "google"
    requested_model: Optional[str] = None
    resolved_model: str


def list_pricings() -> list[GoogleChatPricing]:
    return load_pricing_catalogue(PRICING_PATH, GoogleChatPricing)


def get_pricing(
    model: str, *, effective_at: Optional[datetime] = None
) -> GoogleChatPricing:
    return select_pricing(
        list_pricings(), model, provider_name="google", effective_at=effective_at
    )


def list_models() -> list[str]:
    return sorted({entry.canonical_model_id for entry in list_pricings()})


def calculate_cost_summary(
    *,
    usage: GoogleChatUsageDetails,
    pricing: GoogleChatPricing,
    requested_model: Optional[str],
    resolved_model: str,
) -> CostSummary:
    breakdown = calculate_chat_cost(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        pricing=pricing,
        context_tokens=usage.input_tokens,
    )
    return CostSummary(
        input_cost=breakdown.input_cost,
        output_cost=breakdown.output_cost,
        other_cost=0.0,
        total_cost=breakdown.total_cost,
        currency=pricing.currency,
        provider="google",
        requested_model=requested_model or resolved_model,
        resolved_model=resolved_model,
    )


def calculate_cost_details(
    *,
    usage: GoogleChatUsageDetails,
    pricing: GoogleChatPricing,
    requested_model: Optional[str],
    resolved_model: str,
) -> GoogleChatCostDetails:
    breakdown = calculate_chat_cost(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        pricing=pricing,
        context_tokens=usage.input_tokens,
    )
    return GoogleChatCostDetails(
        input_cost=breakdown.input_cost,
        output_cost=breakdown.output_cost,
        total_cost=breakdown.total_cost,
        currency=pricing.currency,
        requested_model=requested_model,
        resolved_model=resolved_model,
    )
