from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol


class RatedChatPricing(Protocol):
    """
    Structural shape shared by every provider's chat pricing type.

    This is a *mechanical* rate-selection helper (long-context threshold,
    batch fallback, cached-rate fallback to the input rate) — it has no
    opinion on how a provider folds the resulting numbers into a
    ``CostSummary``. Providers own that mapping themselves.
    """

    input_rate: float
    output_rate: float
    cached_read_rate: Optional[float]
    cached_write_rate: Optional[float]
    batch_input_rate: Optional[float]
    batch_output_rate: Optional[float]
    long_context_threshold: Optional[int]
    long_context_input_rate: Optional[float]
    long_context_output_rate: Optional[float]


@dataclass(frozen=True)
class ChatCostBreakdown:
    input_cost: float
    cached_read_cost: float
    cache_write_cost: float
    output_cost: float
    total_cost: float


def calculate_chat_cost(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    pricing: RatedChatPricing,
    pricing_mode: Literal["standard", "batch"] = "standard",
    context_tokens: Optional[int] = None,
) -> ChatCostBreakdown:
    input_rate = pricing.input_rate
    output_rate = pricing.output_rate

    if (
        context_tokens is not None
        and pricing.long_context_threshold is not None
        and context_tokens >= pricing.long_context_threshold
    ):
        input_rate = pricing.long_context_input_rate or input_rate
        output_rate = pricing.long_context_output_rate or output_rate
    elif pricing_mode == "batch":
        input_rate = pricing.batch_input_rate or input_rate
        output_rate = pricing.batch_output_rate or output_rate

    cached_read_rate = pricing.cached_read_rate or input_rate
    cache_write_rate = pricing.cached_write_rate or input_rate

    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    cached_read_cost = (cached_read_tokens / 1_000_000) * cached_read_rate
    cache_write_cost = (cache_write_tokens / 1_000_000) * cache_write_rate

    return ChatCostBreakdown(
        input_cost=input_cost,
        cached_read_cost=cached_read_cost,
        cache_write_cost=cache_write_cost,
        output_cost=output_cost,
        total_cost=input_cost + cached_read_cost + cache_write_cost + output_cost,
    )
