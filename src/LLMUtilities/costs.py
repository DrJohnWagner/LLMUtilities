from __future__ import annotations

from typing import Union

from .exceptions import CostCalculationUnavailableError
from .providers.registry import get_provider
from .types import ChatResponse, CostSummary, ImageResponse


def get_cost_summary(response: Union[ChatResponse, ImageResponse]) -> CostSummary:
    """
    Resolve the provider that produced ``response`` and delegate cost
    calculation to it. Performs no detailed billing calculation itself -
    the provider's summary is authoritative.
    """
    provider = get_provider(response.provider)

    if isinstance(response, ImageResponse):
        get_image_cost_summary = getattr(provider, "get_image_cost_summary", None)
        if get_image_cost_summary is None:
            raise CostCalculationUnavailableError(
                f"Provider {response.provider!r} cannot calculate image costs."
            )
        return get_image_cost_summary(response)

    get_chat_cost_summary = getattr(provider, "get_cost_summary", None)
    if get_chat_cost_summary is None:
        raise CostCalculationUnavailableError(
            f"Provider {response.provider!r} cannot calculate chat costs."
        )
    return get_chat_cost_summary(response)
