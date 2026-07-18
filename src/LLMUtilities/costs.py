from __future__ import annotations

import json
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

from .types import ChatUsage


logger = logging.getLogger(__name__)


def _emit(message: str) -> None:
    logger.info(message)


@dataclass(frozen=True)
class Pricing:
    """
    Pricing is stored in USD per 1M tokens.
    `cached_input_per_million_tokens` is interpreted as the read/hit price.
    """
    input_per_million_tokens: float
    output_per_million_tokens: float
    cached_input_per_million_tokens: Optional[float] = None


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cached_input_cost_usd: float = 0.0
    cache_creation_input_cost_usd: float = 0.0

    total_cost_usd: float = 0.0


@dataclass(frozen=True)
class ImagePricing:
    """
    Image pricing in USD per generated image by size.

    If a requested size is not present in ``per_image_usd`` and
    ``default_per_image_usd`` is set, the default is used.
    """

    per_image_usd: dict[str, float]
    default_per_image_usd: Optional[float] = None


@dataclass(frozen=True)
class ImageCostEstimate:
    model: str
    size: str
    image_count: int
    cost_per_image_usd: float
    total_cost_usd: float

def _pricing_file() -> Path:
    return Path(__file__).with_name("PRICING.json")


def _image_pricing_file() -> Path:
    return Path(__file__).with_name("IMAGE_PRICING.json")


def _load_pricing_table(path: Path) -> dict[str, Pricing]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    table: dict[str, Pricing] = {}

    for model, spec in raw.items():
        table[model] = Pricing(
            input_per_million_tokens=spec["input_per_million_tokens"],
            output_per_million_tokens=spec["output_per_million_tokens"],
            cached_input_per_million_tokens=spec.get("cached_input_per_million_tokens"),
        )

    return table


def _load_image_pricing_table(path: Path) -> dict[str, ImagePricing]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    table: dict[str, ImagePricing] = {}

    for model, spec in raw.items():
        table[model] = ImagePricing(
            per_image_usd=spec["per_image_usd"],
            default_per_image_usd=spec.get("default_per_image_usd"),
        )

    return table


PRICING: dict[str, Pricing] = _load_pricing_table(_pricing_file())


IMAGE_PRICING: dict[str, ImagePricing] = _load_image_pricing_table(_image_pricing_file())


def register_pricing(model: str, pricing: Pricing) -> None:
    PRICING[model] = pricing


def register_image_pricing(model: str, pricing: ImagePricing) -> None:
    IMAGE_PRICING[model] = pricing


def register_pricing_alias(alias: str, target_model: str) -> None:
    PRICING[alias] = get_pricing(target_model)


def register_image_pricing_alias(alias: str, target_model: str) -> None:
    IMAGE_PRICING[alias] = get_image_pricing(target_model)


def get_pricing(model: str) -> Pricing:
    try:
        return PRICING[model]
    except KeyError as exc:
        known = ", ".join(sorted(PRICING))
        raise KeyError(
            f"No pricing registered for model {model!r}. Known models: {known}"
        ) from exc


def get_image_pricing(model: str) -> ImagePricing:
    try:
        return IMAGE_PRICING[model]
    except KeyError as exc:
        known = ", ".join(sorted(IMAGE_PRICING))
        raise KeyError(
            f"No image pricing registered for model {model!r}. Known models: {known}"
        ) from exc


def estimate_image_cost(
    *,
    model: str,
    size: str,
    image_count: int = 1,
) -> ImageCostEstimate:
    if image_count < 1:
        raise ValueError("image_count must be >= 1")

    pricing = get_image_pricing(model)
    if size in pricing.per_image_usd:
        per_image = pricing.per_image_usd[size]
    elif pricing.default_per_image_usd is not None:
        per_image = pricing.default_per_image_usd
    else:
        known_sizes = ", ".join(sorted(pricing.per_image_usd))
        raise KeyError(
            f"No image pricing for size {size!r} on model {model!r}. "
            f"Known sizes: {known_sizes}"
        )

    return ImageCostEstimate(
        model=model,
        size=size,
        image_count=image_count,
        cost_per_image_usd=per_image,
        total_cost_usd=per_image * image_count,
    )


def cost_from_tokens(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    pricing: Pricing,
) -> CostEstimate:
    if input_tokens < 0:
        raise ValueError("input_tokens must be >= 0")

    if output_tokens < 0:
        raise ValueError("output_tokens must be >= 0")

    if cached_input_tokens < 0:
        raise ValueError("cached_input_tokens must be >= 0")

    if cache_creation_input_tokens < 0:
        raise ValueError("cache_creation_input_tokens must be >= 0")

    input_cost_usd = (input_tokens / 1_000_000) * pricing.input_per_million_tokens
    output_cost_usd = (output_tokens / 1_000_000) * pricing.output_per_million_tokens
    cache_creation_input_cost_usd = (
        (cache_creation_input_tokens / 1_000_000) * pricing.input_per_million_tokens
    )

    cached_rate = (
        pricing.cached_input_per_million_tokens
        if pricing.cached_input_per_million_tokens is not None
        else pricing.input_per_million_tokens
    )
    cached_input_cost_usd = (cached_input_tokens / 1_000_000) * cached_rate

    total_cost_usd = (
        input_cost_usd
        + output_cost_usd
        + cached_input_cost_usd
        + cache_creation_input_cost_usd
    )

    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        cached_input_cost_usd=cached_input_cost_usd,
        cache_creation_input_cost_usd=cache_creation_input_cost_usd,
        total_cost_usd=total_cost_usd,
    )


def cost_from_usage(
    *,
    usage: ChatUsage,
    pricing: Pricing,
    cached_input_tokens: Optional[int] = None,
    cache_creation_input_tokens: Optional[int] = None,
) -> CostEstimate:
    return cost_from_tokens(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cached_input_tokens=(
            usage.cached_input_tokens or 0
            if cached_input_tokens is None
            else cached_input_tokens
        ),
        cache_creation_input_tokens=(
            usage.cache_creation_input_tokens or 0
            if cache_creation_input_tokens is None
            else cache_creation_input_tokens
        ),
        pricing=pricing,
    )


def cost_for_model(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> CostEstimate:
    pricing = get_pricing(model)
    return cost_from_tokens(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        pricing=pricing,
    )


def cost_for_response(
    *,
    model: str,
    usage: ChatUsage,
    cached_input_tokens: Optional[int] = None,
    cache_creation_input_tokens: Optional[int] = None,
) -> CostEstimate:
    pricing = get_pricing(model)
    return cost_from_usage(
        usage=usage,
        pricing=pricing,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )


def format_cost(cost: float, decimals: int = 8) -> str:
    return f"${cost:.{decimals}f}"


def print_cost_breakdown(
    *,
    estimate: CostEstimate,
    model: Optional[str] = None,
) -> None:
    if model:
        _emit(f"Model: {model}")

    _emit(f"Input tokens: {estimate.input_tokens}")
    _emit(f"Output tokens: {estimate.output_tokens}")
    _emit(f"Cached input tokens: {estimate.cached_input_tokens}")
    _emit(f"Cache creation input tokens: {estimate.cache_creation_input_tokens}")
    _emit(f"Input cost: {format_cost(estimate.input_cost_usd)}")
    _emit(f"Output cost: {format_cost(estimate.output_cost_usd)}")
    _emit(f"Cached input cost: {format_cost(estimate.cached_input_cost_usd)}")
    _emit(
        f"Cache creation input cost: {format_cost(estimate.cache_creation_input_cost_usd)}"
    )
    _emit(f"Total cost: {format_cost(estimate.total_cost_usd)}")


def print_cost_summary(
    *,
    estimate: CostEstimate,
    model: Optional[str] = None,
) -> None:
    label = model or "Unknown model"
    _emit("---")
    _emit(
        f"Model: {label} | "
        f"Input tokens: {estimate.input_tokens} "
        f"Output tokens: {estimate.output_tokens} "
        f"Cached input tokens: {estimate.cached_input_tokens} "
        f"Cache creation input tokens: {estimate.cache_creation_input_tokens}"
    )
    _emit(
        f"Cached input cost: {format_cost(estimate.cached_input_cost_usd)} "
        f"Cache creation input cost: {format_cost(estimate.cache_creation_input_cost_usd)} "
        f"Total cost: {format_cost(estimate.total_cost_usd)}"
    )
    _emit("---")


def register_default_pricing_aliases() -> None:
    """
    Register a few practical aliases for common local/default model names.

    This is useful when your config or notebooks use shorter or older names
    that should resolve to a canonical pricing entry.
    """
    aliases = {
        # OpenAI
        "gpt-5": "gpt-5.4",
        "gpt-5-mini": "gpt-5.4-mini",
        "gpt-5-nano": "gpt-5.4-nano",

        # Anthropic
        "claude-sonnet-4-0": "claude-sonnet-4.6",
        "claude-sonnet-4": "claude-sonnet-4.6",
        "claude-haiku-4": "claude-haiku-4.5",
        "claude-opus-4": "claude-opus-4.6",

        # Google
        "google-default": "gemini-2.5-flash",
        "gemini-pro": "gemini-2.5-pro",
        "gemini-flash": "gemini-2.5-flash",
        "gemini-flash-lite": "gemini-2.5-flash-lite",
    }

    for alias, target_model in aliases.items():
        register_pricing_alias(alias, target_model)


register_default_pricing_aliases()


def register_default_image_pricing_aliases() -> None:
    aliases = {
        "gpt-image-1": "gpt-image-1.5",
        "openai-image-default": "gpt-image-1.5",
    }

    for alias, target_model in aliases.items():
        register_image_pricing_alias(alias, target_model)


register_default_image_pricing_aliases()


def estimate_cost(
    *,
    model: str,
    usage: "ChatUsage",
    cached_input_tokens: Optional[int] = None,
    cache_creation_input_tokens: Optional[int] = None,
) -> CostEstimate:
    """
    Alias for :func:`cost_for_response`.

    Returns a :class:`CostEstimate` for the given model and usage object.
    """
    return cost_for_response(
        model=model,
        usage=usage,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )
