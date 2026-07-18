from __future__ import annotations

import json
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Literal, Optional

from .types import ChatUsage, ImageResponse, ImageUsage

logger = logging.getLogger(__name__)


def _emit(message: str) -> None:
    logger.info(message)


@dataclass(frozen=True)
class Pricing:
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


@dataclass(frozen=True)
class PricingCatalogue:
    schema_version: int
    catalogue_version: str
    generated_at: str
    entries: tuple[Pricing, ...]

    @property
    def version(self) -> str:
        return self.catalogue_version


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_write_tokens: int = 0

    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cached_input_cost_usd: float = 0.0
    cache_creation_input_cost_usd: float = 0.0
    cache_write_input_cost_usd: float = 0.0

    total_cost_usd: float = 0.0


@dataclass(frozen=True)
class ImagePricing:
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
    reference_image_output_costs: dict[str, dict[str, float]] | None = None
    partial_image_output_tokens: Optional[int] = None
    currency: str = "USD"
    effective_from: Optional[str] = None
    effective_until: Optional[str] = None
    source_url: Optional[str] = None
    last_verified_at: Optional[str] = None


@dataclass(frozen=True)
class ImagePricingCatalogue:
    schema_version: int
    catalogue_version: str
    generated_at: str
    entries: tuple[ImagePricing, ...]

    @property
    def version(self) -> str:
        return self.catalogue_version


@dataclass(frozen=True)
class ImageCostEstimate:
    model: str
    size: Optional[str] = None
    quality: Optional[str] = None
    image_count: int = 1
    pricing_mode: Literal["standard", "batch"] = "standard"
    cost_per_image_usd: float = 0.0

    text_input_tokens: int = 0
    cached_text_input_tokens: int = 0
    text_output_tokens: int = 0
    image_input_tokens: int = 0
    cached_image_input_tokens: int = 0
    image_output_tokens: int = 0
    partial_image_output_tokens: int = 0

    text_input_cost_usd: float = 0.0
    cached_text_input_cost_usd: float = 0.0
    text_output_cost_usd: float = 0.0
    image_input_cost_usd: float = 0.0
    cached_image_input_cost_usd: float = 0.0
    image_output_cost_usd: float = 0.0
    partial_image_output_cost_usd: float = 0.0
    token_based_cost_usd: float = 0.0
    reference_image_output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0


CATALOGUE_SCHEMA_VERSION = 1
DEFAULT_CATALOGUE_VERSION = "2026-07-18"
DEFAULT_LAST_VERIFIED_AT = "2026-07-18T16:55:00Z"


def _pricing_file() -> Path:
    return Path(__file__).with_name("PRICING.json")


def _image_pricing_file() -> Path:
    return Path(__file__).with_name("IMAGE_PRICING.json")


def _infer_provider_from_model(model: str) -> str:
    if model.startswith(
        (
            "gpt-",
            "o",
            "chat-",
            "gpt-image-",
            "gpt-realtime-",
            "o3-",
            "o4-",
            "computer-use-",
        )
    ):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(
        (
            "gemini-",
            "imagen-",
            "sora-",
            "lyria-",
            "veo-",
            "multilingual-",
            "multimodalembedding",
            "model-optimizer",
            "codey",
            "translation",
        )
    ):
        return "google"
    if model.startswith(("kimi-", "moonshot-")):
        return "moonshot"
    if model.startswith("deepseek-"):
        return "deepseek"
    return "unknown"


def _coerce_pricing(model: str, pricing: Pricing | dict[str, Any]) -> Pricing:
    if isinstance(pricing, Pricing):
        return pricing

    return Pricing(
        provider=str(pricing.get("provider", _infer_provider_from_model(model))),
        canonical_model_id=str(pricing.get("canonical_model_id", model)),
        input_rate=float(pricing["input_rate"]),
        output_rate=float(pricing["output_rate"]),
        cached_read_rate=(
            None
            if pricing.get("cached_read_rate") is None
            else float(pricing["cached_read_rate"])
        ),
        cached_write_rate=(
            None
            if pricing.get("cached_write_rate") is None
            else float(pricing["cached_write_rate"])
        ),
        batch_input_rate=(
            None
            if pricing.get("batch_input_rate") is None
            else float(pricing["batch_input_rate"])
        ),
        batch_output_rate=(
            None
            if pricing.get("batch_output_rate") is None
            else float(pricing["batch_output_rate"])
        ),
        long_context_threshold=(
            None
            if pricing.get("long_context_threshold") is None
            else int(pricing["long_context_threshold"])
        ),
        long_context_input_rate=(
            None
            if pricing.get("long_context_input_rate") is None
            else float(pricing["long_context_input_rate"])
        ),
        long_context_output_rate=(
            None
            if pricing.get("long_context_output_rate") is None
            else float(pricing["long_context_output_rate"])
        ),
        currency=str(pricing.get("currency", "USD")),
        effective_from=pricing.get("effective_from"),
        effective_until=pricing.get("effective_until"),
        source_url=pricing.get("source_url"),
        last_verified_at=pricing.get("last_verified_at"),
    )


def _load_pricing_catalogue(path: Path) -> PricingCatalogue:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "entries" in raw:
        entries_raw = raw["entries"]
        schema_version = int(raw.get("schema_version", CATALOGUE_SCHEMA_VERSION))
        catalogue_version = str(raw.get("catalogue_version", DEFAULT_CATALOGUE_VERSION))
        generated_at = str(raw.get("generated_at", DEFAULT_LAST_VERIFIED_AT))
    else:
        schema_version = CATALOGUE_SCHEMA_VERSION
        catalogue_version = DEFAULT_CATALOGUE_VERSION
        generated_at = DEFAULT_LAST_VERIFIED_AT
        entries_raw = [
            {
                "provider": _infer_provider_from_model(model),
                "canonical_model_id": model,
                "input_rate": spec["input_per_million_tokens"],
                "output_rate": spec["output_per_million_tokens"],
                "cached_read_rate": spec.get("cached_input_per_million_tokens"),
                "cached_write_rate": spec.get(
                    "cached_input_per_million_tokens", spec["input_per_million_tokens"]
                ),
                "currency": "USD",
                "effective_from": DEFAULT_LAST_VERIFIED_AT,
                "source_url": None,
                "last_verified_at": DEFAULT_LAST_VERIFIED_AT,
            }
            for model, spec in raw.items()
        ]

    entries = tuple(
        _coerce_pricing(str(entry["canonical_model_id"]), entry)
        for entry in entries_raw
    )
    return PricingCatalogue(
        schema_version=schema_version,
        catalogue_version=catalogue_version,
        generated_at=generated_at,
        entries=entries,
    )


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _coerce_reference_image_output_costs(
    value: Any,
) -> dict[str, dict[str, float]]:
    if value is None:
        return {}

    coerced: dict[str, dict[str, float]] = {}
    for quality, size_map in value.items():
        coerced_sizes: dict[str, float] = {}
        for size, amount in size_map.items():
            coerced_sizes[str(size)] = float(amount)
        coerced[str(quality).lower()] = coerced_sizes
    return coerced


def _coerce_image_pricing(
    model: str, pricing: ImagePricing | dict[str, Any]
) -> ImagePricing:
    if isinstance(pricing, ImagePricing):
        return pricing

    return ImagePricing(
        provider=str(pricing.get("provider", _infer_provider_from_model(model))),
        canonical_model_id=str(pricing.get("canonical_model_id", model)),
        text_input_rate=_coerce_optional_float(pricing.get("text_input_rate")),
        text_output_rate=_coerce_optional_float(pricing.get("text_output_rate")),
        text_cached_read_rate=_coerce_optional_float(
            pricing.get("text_cached_read_rate")
        ),
        text_cached_write_rate=_coerce_optional_float(
            pricing.get("text_cached_write_rate")
        ),
        image_input_rate=_coerce_optional_float(pricing.get("image_input_rate")),
        image_output_rate=_coerce_optional_float(pricing.get("image_output_rate")),
        image_cached_read_rate=_coerce_optional_float(
            pricing.get("image_cached_read_rate")
        ),
        image_cached_write_rate=_coerce_optional_float(
            pricing.get("image_cached_write_rate")
        ),
        batch_text_input_rate=_coerce_optional_float(
            pricing.get("batch_text_input_rate")
        ),
        batch_text_output_rate=_coerce_optional_float(
            pricing.get("batch_text_output_rate")
        ),
        batch_text_cached_read_rate=_coerce_optional_float(
            pricing.get("batch_text_cached_read_rate")
        ),
        batch_text_cached_write_rate=_coerce_optional_float(
            pricing.get("batch_text_cached_write_rate")
        ),
        batch_image_input_rate=_coerce_optional_float(
            pricing.get("batch_image_input_rate")
        ),
        batch_image_output_rate=_coerce_optional_float(
            pricing.get("batch_image_output_rate")
        ),
        batch_image_cached_read_rate=_coerce_optional_float(
            pricing.get("batch_image_cached_read_rate")
        ),
        batch_image_cached_write_rate=_coerce_optional_float(
            pricing.get("batch_image_cached_write_rate")
        ),
        reference_image_output_costs=_coerce_reference_image_output_costs(
            pricing.get("reference_image_output_costs")
        ),
        partial_image_output_tokens=(
            None
            if pricing.get("partial_image_output_tokens") is None
            else int(pricing.get("partial_image_output_tokens"))
        ),
        currency=str(pricing.get("currency", "USD")),
        effective_from=pricing.get("effective_from"),
        effective_until=pricing.get("effective_until"),
        source_url=pricing.get("source_url"),
        last_verified_at=pricing.get("last_verified_at"),
    )


def _load_image_pricing_catalogue(path: Path) -> ImagePricingCatalogue:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict) or "entries" not in raw:
        raise ValueError(
            "IMAGE_PRICING.json must be a schema-v1 catalogue object with an 'entries' array."
        )

    schema_version = int(raw.get("schema_version", CATALOGUE_SCHEMA_VERSION))
    if schema_version != 1:
        raise ValueError(
            f"Unsupported IMAGE_PRICING schema_version {schema_version}; expected 1."
        )

    entries_raw = raw["entries"]
    catalogue_version = str(raw.get("catalogue_version", DEFAULT_CATALOGUE_VERSION))
    generated_at = str(raw.get("generated_at", DEFAULT_LAST_VERIFIED_AT))

    entries = tuple(
        _coerce_image_pricing(str(entry["canonical_model_id"]), entry)
        for entry in entries_raw
    )

    return ImagePricingCatalogue(
        schema_version=schema_version,
        catalogue_version=catalogue_version,
        generated_at=generated_at,
        entries=entries,
    )


PRICING_CATALOGUE = _load_pricing_catalogue(_pricing_file())
_BASE_PRICING_ENTRIES: dict[str, Pricing] = {
    entry.canonical_model_id: entry for entry in PRICING_CATALOGUE.entries
}
PRICING_OVERRIDES: dict[str, Pricing] = {}
PRICING_ALIASES: dict[str, str] = {}
PRICING: dict[str, Pricing] = {}
IMAGE_PRICING_CATALOGUE = _load_image_pricing_catalogue(_image_pricing_file())
_BASE_IMAGE_PRICING_ENTRIES: dict[str, ImagePricing] = {
    entry.canonical_model_id: entry for entry in IMAGE_PRICING_CATALOGUE.entries
}
IMAGE_PRICING_OVERRIDES: dict[str, ImagePricing] = {}
IMAGE_PRICING_ALIASES: dict[str, str] = {}
IMAGE_PRICING: dict[str, ImagePricing] = {}


def _resolve_alias(model: str) -> str:
    resolved = model
    seen: set[str] = set()

    while resolved in PRICING_ALIASES:
        if resolved in seen:
            break
        seen.add(resolved)
        resolved = PRICING_ALIASES[resolved]

    return resolved


def _resolve_pricing(model: str) -> Pricing:
    if model in PRICING_OVERRIDES:
        return PRICING_OVERRIDES[model]

    resolved = _resolve_alias(model)
    if resolved in PRICING_OVERRIDES:
        return PRICING_OVERRIDES[resolved]

    if resolved in _BASE_PRICING_ENTRIES:
        return _BASE_PRICING_ENTRIES[resolved]

    if model in _BASE_PRICING_ENTRIES:
        return _BASE_PRICING_ENTRIES[model]

    known = ", ".join(
        sorted(
            set(_BASE_PRICING_ENTRIES) | set(PRICING_OVERRIDES) | set(PRICING_ALIASES)
        )
    )
    raise KeyError(f"No pricing registered for model {model!r}. Known models: {known}")


def _resolve_image_pricing(model: str) -> ImagePricing:
    if model in IMAGE_PRICING_OVERRIDES:
        return IMAGE_PRICING_OVERRIDES[model]

    resolved = model
    seen: set[str] = set()
    while resolved in IMAGE_PRICING_ALIASES:
        if resolved in seen:
            break
        seen.add(resolved)
        resolved = IMAGE_PRICING_ALIASES[resolved]

    if resolved in IMAGE_PRICING_OVERRIDES:
        return IMAGE_PRICING_OVERRIDES[resolved]

    if resolved in _BASE_IMAGE_PRICING_ENTRIES:
        return _BASE_IMAGE_PRICING_ENTRIES[resolved]

    if model in _BASE_IMAGE_PRICING_ENTRIES:
        return _BASE_IMAGE_PRICING_ENTRIES[model]

    known = ", ".join(
        sorted(
            set(_BASE_IMAGE_PRICING_ENTRIES)
            | set(IMAGE_PRICING_OVERRIDES)
            | set(IMAGE_PRICING_ALIASES)
        )
    )
    raise KeyError(
        f"No image pricing registered for model {model!r}. Known models: {known}"
    )


def _refresh_pricing_index() -> None:
    PRICING.clear()

    for model, entry in _BASE_PRICING_ENTRIES.items():
        PRICING[model] = entry

    for model, entry in PRICING_OVERRIDES.items():
        PRICING[model] = entry

    for alias, target_model in PRICING_ALIASES.items():
        PRICING[alias] = _resolve_pricing(target_model)

    IMAGE_PRICING.clear()

    for model, entry in _BASE_IMAGE_PRICING_ENTRIES.items():
        IMAGE_PRICING[model] = entry

    for model, entry in IMAGE_PRICING_OVERRIDES.items():
        IMAGE_PRICING[model] = entry

    for alias, target_model in IMAGE_PRICING_ALIASES.items():
        IMAGE_PRICING[alias] = _resolve_image_pricing(target_model)


def register_pricing(model: str, pricing: Pricing | dict[str, Any]) -> None:
    PRICING_OVERRIDES[model] = _coerce_pricing(model, pricing)
    _refresh_pricing_index()


def register_image_pricing(model: str, pricing: ImagePricing | dict[str, Any]) -> None:
    IMAGE_PRICING_OVERRIDES[model] = _coerce_image_pricing(model, pricing)
    _refresh_pricing_index()


def register_pricing_alias(alias: str, target_model: str) -> None:
    PRICING_ALIASES[alias] = target_model
    _refresh_pricing_index()


def register_image_pricing_alias(alias: str, target_model: str) -> None:
    IMAGE_PRICING_ALIASES[alias] = target_model
    _refresh_pricing_index()


def get_pricing(model: str) -> Pricing:
    return _resolve_pricing(model)


def get_image_pricing(model: str) -> ImagePricing:
    return _resolve_image_pricing(model)


_DIMENSION_PATTERN = re.compile(r"^(\d+)x(\d+)$")


def _normalise_quality(quality: str) -> str:
    return quality.strip().lower()


def _validate_gpt_image_2_size(size: str) -> None:
    if size == "auto":
        return

    match = _DIMENSION_PATTERN.fullmatch(size)
    if match is None:
        raise ValueError(
            "gpt-image-2 size must be 'auto' or WIDTHxHEIGHT with positive integers."
        )

    width = int(match.group(1))
    height = int(match.group(2))
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive integers.")
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("gpt-image-2 dimensions must be multiples of 16.")
    if max(width, height) > 3840:
        raise ValueError("gpt-image-2 maximum edge is 3840 pixels.")

    ratio = max(width / height, height / width)
    if ratio > 3.0:
        raise ValueError("gpt-image-2 maximum aspect ratio is 3:1.")

    pixels = width * height
    if pixels < 655_360 or pixels > 8_294_400:
        raise ValueError(
            "gpt-image-2 total pixel count must be between 655360 and 8294400."
        )


def validate_image_size_for_model(model: str, size: str) -> None:
    pricing = get_image_pricing(model)
    canonical_model_id = pricing.canonical_model_id
    if canonical_model_id == "gpt-image-2":
        _validate_gpt_image_2_size(size)


def _select_image_pricing_rates(
    pricing: ImagePricing,
    *,
    pricing_mode: Literal["standard", "batch"] = "standard",
) -> tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    if pricing_mode == "batch":
        text_input_rate = (
            pricing.batch_text_input_rate
            if pricing.batch_text_input_rate is not None
            else pricing.text_input_rate
        )
        text_output_rate = (
            pricing.batch_text_output_rate
            if pricing.batch_text_output_rate is not None
            else pricing.text_output_rate
        )
        text_cached_read_rate = (
            pricing.batch_text_cached_read_rate
            if pricing.batch_text_cached_read_rate is not None
            else pricing.text_cached_read_rate
        )
        image_input_rate = (
            pricing.batch_image_input_rate
            if pricing.batch_image_input_rate is not None
            else pricing.image_input_rate
        )
        image_output_rate = (
            pricing.batch_image_output_rate
            if pricing.batch_image_output_rate is not None
            else pricing.image_output_rate
        )
        image_cached_read_rate = (
            pricing.batch_image_cached_read_rate
            if pricing.batch_image_cached_read_rate is not None
            else pricing.image_cached_read_rate
        )
    else:
        text_input_rate = pricing.text_input_rate
        text_output_rate = pricing.text_output_rate
        text_cached_read_rate = pricing.text_cached_read_rate
        image_input_rate = pricing.image_input_rate
        image_output_rate = pricing.image_output_rate
        image_cached_read_rate = pricing.image_cached_read_rate

    if text_cached_read_rate is None:
        text_cached_read_rate = text_input_rate
    if image_cached_read_rate is None:
        image_cached_read_rate = image_input_rate

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


def _strict_rate_cost(tokens: int, rate: Optional[float], *, label: str) -> float:
    if tokens <= 0:
        return 0.0
    if rate is None:
        raise ValueError(
            f"No pricing rate is available for {label} when tokens are present."
        )
    return (tokens / 1_000_000) * rate


def _resolve_partial_image_output_tokens(
    pricing: ImagePricing,
    *,
    partial_image_output_tokens: Optional[int],
    partial_image_count: Optional[int],
) -> int:
    if partial_image_output_tokens is not None:
        return partial_image_output_tokens

    if partial_image_count is None:
        return 0

    if pricing.partial_image_output_tokens is None:
        return 0

    return partial_image_count * pricing.partial_image_output_tokens


def _reference_per_image_cost(
    pricing: ImagePricing,
    *,
    size: str,
    quality: str,
) -> float:
    if size == "auto":
        raise ValueError(
            "Offline image estimates require an explicit listed size, not 'auto'."
        )

    quality_key = _normalise_quality(quality)
    references = pricing.reference_image_output_costs or {}
    if quality_key not in references:
        known_qualities = ", ".join(sorted(references))
        raise KeyError(
            f"No reference image output pricing for quality {quality!r} on model "
            f"{pricing.canonical_model_id!r}. Known qualities: {known_qualities}"
        )

    quality_table = references[quality_key]
    if size not in quality_table:
        known_sizes = ", ".join(sorted(quality_table))
        raise KeyError(
            f"No reference image output pricing for size {size!r} at quality "
            f"{quality!r} on model {pricing.canonical_model_id!r}. "
            f"Known sizes: {known_sizes}"
        )

    return quality_table[size]


def _reference_image_output_cost(
    pricing: ImagePricing,
    *,
    size: str,
    quality: str,
    image_count: int,
    pricing_mode: Literal["standard", "batch"],
) -> float:
    reference_per_image = _reference_per_image_cost(
        pricing,
        size=size,
        quality=quality,
    )

    if pricing_mode == "batch":
        if pricing.batch_image_output_rate is None:
            raise ValueError(
                f"No batch image output rate is available for model {pricing.canonical_model_id!r}."
            )
        if pricing.image_output_rate is None or pricing.image_output_rate <= 0:
            raise ValueError(
                f"No standard image output rate is available for model {pricing.canonical_model_id!r}."
            )
        reference_per_image *= (
            pricing.batch_image_output_rate / pricing.image_output_rate
        )

    return reference_per_image * image_count


def _calculate_exact_image_cost(
    *,
    model: str,
    pricing: ImagePricing,
    size: Optional[str],
    quality: Optional[str],
    image_count: int,
    text_input_tokens: int,
    cached_text_input_tokens: int,
    text_output_tokens: int,
    image_input_tokens: int,
    cached_image_input_tokens: int,
    image_output_tokens: int,
    partial_image_output_tokens: Optional[int],
    partial_image_count: Optional[int],
    pricing_mode: Literal["standard", "batch"],
) -> ImageCostEstimate:
    (
        text_input_rate,
        text_output_rate,
        text_cached_read_rate,
        image_input_rate,
        image_output_rate,
        image_cached_read_rate,
    ) = _select_image_pricing_rates(pricing, pricing_mode=pricing_mode)

    partial_tokens = _resolve_partial_image_output_tokens(
        pricing,
        partial_image_output_tokens=partial_image_output_tokens,
        partial_image_count=partial_image_count,
    )
    billable_image_output_tokens = max(image_output_tokens - partial_tokens, 0)

    text_input_cost_usd = _strict_rate_cost(
        text_input_tokens,
        text_input_rate,
        label=f"text input pricing for model {pricing.canonical_model_id!r}",
    )
    cached_text_input_cost_usd = _strict_rate_cost(
        cached_text_input_tokens,
        text_cached_read_rate,
        label=f"cached text input pricing for model {pricing.canonical_model_id!r}",
    )
    text_output_cost_usd = _strict_rate_cost(
        text_output_tokens,
        text_output_rate,
        label=f"text output pricing for model {pricing.canonical_model_id!r}",
    )
    image_input_cost_usd = _strict_rate_cost(
        image_input_tokens,
        image_input_rate,
        label=f"image input pricing for model {pricing.canonical_model_id!r}",
    )
    cached_image_input_cost_usd = _strict_rate_cost(
        cached_image_input_tokens,
        image_cached_read_rate,
        label=f"cached image input pricing for model {pricing.canonical_model_id!r}",
    )
    image_output_cost_usd = _strict_rate_cost(
        billable_image_output_tokens,
        image_output_rate,
        label=f"image output pricing for model {pricing.canonical_model_id!r}",
    )
    partial_image_output_cost_usd = _strict_rate_cost(
        partial_tokens,
        image_output_rate,
        label=f"partial image output pricing for model {pricing.canonical_model_id!r}",
    )

    total_cost_usd = (
        text_input_cost_usd
        + cached_text_input_cost_usd
        + text_output_cost_usd
        + image_input_cost_usd
        + cached_image_input_cost_usd
        + image_output_cost_usd
        + partial_image_output_cost_usd
    )

    return ImageCostEstimate(
        model=model,
        size=size,
        quality=quality,
        image_count=image_count,
        pricing_mode=pricing_mode,
        cost_per_image_usd=0.0,
        text_input_tokens=text_input_tokens,
        cached_text_input_tokens=cached_text_input_tokens,
        text_output_tokens=text_output_tokens,
        image_input_tokens=image_input_tokens,
        cached_image_input_tokens=cached_image_input_tokens,
        image_output_tokens=billable_image_output_tokens,
        partial_image_output_tokens=partial_tokens,
        text_input_cost_usd=text_input_cost_usd,
        cached_text_input_cost_usd=cached_text_input_cost_usd,
        text_output_cost_usd=text_output_cost_usd,
        image_input_cost_usd=image_input_cost_usd,
        cached_image_input_cost_usd=cached_image_input_cost_usd,
        image_output_cost_usd=image_output_cost_usd,
        partial_image_output_cost_usd=partial_image_output_cost_usd,
        token_based_cost_usd=total_cost_usd,
        reference_image_output_cost_usd=0.0,
        total_cost_usd=total_cost_usd,
    )


def estimate_image_cost(
    *,
    model: str,
    size: str,
    quality: str,
    image_count: int = 1,
    text_input_tokens: int = 0,
    cached_text_input_tokens: int = 0,
    text_output_tokens: int = 0,
    image_input_tokens: int = 0,
    cached_image_input_tokens: int = 0,
    image_output_tokens: int = 0,
    partial_image_output_tokens: Optional[int] = None,
    partial_image_count: Optional[int] = None,
    pricing_mode: Literal["standard", "batch"] = "standard",
) -> ImageCostEstimate:
    if image_count < 1:
        raise ValueError("image_count must be >= 1")

    if text_input_tokens < 0:
        raise ValueError("text_input_tokens must be >= 0")
    if cached_text_input_tokens < 0:
        raise ValueError("cached_text_input_tokens must be >= 0")
    if text_output_tokens < 0:
        raise ValueError("text_output_tokens must be >= 0")
    if image_input_tokens < 0:
        raise ValueError("image_input_tokens must be >= 0")
    if cached_image_input_tokens < 0:
        raise ValueError("cached_image_input_tokens must be >= 0")

    pricing = get_image_pricing(model)
    validate_image_size_for_model(model, size)

    (
        text_input_rate,
        _text_output_rate,
        text_cached_read_rate,
        image_input_rate,
        _image_output_rate,
        image_cached_read_rate,
    ) = _select_image_pricing_rates(pricing, pricing_mode=pricing_mode)

    text_input_cost_usd = _strict_rate_cost(
        text_input_tokens,
        text_input_rate,
        label=f"text input pricing for model {pricing.canonical_model_id!r}",
    )
    cached_text_input_cost_usd = _strict_rate_cost(
        cached_text_input_tokens,
        text_cached_read_rate,
        label=f"cached text input pricing for model {pricing.canonical_model_id!r}",
    )
    image_input_cost_usd = _strict_rate_cost(
        image_input_tokens,
        image_input_rate,
        label=f"image input pricing for model {pricing.canonical_model_id!r}",
    )
    cached_image_input_cost_usd = _strict_rate_cost(
        cached_image_input_tokens,
        image_cached_read_rate,
        label=f"cached image input pricing for model {pricing.canonical_model_id!r}",
    )

    reference_total = _reference_image_output_cost(
        pricing,
        size=size,
        quality=quality,
        image_count=image_count,
        pricing_mode=pricing_mode,
    )

    return ImageCostEstimate(
        model=model,
        size=size,
        quality=quality,
        image_count=image_count,
        pricing_mode=pricing_mode,
        cost_per_image_usd=(reference_total / image_count if image_count else 0.0),
        text_input_tokens=text_input_tokens,
        cached_text_input_tokens=cached_text_input_tokens,
        text_output_tokens=text_output_tokens,
        image_input_tokens=image_input_tokens,
        cached_image_input_tokens=cached_image_input_tokens,
        image_output_tokens=0,
        partial_image_output_tokens=0,
        text_input_cost_usd=text_input_cost_usd,
        cached_text_input_cost_usd=cached_text_input_cost_usd,
        text_output_cost_usd=0.0,
        image_input_cost_usd=image_input_cost_usd,
        cached_image_input_cost_usd=cached_image_input_cost_usd,
        image_output_cost_usd=0.0,
        partial_image_output_cost_usd=0.0,
        token_based_cost_usd=(
            text_input_cost_usd
            + cached_text_input_cost_usd
            + image_input_cost_usd
            + cached_image_input_cost_usd
        ),
        reference_image_output_cost_usd=reference_total,
        total_cost_usd=(
            reference_total
            + text_input_cost_usd
            + cached_text_input_cost_usd
            + image_input_cost_usd
            + cached_image_input_cost_usd
        ),
    )


def _as_dict_or_none(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return None


def _extract_int(mapping: dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping.get(key)
        if value is None:
            continue
        return int(value)
    return None


def normalise_image_usage(
    usage: ImageUsage | dict[str, Any] | Any | None,
) -> ImageUsage:
    if usage is None:
        return ImageUsage()
    if isinstance(usage, ImageUsage):
        return usage

    usage_map = _as_dict_or_none(usage)
    if usage_map is None:
        return ImageUsage()

    input_details = _as_dict_or_none(usage_map.get("input_tokens_details")) or {}
    output_details = _as_dict_or_none(usage_map.get("output_tokens_details")) or {}

    text_input_tokens = _extract_int(
        usage_map,
        "text_input_tokens",
    )
    cached_text_input_tokens = _extract_int(
        usage_map,
        "cached_text_input_tokens",
    )
    text_output_tokens = _extract_int(
        usage_map,
        "text_output_tokens",
    )

    image_input_tokens = _extract_int(
        usage_map,
        "image_input_tokens",
    )
    cached_image_input_tokens = _extract_int(
        usage_map,
        "cached_image_input_tokens",
    )
    image_output_tokens = _extract_int(
        usage_map,
        "image_output_tokens",
    )
    partial_image_output_tokens = _extract_int(
        usage_map,
        "partial_image_output_tokens",
        "streamed_partial_image_output_tokens",
    )
    partial_image_count = _extract_int(
        usage_map,
        "partial_image_count",
        "streamed_partial_image_count",
    )

    if text_input_tokens is None:
        text_input_tokens = _extract_int(
            input_details, "text_tokens", "text_input_tokens"
        )
    if cached_text_input_tokens is None:
        cached_text_input_tokens = _extract_int(
            input_details,
            "cached_text_tokens",
            "cached_text_input_tokens",
        )
    if text_output_tokens is None:
        text_output_tokens = _extract_int(
            output_details,
            "text_tokens",
            "text_output_tokens",
        )

    if image_input_tokens is None:
        image_input_tokens = _extract_int(
            input_details,
            "image_tokens",
            "image_input_tokens",
        )
    if cached_image_input_tokens is None:
        cached_image_input_tokens = _extract_int(
            input_details,
            "cached_image_tokens",
            "cached_image_input_tokens",
        )
    if image_output_tokens is None:
        image_output_tokens = _extract_int(
            output_details,
            "image_tokens",
            "image_output_tokens",
        )
    if partial_image_output_tokens is None:
        partial_image_output_tokens = _extract_int(
            output_details,
            "partial_image_tokens",
            "partial_image_output_tokens",
        )
    if partial_image_count is None:
        partial_image_count = _extract_int(
            output_details,
            "partial_image_count",
        )

    input_tokens = _extract_int(usage_map, "input_tokens")
    cached_input_tokens = _extract_int(
        usage_map,
        "cached_input_tokens",
        "cache_read_input_tokens",
    )
    output_tokens = _extract_int(usage_map, "output_tokens")
    total_tokens = _extract_int(usage_map, "total_tokens")

    if cached_input_tokens is None:
        cached_input_tokens = (cached_text_input_tokens or 0) + (
            cached_image_input_tokens or 0
        )

    if output_tokens is None:
        output_tokens = (text_output_tokens or 0) + (image_output_tokens or 0)

    if total_tokens is None:
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        else:
            total_tokens = None

    return ImageUsage(
        text_input_tokens=text_input_tokens,
        cached_text_input_tokens=cached_text_input_tokens,
        text_output_tokens=text_output_tokens,
        image_input_tokens=image_input_tokens,
        cached_image_input_tokens=cached_image_input_tokens,
        image_output_tokens=image_output_tokens,
        partial_image_output_tokens=partial_image_output_tokens,
        partial_image_count=partial_image_count,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def cost_for_image_usage(
    *,
    model: str,
    usage: ImageUsage | dict[str, Any] | Any,
    size: Optional[str] = None,
    quality: Optional[str] = None,
    image_count: int = 1,
    pricing_mode: Literal["standard", "batch"] = "standard",
) -> ImageCostEstimate:
    normalised_usage = normalise_image_usage(usage)

    pricing = get_image_pricing(model)
    usable_image_output_tokens = (
        normalised_usage.image_output_tokens
        if normalised_usage.image_output_tokens is not None
        else normalised_usage.output_tokens
    )

    if usable_image_output_tokens is None:
        if size is None or quality is None:
            raise ValueError(
                "size and quality are required when image-output usage is unavailable."
            )
        input_only_cost_usd = _calculate_exact_image_cost(
            model=model,
            pricing=pricing,
            size=size,
            quality=quality,
            image_count=image_count,
            text_input_tokens=normalised_usage.text_input_tokens or 0,
            cached_text_input_tokens=normalised_usage.cached_text_input_tokens or 0,
            text_output_tokens=normalised_usage.text_output_tokens or 0,
            image_input_tokens=normalised_usage.image_input_tokens or 0,
            cached_image_input_tokens=normalised_usage.cached_image_input_tokens or 0,
            image_output_tokens=0,
            partial_image_output_tokens=None,
            partial_image_count=None,
            pricing_mode=pricing_mode,
        )
        reference_total = _reference_image_output_cost(
            pricing,
            size=size,
            quality=quality,
            image_count=image_count,
            pricing_mode=pricing_mode,
        )
        total_cost_usd = input_only_cost_usd.total_cost_usd + reference_total
        return ImageCostEstimate(
            model=model,
            size=size,
            quality=quality,
            image_count=image_count,
            pricing_mode=pricing_mode,
            cost_per_image_usd=(reference_total / image_count if image_count else 0.0),
            text_input_tokens=input_only_cost_usd.text_input_tokens,
            cached_text_input_tokens=input_only_cost_usd.cached_text_input_tokens,
            text_output_tokens=input_only_cost_usd.text_output_tokens,
            image_input_tokens=input_only_cost_usd.image_input_tokens,
            cached_image_input_tokens=input_only_cost_usd.cached_image_input_tokens,
            image_output_tokens=0,
            partial_image_output_tokens=0,
            text_input_cost_usd=input_only_cost_usd.text_input_cost_usd,
            cached_text_input_cost_usd=input_only_cost_usd.cached_text_input_cost_usd,
            text_output_cost_usd=input_only_cost_usd.text_output_cost_usd,
            image_input_cost_usd=input_only_cost_usd.image_input_cost_usd,
            cached_image_input_cost_usd=input_only_cost_usd.cached_image_input_cost_usd,
            image_output_cost_usd=0.0,
            partial_image_output_cost_usd=0.0,
            token_based_cost_usd=input_only_cost_usd.token_based_cost_usd,
            reference_image_output_cost_usd=reference_total,
            total_cost_usd=total_cost_usd,
        )

    exact_image_output_tokens = usable_image_output_tokens
    exact_estimate = _calculate_exact_image_cost(
        model=model,
        pricing=pricing,
        size=size,
        quality=quality,
        image_count=image_count,
        text_input_tokens=normalised_usage.text_input_tokens or 0,
        cached_text_input_tokens=normalised_usage.cached_text_input_tokens or 0,
        text_output_tokens=normalised_usage.text_output_tokens or 0,
        image_input_tokens=normalised_usage.image_input_tokens or 0,
        cached_image_input_tokens=normalised_usage.cached_image_input_tokens or 0,
        image_output_tokens=exact_image_output_tokens or 0,
        partial_image_output_tokens=normalised_usage.partial_image_output_tokens,
        partial_image_count=normalised_usage.partial_image_count,
        pricing_mode=pricing_mode,
    )

    return exact_estimate


def cost_for_image_response(
    *,
    response: ImageResponse,
    size: Optional[str] = None,
    quality: Optional[str] = None,
    image_count: Optional[int] = None,
    pricing_mode: Literal["standard", "batch"] = "standard",
) -> ImageCostEstimate:
    return cost_for_image_usage(
        model=response.model,
        usage=response.usage or ImageUsage(),
        size=size,
        quality=quality,
        image_count=(len(response.artifacts) if image_count is None else image_count),
        pricing_mode=pricing_mode,
    )


def _select_pricing_rates(
    pricing: Pricing,
    *,
    pricing_mode: Literal["standard", "batch"] = "standard",
    context_tokens: Optional[int] = None,
) -> tuple[float, float, float, float]:
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
    cached_write_rate = pricing.cached_write_rate or input_rate
    return input_rate, output_rate, cached_read_rate, cached_write_rate


def cost_from_tokens(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    pricing: Pricing,
    pricing_mode: Literal["standard", "batch"] = "standard",
    context_tokens: Optional[int] = None,
) -> CostEstimate:
    if input_tokens < 0:
        raise ValueError("input_tokens must be >= 0")

    if output_tokens < 0:
        raise ValueError("output_tokens must be >= 0")

    if cached_input_tokens < 0:
        raise ValueError("cached_input_tokens must be >= 0")

    if cache_creation_input_tokens < 0:
        raise ValueError("cache_creation_input_tokens must be >= 0")

    if cache_write_tokens < 0:
        raise ValueError("cache_write_tokens must be >= 0")

    input_rate, output_rate, cached_rate, cache_write_rate = _select_pricing_rates(
        pricing,
        pricing_mode=pricing_mode,
        context_tokens=context_tokens,
    )

    input_cost_usd = (input_tokens / 1_000_000) * input_rate
    output_cost_usd = (output_tokens / 1_000_000) * output_rate
    cache_creation_input_cost_usd = (
        cache_creation_input_tokens / 1_000_000
    ) * input_rate
    cache_write_input_cost_usd = (cache_write_tokens / 1_000_000) * cache_write_rate
    cached_input_cost_usd = (cached_input_tokens / 1_000_000) * cached_rate

    total_cost_usd = (
        input_cost_usd
        + output_cost_usd
        + cached_input_cost_usd
        + cache_creation_input_cost_usd
        + cache_write_input_cost_usd
    )

    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_write_tokens=cache_write_tokens,
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        cached_input_cost_usd=cached_input_cost_usd,
        cache_creation_input_cost_usd=cache_creation_input_cost_usd,
        cache_write_input_cost_usd=cache_write_input_cost_usd,
        total_cost_usd=total_cost_usd,
    )


def cost_from_usage(
    *,
    usage: ChatUsage,
    pricing: Pricing,
    cached_input_tokens: Optional[int] = None,
    cache_creation_input_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    pricing_mode: Literal["standard", "batch"] = "standard",
    context_tokens: Optional[int] = None,
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
        cache_write_tokens=(0 if cache_write_tokens is None else cache_write_tokens),
        pricing=pricing,
        pricing_mode=pricing_mode,
        context_tokens=context_tokens,
    )


def cost_for_model(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    pricing_mode: Literal["standard", "batch"] = "standard",
    context_tokens: Optional[int] = None,
) -> CostEstimate:
    pricing = get_pricing(model)
    return cost_from_tokens(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_write_tokens=cache_write_tokens,
        pricing=pricing,
        pricing_mode=pricing_mode,
        context_tokens=context_tokens,
    )


def cost_for_response(
    *,
    model: str,
    usage: ChatUsage,
    cached_input_tokens: Optional[int] = None,
    cache_creation_input_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    pricing_mode: Literal["standard", "batch"] = "standard",
    context_tokens: Optional[int] = None,
) -> CostEstimate:
    pricing = get_pricing(model)
    return cost_from_usage(
        usage=usage,
        pricing=pricing,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_write_tokens=cache_write_tokens,
        pricing_mode=pricing_mode,
        context_tokens=context_tokens,
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
    _emit(f"Cache write tokens: {estimate.cache_write_tokens}")
    _emit(f"Input cost: {format_cost(estimate.input_cost_usd)}")
    _emit(f"Output cost: {format_cost(estimate.output_cost_usd)}")
    _emit(f"Cached input cost: {format_cost(estimate.cached_input_cost_usd)}")
    _emit(
        f"Cache creation input cost: {format_cost(estimate.cache_creation_input_cost_usd)}"
    )
    _emit(f"Cache write cost: {format_cost(estimate.cache_write_input_cost_usd)}")
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
        f"Cache creation input tokens: {estimate.cache_creation_input_tokens} "
        f"Cache write tokens: {estimate.cache_write_tokens}"
    )
    _emit(
        f"Cached input cost: {format_cost(estimate.cached_input_cost_usd)} "
        f"Cache creation input cost: {format_cost(estimate.cache_creation_input_cost_usd)} "
        f"Cache write cost: {format_cost(estimate.cache_write_input_cost_usd)} "
        f"Total cost: {format_cost(estimate.total_cost_usd)}"
    )
    _emit("---")


def print_image_cost_breakdown(
    *,
    estimate: ImageCostEstimate,
    model: Optional[str] = None,
) -> None:
    label = model or estimate.model
    _emit(f"Model: {label}")
    _emit(f"Size: {estimate.size}")
    _emit(f"Quality: {estimate.quality}")
    _emit(f"Image count: {estimate.image_count}")
    _emit(f"Pricing mode: {estimate.pricing_mode}")
    _emit(f"Text input tokens: {estimate.text_input_tokens}")
    _emit(f"Cached text input tokens: {estimate.cached_text_input_tokens}")
    _emit(f"Text output tokens: {estimate.text_output_tokens}")
    _emit(f"Image input tokens: {estimate.image_input_tokens}")
    _emit(f"Cached image input tokens: {estimate.cached_image_input_tokens}")
    _emit(f"Image output tokens: {estimate.image_output_tokens}")
    _emit(f"Partial image output tokens: {estimate.partial_image_output_tokens}")
    _emit(f"Reference cost per image: {format_cost(estimate.cost_per_image_usd)}")
    _emit(
        "Reference image output cost: "
        f"{format_cost(estimate.reference_image_output_cost_usd)}"
    )
    _emit(f"Text input cost: {format_cost(estimate.text_input_cost_usd)}")
    _emit(f"Cached text input cost: {format_cost(estimate.cached_text_input_cost_usd)}")
    _emit(f"Text output cost: {format_cost(estimate.text_output_cost_usd)}")
    _emit(f"Image input cost: {format_cost(estimate.image_input_cost_usd)}")
    _emit(
        f"Cached image input cost: {format_cost(estimate.cached_image_input_cost_usd)}"
    )
    _emit(f"Image output cost: {format_cost(estimate.image_output_cost_usd)}")
    _emit(
        "Partial image output cost: "
        f"{format_cost(estimate.partial_image_output_cost_usd)}"
    )
    _emit(f"Token-based cost: {format_cost(estimate.token_based_cost_usd)}")
    _emit(f"Total cost: {format_cost(estimate.total_cost_usd)}")


def print_image_cost_summary(
    *,
    estimate: ImageCostEstimate,
    model: Optional[str] = None,
) -> None:
    label = model or estimate.model
    _emit("---")
    _emit(
        f"Model: {label} | Size: {estimate.size} | Quality: {estimate.quality} | "
        f"Images: {estimate.image_count} | Pricing mode: {estimate.pricing_mode}"
    )
    _emit(
        f"Reference output cost: {format_cost(estimate.reference_image_output_cost_usd)} "
        f"Token-based cost: {format_cost(estimate.token_based_cost_usd)} "
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
    cache_write_tokens: Optional[int] = None,
    pricing_mode: Literal["standard", "batch"] = "standard",
    context_tokens: Optional[int] = None,
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
        cache_write_tokens=cache_write_tokens,
        pricing_mode=pricing_mode,
        context_tokens=context_tokens,
    )


_refresh_pricing_index()
