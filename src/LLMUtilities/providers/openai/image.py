from __future__ import annotations

import re
from typing import Any, Optional

from ...exceptions import (
    AuthenticationError,
    ConfigurationError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)
from ...types import ImageArtifact, ImageRequest, ImageResponse
from .pricing import OpenAIImageUsageDetails, get_image_pricing

try:
    from openai import OpenAI as _OpenAIClient
    from openai import APIConnectionError as _OAIConnectionError
    from openai import APIStatusError as _OAIStatusError
    from openai import AuthenticationError as _OAIAuthError
    from openai import RateLimitError as _OAIRateLimitError
except ImportError:
    _OpenAIClient = None

_DIMENSION_PATTERN = re.compile(r"(\d+)x(\d+)")


def is_available() -> bool:
    return _OpenAIClient is not None


def resolve_model_name(request: ImageRequest, default_model: Optional[str]) -> str:
    model_name = request.model or default_model
    if not model_name:
        raise ConfigurationError("No image model configured for provider 'openai'.")
    return model_name


def list_image_models() -> list[str]:
    from .pricing import list_image_pricings

    return sorted({entry.canonical_model_id for entry in list_image_pricings()})


def validate_image_size(model_name: str, size: str) -> None:
    pricing = get_image_pricing(model_name)
    if pricing.canonical_model_id == "gpt-image-2":
        _validate_gpt_image_2_size(size)


def _validate_gpt_image_2_size(size: str) -> None:
    if size == "auto":
        return

    match = _DIMENSION_PATTERN.fullmatch(size)
    if match is None:
        raise ValueError(
            "gpt-image-2 size must be 'auto' or WIDTHxHEIGHT with positive integers."
        )

    width, height = int(match.group(1)), int(match.group(2))
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


def run_generate(
    *,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    request: ImageRequest,
    model_name: str,
    default_size: Optional[str],
    default_quality: Optional[str],
    default_background: Optional[str],
    default_format: Optional[str],
) -> ImageResponse:
    if _OpenAIClient is None:
        raise MissingDependencyError(
            "The 'openai' package is required for OpenAI image generation. "
            "Install it with: pip install openai"
        )

    client = _OpenAIClient(
        api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
    )

    resolved_size = request.size or default_size
    resolved_quality = request.quality or default_quality
    resolved_background = request.background or default_background
    resolved_format = request.format or default_format

    kwargs: dict[str, Any] = {
        "model": model_name,
        "prompt": request.prompt,
        "n": request.n,
    }
    if resolved_size is not None:
        validate_image_size(model_name, resolved_size)
        kwargs["size"] = resolved_size
    if resolved_quality is not None:
        kwargs["quality"] = resolved_quality
    if resolved_background is not None:
        kwargs["background"] = resolved_background
    if resolved_format is not None:
        kwargs["output_format"] = resolved_format
    if request.user is not None:
        kwargs["user"] = request.user

    try:
        response = client.images.generate(**kwargs)
    except _OAIAuthError as exc:
        raise AuthenticationError(f"OpenAI authentication failed: {exc}") from exc
    except _OAIRateLimitError as exc:
        raise RateLimitError(f"OpenAI rate limit exceeded: {exc}") from exc
    except _OAIConnectionError as exc:
        raise RequestError(f"OpenAI connection error: {exc}") from exc
    except _OAIStatusError as exc:
        raise ProviderError(
            f"OpenAI image API error ({exc.status_code}): {exc.message}"
        ) from exc
    except Exception as exc:
        raise RequestError(f"Unexpected OpenAI image request failure: {exc}") from exc

    try:
        artifacts = _extract_artifacts(response, resolved_format)
        return ImageResponse(
            provider="openai",
            requested_model=request.model,
            resolved_model=model_name,
            artifacts=artifacts,
            usage=None,
            raw=response,
        )
    except (ResponseError, ValueError):
        raise
    except Exception as exc:
        raise ResponseError(f"Failed to parse OpenAI image response: {exc}") from exc


def _extract_artifacts(
    response: Any, requested_format: Optional[str]
) -> list[ImageArtifact]:
    data = getattr(response, "data", None)
    if not data:
        raise ResponseError("OpenAI image response contained no data artifacts.")

    mime_type = f"image/{requested_format}" if requested_format else "image/png"
    artifacts: list[ImageArtifact] = []
    for item in data:
        b64_data = getattr(item, "b64_json", None)
        url = getattr(item, "url", None)
        revised_prompt = getattr(item, "revised_prompt", None)

        if not b64_data and not url:
            continue

        artifacts.append(
            ImageArtifact(
                mime_type=mime_type,
                b64_data=b64_data,
                url=url,
                revised_prompt=revised_prompt,
            )
        )

    if not artifacts:
        raise ResponseError("OpenAI image response artifacts were empty.")

    return artifacts


def extract_usage_details(response: Any) -> OpenAIImageUsageDetails:
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return OpenAIImageUsageDetails()

    input_details = getattr(usage_obj, "input_tokens_details", None)
    text_input = getattr(input_details, "text_tokens", None) or 0
    image_input = getattr(input_details, "image_tokens", None) or 0

    return OpenAIImageUsageDetails(
        text_input_tokens=text_input,
        image_input_tokens=image_input,
        text_output_tokens=0,
        image_output_tokens=getattr(usage_obj, "output_tokens", None) or 0,
    )
