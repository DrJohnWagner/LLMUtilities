from __future__ import annotations

from typing import Any, Optional

from .capabilities import ImageGenerationCapability, require_capability
from .chat import resolve_provider
from .exceptions import ResponseError
from .types import ImageRequest, ImageResponse


def generate_image(
    *,
    provider: Any = None,
    provider_name: Optional[str] = None,
    prompt: str,
    model: Optional[str] = None,
    size: Optional[str] = None,
    quality: Optional[str] = None,
    background: Optional[str] = None,
    format: Optional[str] = None,
    n: int = 1,
    seed: Optional[int] = None,
    user: Optional[str] = None,
) -> ImageResponse:
    request = ImageRequest(
        prompt=prompt,
        model=model,
        size=size,
        quality=quality,
        background=background,
        format=format,
        n=n,
        seed=seed,
        user=user,
    )

    resolved_provider = resolve_provider(provider, provider_name)
    require_capability(
        resolved_provider, ImageGenerationCapability, "ImageGenerationCapability"
    )
    return resolved_provider.generate_image(request)


def generate_image_b64(
    *,
    provider: Any = None,
    provider_name: Optional[str] = None,
    prompt: str,
    model: Optional[str] = None,
    size: Optional[str] = None,
    quality: Optional[str] = None,
    background: Optional[str] = None,
    format: Optional[str] = None,
    n: int = 1,
    seed: Optional[int] = None,
    user: Optional[str] = None,
) -> str:
    response = generate_image(
        provider=provider,
        provider_name=provider_name,
        prompt=prompt,
        model=model,
        size=size,
        quality=quality,
        background=background,
        format=format,
        n=n,
        seed=seed,
        user=user,
    )

    first = response.artifacts[0]
    if first.b64_data:
        return first.b64_data

    raise ResponseError(
        "Image artifact did not include base64 data. Request URL output or use generate_image()."
    )
