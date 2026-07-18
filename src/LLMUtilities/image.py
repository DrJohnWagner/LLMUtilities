from __future__ import annotations

from typing import Optional

from .config import settings
from .exceptions import ConfigurationError, ResponseError
from .types import ImageRequest, ImageResponse


def get_image_model(provider: Optional[str] = None):
    provider_name = (provider or settings.default_provider).strip().lower()

    if provider_name == "openai":
        from .providers.openai_image import OpenAIImageModel

        return OpenAIImageModel()

    if provider_name in {"anthropic", "google"}:
        raise ConfigurationError(
            f"Image generation is not implemented for provider {provider_name!r}."
        )

    raise ConfigurationError(
        f"Unsupported image provider: {provider_name!r}. "
        f"Expected one of: 'openai', 'anthropic', 'google'."
    )


def generate_image(
    *,
    provider=None,
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

    resolved_provider = provider or get_image_model(provider_name)
    return resolved_provider.generate(request)


def generate_image_b64(
    *,
    provider=None,
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
