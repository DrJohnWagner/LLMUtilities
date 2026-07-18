from __future__ import annotations

from typing import Any, Optional

from ..config import settings
from ..costs import normalise_image_usage, validate_image_size_for_model
from ..exceptions import (
    AuthenticationError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)
from ..types import ImageArtifact, ImageRequest, ImageResponse, ImageUsage
from .base_image import BaseImageModel


class OpenAIImageModel(BaseImageModel):
    provider_name = "openai"
    default_model = settings.openai.image_model or "gpt-image-1.5"
    api_key = settings.openai_api_key

    def generate(self, request: ImageRequest) -> ImageResponse:
        try:
            from openai import OpenAI
            from openai import AuthenticationError as _OAIAuthError
            from openai import RateLimitError as _OAIRateLimitError
            from openai import APIConnectionError as _OAIConnectionError
            from openai import APIStatusError as _OAIStatusError
        except ImportError as exc:
            raise MissingDependencyError(
                "The 'openai' package is required for OpenAI image generation. "
                "Install it with: pip install openai"
            ) from exc

        request = self.prepare_request(request)
        model_name = self.get_model_name(request)
        api_key = self.require_api_key()

        client = OpenAI(
            api_key=api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

        kwargs: dict[str, Any] = {
            "model": model_name,
            "prompt": request.prompt,
            "n": request.n,
        }
        resolved_size = request.size or settings.openai.image_size
        resolved_quality = request.quality or settings.openai.image_quality
        resolved_background = request.background or settings.openai.image_background
        resolved_format = request.format or settings.openai.image_format

        if resolved_size is not None:
            validate_image_size_for_model(model_name, resolved_size)
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
            usage = _extract_usage(response)
            return ImageResponse(
                provider=self.provider_name,
                model=model_name,
                artifacts=artifacts,
                usage=usage,
                raw=response,
            )
        except (ResponseError, ValueError):
            raise
        except Exception as exc:
            raise ResponseError(f"Failed to parse OpenAI image response: {exc}") from exc


def _extract_artifacts(response: Any, requested_format: Optional[str]) -> list[ImageArtifact]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not data:
        raise ResponseError("OpenAI image response contained no data artifacts.")

    mime_type = f"image/{requested_format}" if requested_format else "image/png"
    artifacts: list[ImageArtifact] = []
    for item in data:
        if isinstance(item, dict):
            b64_data = item.get("b64_json")
            url = item.get("url")
            revised_prompt = item.get("revised_prompt")
        else:
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


def _extract_usage(response: Any) -> ImageUsage:
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None and isinstance(response, dict):
        usage_obj = response.get("usage")
    if usage_obj is None:
        return ImageUsage()

    return normalise_image_usage(usage_obj)
