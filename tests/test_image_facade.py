from __future__ import annotations

import pytest

from LLMUtilities.exceptions import ResponseError, UnsupportedCapabilityError
from LLMUtilities.image import generate_image, generate_image_b64
from LLMUtilities.types import ImageArtifact, ImageRequest, ImageResponse


class _FakeImageProvider:
    name = "fake"

    def __init__(self, artifact: ImageArtifact):
        self._artifact = artifact

    def generate_image(self, request: ImageRequest) -> ImageResponse:
        return ImageResponse(
            provider="fake",
            requested_model=request.model,
            resolved_model=request.model or "fake-image-model",
            artifacts=[self._artifact],
            usage=None,
            raw=None,
        )


class _NotAnImageProvider:
    name = "not-image"


def test_generate_image_delegates_to_provider():
    provider = _FakeImageProvider(ImageArtifact(b64_data="abc123"))
    response = generate_image(provider=provider, prompt="a cat")
    assert response.artifacts[0].b64_data == "abc123"


def test_generate_image_b64_returns_b64_data():
    provider = _FakeImageProvider(ImageArtifact(b64_data="abc123"))
    assert generate_image_b64(provider=provider, prompt="a cat") == "abc123"


def test_generate_image_b64_raises_when_only_url_present():
    provider = _FakeImageProvider(ImageArtifact(url="https://example.com/cat.png"))
    with pytest.raises(ResponseError):
        generate_image_b64(provider=provider, prompt="a cat")


def test_generate_image_rejects_provider_without_capability():
    with pytest.raises(UnsupportedCapabilityError):
        generate_image(provider=_NotAnImageProvider(), prompt="a cat")
