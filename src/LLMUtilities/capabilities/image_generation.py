from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import ImageRequest, ImageResponse


@runtime_checkable
class ImageGenerationCapability(Protocol):
    def generate_image(self, request: ImageRequest) -> ImageResponse: ...
