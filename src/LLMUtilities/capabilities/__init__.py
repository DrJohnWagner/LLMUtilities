from __future__ import annotations

from typing import Any, Type

from ..exceptions import UnsupportedCapabilityError
from .chat import ChatCapability
from .embedding import EmbeddingCapability
from .image_generation import ImageGenerationCapability
from .token_counting import TokenCountingCapability, TokenCountResult


def require_capability(
    provider: Any,
    capability: Type[Any],
    capability_name: str,
) -> None:
    """
    Raise ``UnsupportedCapabilityError`` unless ``provider`` implements ``capability``.

    Providers only need to implement the methods a capability protocol declares;
    ``isinstance`` against a ``@runtime_checkable`` Protocol checks that structurally.
    """
    if not isinstance(provider, capability):
        provider_name = getattr(provider, "name", type(provider).__name__)
        raise UnsupportedCapabilityError(
            f"Provider {provider_name!r} does not implement {capability_name!r}."
        )


__all__ = [
    "ChatCapability",
    "EmbeddingCapability",
    "ImageGenerationCapability",
    "TokenCountingCapability",
    "TokenCountResult",
    "require_capability",
]
