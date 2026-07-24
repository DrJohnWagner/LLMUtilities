from __future__ import annotations

import pytest

from LLMUtilities.capabilities import (
    ChatCapability,
    EmbeddingCapability,
    ImageGenerationCapability,
    TokenCountingCapability,
    require_capability,
)
from LLMUtilities.exceptions import UnsupportedCapabilityError
from LLMUtilities.providers.registry import get_provider


def test_anthropic_does_not_implement_image_generation():
    anthropic = get_provider("anthropic")
    with pytest.raises(UnsupportedCapabilityError, match="anthropic.*ImageGenerationCapability"):
        require_capability(anthropic, ImageGenerationCapability, "ImageGenerationCapability")


def test_anthropic_does_not_implement_embedding():
    anthropic = get_provider("anthropic")
    with pytest.raises(UnsupportedCapabilityError):
        require_capability(anthropic, EmbeddingCapability, "EmbeddingCapability")


def test_moonshot_only_implements_chat():
    moonshot = get_provider("moonshot")
    require_capability(moonshot, ChatCapability, "ChatCapability")
    for capability, name in [
        (ImageGenerationCapability, "ImageGenerationCapability"),
        (EmbeddingCapability, "EmbeddingCapability"),
        (TokenCountingCapability, "TokenCountingCapability"),
    ]:
        with pytest.raises(UnsupportedCapabilityError):
            require_capability(moonshot, capability, name)


def test_openai_implements_every_capability():
    openai = get_provider("openai")
    require_capability(openai, ChatCapability, "ChatCapability")
    require_capability(openai, ImageGenerationCapability, "ImageGenerationCapability")
    require_capability(openai, EmbeddingCapability, "EmbeddingCapability")
    require_capability(openai, TokenCountingCapability, "TokenCountingCapability")
