"""
LLMUtilities — a small, provider-normalised LLM utility library.

Public entry points::

    from LLMUtilities.chat import chat, chat_text, chat_usage
    from LLMUtilities.types import ChatRequest, ChatResponse, Message, ChatUsage
    from LLMUtilities.costs import estimate_cost
    from LLMUtilities.embeddings import embed_texts, cosine_similarity
    from LLMUtilities.parsing.structured_output import structured_output
    from LLMUtilities.compare import compare_outputs
"""

from .types import (
    ChatRequest,
    ChatResponse,
    ChatUsage,
    ImageArtifact,
    ImageRequest,
    ImageResponse,
    ImageUsage,
    Message,
    TextContentPart,
    ImageContentPart,
)
from .chat import chat, chat_text, chat_usage
from .image import generate_image, generate_image_b64
from .costs import (
    estimate_cost,
    estimate_image_cost,
    ImagePricing,
    ImageCostEstimate,
)
from .embeddings import embed_texts, cosine_similarity
from .exceptions import (
    LLMUtilitiesError,
    ConfigurationError,
    AuthenticationError,
    MissingDependencyError,
    RateLimitError,
    RequestError,
    ResponseError,
    ResponseFormatError,
    ProviderError,
)

__all__ = [
    # types
    "ChatRequest",
    "ChatResponse",
    "ChatUsage",
    "Message",
    "TextContentPart",
    "ImageContentPart",
    "ImageRequest",
    "ImageResponse",
    "ImageArtifact",
    "ImageUsage",
    # chat
    "chat",
    "chat_text",
    "chat_usage",
    # image
    "generate_image",
    "generate_image_b64",
    # costs
    "estimate_cost",
    "estimate_image_cost",
    "ImagePricing",
    "ImageCostEstimate",
    # embeddings
    "embed_texts",
    "cosine_similarity",
    # exceptions
    "LLMUtilitiesError",
    "ConfigurationError",
    "AuthenticationError",
    "MissingDependencyError",
    "RateLimitError",
    "RequestError",
    "ResponseError",
    "ResponseFormatError",
    "ProviderError",
]
