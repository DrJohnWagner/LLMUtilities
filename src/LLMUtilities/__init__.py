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
    cost_for_image_response,
    cost_for_image_usage,
    estimate_cost,
    estimate_image_cost,
    get_image_pricing,
    get_pricing,
    ImagePricingCatalogue,
    ImagePricing,
    ImageCostEstimate,
    Pricing,
    PricingCatalogue,
    print_image_cost_breakdown,
    print_image_cost_summary,
    register_image_pricing,
    register_image_pricing_alias,
    register_pricing,
    validate_image_size_for_model,
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
    "cost_for_image_usage",
    "cost_for_image_response",
    "get_pricing",
    "get_image_pricing",
    "validate_image_size_for_model",
    "Pricing",
    "PricingCatalogue",
    "ImagePricingCatalogue",
    "ImagePricing",
    "ImageCostEstimate",
    "register_pricing",
    "register_image_pricing",
    "register_image_pricing_alias",
    "print_image_cost_breakdown",
    "print_image_cost_summary",
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
