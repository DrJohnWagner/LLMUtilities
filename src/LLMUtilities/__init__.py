"""
LLMUtilities — a small, provider-normalised LLM utility library.

Public entry points::

    from LLMUtilities.chat import chat, chat_text, chat_usage
    from LLMUtilities.types import ChatRequest, ChatResponse, Message, ChatUsage
    from LLMUtilities.costs import cost_for_response
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
    ImageContentPart,
    Message,
    TextContentPart,
)
from .chat import chat, chat_text, chat_usage
from .config import get_settings, reload_settings, settings
from .image import generate_image, generate_image_b64
from .costs import (
    CostEstimate,
    ImageCostEstimate,
    ImagePricing,
    ImagePricingCatalogue,
    Pricing,
    PricingCatalogue,
    cost_for_image_response,
    cost_for_image_usage,
    cost_for_model,
    cost_for_response,
    cost_from_tokens,
    cost_from_usage,
    estimate_image_cost,
    get_image_pricing,
    get_pricing,
    print_image_cost_breakdown,
    print_image_cost_summary,
    validate_image_size_for_model,
)
from .embeddings import cosine_similarity, embed_texts
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    LLMUtilitiesError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
    ResponseFormatError,
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
    # config
    "settings",
    "get_settings",
    "reload_settings",
    # image
    "generate_image",
    "generate_image_b64",
    # text costs
    "cost_from_tokens",
    "cost_from_usage",
    "cost_for_model",
    "cost_for_response",
    "get_pricing",
    "Pricing",
    "PricingCatalogue",
    "CostEstimate",
    # image costs
    "estimate_image_cost",
    "cost_for_image_usage",
    "cost_for_image_response",
    "get_image_pricing",
    "validate_image_size_for_model",
    "ImagePricing",
    "ImagePricingCatalogue",
    "ImageCostEstimate",
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
