"""
LLMUtilities - a provider-neutral Python interface over several inconsistent
LLM provider SDKs.

Public entry points::

    from LLMUtilities import chat, chat_text, list_providers, get_provider
    from LLMUtilities.types import ChatRequest, ChatResponse, Message, CostSummary
    from LLMUtilities.costs import get_cost_summary
    from LLMUtilities.embeddings import embed_texts, cosine_similarity
"""

from .chat import chat, chat_text
from .config import get_settings, reload_settings
from .costs import get_cost_summary
from .embeddings import cosine_similarity, embed_texts
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    CostCalculationUnavailableError,
    LLMUtilitiesError,
    MissingDependencyError,
    PricingUnavailableError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
    ResponseFormatError,
    UnsupportedCapabilityError,
)
from .image import generate_image, generate_image_b64
from .providers.registry import get_provider, list_providers
from .tokens import count_chat_request_tokens, count_message_tokens, count_text_tokens
from .types import (
    ChatRequest,
    ChatResponse,
    CommonUsage,
    CostSummary,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageArtifact,
    ImageRequest,
    ImageResponse,
    Message,
)

__all__ = [
    # providers
    "list_providers",
    "get_provider",
    # chat
    "chat",
    "chat_text",
    # image
    "generate_image",
    "generate_image_b64",
    # embeddings
    "embed_texts",
    "cosine_similarity",
    # tokens
    "count_text_tokens",
    "count_message_tokens",
    "count_chat_request_tokens",
    # costs
    "get_cost_summary",
    # config
    "get_settings",
    "reload_settings",
    # types
    "ChatRequest",
    "ChatResponse",
    "CommonUsage",
    "CostSummary",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ImageArtifact",
    "ImageRequest",
    "ImageResponse",
    "Message",
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
    "UnsupportedCapabilityError",
    "PricingUnavailableError",
    "CostCalculationUnavailableError",
]
