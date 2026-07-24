class LLMUtilitiesError(Exception):
    """Base exception for all LLMUtilities errors."""


class ConfigurationError(LLMUtilitiesError):
    """Raised when required configuration is missing or invalid."""


class AuthenticationError(LLMUtilitiesError):
    """Raised when a provider rejects authentication credentials."""


class RateLimitError(LLMUtilitiesError):
    """Raised when a provider rate limit is hit."""


class RequestError(LLMUtilitiesError):
    """Raised when a request to a provider fails before a valid response is returned."""


class ResponseError(LLMUtilitiesError):
    """Raised when a provider response is malformed or unusable."""


class ResponseFormatError(ResponseError):
    """Raised when a provider response cannot be parsed into the expected format."""


class ProviderError(LLMUtilitiesError):
    """Raised for provider-specific failures not covered by a more specific exception."""


class MissingDependencyError(LLMUtilitiesError):
    """
    Raised when an optional provider SDK is required but not installed.

    The message always names the missing package and the affected provider.
    """


class UnsupportedCapabilityError(LLMUtilitiesError):
    """
    Raised when a provider is asked to perform an operation for a capability
    it does not implement, e.g. image generation on a chat-only provider.
    """


class PricingUnavailableError(LLMUtilitiesError):
    """Raised when no pricing record can be resolved for a requested model."""


class CostCalculationUnavailableError(LLMUtilitiesError):
    """
    Raised when a cost cannot be calculated for an otherwise valid response,
    e.g. usage information is missing from the provider response.
    """
