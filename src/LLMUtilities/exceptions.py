class LLMUtilitiesError(Exception):
    """
    Base exception for all LLMUtilities errors.
    """
    pass


class ConfigurationError(LLMUtilitiesError):
    """
    Raised when required configuration is missing or invalid.
    """
    pass


class AuthenticationError(LLMUtilitiesError):
    """
    Raised when a provider rejects authentication credentials.
    """
    pass


class RateLimitError(LLMUtilitiesError):
    """
    Raised when a provider rate limit is hit.
    """
    pass


class RequestError(LLMUtilitiesError):
    """
    Raised when a request to a provider fails before a valid response is returned.
    """
    pass


class ResponseError(LLMUtilitiesError):
    """
    Raised when a provider response is malformed or unusable.
    """
    pass


class ResponseFormatError(ResponseError):
    """
    Raised when a provider response cannot be parsed into the expected format.
    """
    pass


class ProviderError(LLMUtilitiesError):
    """
    Raised for provider-specific failures not covered by a more specific exception.
    """
    pass


class MissingDependencyError(LLMUtilitiesError):
    """
    Raised when an optional provider SDK is required but not installed.

    The message always names the missing package and the affected provider.
    """
    pass