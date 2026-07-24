from __future__ import annotations

from typing import Optional

from ...exceptions import (
    AuthenticationError,
    ConfigurationError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
    RequestError,
)
from ...types import EmbeddingRequest, EmbeddingResponse

try:
    from openai import OpenAI as _OpenAIClient
    from openai import APIConnectionError as _OAIConnectionError
    from openai import APIStatusError as _OAIStatusError
    from openai import AuthenticationError as _OAIAuthError
    from openai import RateLimitError as _OAIRateLimitError
except ImportError:
    _OpenAIClient = None

EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]


def is_available() -> bool:
    return _OpenAIClient is not None


def resolve_model_name(request: EmbeddingRequest, default_model: Optional[str]) -> str:
    model_name = request.model or default_model
    if not model_name:
        raise ConfigurationError(
            "No embedding model configured for provider 'openai'."
        )
    return model_name


def run_embed(
    *,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    request: EmbeddingRequest,
    model_name: str,
) -> EmbeddingResponse:
    if _OpenAIClient is None:
        raise MissingDependencyError(
            "The 'openai' package is required for OpenAI embeddings. "
            "Install it with: pip install openai"
        )

    client = _OpenAIClient(
        api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
    )

    kwargs: dict = {"input": request.texts, "model": model_name}
    if request.dimensions is not None:
        kwargs["dimensions"] = request.dimensions

    try:
        response = client.embeddings.create(**kwargs)
    except _OAIAuthError as exc:
        raise AuthenticationError(f"OpenAI authentication failed: {exc}") from exc
    except _OAIRateLimitError as exc:
        raise RateLimitError(f"OpenAI rate limit exceeded: {exc}") from exc
    except _OAIConnectionError as exc:
        raise RequestError(f"OpenAI connection error: {exc}") from exc
    except _OAIStatusError as exc:
        raise ProviderError(
            f"OpenAI API error ({exc.status_code}): {exc.message}"
        ) from exc
    except Exception as exc:
        raise RequestError(f"Unexpected OpenAI embedding request failure: {exc}") from exc

    vectors = [item.embedding for item in response.data]
    return EmbeddingResponse(
        provider="openai",
        requested_model=request.model,
        resolved_model=model_name,
        vectors=vectors,
        usage=None,
        raw=response,
    )
