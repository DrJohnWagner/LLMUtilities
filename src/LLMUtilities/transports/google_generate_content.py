from __future__ import annotations

from typing import Any

from ..exceptions import (
    AuthenticationError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)

try:
    from google import genai as _genai
    from google.genai import errors as _genai_errors
    from google.genai import types as _genai_types
except ImportError:
    _genai = None
    _genai_errors = None
    _genai_types = None


class GoogleGenerateContentTransport:
    """Reusable transport mechanics for Google's ``generate_content`` API."""

    def __init__(self, *, api_key: str) -> None:
        if _genai is None:
            raise MissingDependencyError(
                "The 'google-genai' package is required for the Google provider. "
                "Install it with: pip install google-genai"
            )
        self._client = _genai.Client(api_key=api_key)

    def generate_content(self, **kwargs: Any) -> Any:
        try:
            return self._client.models.generate_content(**kwargs)
        except _genai_errors.APIError as exc:
            _raise_provider_exception(exc)
        except Exception as exc:
            raise RequestError(f"Unexpected Google request failure: {exc}") from exc

    def count_tokens(self, **kwargs: Any) -> Any:
        try:
            return self._client.models.count_tokens(**kwargs)
        except _genai_errors.APIError as exc:
            _raise_provider_exception(exc)
        except Exception as exc:
            raise RequestError(
                f"Unexpected Google token-count request failure: {exc}"
            ) from exc

    def embed_content(self, **kwargs: Any) -> Any:
        try:
            return self._client.models.embed_content(**kwargs)
        except _genai_errors.APIError as exc:
            _raise_provider_exception(exc)
        except Exception as exc:
            raise RequestError(
                f"Unexpected Google embedding request failure: {exc}"
            ) from exc

    @property
    def client(self) -> Any:
        return self._client

    @property
    def types(self) -> Any:
        return _genai_types


def _raise_provider_exception(exc: Any) -> None:
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", str(exc))

    if code in (401, 403):
        raise AuthenticationError(f"Google authentication failed: {message}") from exc
    if code == 429:
        raise RateLimitError(f"Google rate limit or quota exceeded: {message}") from exc
    if code in (408, 500, 502, 503, 504):
        raise RequestError(f"Google request failed: {message}") from exc

    raise ProviderError(f"Google API error ({code}): {message}") from exc


def extract_text(response: Any) -> str:
    """Concatenate all text parts across all candidates in order."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise ResponseError("Google response contained no candidates.")

    segments: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text_val = getattr(part, "text", None)
            if text_val:
                segments.append(text_val)

    if not segments:
        raise ResponseError("Google response candidates contained no text parts.")

    return "".join(segments)


def is_available() -> bool:
    return _genai is not None
