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
    import anthropic as _anthropic
except ImportError:
    _anthropic = None


class AnthropicMessagesTransport:
    """Reusable transport mechanics for the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        if _anthropic is None:
            raise MissingDependencyError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "Install it with: pip install anthropic"
            )
        self._client = _anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def create_message(self, **kwargs: Any) -> Any:
        try:
            return self._client.messages.create(**kwargs)
        except _anthropic.AuthenticationError as exc:
            raise AuthenticationError(f"Anthropic authentication failed: {exc}") from exc
        except _anthropic.RateLimitError as exc:
            raise RateLimitError(f"Anthropic rate limit exceeded: {exc}") from exc
        except _anthropic.APIConnectionError as exc:
            raise RequestError(f"Anthropic connection error: {exc}") from exc
        except _anthropic.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic API error ({exc.status_code}): {exc.message}"
            ) from exc
        except Exception as exc:
            raise RequestError(f"Unexpected Anthropic request failure: {exc}") from exc

    def count_tokens(self, **kwargs: Any) -> Any:
        try:
            return self._client.messages.count_tokens(**kwargs)
        except _anthropic.AuthenticationError as exc:
            raise AuthenticationError(f"Anthropic authentication failed: {exc}") from exc
        except _anthropic.RateLimitError as exc:
            raise RateLimitError(f"Anthropic rate limit exceeded: {exc}") from exc
        except Exception as exc:
            raise RequestError(
                f"Unexpected Anthropic token-count request failure: {exc}"
            ) from exc

    @property
    def client(self) -> Any:
        return self._client


def extract_text(response: Any) -> str:
    """Concatenate all text content blocks from an Anthropic response in order."""
    content = getattr(response, "content", None)
    if content is None:
        raise ResponseError("Anthropic response is missing the 'content' field.")

    segments: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            text_val = getattr(block, "text", None)
            if text_val:
                segments.append(text_val)

    if not segments:
        raise ResponseError(
            "Anthropic response contained no text content blocks. "
            f"Stop reason: {getattr(response, 'stop_reason', 'unknown')!r}"
        )

    return "".join(segments)


def is_available() -> bool:
    return _anthropic is not None
