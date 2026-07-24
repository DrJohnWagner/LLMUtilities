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
    from openai import OpenAI as _OpenAIClient
    from openai import APIConnectionError as _OAIConnectionError
    from openai import APIStatusError as _OAIStatusError
    from openai import AuthenticationError as _OAIAuthError
    from openai import RateLimitError as _OAIRateLimitError
except ImportError:
    _OpenAIClient = None


class OpenAIResponsesTransport:
    """
    Reusable transport mechanics for the OpenAI ``responses`` API.

    Owns client construction, request submission and SDK exception translation.
    Does not interpret usage, calculate cost or apply provider-specific validation.
    """

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        if _OpenAIClient is None:
            raise MissingDependencyError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            )
        self._client = _OpenAIClient(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def create_response(self, **kwargs: Any) -> Any:
        try:
            return self._client.responses.create(**kwargs)
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
            raise RequestError(f"Unexpected OpenAI request failure: {exc}") from exc

    @property
    def client(self) -> Any:
        return self._client


def extract_text(response: Any) -> str:
    """
    Collect all assistant text segments from a ``responses`` output in order.
    """
    output = getattr(response, "output", None)
    if output is None:
        raise ResponseError("OpenAI response is missing the 'output' field.")

    segments: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if content is None:
            continue
        for part in content:
            if getattr(part, "type", None) == "output_text":
                text_val = getattr(part, "text", None)
                if text_val:
                    segments.append(text_val)

    if not segments:
        raise ResponseError(
            "OpenAI response contained no text output. "
            f"Response status: {getattr(response, 'status', 'unknown')!r}"
        )

    return "".join(segments)


def is_available() -> bool:
    return _OpenAIClient is not None
