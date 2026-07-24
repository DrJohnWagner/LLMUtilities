from __future__ import annotations

from typing import Any, Optional

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


class OpenAIChatCompletionsTransport:
    """
    Reusable transport mechanics for OpenAI Chat Completions-compatible endpoints.

    Used directly by ``OpenAIProvider`` in compatibility mode and composed by
    Moonshot and DeepSeek, which speak the Chat Completions protocol through
    the OpenAI Python SDK against a different ``base_url``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str],
        timeout_seconds: float,
        max_retries: int,
        display_name: str,
    ) -> None:
        if _OpenAIClient is None:
            raise MissingDependencyError(
                "The 'openai' package is required for this provider. "
                "Install it with: pip install openai"
            )
        self._display_name = display_name
        self._client = _OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def create_completion(self, **kwargs: Any) -> Any:
        try:
            return self._client.chat.completions.create(**kwargs)
        except _OAIAuthError as exc:
            raise AuthenticationError(
                f"{self._display_name} authentication failed: {exc}"
            ) from exc
        except _OAIRateLimitError as exc:
            raise RateLimitError(
                f"{self._display_name} rate limit exceeded: {exc}"
            ) from exc
        except _OAIConnectionError as exc:
            raise RequestError(
                f"{self._display_name} connection error: {exc}"
            ) from exc
        except _OAIStatusError as exc:
            raise ProviderError(
                f"{self._display_name} API error ({exc.status_code}): {exc.message}"
            ) from exc
        except Exception as exc:
            raise RequestError(
                f"Unexpected {self._display_name} request failure: {exc}"
            ) from exc

    @property
    def client(self) -> Any:
        return self._client


def first_choice(response: Any) -> Any:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ResponseError("Chat completion response contained no choices.")
    return choices[0]


def extract_text(choice: Any) -> str:
    message = getattr(choice, "message", None)
    if message is None:
        raise ResponseError("Chat completion choice is missing the message field.")

    content = getattr(message, "content", None)
    text = _collect_text(content)
    if not text:
        raise ResponseError("Chat completion choice contained no text output.")
    return text


def _collect_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        segments: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    segments.append(part["text"])
            elif getattr(part, "type", None) == "text" and getattr(part, "text", None):
                segments.append(part.text)
        return "".join(segments)

    if content is None:
        return ""

    return str(content)


def is_available() -> bool:
    return _OpenAIClient is not None
