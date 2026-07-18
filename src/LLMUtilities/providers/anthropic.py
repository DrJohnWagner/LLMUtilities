from __future__ import annotations

from typing import Any, Optional

from ..config import settings
from ..exceptions import (
    AuthenticationError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)
from ..types import ChatRequest, ChatResponse, ChatUsage
from .base import BaseChatModel

_DEFAULT_MAX_TOKENS = 8192


class AnthropicChatModel(BaseChatModel):
    """Anthropic chat adapter."""

    provider_name = "anthropic"
    default_model = settings.anthropic.chat_model
    api_key = settings.anthropic_api_key

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to Anthropic and return a normalised ChatResponse."""
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise MissingDependencyError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "Install it with: pip install anthropic"
            ) from exc

        request = self.prepare_request(request)
        model_name = self.get_model_name(request)
        api_key = self.require_api_key()

        client = _anthropic.Anthropic(
            api_key=api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

        # Anthropic keeps system prompts separate from the messages list.
        # Use explicit block-level prompt caching on the final static system block.
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, str]] = []

        for msg in request.messages:
            content_str = _normalize_content(msg.content)
            if msg.role == "system":
                system_parts.append(content_str)
            else:
                anthropic_messages.append({"role": msg.role, "content": content_str})

        max_tokens = (
            request.max_output_tokens
            or settings.anthropic.max_output_tokens
            or _DEFAULT_MAX_TOKENS
        )

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }

        if system_parts:
            system_text = "\n\n".join(system_parts)
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        try:
            response = client.messages.create(**kwargs)
        except _anthropic.AuthenticationError as exc:
            raise AuthenticationError(
                f"Anthropic authentication failed: {exc}"
            ) from exc
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

        try:
            text = _extract_text(response)
            usage = _extract_usage(response)

            return ChatResponse(
                text=text,
                provider=self.provider_name,
                model=model_name,
                usage=usage,
                stop_reason=getattr(response, "stop_reason", None),
                raw=response,
            )
        except (ResponseError, ValueError):
            raise
        except Exception as exc:
            raise ResponseError(f"Failed to parse Anthropic response: {exc}") from exc


def _extract_text(response: Any) -> str:
    """
    Concatenate all text content blocks from an Anthropic response in order.

    Ignores non-text blocks (e.g. tool_use). Raises ``ResponseError`` if
    ``response.content`` is missing or yields no text at all.
    """
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


def _normalize_content(content: Any) -> str:
    """
    Normalize message content to a string for Anthropic API.

    Handles both backward-compatible string content and new multimodal content parts.
    For multimodal content, extracts all text parts and concatenates them.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    text_parts.append(part["text"])
            elif (
                hasattr(part, "type") and part.type == "text" and hasattr(part, "text")
            ):
                text_parts.append(part.text)
        return " ".join(text_parts) if text_parts else ""

    return str(content)


def _extract_usage(response: Any) -> ChatUsage:
    """
    Normalise Anthropic usage into the package's ``ChatUsage`` schema.

    Anthropic usage may include:
    - input_tokens
    - output_tokens
    - cache_creation_input_tokens
    - cache_read_input_tokens
    """
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return ChatUsage()

    input_tokens = getattr(usage_obj, "input_tokens", None)
    output_tokens = getattr(usage_obj, "output_tokens", None)
    cache_creation_input_tokens = getattr(
        usage_obj, "cache_creation_input_tokens", None
    )
    cache_read_input_tokens = getattr(usage_obj, "cache_read_input_tokens", None)

    total = None
    if input_tokens is not None or output_tokens is not None:
        total = (input_tokens or 0) + (output_tokens or 0)

    return ChatUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cached_input_tokens=cache_read_input_tokens,
    )
