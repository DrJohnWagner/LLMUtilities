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


class OpenAIChatModel(BaseChatModel):
    """OpenAI chat adapter (uses the ``responses`` API)."""

    provider_name = "openai"
    default_model = settings.openai.chat_model
    api_key = settings.openai_api_key

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
        """Send a chat request via OpenAI ``responses.create`` and return a normalised ChatResponse."""
        try:
            from openai import OpenAI
            from openai import AuthenticationError as _OAIAuthError
            from openai import RateLimitError as _OAIRateLimitError
            from openai import APIConnectionError as _OAIConnectionError
            from openai import APIStatusError as _OAIStatusError
        except ImportError as exc:
            raise MissingDependencyError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            ) from exc

        request = self.prepare_request(request)
        model_name = self.get_model_name(request)
        api_key = self.require_api_key()

        client = OpenAI(
            api_key=api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

        # Separate system instructions from conversational turns.
        # The responses API accepts system content via the top-level
        # `instructions` parameter; user/assistant turns go in `input`.
        system_parts: list[str] = []
        input_messages: list[dict[str, str]] = []

        for msg in request.messages:
            content_str = _normalize_content(msg.content)
            if msg.role == "system":
                system_parts.append(content_str)
            else:
                input_messages.append({"role": msg.role, "content": content_str})

        kwargs: dict[str, Any] = {
            "model": model_name,
            "input": input_messages,
        }
        if system_parts:
            kwargs["instructions"] = "\n\n".join(system_parts)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens

        try:
            response = client.responses.create(**kwargs)
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

        try:
            text = _extract_text(response)
            usage = _extract_usage(response)
            # The responses API does not expose a per-completion finish reason
            # equivalent to chat.completions finish_reason; response.status is
            # request lifecycle state, not a stop reason, so we leave this None.
            stop_reason = None

            return ChatResponse(
                text=text,
                provider=self.provider_name,
                model=model_name,
                usage=usage,
                stop_reason=stop_reason,
                raw=response,
            )
        except (ResponseError, ValueError):
            raise
        except Exception as exc:
            raise ResponseError(f"Failed to parse OpenAI response: {exc}") from exc


def _extract_text(response: Any) -> str:
    """
    Collect all assistant text segments from a ``responses`` output in order.

    Iterates ``response.output`` items, then each item's ``content`` parts,
    and concatenates every ``output_text`` segment found.  Raises
    ``ResponseError`` if no text content is present.
    """
    output = getattr(response, "output", None)
    if output is None:
        raise ResponseError(
            "OpenAI response is missing the 'output' field."
        )

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


def _normalize_content(content: Any) -> str:
    """
    Normalize message content to a string for OpenAI API.

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
    Extract normalised token usage from an OpenAI ``responses`` result.

    The responses API exposes ``input_tokens`` and ``output_tokens`` directly,
    matching the package's normalised ``ChatUsage`` field names.
    """
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return ChatUsage()

    input_tokens = getattr(usage_obj, "input_tokens", None)
    output_tokens = getattr(usage_obj, "output_tokens", None)
    total_tokens = getattr(usage_obj, "total_tokens", None)

    # Compute total if the field is absent but the components are present.
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return ChatUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
