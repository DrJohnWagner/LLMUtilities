from __future__ import annotations

from typing import Any, Optional

from ...exceptions import ConfigurationError, ResponseError
from ...transports.anthropic_messages import AnthropicMessagesTransport, extract_text
from ...types import ChatRequest, ChatResponse, CommonUsage
from ...utils import content_to_text
from .pricing import AnthropicChatUsageDetails

_DEFAULT_MAX_TOKENS = 8192


def resolve_model_name(request: ChatRequest, default_model: Optional[str]) -> str:
    model_name = request.model or default_model
    if not model_name:
        raise ConfigurationError("No chat model configured for provider 'anthropic'.")
    return model_name


def run_chat(
    *,
    transport: AnthropicMessagesTransport,
    request: ChatRequest,
    model_name: str,
    default_max_output_tokens: Optional[int],
) -> ChatResponse:
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, str]] = []

    for message in request.messages:
        content_str = content_to_text(message.content)
        if message.role == "system":
            system_parts.append(content_str)
        else:
            anthropic_messages.append({"role": message.role, "content": content_str})

    max_tokens = (
        request.max_output_tokens or default_max_output_tokens or _DEFAULT_MAX_TOKENS
    )

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
    }

    if system_parts:
        kwargs["system"] = [
            {
                "type": "text",
                "text": "\n\n".join(system_parts),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    if request.temperature is not None:
        kwargs["temperature"] = request.temperature

    response = transport.create_message(**kwargs)

    try:
        text = extract_text(response)
        usage = extract_usage_details(response)
        resolved_model = getattr(response, "model", None) or model_name

        return ChatResponse(
            text=text,
            provider="anthropic",
            requested_model=request.model,
            resolved_model=resolved_model,
            usage=CommonUsage(
                total_input_tokens=usage.input_tokens
                + usage.cache_creation_input_tokens
                + usage.cache_read_input_tokens,
                total_output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            ),
            stop_reason=getattr(response, "stop_reason", None),
            raw=response,
        )
    except (ResponseError, ValueError):
        raise
    except Exception as exc:
        raise ResponseError(f"Failed to parse Anthropic response: {exc}") from exc


def extract_usage_details(response: Any) -> AnthropicChatUsageDetails:
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return AnthropicChatUsageDetails()

    input_tokens = getattr(usage_obj, "input_tokens", None) or 0
    output_tokens = getattr(usage_obj, "output_tokens", None) or 0
    cache_creation_input_tokens = (
        getattr(usage_obj, "cache_creation_input_tokens", None) or 0
    )
    cache_read_input_tokens = getattr(usage_obj, "cache_read_input_tokens", None) or 0

    total = (
        input_tokens
        + output_tokens
        + cache_creation_input_tokens
        + cache_read_input_tokens
    )

    return AnthropicChatUsageDetails(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        total_tokens=total,
    )
