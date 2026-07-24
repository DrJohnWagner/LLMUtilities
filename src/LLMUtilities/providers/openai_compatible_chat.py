from __future__ import annotations

from typing import Any, Optional

from ..exceptions import ConfigurationError, ResponseError
from ..transports.openai_chat_completions import (
    OpenAIChatCompletionsTransport,
    extract_text,
    first_choice,
)
from ..types import ChatRequest, ChatResponse, CommonUsage
from ..utils import content_to_text

"""
Shared chat logic for OpenAI Chat Completions-compatible providers (Moonshot,
DeepSeek). Only the transport's client construction (base_url, api_key) and
the provider's own pricing/model catalogue differ between them; message
framing, text extraction and usage mapping are identical because they all
speak the same Chat Completions wire protocol.
"""


def resolve_model_name(
    request: ChatRequest, default_model: Optional[str], provider_name: str
) -> str:
    model_name = request.model or default_model
    if not model_name:
        raise ConfigurationError(f"No chat model configured for provider {provider_name!r}.")
    return model_name


def run_chat(
    *,
    transport: OpenAIChatCompletionsTransport,
    request: ChatRequest,
    model_name: str,
    provider_name: str,
) -> ChatResponse:
    system_parts: list[str] = []
    input_messages: list[dict[str, str]] = []

    for message in request.messages:
        content_str = content_to_text(message.content)
        if message.role == "system":
            system_parts.append(content_str)
        else:
            input_messages.append({"role": message.role, "content": content_str})

    if system_parts:
        input_messages = [
            {"role": "system", "content": "\n\n".join(system_parts)}
        ] + input_messages

    kwargs: dict[str, Any] = {"model": model_name, "messages": input_messages}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        kwargs["max_tokens"] = request.max_output_tokens

    response = transport.create_completion(**kwargs)

    try:
        choice = first_choice(response)
        text = extract_text(choice)
        usage = extract_usage_totals(response)
        resolved_model = getattr(response, "model", None) or model_name

        return ChatResponse(
            text=text,
            provider=provider_name,
            requested_model=request.model,
            resolved_model=resolved_model,
            usage=CommonUsage(
                total_input_tokens=usage[0],
                total_output_tokens=usage[1],
                total_tokens=usage[2],
            ),
            stop_reason=getattr(choice, "finish_reason", None),
            raw=response,
        )
    except (ResponseError, ValueError):
        raise
    except Exception as exc:
        raise ResponseError(
            f"Failed to parse {provider_name} response: {exc}"
        ) from exc


def extract_usage_totals(response: Any) -> tuple[int, int, int, int]:
    """Returns (input_tokens, output_tokens, total_tokens, cached_input_tokens)."""
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return (0, 0, 0, 0)

    input_tokens = getattr(usage_obj, "prompt_tokens", None) or 0
    output_tokens = getattr(usage_obj, "completion_tokens", None) or 0
    total_tokens = getattr(usage_obj, "total_tokens", None) or (
        input_tokens + output_tokens
    )
    cached_input_tokens = _extract_cached_input_tokens(usage_obj)

    return (input_tokens, output_tokens, total_tokens, cached_input_tokens)


def _extract_cached_input_tokens(usage_obj: Any) -> int:
    prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
    if prompt_details is None:
        return getattr(usage_obj, "cached_tokens", None) or 0

    cached_tokens = getattr(prompt_details, "cached_tokens", None)
    return cached_tokens or 0
