from __future__ import annotations

from typing import Any, Optional

from ...exceptions import ConfigurationError, ResponseError
from ...transports.openai_responses import OpenAIResponsesTransport, extract_text
from ...types import ChatRequest, ChatResponse, CommonUsage
from ...utils import content_to_text
from .pricing import OpenAIChatUsageDetails


def build_input(request: ChatRequest) -> tuple[Optional[str], list[dict[str, str]]]:
    system_parts: list[str] = []
    input_messages: list[dict[str, str]] = []

    for message in request.messages:
        content_str = content_to_text(message.content)
        if message.role == "system":
            system_parts.append(content_str)
        else:
            input_messages.append({"role": message.role, "content": content_str})

    instructions = "\n\n".join(system_parts) if system_parts else None
    return instructions, input_messages


def run_chat(
    *,
    transport: OpenAIResponsesTransport,
    request: ChatRequest,
    model_name: str,
) -> ChatResponse:
    instructions, input_messages = build_input(request)

    kwargs: dict[str, Any] = {"model": model_name, "input": input_messages}
    if instructions is not None:
        kwargs["instructions"] = instructions
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        kwargs["max_output_tokens"] = request.max_output_tokens

    response = transport.create_response(**kwargs)

    try:
        text = extract_text(response)
        usage = extract_usage_details(response)
        resolved_model = getattr(response, "model", None) or model_name

        return ChatResponse(
            text=text,
            provider="openai",
            requested_model=request.model,
            resolved_model=resolved_model,
            usage=CommonUsage(
                total_input_tokens=usage.input_tokens,
                total_output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            ),
            stop_reason=None,
            raw=response,
        )
    except (ResponseError, ValueError):
        raise
    except Exception as exc:
        raise ResponseError(f"Failed to parse OpenAI response: {exc}") from exc


def extract_usage_details(response: Any) -> OpenAIChatUsageDetails:
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return OpenAIChatUsageDetails()

    input_tokens = getattr(usage_obj, "input_tokens", None) or 0
    output_tokens = getattr(usage_obj, "output_tokens", None) or 0
    total_tokens = getattr(usage_obj, "total_tokens", None) or (
        input_tokens + output_tokens
    )

    cached_input_tokens = 0
    input_details = getattr(usage_obj, "input_tokens_details", None)
    if input_details is not None:
        cached_input_tokens = getattr(input_details, "cached_tokens", None) or 0

    return OpenAIChatUsageDetails(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def resolve_model_name(request: ChatRequest, default_model: Optional[str]) -> str:
    model_name = request.model or default_model
    if not model_name:
        raise ConfigurationError("No chat model configured for provider 'openai'.")
    return model_name
