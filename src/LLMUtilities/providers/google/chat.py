from __future__ import annotations

from typing import Any, Optional

from ...exceptions import ConfigurationError, ResponseError
from ...transports.google_generate_content import (
    GoogleGenerateContentTransport,
    extract_text,
)
from ...types import ChatRequest, ChatResponse, CommonUsage
from ...utils import content_to_text
from .pricing import GoogleChatUsageDetails


def resolve_model_name(request: ChatRequest, default_model: Optional[str]) -> str:
    model_name = request.model or default_model
    if not model_name:
        raise ConfigurationError("No chat model configured for provider 'google'.")
    return model_name


def _map_role(role: str) -> str:
    if role == "user":
        return "user"
    if role == "assistant":
        return "model"
    raise ValueError(f"Unsupported Google message role: {role!r}")


def _split_messages(
    request: ChatRequest, genai_types: Any
) -> tuple[Optional[str], list[Any]]:
    system_parts: list[str] = []
    contents: list[Any] = []

    for message in request.messages:
        content_str = content_to_text(message.content)
        if message.role == "system":
            system_parts.append(content_str)
            continue

        contents.append(
            genai_types.Content(
                role=_map_role(message.role),
                parts=[genai_types.Part(text=content_str)],
            )
        )

    if not contents:
        raise ValueError(
            "Google chat requests must include at least one non-system message."
        )

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def run_chat(
    *,
    transport: GoogleGenerateContentTransport,
    request: ChatRequest,
    model_name: str,
) -> ChatResponse:
    genai_types = transport.types
    system_instruction, contents = _split_messages(request, genai_types)

    config = None
    if (
        system_instruction is not None
        or request.temperature is not None
        or request.max_output_tokens is not None
    ):
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
        )

    response = transport.generate_content(
        model=model_name, contents=contents, config=config
    )

    try:
        text = extract_text(response)
        usage = extract_usage_details(response)
        stop_reason = _extract_stop_reason(response)
        resolved_model = getattr(response, "model_version", None) or model_name

        return ChatResponse(
            text=text,
            provider="google",
            requested_model=request.model,
            resolved_model=resolved_model,
            usage=CommonUsage(
                total_input_tokens=usage.input_tokens,
                total_output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            ),
            stop_reason=stop_reason,
            raw=response,
        )
    except (ResponseError, ValueError):
        raise
    except Exception as exc:
        raise ResponseError(f"Failed to parse Google response: {exc}") from exc


def extract_usage_details(response: Any) -> GoogleChatUsageDetails:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return GoogleChatUsageDetails()

    input_tokens = getattr(usage, "prompt_token_count", None) or 0
    output_tokens = getattr(usage, "candidates_token_count", None) or 0
    total_tokens = getattr(usage, "total_token_count", None) or (
        input_tokens + output_tokens
    )

    return GoogleChatUsageDetails(
        input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens
    )


def _extract_stop_reason(response: Any) -> Optional[str]:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    finish_reason = getattr(candidates[0], "finish_reason", None)
    return str(finish_reason) if finish_reason is not None else None
