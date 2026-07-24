from __future__ import annotations

from typing import Sequence

from ...capabilities.token_counting import TokenCountResult
from ...transports.anthropic_messages import AnthropicMessagesTransport
from ...types import Message
from ...utils import content_to_text


def count_text_tokens(
    *,
    transport: AnthropicMessagesTransport,
    text: str,
    model: str,
) -> TokenCountResult:
    response = transport.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return TokenCountResult(
        count=response.input_tokens,
        provider="anthropic",
        model=model,
        method="provider_reported",
    )


def count_message_tokens(
    *,
    transport: AnthropicMessagesTransport,
    messages: Sequence[Message],
    model: str,
) -> TokenCountResult:
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, str]] = []

    for message in messages:
        if message.role == "system":
            system_parts.append(content_to_text(message.content))
            continue
        anthropic_messages.append(
            {"role": message.role, "content": content_to_text(message.content)}
        )

    if not anthropic_messages:
        anthropic_messages.append({"role": "user", "content": ""})

    response = transport.count_tokens(
        model=model,
        system="\n\n".join(system_parts) if system_parts else None,
        messages=anthropic_messages,
    )
    return TokenCountResult(
        count=response.input_tokens,
        provider="anthropic",
        model=model,
        method="provider_reported",
    )
