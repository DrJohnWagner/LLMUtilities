from __future__ import annotations

from typing import Sequence

from ...capabilities.token_counting import TokenCountResult
from ...transports.google_generate_content import GoogleGenerateContentTransport
from ...types import Message
from ...utils import content_to_text


def count_text_tokens(
    *, transport: GoogleGenerateContentTransport, text: str, model: str
) -> TokenCountResult:
    response = transport.count_tokens(model=model, contents=text)
    return TokenCountResult(
        count=response.total_tokens,
        provider="google",
        model=model,
        method="provider_reported",
    )


def count_message_tokens(
    *,
    transport: GoogleGenerateContentTransport,
    messages: Sequence[Message],
    model: str,
) -> TokenCountResult:
    contents: list[dict[str, object]] = []
    system_parts: list[str] = []

    for message in messages:
        if message.role == "system":
            system_parts.append(content_to_text(message.content))
            continue

        role = "user" if message.role == "user" else "model"
        contents.append(
            {"role": role, "parts": [{"text": content_to_text(message.content)}]}
        )

    total_tokens = 0
    if contents:
        response = transport.count_tokens(model=model, contents=contents)
        total_tokens += response.total_tokens

    if system_parts:
        response = transport.count_tokens(
            model=model, contents="\n\n".join(system_parts)
        )
        total_tokens += response.total_tokens

    return TokenCountResult(
        count=total_tokens, provider="google", model=model, method="provider_reported"
    )
