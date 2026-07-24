from __future__ import annotations

from typing import Any, Optional, Sequence

from .capabilities import TokenCountingCapability, TokenCountResult, require_capability
from .chat import build_messages, resolve_provider
from .types import Message


def count_text_tokens(
    text: str,
    *,
    provider: Any = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
) -> TokenCountResult:
    resolved_provider = resolve_provider(provider, provider_name)
    require_capability(
        resolved_provider, TokenCountingCapability, "TokenCountingCapability"
    )
    return resolved_provider.count_text_tokens(text, model=model)


def count_message_tokens(
    messages: Sequence[Message],
    *,
    provider: Any = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
) -> TokenCountResult:
    resolved_provider = resolve_provider(provider, provider_name)
    require_capability(
        resolved_provider, TokenCountingCapability, "TokenCountingCapability"
    )
    return resolved_provider.count_message_tokens(messages, model=model)


def count_chat_request_tokens(
    *,
    system: Optional[str] = None,
    user: Optional[str] = None,
    assistant: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
    provider: Any = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
) -> TokenCountResult:
    built_messages = build_messages(
        system=system, user=user, assistant=assistant, messages=messages
    )
    return count_message_tokens(
        built_messages, provider=provider, provider_name=provider_name, model=model
    )
