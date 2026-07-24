from __future__ import annotations

from typing import Any, Optional, Sequence

from .capabilities import ChatCapability, require_capability
from .config import get_settings
from .providers.registry import get_provider
from .types import ChatRequest, ChatResponse, Message


def build_messages(
    user: Optional[str] = None,
    system: Optional[str] = None,
    assistant: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
) -> list[Message]:
    """
    Build a message list for a chat request.

    Pass an existing ``messages`` sequence, or provide one or more of
    ``system``, ``user``, ``assistant`` to construct one.
    """
    if messages is not None:
        return list(messages)

    built: list[Message] = []
    if system:
        built.append(Message(role="system", content=system))
    if user:
        built.append(Message(role="user", content=user))
    if assistant:
        built.append(Message(role="assistant", content=assistant))
    return built


def resolve_provider(provider: Any = None, provider_name: Optional[str] = None) -> Any:
    if provider is not None:
        return provider
    name = provider_name or get_settings().default_provider
    return get_provider(name)


def chat(
    *,
    provider: Any = None,
    provider_name: Optional[str] = None,
    user: Optional[str] = None,
    system: Optional[str] = None,
    assistant: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> ChatResponse:
    """
    Send a chat request through the selected provider.

    Provide either ``provider`` (an already-resolved provider object) or
    ``provider_name`` ('openai', 'anthropic', 'google', 'moonshot',
    'deepseek'). If neither is given, the configured default provider is used.
    """
    built_messages = build_messages(
        user=user, system=system, assistant=assistant, messages=messages
    )
    if not built_messages:
        raise ValueError("At least one message must be provided.")

    request = ChatRequest(
        messages=built_messages,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    resolved_provider = resolve_provider(provider, provider_name)
    require_capability(resolved_provider, ChatCapability, "ChatCapability")
    return resolved_provider.chat(request)


def chat_text(
    *,
    provider: Any = None,
    provider_name: Optional[str] = None,
    user: Optional[str] = None,
    system: Optional[str] = None,
    assistant: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> str:
    """Convenience wrapper - returns only the response text."""
    return chat(
        provider=provider,
        provider_name=provider_name,
        user=user,
        system=system,
        assistant=assistant,
        messages=messages,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    ).text
