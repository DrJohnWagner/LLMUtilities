from __future__ import annotations

from typing import Optional, Sequence

from .config import settings
from .exceptions import ConfigurationError
from .types import ChatRequest, ChatResponse, ChatUsage, Message


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


def get_chat_model(provider: Optional[str] = None):
    """
    Resolve and return a provider-specific chat model instance.

    Imports are deferred so that missing provider SDKs do not prevent
    the package from loading.  A clear error is raised only when the
    requested provider is actually instantiated.

    Supported providers: ``'openai'``, ``'anthropic'``, ``'google'``.
    """
    provider_name = (provider or settings.default_provider).strip().lower()

    if provider_name == "openai":
        from .providers.openai import OpenAIChatModel
        return OpenAIChatModel()

    if provider_name == "anthropic":
        from .providers.anthropic import AnthropicChatModel
        return AnthropicChatModel()

    if provider_name == "google":
        from .providers.google import GoogleChatModel
        return GoogleChatModel()

    raise ConfigurationError(
        f"Unsupported provider: {provider_name!r}. "
        f"Expected one of: 'openai', 'anthropic', 'google'."
    )


def make_chat_request(
    *,
    user: Optional[str] = None,
    system: Optional[str] = None,
    assistant: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> ChatRequest:
    """Construct a ChatRequest from raw text or an existing message list."""
    built_messages = build_messages(
        user=user,
        system=system,
        assistant=assistant,
        messages=messages,
    )

    if not built_messages:
        raise ValueError("At least one message must be provided.")

    return ChatRequest(
        messages=built_messages,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def chat(
    *,
    provider=None,
    provider_name: Optional[str] = None,
    user: Optional[str] = None,
    system: Optional[str] = None,
    assistant: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> ChatResponse:
    """
    Send a chat request through the selected provider.

    Provide either:

    - ``provider``: an instantiated provider object with a ``.chat(...)`` method, or
    - ``provider_name``: ``'openai'``, ``'anthropic'``, or ``'google'``

    If neither is given, the default provider from config is used.
    """
    request = make_chat_request(
        user=user,
        system=system,
        assistant=assistant,
        messages=messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        model=model,
    )

    resolved_provider = provider or get_chat_model(provider_name)
    return resolved_provider.chat(request)


def chat_text(
    *,
    provider=None,
    provider_name: Optional[str] = None,
    user: Optional[str] = None,
    system: Optional[str] = None,
    assistant: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> str:
    """Convenience wrapper — returns only the response text."""
    return chat(
        provider=provider,
        provider_name=provider_name,
        user=user,
        system=system,
        assistant=assistant,
        messages=messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        model=model,
    ).text


def chat_usage(
    *,
    provider=None,
    provider_name: Optional[str] = None,
    user: Optional[str] = None,
    system: Optional[str] = None,
    assistant: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> ChatUsage:
    """Convenience wrapper — returns only the usage object."""
    return chat(
        provider=provider,
        provider_name=provider_name,
        user=user,
        system=system,
        assistant=assistant,
        messages=messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        model=model,
    ).usage
