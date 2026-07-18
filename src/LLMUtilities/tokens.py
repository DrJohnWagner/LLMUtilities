from __future__ import annotations

import logging
from typing import Optional, Sequence

from .config import settings
from .exceptions import ConfigurationError, MissingDependencyError
from .types import Message

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from google import genai
except ImportError:
    genai = None


logger = logging.getLogger(__name__)


def _emit(message: str) -> None:
    logger.info(message)


def count_text_tokens(
    text: str,
    *,
    provider: str = "openai",
    model: Optional[str] = None,
) -> int:
    provider_name = provider.strip().lower()

    if provider_name == "openai":
        return _count_openai_text_tokens(text, model=model)

    if provider_name == "anthropic":
        return _count_anthropic_text_tokens(text)

    if provider_name == "google":
        return _count_google_text_tokens(text, model=model)

    raise ConfigurationError(
        f"Unsupported token-counting provider: {provider_name!r}. "
        f"Expected one of: 'openai', 'anthropic', 'google'."
    )


def count_message_tokens(
    messages: Sequence[Message],
    *,
    provider: str = "openai",
    model: Optional[str] = None,
) -> int:
    if not messages:
        return 0

    provider_name = provider.strip().lower()

    if provider_name == "openai":
        return _count_openai_message_tokens(messages, model=model)

    if provider_name == "anthropic":
        return _count_anthropic_message_tokens(messages)

    if provider_name == "google":
        return _count_google_message_tokens(messages, model=model)

    raise ConfigurationError(
        f"Unsupported token-counting provider: {provider_name!r}. "
        f"Expected one of: 'openai', 'anthropic', 'google'."
    )


def count_chat_request_tokens(
    *,
    system: Optional[str] = None,
    user: Optional[str] = None,
    assistant: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
    provider: str = "openai",
    model: Optional[str] = None,
) -> int:
    built_messages = _build_messages(
        system=system,
        user=user,
        assistant=assistant,
        messages=messages,
    )
    return count_message_tokens(
        built_messages,
        provider=provider,
        model=model,
    )


def print_token_count(
    *,
    text: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
    provider: str = "openai",
    model: Optional[str] = None,
) -> None:
    if text is not None:
        count = count_text_tokens(text, provider=provider, model=model)
        _emit(f"Token count: {count}")
        return

    if messages is not None:
        count = count_message_tokens(messages, provider=provider, model=model)
        _emit(f"Token count: {count}")
        return

    raise ValueError("Either `text` or `messages` must be provided.")


def _build_messages(
    *,
    system: Optional[str] = None,
    user: Optional[str] = None,
    assistant: Optional[str] = None,
    messages: Optional[Sequence[Message]] = None,
) -> list[Message]:
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


# -------------------------
# OpenAI
# -------------------------

def _count_openai_text_tokens(text: str, model: Optional[str] = None) -> int:
    if tiktoken is None:
        raise MissingDependencyError(
            "The 'tiktoken' package is required for OpenAI token counting. "
            "Install it with: pip install tiktoken"
        )

    encoding = _get_openai_encoding(model)
    return len(encoding.encode(text))


def _count_openai_message_tokens(
    messages: Sequence[Message],
    model: Optional[str] = None,
) -> int:
    if tiktoken is None:
        raise MissingDependencyError(
            "The 'tiktoken' package is required for OpenAI token counting. "
            "Install it with: pip install tiktoken"
        )

    encoding = _get_openai_encoding(model)

    total = 0
    for message in messages:
        total += len(encoding.encode(message.role))
        total += len(encoding.encode(message.content))

    return total


def _get_openai_encoding(model: Optional[str] = None):
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            pass
    return tiktoken.get_encoding("cl100k_base")


# -------------------------
# Anthropic
# -------------------------

def _count_anthropic_text_tokens(text: str) -> int:
    if Anthropic is None:
        raise MissingDependencyError(
            "The 'anthropic' package is required for Anthropic token counting. "
            "Install it with: pip install anthropic"
        )

    api_key = settings.anthropic_api_key
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY is not configured.")

    client = Anthropic(api_key=api_key)

    response = client.messages.count_tokens(
        model=settings.anthropic.chat_model or "claude-sonnet-4-6",
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
    )
    return response.input_tokens


def _count_anthropic_message_tokens(messages: Sequence[Message]) -> int:
    if Anthropic is None:
        raise MissingDependencyError(
            "The 'anthropic' package is required for Anthropic token counting. "
            "Install it with: pip install anthropic"
        )

    api_key = settings.anthropic_api_key
    if not api_key:
        raise ConfigurationError("ANTHROPIC_API_KEY is not configured.")

    system_parts: list[str] = []
    anthropic_messages: list[dict[str, str]] = []

    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue

        anthropic_messages.append(
            {
                "role": message.role,
                "content": message.content,
            }
        )

    if not anthropic_messages:
        anthropic_messages.append(
            {
                "role": "user",
                "content": "",
            }
        )

    client = Anthropic(api_key=api_key)

    response = client.messages.count_tokens(
        model=settings.anthropic.chat_model or "claude-sonnet-4-6",
        system="\n\n".join(system_parts) if system_parts else None,
        messages=anthropic_messages,
    )
    return response.input_tokens


# -------------------------
# Google
# -------------------------

def _count_google_text_tokens(text: str, model: Optional[str] = None) -> int:
    if genai is None:
        raise MissingDependencyError(
            "The 'google-genai' package is required for Google token counting. "
            "Install it with: pip install google-genai"
        )

    api_key = settings.google_api_key
    if not api_key:
        raise ConfigurationError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not configured.")

    client = genai.Client(api_key=api_key)

    response = client.models.count_tokens(
        model=model or settings.google.chat_model or "gemini-2.5-flash",
        contents=text,
    )
    return response.total_tokens


def _count_google_message_tokens(
    messages: Sequence[Message],
    model: Optional[str] = None,
) -> int:
    if genai is None:
        raise MissingDependencyError(
            "The 'google-genai' package is required for Google token counting. "
            "Install it with: pip install google-genai"
        )

    api_key = settings.google_api_key
    if not api_key:
        raise ConfigurationError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not configured.")

    contents: list[dict[str, object]] = []
    system_parts: list[str] = []

    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue

        role = "user" if message.role == "user" else "model"
        contents.append(
            {
                "role": role,
                "parts": [{"text": message.content}],
            }
        )

    if system_parts:
        contents.insert(
            0,
            {
                "role": "user",
                "parts": [{"text": "\n\n".join(system_parts)}],
            },
        )

    client = genai.Client(api_key=api_key)

    response = client.models.count_tokens(
        model=model or settings.google.chat_model or "gemini-2.5-flash",
        contents=contents,
    )
    return response.total_tokens
