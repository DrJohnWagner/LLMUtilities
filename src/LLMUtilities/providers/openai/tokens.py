from __future__ import annotations

from typing import Optional, Sequence

from ...capabilities.token_counting import TokenCountResult
from ...exceptions import MissingDependencyError
from ...types import Message
from ...utils import content_to_text

try:
    import tiktoken as _tiktoken
except ImportError:
    _tiktoken = None


def is_available() -> bool:
    return _tiktoken is not None


def _encoding_for_model(model: Optional[str]):
    if model:
        try:
            return _tiktoken.encoding_for_model(model)
        except Exception:
            pass
    return _tiktoken.get_encoding("cl100k_base")


def count_text_tokens(text: str, *, model: Optional[str] = None) -> TokenCountResult:
    if _tiktoken is None:
        raise MissingDependencyError(
            "The 'tiktoken' package is required for OpenAI token counting. "
            "Install it with: pip install tiktoken"
        )

    encoding = _encoding_for_model(model)
    return TokenCountResult(
        count=len(encoding.encode(text)),
        provider="openai",
        model=model,
        method="local_estimate",
    )


def count_message_tokens(
    messages: Sequence[Message], *, model: Optional[str] = None
) -> TokenCountResult:
    if _tiktoken is None:
        raise MissingDependencyError(
            "The 'tiktoken' package is required for OpenAI token counting. "
            "Install it with: pip install tiktoken"
        )

    encoding = _encoding_for_model(model)

    system_parts: list[str] = []
    total = 0
    for message in messages:
        content_text = content_to_text(message.content)
        if message.role == "system":
            system_parts.append(content_text)
            continue

        total += len(encoding.encode(message.role))
        total += len(encoding.encode(content_text))

    if system_parts:
        total += len(encoding.encode("\n\n".join(system_parts)))

    return TokenCountResult(
        count=total, provider="openai", model=model, method="local_estimate"
    )
