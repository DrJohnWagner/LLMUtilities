from __future__ import annotations

import pytest

from LLMUtilities.capabilities.token_counting import TokenCountResult
from LLMUtilities.exceptions import UnsupportedCapabilityError
from LLMUtilities.tokens import count_chat_request_tokens, count_message_tokens, count_text_tokens
from LLMUtilities.types import Message


class _FakeTokenCountingProvider:
    name = "fake"

    def count_text_tokens(self, text, *, model=None):
        return TokenCountResult(count=len(text.split()), provider="fake", model=model, method="exact")

    def count_message_tokens(self, messages, *, model=None):
        total = sum(len(m.content.split()) for m in messages)
        return TokenCountResult(count=total, provider="fake", model=model, method="exact")


class _NotATokenCountingProvider:
    name = "not-tokens"


def test_count_text_tokens_delegates_to_provider():
    result = count_text_tokens("one two three", provider=_FakeTokenCountingProvider())
    assert result.count == 3
    assert result.method == "exact"


def test_count_message_tokens_delegates_to_provider():
    messages = [Message(role="user", content="one two"), Message(role="assistant", content="three")]
    result = count_message_tokens(messages, provider=_FakeTokenCountingProvider())
    assert result.count == 3


def test_count_chat_request_tokens_builds_messages_first():
    result = count_chat_request_tokens(
        system="a b", user="c d e", provider=_FakeTokenCountingProvider()
    )
    assert result.count == 5


def test_count_text_tokens_rejects_provider_without_capability():
    with pytest.raises(UnsupportedCapabilityError):
        count_text_tokens("hi", provider=_NotATokenCountingProvider())
