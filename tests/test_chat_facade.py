from __future__ import annotations

import pytest

from LLMUtilities.chat import chat, chat_text
from LLMUtilities.exceptions import UnsupportedCapabilityError
from LLMUtilities.types import ChatRequest, ChatResponse, Message


class _FakeChatProvider:
    name = "fake"

    def __init__(self):
        self.last_request = None

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.last_request = request
        return ChatResponse(
            text="fake reply",
            provider="fake",
            requested_model=request.model,
            resolved_model=request.model or "fake-default",
            usage=None,
            raw=None,
        )


class _NotAChatProvider:
    name = "not-chat"


def test_chat_builds_request_from_system_and_user_text():
    provider = _FakeChatProvider()
    response = chat(provider=provider, system="be terse", user="hello", model="x")

    assert response.text == "fake reply"
    assert len(provider.last_request.messages) == 2
    assert provider.last_request.messages[0].role == "system"
    assert provider.last_request.messages[1].role == "user"


def test_chat_accepts_explicit_messages():
    provider = _FakeChatProvider()
    messages = [Message(role="user", content="hi")]
    chat(provider=provider, messages=messages)
    assert provider.last_request.messages == messages


def test_chat_requires_at_least_one_message():
    with pytest.raises(ValueError):
        chat(provider=_FakeChatProvider())


def test_chat_text_returns_only_the_text():
    assert chat_text(provider=_FakeChatProvider(), user="hi") == "fake reply"


def test_chat_rejects_provider_without_chat_capability():
    with pytest.raises(UnsupportedCapabilityError):
        chat(provider=_NotAChatProvider(), user="hi")
