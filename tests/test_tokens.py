"""Tests for text and chat-message token counting."""

from __future__ import annotations

import types
from dataclasses import replace
from typing import Any

import pytest

import LLMUtilities.tokens as tokens
from LLMUtilities.exceptions import (
    ConfigurationError,
    MissingDependencyError,
)
from LLMUtilities.types import (
    ImageContentPart,
    Message,
    TextContentPart,
)


# ---------------------------------------------------------------------------
# Fake token-counting dependencies
# ---------------------------------------------------------------------------


class _CharacterEncoding:
    """Simple deterministic encoding: one token per character."""

    @staticmethod
    def encode(text: str) -> list[str]:
        return list(text)


def _install_fake_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_encoding_error: Exception | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {
        "encoding_for_model_calls": [],
        "get_encoding_calls": [],
    }

    class _FakeTiktoken:
        @staticmethod
        def encoding_for_model(model: str) -> _CharacterEncoding:
            captured["encoding_for_model_calls"].append(model)

            if model_encoding_error is not None:
                raise model_encoding_error

            return _CharacterEncoding()

        @staticmethod
        def get_encoding(name: str) -> _CharacterEncoding:
            captured["get_encoding_calls"].append(name)
            return _CharacterEncoding()

    monkeypatch.setattr(
        tokens,
        "tiktoken",
        _FakeTiktoken(),
    )

    return captured


def _install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    *,
    token_count: int = 11,
    error: Exception | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _Messages:
        @staticmethod
        def count_tokens(**kwargs: Any) -> Any:
            captured.setdefault(
                "count_tokens_calls",
                [],
            ).append(kwargs)

            if error is not None:
                raise error

            return types.SimpleNamespace(
                input_tokens=token_count,
            )

    class _AnthropicClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.messages = _Messages()

    monkeypatch.setattr(
        tokens,
        "Anthropic",
        _AnthropicClient,
    )

    return captured


def _install_fake_google(
    monkeypatch: pytest.MonkeyPatch,
    *,
    token_counts: list[int] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    remaining_counts = list(
        token_counts
        if token_counts is not None
        else [13]
    )

    class _Models:
        @staticmethod
        def count_tokens(**kwargs: Any) -> Any:
            captured.setdefault(
                "count_tokens_calls",
                [],
            ).append(kwargs)

            if error is not None:
                raise error

            count = (
                remaining_counts.pop(0)
                if remaining_counts
                else 0
            )

            return types.SimpleNamespace(
                total_tokens=count,
            )

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.models = _Models()

    fake_genai = types.SimpleNamespace(
        Client=_Client,
    )

    monkeypatch.setattr(
        tokens,
        "genai",
        fake_genai,
    )

    return captured


# ---------------------------------------------------------------------------
# Public provider dispatch
# ---------------------------------------------------------------------------


class TestTokenCountingDispatch:
    def test_text_count_defaults_to_openai(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def _count(
            text: str,
            model: str | None = None,
        ) -> int:
            captured["text"] = text
            captured["model"] = model
            return 7

        monkeypatch.setattr(
            tokens,
            "_count_openai_text_tokens",
            _count,
        )

        result = tokens.count_text_tokens(
            "hello",
            model="test-model",
        )

        assert result == 7
        assert captured == {
            "text": "hello",
            "model": "test-model",
        }

    @pytest.mark.parametrize(
        (
            "provider",
            "function_name",
        ),
        [
            (
                "openai",
                "_count_openai_text_tokens",
            ),
            (
                "ANTHROPIC",
                "_count_anthropic_text_tokens",
            ),
            (
                "  Google  ",
                "_count_google_text_tokens",
            ),
        ],
    )
    def test_text_count_dispatches_by_normalised_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        function_name: str,
    ) -> None:
        captured: dict[str, Any] = {}

        def _count(
            text: str,
            model: str | None = None,
        ) -> int:
            captured["text"] = text
            captured["model"] = model
            return 9

        monkeypatch.setattr(
            tokens,
            function_name,
            _count,
        )

        result = tokens.count_text_tokens(
            "hello",
            provider=provider,
            model="custom-model",
        )

        assert result == 9
        assert captured == {
            "text": "hello",
            "model": "custom-model",
        }

    @pytest.mark.parametrize(
        (
            "provider",
            "function_name",
        ),
        [
            (
                "openai",
                "_count_openai_message_tokens",
            ),
            (
                "ANTHROPIC",
                "_count_anthropic_message_tokens",
            ),
            (
                "  Google  ",
                "_count_google_message_tokens",
            ),
        ],
    )
    def test_message_count_dispatches_by_normalised_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        function_name: str,
    ) -> None:
        message = Message(
            role="user",
            content="hello",
        )
        captured: dict[str, Any] = {}

        def _count(
            messages: list[Message],
            model: str | None = None,
        ) -> int:
            captured["messages"] = messages
            captured["model"] = model
            return 12

        monkeypatch.setattr(
            tokens,
            function_name,
            _count,
        )

        result = tokens.count_message_tokens(
            [message],
            provider=provider,
            model="custom-model",
        )

        assert result == 12
        assert captured["messages"] == [message]
        assert captured["model"] == "custom-model"

    def test_unknown_text_provider_raises_configuration_error(
        self,
    ) -> None:
        with pytest.raises(
            ConfigurationError,
            match="Unsupported token-counting provider",
        ):
            tokens.count_text_tokens(
                "hello",
                provider="no-such-provider",
            )

    def test_unknown_message_provider_raises_configuration_error(
        self,
    ) -> None:
        with pytest.raises(
            ConfigurationError,
            match="Unsupported token-counting provider",
        ):
            tokens.count_message_tokens(
                [
                    Message(
                        role="user",
                        content="hello",
                    )
                ],
                provider="no-such-provider",
            )

    def test_empty_message_sequence_returns_zero(self) -> None:
        assert (
            tokens.count_message_tokens(
                [],
                provider="openai",
            )
            == 0
        )

    def test_empty_messages_return_zero_before_provider_validation(
        self,
    ) -> None:
        assert (
            tokens.count_message_tokens(
                [],
                provider="no-such-provider",
            )
            == 0
        )


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


class TestMessageConstruction:
    def test_build_messages_in_role_order(self) -> None:
        messages = tokens._build_messages(
            system="System instruction.",
            user="Question.",
            assistant="Previous answer.",
        )

        assert messages == [
            Message(
                role="system",
                content="System instruction.",
            ),
            Message(
                role="user",
                content="Question.",
            ),
            Message(
                role="assistant",
                content="Previous answer.",
            ),
        ]

    def test_build_messages_omits_missing_values(self) -> None:
        messages = tokens._build_messages(
            user="Question.",
        )

        assert messages == [
            Message(
                role="user",
                content="Question.",
            )
        ]

    def test_build_messages_omits_empty_strings(self) -> None:
        assert (
            tokens._build_messages(
                system="",
                user="",
                assistant="",
            )
            == []
        )

    def test_explicit_messages_take_precedence(
        self,
    ) -> None:
        supplied = [
            Message(
                role="user",
                content="Explicit message.",
            )
        ]

        result = tokens._build_messages(
            system="Ignored system.",
            user="Ignored user.",
            assistant="Ignored assistant.",
            messages=supplied,
        )

        assert result == supplied
        assert result is not supplied

    def test_explicit_empty_message_list_takes_precedence(
        self,
    ) -> None:
        result = tokens._build_messages(
            system="Ignored system.",
            user="Ignored user.",
            messages=[],
        )

        assert result == []

    def test_count_chat_request_tokens_builds_and_dispatches(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def _count(
            messages: list[Message],
            *,
            provider: str,
            model: str | None,
        ) -> int:
            captured["messages"] = messages
            captured["provider"] = provider
            captured["model"] = model
            return 21

        monkeypatch.setattr(
            tokens,
            "count_message_tokens",
            _count,
        )

        result = tokens.count_chat_request_tokens(
            system="System.",
            user="Question.",
            assistant="Answer.",
            provider="anthropic",
            model="custom-model",
        )

        assert result == 21
        assert captured["provider"] == "anthropic"
        assert captured["model"] == "custom-model"
        assert captured["messages"] == [
            Message(
                role="system",
                content="System.",
            ),
            Message(
                role="user",
                content="Question.",
            ),
            Message(
                role="assistant",
                content="Answer.",
            ),
        ]

    def test_count_chat_request_tokens_preserves_explicit_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        supplied = [
            Message(
                role="user",
                content="Explicit.",
            )
        ]
        captured: dict[str, Any] = {}

        def _count(
            messages: list[Message],
            *,
            provider: str,
            model: str | None,
        ) -> int:
            captured["messages"] = messages
            return 5

        monkeypatch.setattr(
            tokens,
            "count_message_tokens",
            _count,
        )

        result = tokens.count_chat_request_tokens(
            system="Ignored.",
            user="Ignored.",
            messages=supplied,
        )

        assert result == 5
        assert captured["messages"] == supplied


# ---------------------------------------------------------------------------
# Token-count logging
# ---------------------------------------------------------------------------


class TestPrintTokenCount:
    def test_prints_text_token_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            tokens,
            "count_text_tokens",
            lambda *args, **kwargs: 17,
        )
        caplog.set_level(
            "INFO",
            logger="LLMUtilities.tokens",
        )

        tokens.print_token_count(
            text="hello",
            provider="openai",
            model="test-model",
        )

        assert "Token count: 17" in caplog.text

    def test_prints_message_token_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            tokens,
            "count_message_tokens",
            lambda *args, **kwargs: 23,
        )
        caplog.set_level(
            "INFO",
            logger="LLMUtilities.tokens",
        )

        tokens.print_token_count(
            messages=[
                Message(
                    role="user",
                    content="hello",
                )
            ],
            provider="google",
        )

        assert "Token count: 23" in caplog.text

    def test_text_takes_precedence_over_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        def _text_count(*args: Any, **kwargs: Any) -> int:
            calls.append("text")
            return 1

        def _message_count(*args: Any, **kwargs: Any) -> int:
            calls.append("messages")
            return 2

        monkeypatch.setattr(
            tokens,
            "count_text_tokens",
            _text_count,
        )
        monkeypatch.setattr(
            tokens,
            "count_message_tokens",
            _message_count,
        )

        tokens.print_token_count(
            text="hello",
            messages=[
                Message(
                    role="user",
                    content="ignored",
                )
            ],
        )

        assert calls == ["text"]

    def test_missing_text_and_messages_raises_value_error(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="Either `text` or `messages`",
        ):
            tokens.print_token_count()


# ---------------------------------------------------------------------------
# OpenAI token counting
# ---------------------------------------------------------------------------


class TestOpenAITokenCounting:
    def test_missing_tiktoken_raises_package_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            tokens,
            "tiktoken",
            None,
        )

        with pytest.raises(
            MissingDependencyError,
            match="tiktoken",
        ):
            tokens._count_openai_text_tokens("hello")

    def test_message_count_missing_tiktoken_raises_package_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            tokens,
            "tiktoken",
            None,
        )

        with pytest.raises(
            MissingDependencyError,
            match="tiktoken",
        ):
            tokens._count_openai_message_tokens(
                [
                    Message(
                        role="user",
                        content="hello",
                    )
                ]
            )

    def test_text_count_uses_encoding_length(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_tiktoken(monkeypatch)

        assert tokens._count_openai_text_tokens("hello") == 5

    def test_explicit_model_uses_model_encoding(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_tiktoken(monkeypatch)

        count = tokens._count_openai_text_tokens(
            "hello",
            model="gpt-test",
        )

        assert count == 5
        assert captured["encoding_for_model_calls"] == [
            "gpt-test"
        ]
        assert captured["get_encoding_calls"] == []

    def test_missing_model_uses_cl100k_base(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_tiktoken(monkeypatch)

        count = tokens._count_openai_text_tokens("hello")

        assert count == 5
        assert captured["encoding_for_model_calls"] == []
        assert captured["get_encoding_calls"] == [
            "cl100k_base"
        ]

    def test_unknown_model_falls_back_to_cl100k_base(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_tiktoken(
            monkeypatch,
            model_encoding_error=KeyError("unknown model"),
        )

        count = tokens._count_openai_text_tokens(
            "hello",
            model="unknown-model",
        )

        assert count == 5
        assert captured["encoding_for_model_calls"] == [
            "unknown-model"
        ]
        assert captured["get_encoding_calls"] == [
            "cl100k_base"
        ]

    def test_message_count_includes_roles_and_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_tiktoken(monkeypatch)

        count = tokens._count_openai_message_tokens(
            [
                Message(
                    role="user",
                    content="hello",
                ),
                Message(
                    role="assistant",
                    content="ok",
                ),
            ]
        )

        expected = (
            len("user")
            + len("hello")
            + len("assistant")
            + len("ok")
        )

        assert count == expected

    def test_system_messages_are_joined_without_role_tokens(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_tiktoken(monkeypatch)

        count = tokens._count_openai_message_tokens(
            [
                Message(
                    role="system",
                    content="First",
                ),
                Message(
                    role="user",
                    content="hello",
                ),
                Message(
                    role="system",
                    content="Second",
                ),
            ]
        )

        expected = (
            len("user")
            + len("hello")
            + len("First\n\nSecond")
        )

        assert count == expected

    def test_multimodal_content_uses_text_parts_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_tiktoken(monkeypatch)

        message = Message(
            role="user",
            content=[
                TextContentPart(
                    type="text",
                    text="hello",
                ),
                ImageContentPart(
                    type="image",
                    source={
                        "type": "url",
                        "url": "https://example.com/image.png",
                    },
                ),
                TextContentPart(
                    type="text",
                    text="world",
                ),
            ],
        )

        count = tokens.count_message_tokens(
            [message],
            provider="openai",
        )

        assert count == len("user") + len("hello world")


# ---------------------------------------------------------------------------
# Anthropic token counting
# ---------------------------------------------------------------------------


class TestAnthropicTokenCounting:
    def test_missing_sdk_raises_package_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            tokens,
            "Anthropic",
            None,
        )

        with pytest.raises(
            MissingDependencyError,
            match="anthropic",
        ):
            tokens._count_anthropic_text_tokens("hello")

    def test_message_count_missing_sdk_raises_package_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            tokens,
            "Anthropic",
            None,
        )

        with pytest.raises(
            MissingDependencyError,
            match="anthropic",
        ):
            tokens._count_anthropic_message_tokens(
                [
                    Message(
                        role="user",
                        content="hello",
                    )
                ]
            )

    def test_missing_api_key_raises_configuration_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_anthropic(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                anthropic_api_key=None,
            ),
        )

        with pytest.raises(
            ConfigurationError,
            match="ANTHROPIC_API_KEY",
        ):
            tokens._count_anthropic_text_tokens("hello")

    def test_message_count_missing_api_key_raises_configuration_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_anthropic(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                anthropic_api_key=None,
            ),
        )

        with pytest.raises(
            ConfigurationError,
            match="ANTHROPIC_API_KEY",
        ):
            tokens._count_anthropic_message_tokens(
                [
                    Message(
                        role="user",
                        content="hello",
                    )
                ]
            )

    def test_text_count_constructs_provider_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(
            monkeypatch,
            token_count=11,
        )
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                anthropic_api_key="fake-key",
            ),
        )

        count = tokens._count_anthropic_text_tokens(
            "hello",
            model="custom-model",
        )

        assert count == 11
        assert captured["client_kwargs"] == {
            "api_key": "fake-key",
        }
        assert captured["count_tokens_calls"] == [
            {
                "model": "custom-model",
                "messages": [
                    {
                        "role": "user",
                        "content": "hello",
                    }
                ],
            }
        ]

    def test_text_count_uses_configured_default_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                anthropic_api_key="fake-key",
                anthropic=replace(
                    tokens.settings.anthropic,
                    chat_model="configured-anthropic-model",
                ),
            ),
        )

        tokens._count_anthropic_text_tokens("hello")

        assert (
            captured["count_tokens_calls"][0]["model"]
            == "configured-anthropic-model"
        )

    def test_text_count_uses_builtin_model_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                anthropic_api_key="fake-key",
                anthropic=replace(
                    tokens.settings.anthropic,
                    chat_model=None,
                ),
            ),
        )

        tokens._count_anthropic_text_tokens("hello")

        assert (
            captured["count_tokens_calls"][0]["model"]
            == "claude-sonnet-4-6"
        )

    def test_message_count_separates_system_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(
            monkeypatch,
            token_count=29,
        )
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                anthropic_api_key="fake-key",
            ),
        )

        count = tokens._count_anthropic_message_tokens(
            [
                Message(
                    role="system",
                    content="First instruction.",
                ),
                Message(
                    role="system",
                    content="Second instruction.",
                ),
                Message(
                    role="user",
                    content="Question.",
                ),
                Message(
                    role="assistant",
                    content="Previous answer.",
                ),
            ],
            model="custom-model",
        )

        assert count == 29
        assert captured["count_tokens_calls"] == [
            {
                "model": "custom-model",
                "system": (
                    "First instruction.\n\n"
                    "Second instruction."
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": "Question.",
                    },
                    {
                        "role": "assistant",
                        "content": "Previous answer.",
                    },
                ],
            }
        ]

    def test_message_count_without_system_passes_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                anthropic_api_key="fake-key",
            ),
        )

        tokens._count_anthropic_message_tokens(
            [
                Message(
                    role="user",
                    content="hello",
                )
            ]
        )

        assert (
            captured["count_tokens_calls"][0]["system"]
            is None
        )

    def test_system_only_request_inserts_empty_user_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                anthropic_api_key="fake-key",
            ),
        )

        tokens._count_anthropic_message_tokens(
            [
                Message(
                    role="system",
                    content="System instruction.",
                )
            ]
        )

        assert captured["count_tokens_calls"][0] == {
            "model": tokens.settings.anthropic.chat_model,
            "system": "System instruction.",
            "messages": [
                {
                    "role": "user",
                    "content": "",
                }
            ],
        }

    def test_multimodal_content_is_flattened_to_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                anthropic_api_key="fake-key",
            ),
        )

        tokens._count_anthropic_message_tokens(
            [
                Message(
                    role="user",
                    content=[
                        TextContentPart(
                            type="text",
                            text="Part one",
                        ),
                        ImageContentPart(
                            type="image",
                            source={
                                "type": "url",
                                "url": "https://example.com/image.png",
                            },
                        ),
                        TextContentPart(
                            type="text",
                            text="part two",
                        ),
                    ],
                )
            ]
        )

        assert captured["count_tokens_calls"][0]["messages"] == [
            {
                "role": "user",
                "content": "Part one part two",
            }
        ]


# ---------------------------------------------------------------------------
# Google token counting
# ---------------------------------------------------------------------------


class TestGoogleTokenCounting:
    def test_missing_sdk_raises_package_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            tokens,
            "genai",
            None,
        )

        with pytest.raises(
            MissingDependencyError,
            match="google-genai",
        ):
            tokens._count_google_text_tokens("hello")

    def test_message_count_missing_sdk_raises_package_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            tokens,
            "genai",
            None,
        )

        with pytest.raises(
            MissingDependencyError,
            match="google-genai",
        ):
            tokens._count_google_message_tokens(
                [
                    Message(
                        role="user",
                        content="hello",
                    )
                ]
            )

    def test_missing_api_key_raises_configuration_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_google(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                google_api_key=None,
            ),
        )

        with pytest.raises(
            ConfigurationError,
            match="GOOGLE_API_KEY",
        ):
            tokens._count_google_text_tokens("hello")

    def test_message_count_missing_api_key_raises_configuration_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_google(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                google_api_key=None,
            ),
        )

        with pytest.raises(
            ConfigurationError,
            match="GOOGLE_API_KEY",
        ):
            tokens._count_google_message_tokens(
                [
                    Message(
                        role="user",
                        content="hello",
                    )
                ]
            )

    def test_text_count_constructs_provider_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_google(
            monkeypatch,
            token_counts=[13],
        )
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                google_api_key="fake-key",
            ),
        )

        count = tokens._count_google_text_tokens(
            "hello",
            model="custom-model",
        )

        assert count == 13
        assert captured["client_kwargs"] == {
            "api_key": "fake-key",
        }
        assert captured["count_tokens_calls"] == [
            {
                "model": "custom-model",
                "contents": "hello",
            }
        ]

    def test_text_count_uses_configured_default_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_google(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                google_api_key="fake-key",
                google=replace(
                    tokens.settings.google,
                    chat_model="configured-google-model",
                ),
            ),
        )

        tokens._count_google_text_tokens("hello")

        assert (
            captured["count_tokens_calls"][0]["model"]
            == "configured-google-model"
        )

    def test_text_count_uses_builtin_model_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_google(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                google_api_key="fake-key",
                google=replace(
                    tokens.settings.google,
                    chat_model=None,
                ),
            ),
        )

        tokens._count_google_text_tokens("hello")

        assert (
            captured["count_tokens_calls"][0]["model"]
            == "gemini-2.5-flash"
        )

    def test_message_count_maps_roles_and_counts_system_separately(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_google(
            monkeypatch,
            token_counts=[
                20,
                7,
            ],
        )
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                google_api_key="fake-key",
            ),
        )

        count = tokens._count_google_message_tokens(
            [
                Message(
                    role="system",
                    content="First instruction.",
                ),
                Message(
                    role="system",
                    content="Second instruction.",
                ),
                Message(
                    role="user",
                    content="Question.",
                ),
                Message(
                    role="assistant",
                    content="Previous answer.",
                ),
            ],
            model="custom-model",
        )

        assert count == 27
        assert captured["count_tokens_calls"] == [
            {
                "model": "custom-model",
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": "Question.",
                            }
                        ],
                    },
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": "Previous answer.",
                            }
                        ],
                    },
                ],
            },
            {
                "model": "custom-model",
                "contents": (
                    "First instruction.\n\n"
                    "Second instruction."
                ),
            },
        ]

    def test_message_count_without_system_uses_one_api_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_google(
            monkeypatch,
            token_counts=[8],
        )
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                google_api_key="fake-key",
            ),
        )

        count = tokens._count_google_message_tokens(
            [
                Message(
                    role="user",
                    content="hello",
                )
            ]
        )

        assert count == 8
        assert len(captured["count_tokens_calls"]) == 1
        assert isinstance(
            captured["count_tokens_calls"][0]["contents"],
            list,
        )

    def test_system_only_request_uses_one_api_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_google(
            monkeypatch,
            token_counts=[6],
        )
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                google_api_key="fake-key",
            ),
        )

        count = tokens._count_google_message_tokens(
            [
                Message(
                    role="system",
                    content="System instruction.",
                )
            ]
        )

        assert count == 6
        assert captured["count_tokens_calls"] == [
            {
                "model": tokens.settings.google.chat_model,
                "contents": "System instruction.",
            }
        ]

    def test_multimodal_content_is_flattened_to_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_google(monkeypatch)
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                google_api_key="fake-key",
            ),
        )

        tokens._count_google_message_tokens(
            [
                Message(
                    role="user",
                    content=[
                        TextContentPart(
                            type="text",
                            text="Part one",
                        ),
                        ImageContentPart(
                            type="image",
                            source={
                                "type": "url",
                                "url": "https://example.com/image.png",
                            },
                        ),
                        TextContentPart(
                            type="text",
                            text="part two",
                        ),
                    ],
                )
            ]
        )

        assert captured["count_tokens_calls"][0]["contents"] == [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Part one part two",
                    }
                ],
            }
        ]

    def test_provider_errors_are_not_silently_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_google(
            monkeypatch,
            error=RuntimeError("provider failure"),
        )
        monkeypatch.setattr(
            tokens,
            "settings",
            replace(
                tokens.settings,
                google_api_key="fake-key",
            ),
        )

        with pytest.raises(
            RuntimeError,
            match="provider failure",
        ):
            tokens._count_google_text_tokens("hello")