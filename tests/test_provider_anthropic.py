"""Tests for the Anthropic chat provider."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from typing import Any

import pytest

import LLMUtilities.providers.anthropic as anthropic_module
from LLMUtilities.exceptions import (
    AuthenticationError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)
from LLMUtilities.providers.anthropic import (
    AnthropicChatModel,
    _extract_text,
    _extract_usage,
    _normalize_content,
)
from LLMUtilities.types import (
    ChatRequest,
    ImageContentPart,
    Message,
    TextContentPart,
)


# ---------------------------------------------------------------------------
# Fake Anthropic response objects
# ---------------------------------------------------------------------------


def _make_text_block(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        type="text",
        text=text,
    )


def _make_tool_block() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        type="tool_use",
        id="tool-1",
        name="example_tool",
        input={},
    )


def _make_response(
    *,
    content: Any,
    usage: Any = None,
    stop_reason: str | None = "end_turn",
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        content=content,
        usage=usage,
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# Fake Anthropic SDK
# ---------------------------------------------------------------------------


class _AnthropicAuthenticationError(Exception):
    pass


class _AnthropicRateLimitError(Exception):
    pass


class _AnthropicConnectionError(Exception):
    pass


class _AnthropicStatusError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: Any = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    """
    Install a fake ``anthropic`` module.

    The production adapter imports Anthropic lazily inside ``chat()``, so
    replacing ``sys.modules["anthropic"]`` exercises the complete adapter
    without making a network request.
    """
    captured: dict[str, Any] = {}

    class _Messages:
        @staticmethod
        def create(**kwargs: Any) -> Any:
            captured["request_kwargs"] = kwargs

            if error is not None:
                raise error

            return response

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.messages = _Messages()

    fake_module = types.ModuleType("anthropic")
    setattr(fake_module, "Anthropic", _Client)
    setattr(
        fake_module,
        "AuthenticationError",
        _AnthropicAuthenticationError,
    )
    setattr(
        fake_module,
        "RateLimitError",
        _AnthropicRateLimitError,
    )
    setattr(
        fake_module,
        "APIConnectionError",
        _AnthropicConnectionError,
    )
    setattr(
        fake_module,
        "APIStatusError",
        _AnthropicStatusError,
    )

    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    return captured


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


class TestAnthropicTextExtraction:
    def test_missing_content_field_raises_response_error(self) -> None:
        response = types.SimpleNamespace(stop_reason="end_turn")

        with pytest.raises(
            ResponseError,
            match="missing the 'content' field",
        ):
            _extract_text(response)

    def test_none_content_raises_response_error(self) -> None:
        response = _make_response(content=None)

        with pytest.raises(
            ResponseError,
            match="missing the 'content' field",
        ):
            _extract_text(response)

    def test_empty_content_raises_response_error(self) -> None:
        response = _make_response(content=[])

        with pytest.raises(
            ResponseError,
            match="no text content blocks",
        ):
            _extract_text(response)

    def test_tool_only_content_raises_response_error(self) -> None:
        response = _make_response(
            content=[_make_tool_block()],
            stop_reason="tool_use",
        )

        with pytest.raises(
            ResponseError,
            match="no text content blocks",
        ):
            _extract_text(response)

    def test_empty_text_block_raises_response_error(self) -> None:
        response = _make_response(
            content=[_make_text_block("")],
        )

        with pytest.raises(
            ResponseError,
            match="no text content blocks",
        ):
            _extract_text(response)

    def test_single_text_block(self) -> None:
        response = _make_response(
            content=[_make_text_block("hello world")],
        )

        assert _extract_text(response) == "hello world"

    def test_multiple_text_blocks_are_concatenated(self) -> None:
        response = _make_response(
            content=[
                _make_text_block("first"),
                _make_text_block(" second"),
                _make_text_block(" third"),
            ],
        )

        assert _extract_text(response) == "first second third"

    def test_non_text_blocks_are_skipped(self) -> None:
        response = _make_response(
            content=[
                _make_text_block("before"),
                _make_tool_block(),
                _make_text_block(" after"),
            ],
        )

        assert _extract_text(response) == "before after"

    def test_error_includes_stop_reason(self) -> None:
        response = _make_response(
            content=[],
            stop_reason="max_tokens",
        )

        with pytest.raises(
            ResponseError,
            match="max_tokens",
        ):
            _extract_text(response)


# ---------------------------------------------------------------------------
# Content normalisation
# ---------------------------------------------------------------------------


class TestAnthropicContentNormalisation:
    def test_string_content_is_unchanged(self) -> None:
        assert _normalize_content("hello") == "hello"

    def test_dictionary_text_parts_are_joined(self) -> None:
        content = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]

        assert _normalize_content(content) == "Part 1 Part 2"

    def test_text_content_objects_are_joined(self) -> None:
        content = [
            TextContentPart(
                type="text",
                text="Part 1",
            ),
            TextContentPart(
                type="text",
                text="Part 2",
            ),
        ]

        assert _normalize_content(content) == "Part 1 Part 2"

    def test_non_text_dictionary_parts_are_ignored(self) -> None:
        content = [
            {"type": "text", "text": "before"},
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/image.png",
                },
            },
            {"type": "text", "text": "after"},
        ]

        assert _normalize_content(content) == "before after"

    def test_non_text_content_objects_are_ignored(self) -> None:
        content = [
            TextContentPart(
                type="text",
                text="before",
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
                text="after",
            ),
        ]

        assert _normalize_content(content) == "before after"

    def test_content_with_no_text_parts_returns_empty_string(self) -> None:
        content = [
            ImageContentPart(
                type="image",
                source={
                    "type": "url",
                    "url": "https://example.com/image.png",
                },
            )
        ]

        assert _normalize_content(content) == ""

    def test_unrecognised_content_is_converted_to_string(self) -> None:
        assert _normalize_content(42) == "42"


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


class TestAnthropicUsageExtraction:
    def test_usage_fields_are_normalised(self) -> None:
        response = _make_response(
            content=[],
            usage=types.SimpleNamespace(
                input_tokens=20,
                output_tokens=8,
                cache_creation_input_tokens=6,
                cache_read_input_tokens=12,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens == 20
        assert usage.output_tokens == 8
        assert usage.total_tokens == 28
        assert usage.cache_creation_input_tokens == 6
        assert usage.cache_read_input_tokens == 12
        assert usage.cached_input_tokens == 12

    def test_total_is_computed_when_both_components_exist(self) -> None:
        response = _make_response(
            content=[],
            usage=types.SimpleNamespace(
                input_tokens=11,
                output_tokens=4,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.total_tokens == 15

    def test_total_is_computed_with_missing_input_tokens(self) -> None:
        response = _make_response(
            content=[],
            usage=types.SimpleNamespace(
                input_tokens=None,
                output_tokens=4,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens is None
        assert usage.output_tokens == 4
        assert usage.total_tokens == 4

    def test_total_is_computed_with_missing_output_tokens(self) -> None:
        response = _make_response(
            content=[],
            usage=types.SimpleNamespace(
                input_tokens=11,
                output_tokens=None,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens == 11
        assert usage.output_tokens is None
        assert usage.total_tokens == 11

    def test_cache_read_tokens_populate_compatibility_field(self) -> None:
        response = _make_response(
            content=[],
            usage=types.SimpleNamespace(
                input_tokens=10,
                output_tokens=2,
                cache_creation_input_tokens=3,
                cache_read_input_tokens=7,
            ),
        )

        usage = _extract_usage(response)

        assert usage.cache_read_input_tokens == 7
        assert usage.cached_input_tokens == 7

    def test_missing_cache_fields_are_allowed(self) -> None:
        response = _make_response(
            content=[],
            usage=types.SimpleNamespace(
                input_tokens=10,
                output_tokens=2,
            ),
        )

        usage = _extract_usage(response)

        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None
        assert usage.cached_input_tokens is None

    def test_missing_usage_returns_empty_usage(self) -> None:
        response = _make_response(
            content=[],
            usage=None,
        )

        usage = _extract_usage(response)

        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert usage.total_tokens is None
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None
        assert usage.cached_input_tokens is None


# ---------------------------------------------------------------------------
# Complete adapter behaviour
# ---------------------------------------------------------------------------


class TestAnthropicChatModel:
    def test_missing_sdk_raises_package_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setitem(sys.modules, "anthropic", None)

        provider = AnthropicChatModel(
            model="claude-test",
            api_key="fake-key",
        )
        request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content="hello",
                )
            ]
        )

        with pytest.raises(
            MissingDependencyError,
            match="anthropic",
        ):
            provider.chat(request)

    def test_chat_constructs_request_and_normalises_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        raw_response = _make_response(
            content=[
                _make_text_block("Hello"),
                _make_text_block(" world"),
            ],
            usage=types.SimpleNamespace(
                input_tokens=12,
                output_tokens=3,
                cache_creation_input_tokens=5,
                cache_read_input_tokens=7,
            ),
            stop_reason="end_turn",
        )

        captured = _install_fake_anthropic(
            monkeypatch,
            response=raw_response,
        )

        provider = AnthropicChatModel(
            model="claude-test",
            api_key="fake-key",
            timeout_seconds=12.5,
            max_retries=0,
        )

        request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content="First system instruction.",
                ),
                Message(
                    role="system",
                    content="Second system instruction.",
                ),
                Message(
                    role="user",
                    content="Hello?",
                ),
                Message(
                    role="assistant",
                    content="Previous answer.",
                ),
            ],
            temperature=0.25,
            max_output_tokens=321,
        )

        response = provider.chat(request)

        assert captured["client_kwargs"] == {
            "api_key": "fake-key",
            "timeout": 12.5,
            "max_retries": 0,
        }

        assert captured["request_kwargs"] == {
            "model": "claude-test",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello?",
                },
                {
                    "role": "assistant",
                    "content": "Previous answer.",
                },
            ],
            "max_tokens": 321,
            "system": [
                {
                    "type": "text",
                    "text": (
                        "First system instruction.\n\n"
                        "Second system instruction."
                    ),
                    "cache_control": {
                        "type": "ephemeral",
                    },
                }
            ],
            "temperature": 0.25,
        }

        assert response.text == "Hello world"
        assert response.provider == "anthropic"
        assert response.model == "claude-test"
        assert response.usage is not None
        assert response.usage.input_tokens == 12
        assert response.usage.output_tokens == 3
        assert response.usage.total_tokens == 15
        assert response.usage.cache_creation_input_tokens == 5
        assert response.usage.cache_read_input_tokens == 7
        assert response.usage.cached_input_tokens == 7
        assert response.stop_reason == "end_turn"
        assert response.raw is raw_response

    def test_system_messages_are_removed_from_conversation_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(
            monkeypatch,
            response=_make_response(
                content=[_make_text_block("ok")],
            ),
        )

        provider = AnthropicChatModel(
            model="claude-test",
            api_key="fake-key",
        )

        provider.chat(
            ChatRequest(
                messages=[
                    Message(
                        role="system",
                        content="System instruction.",
                    ),
                    Message(
                        role="user",
                        content="User message.",
                    ),
                ]
            )
        )

        request_kwargs = captured["request_kwargs"]

        assert request_kwargs["messages"] == [
            {
                "role": "user",
                "content": "User message.",
            }
        ]
        assert request_kwargs["system"][0]["text"] == "System instruction."

    def test_system_prompt_receives_ephemeral_cache_control(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(
            monkeypatch,
            response=_make_response(
                content=[_make_text_block("ok")],
            ),
        )

        provider = AnthropicChatModel(
            model="claude-test",
            api_key="fake-key",
        )

        provider.chat(
            ChatRequest(
                messages=[
                    Message(
                        role="system",
                        content="Static instructions.",
                    ),
                    Message(
                        role="user",
                        content="hello",
                    ),
                ]
            )
        )

        assert captured["request_kwargs"]["system"] == [
            {
                "type": "text",
                "text": "Static instructions.",
                "cache_control": {
                    "type": "ephemeral",
                },
            }
        ]

    def test_request_without_system_message_omits_system_argument(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(
            monkeypatch,
            response=_make_response(
                content=[_make_text_block("ok")],
            ),
        )

        provider = AnthropicChatModel(
            model="claude-test",
            api_key="fake-key",
        )

        provider.chat(
            ChatRequest(
                messages=[
                    Message(
                        role="user",
                        content="hello",
                    )
                ]
            )
        )

        assert "system" not in captured["request_kwargs"]

    def test_request_model_overrides_provider_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(
            monkeypatch,
            response=_make_response(
                content=[_make_text_block("ok")],
            ),
        )

        provider = AnthropicChatModel(
            model="provider-model",
            api_key="fake-key",
        )

        response = provider.chat(
            ChatRequest(
                model="request-model",
                messages=[
                    Message(
                        role="user",
                        content="hello",
                    )
                ],
            )
        )

        assert captured["request_kwargs"]["model"] == "request-model"
        assert response.model == "request-model"

    def test_request_max_output_tokens_takes_precedence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        configured_settings = replace(
            anthropic_module.settings,
            anthropic=replace(
                anthropic_module.settings.anthropic,
                max_output_tokens=999,
            ),
        )
        monkeypatch.setattr(
            anthropic_module,
            "settings",
            configured_settings,
        )

        captured = _install_fake_anthropic(
            monkeypatch,
            response=_make_response(
                content=[_make_text_block("ok")],
            ),
        )

        provider = AnthropicChatModel(
            model="claude-test",
            api_key="fake-key",
        )

        provider.chat(
            ChatRequest(
                messages=[
                    Message(
                        role="user",
                        content="hello",
                    )
                ],
                max_output_tokens=123,
            )
        )

        assert captured["request_kwargs"]["max_tokens"] == 123

    def test_configured_max_output_tokens_is_used_when_request_omits_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        configured_settings = replace(
            anthropic_module.settings,
            anthropic=replace(
                anthropic_module.settings.anthropic,
                max_output_tokens=999,
            ),
        )
        monkeypatch.setattr(
            anthropic_module,
            "settings",
            configured_settings,
        )

        captured = _install_fake_anthropic(
            monkeypatch,
            response=_make_response(
                content=[_make_text_block("ok")],
            ),
        )

        provider = AnthropicChatModel(
            model="claude-test",
            api_key="fake-key",
        )

        provider.chat(
            ChatRequest(
                messages=[
                    Message(
                        role="user",
                        content="hello",
                    )
                ]
            )
        )

        assert captured["request_kwargs"]["max_tokens"] == 999

    def test_builtin_max_tokens_fallback_is_used(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        configured_settings = replace(
            anthropic_module.settings,
            anthropic=replace(
                anthropic_module.settings.anthropic,
                max_output_tokens=None,
            ),
        )
        monkeypatch.setattr(
            anthropic_module,
            "settings",
            configured_settings,
        )

        captured = _install_fake_anthropic(
            monkeypatch,
            response=_make_response(
                content=[_make_text_block("ok")],
            ),
        )

        provider = AnthropicChatModel(
            model="claude-test",
            api_key="fake-key",
        )

        provider.chat(
            ChatRequest(
                messages=[
                    Message(
                        role="user",
                        content="hello",
                    )
                ]
            )
        )

        assert captured["request_kwargs"]["max_tokens"] == 8192

    def test_temperature_is_omitted_when_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_anthropic(
            monkeypatch,
            response=_make_response(
                content=[_make_text_block("ok")],
            ),
        )

        provider = AnthropicChatModel(
            model="claude-test",
            api_key="fake-key",
        )

        provider.chat(
            ChatRequest(
                messages=[
                    Message(
                        role="user",
                        content="hello",
                    )
                ]
            )
        )

        assert "temperature" not in captured["request_kwargs"]

    def test_malformed_response_raises_response_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_anthropic(
            monkeypatch,
            response=_make_response(
                content=[],
                stop_reason="end_turn",
            ),
        )

        provider = AnthropicChatModel(
            model="claude-test",
            api_key="fake-key",
        )

        with pytest.raises(
            ResponseError,
            match="no text content blocks",
        ):
            provider.chat(
                ChatRequest(
                    messages=[
                        Message(
                            role="user",
                            content="hello",
                        )
                    ]
                )
            )


# ---------------------------------------------------------------------------
# SDK exception mapping
# ---------------------------------------------------------------------------


class TestAnthropicExceptionMapping:
    @pytest.mark.parametrize(
        (
            "sdk_error",
            "expected_error",
            "match",
        ),
        [
            (
                _AnthropicAuthenticationError("bad key"),
                AuthenticationError,
                "authentication failed",
            ),
            (
                _AnthropicRateLimitError("too many requests"),
                RateLimitError,
                "rate limit exceeded",
            ),
            (
                _AnthropicConnectionError("network unavailable"),
                RequestError,
                "connection error",
            ),
            (
                _AnthropicStatusError(
                    418,
                    "teapot",
                ),
                ProviderError,
                r"API error \(418\): teapot",
            ),
            (
                RuntimeError("unexpected failure"),
                RequestError,
                "Unexpected Anthropic request failure",
            ),
        ],
    )
    def test_sdk_errors_are_mapped_to_package_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sdk_error: Exception,
        expected_error: type[Exception],
        match: str,
    ) -> None:
        _install_fake_anthropic(
            monkeypatch,
            error=sdk_error,
        )

        provider = AnthropicChatModel(
            model="claude-test",
            api_key="fake-key",
        )

        request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content="hello",
                )
            ]
        )

        with pytest.raises(
            expected_error,
            match=match,
        ):
            provider.chat(request)