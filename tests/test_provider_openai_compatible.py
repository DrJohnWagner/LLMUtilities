"""Tests for providers using the OpenAI Chat Completions interface."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from LLMUtilities.config import get_settings
from LLMUtilities.exceptions import (
    AuthenticationError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)
from LLMUtilities.providers.deepseek import DeepSeekChatModel
from LLMUtilities.providers.moonshot import MoonshotChatModel
from LLMUtilities.providers.openai_chat_completions import (
    BaseOpenAIChatCompletionsModel,
    _collect_text,
    _extract_cached_input_tokens,
    _extract_text,
    _extract_usage,
    _first_choice,
    _prefix_system_message,
)
from LLMUtilities.types import (
    ChatRequest,
    Message,
    TextContentPart,
)

# ---------------------------------------------------------------------------
# Provider cases
# ---------------------------------------------------------------------------


_PROVIDER_CASES = [
    (
        MoonshotChatModel,
        "moonshot",
        "https://api.moonshot.ai/v1",
    ),
    (
        DeepSeekChatModel,
        "deepseek",
        "https://api.deepseek.com",
    ),
]


# ---------------------------------------------------------------------------
# Fake Chat Completions responses
# ---------------------------------------------------------------------------


def _make_choice(
    content: Any = "hello",
    *,
    finish_reason: str | None = "stop",
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        message=types.SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )


def _make_response(
    *,
    choices: Any,
    usage: Any = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        choices=choices,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Fake OpenAI SDK
# ---------------------------------------------------------------------------


class _OpenAIAuthenticationError(Exception):
    pass


class _OpenAIRateLimitError(Exception):
    pass


class _OpenAIConnectionError(Exception):
    pass


class _OpenAIStatusError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: Any = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    """
    Install a fake ``openai`` module and return captured call arguments.

    Moonshot and DeepSeek both use the OpenAI SDK's Chat Completions
    interface with provider-specific base URLs.
    """
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=self._create,
                )
            )

        @staticmethod
        def _create(**kwargs: Any) -> Any:
            captured["request_kwargs"] = kwargs

            if error is not None:
                raise error

            return response

    fake_module = types.ModuleType("openai")
    setattr(fake_module, "OpenAI", _Client)
    setattr(
        fake_module,
        "AuthenticationError",
        _OpenAIAuthenticationError,
    )
    setattr(
        fake_module,
        "RateLimitError",
        _OpenAIRateLimitError,
    )
    setattr(
        fake_module,
        "APIConnectionError",
        _OpenAIConnectionError,
    )
    setattr(
        fake_module,
        "APIStatusError",
        _OpenAIStatusError,
    )

    monkeypatch.setitem(sys.modules, "openai", fake_module)

    return captured


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


class TestOpenAICompatibleProviderConfiguration:
    def test_moonshot_configuration(self) -> None:
        configured = get_settings()

        assert MoonshotChatModel.provider_name == "moonshot"
        assert MoonshotChatModel.openai_base_url == "https://api.moonshot.ai/v1"
        assert MoonshotChatModel.default_model == configured.moonshot.chat_model
        assert MoonshotChatModel.api_key == configured.moonshot_api_key

    def test_deepseek_configuration(self) -> None:
        configured = get_settings()

        assert DeepSeekChatModel.provider_name == "deepseek"
        assert DeepSeekChatModel.openai_base_url == "https://api.deepseek.com"
        assert DeepSeekChatModel.default_model == configured.deepseek.chat_model
        assert DeepSeekChatModel.api_key == configured.deepseek_api_key

    @pytest.mark.parametrize(
        (
            "provider_cls",
            "provider_name",
            "base_url",
        ),
        _PROVIDER_CASES,
    )
    def test_provider_inherits_shared_adapter(
        self,
        provider_cls: type[BaseOpenAIChatCompletionsModel],
        provider_name: str,
        base_url: str,
    ) -> None:
        assert issubclass(
            provider_cls,
            BaseOpenAIChatCompletionsModel,
        )
        assert provider_cls.provider_name == provider_name
        assert provider_cls.openai_base_url == base_url


# ---------------------------------------------------------------------------
# System-message prefixing
# ---------------------------------------------------------------------------


class TestSystemMessagePrefixing:
    def test_single_system_message_is_prefixed(self) -> None:
        input_messages = [
            {
                "role": "user",
                "content": "hello",
            }
        ]

        result = _prefix_system_message(
            input_messages,
            ["You are helpful."],
        )

        assert result == [
            {
                "role": "system",
                "content": "You are helpful.",
            },
            {
                "role": "user",
                "content": "hello",
            },
        ]

    def test_multiple_system_messages_are_joined(self) -> None:
        input_messages = [
            {
                "role": "user",
                "content": "hello",
            }
        ]

        result = _prefix_system_message(
            input_messages,
            [
                "First instruction.",
                "Second instruction.",
            ],
        )

        assert result[0] == {
            "role": "system",
            "content": (
                "First instruction.\n\n"
                "Second instruction."
            ),
        }

    def test_input_message_list_is_not_mutated(self) -> None:
        input_messages = [
            {
                "role": "user",
                "content": "hello",
            }
        ]

        _prefix_system_message(
            input_messages,
            ["System instruction."],
        )

        assert input_messages == [
            {
                "role": "user",
                "content": "hello",
            }
        ]


# ---------------------------------------------------------------------------
# Choice extraction
# ---------------------------------------------------------------------------


class TestFirstChoice:
    def test_missing_choices_raises_response_error(self) -> None:
        response = types.SimpleNamespace()

        with pytest.raises(
            ResponseError,
            match="no choices",
        ):
            _first_choice(response)

    def test_none_choices_raises_response_error(self) -> None:
        response = _make_response(choices=None)

        with pytest.raises(
            ResponseError,
            match="no choices",
        ):
            _first_choice(response)

    def test_empty_choices_raises_response_error(self) -> None:
        response = _make_response(choices=[])

        with pytest.raises(
            ResponseError,
            match="no choices",
        ):
            _first_choice(response)

    def test_first_choice_is_returned(self) -> None:
        first = _make_choice("first")
        second = _make_choice("second")

        response = _make_response(
            choices=[
                first,
                second,
            ]
        )

        assert _first_choice(response) is first


# ---------------------------------------------------------------------------
# Text collection and extraction
# ---------------------------------------------------------------------------


class TestChatCompletionTextExtraction:
    def test_string_content_is_returned(self) -> None:
        assert _collect_text("hello") == "hello"

    def test_dictionary_text_parts_are_concatenated(self) -> None:
        content = [
            {
                "type": "text",
                "text": "foo",
            },
            {
                "type": "text",
                "text": " bar",
            },
        ]

        assert _collect_text(content) == "foo bar"

    def test_object_text_parts_are_concatenated(self) -> None:
        content = [
            types.SimpleNamespace(
                type="text",
                text="foo",
            ),
            types.SimpleNamespace(
                type="text",
                text=" bar",
            ),
        ]

        assert _collect_text(content) == "foo bar"

    def test_non_text_parts_are_ignored(self) -> None:
        content = [
            {
                "type": "tool_call",
                "text": "ignored",
            },
            {
                "type": "text",
                "text": "kept",
            },
        ]

        assert _collect_text(content) == "kept"

    def test_empty_text_parts_are_ignored(self) -> None:
        content = [
            {
                "type": "text",
                "text": "",
            },
            {
                "type": "text",
                "text": "kept",
            },
        ]

        assert _collect_text(content) == "kept"

    def test_none_content_returns_empty_string(self) -> None:
        assert _collect_text(None) == ""

    def test_unrecognised_content_is_converted_to_string(self) -> None:
        assert _collect_text(42) == "42"

    def test_missing_message_raises_response_error(self) -> None:
        choice = types.SimpleNamespace(
            finish_reason="stop",
        )

        with pytest.raises(
            ResponseError,
            match="missing the message field",
        ):
            _extract_text(choice)

    def test_none_message_raises_response_error(self) -> None:
        choice = types.SimpleNamespace(
            message=None,
            finish_reason="stop",
        )

        with pytest.raises(
            ResponseError,
            match="missing the message field",
        ):
            _extract_text(choice)

    def test_empty_content_raises_response_error(self) -> None:
        choice = _make_choice("")

        with pytest.raises(
            ResponseError,
            match="no text output",
        ):
            _extract_text(choice)

    def test_none_content_raises_response_error(self) -> None:
        choice = _make_choice(None)

        with pytest.raises(
            ResponseError,
            match="no text output",
        ):
            _extract_text(choice)

    def test_string_content_is_extracted(self) -> None:
        choice = _make_choice("hello world")

        assert _extract_text(choice) == "hello world"

    def test_structured_content_is_extracted(self) -> None:
        choice = _make_choice(
            [
                {
                    "type": "text",
                    "text": "hello",
                },
                {
                    "type": "text",
                    "text": " world",
                },
            ]
        )

        assert _extract_text(choice) == "hello world"


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


class TestOpenAICompatibleUsageExtraction:
    def test_standard_usage_fields_are_normalised(self) -> None:
        response = _make_response(
            choices=[],
            usage=types.SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=14,
                prompt_tokens_details=None,
                cached_tokens=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens == 10
        assert usage.output_tokens == 4
        assert usage.total_tokens == 14
        assert usage.cached_input_tokens is None
        assert usage.cache_read_input_tokens is None

    def test_total_is_computed_when_absent(self) -> None:
        response = _make_response(
            choices=[],
            usage=types.SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=None,
                prompt_tokens_details=None,
                cached_tokens=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.total_tokens == 14

    def test_total_is_computed_with_missing_input_tokens(self) -> None:
        response = _make_response(
            choices=[],
            usage=types.SimpleNamespace(
                prompt_tokens=None,
                completion_tokens=4,
                total_tokens=None,
                prompt_tokens_details=None,
                cached_tokens=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens is None
        assert usage.output_tokens == 4
        assert usage.total_tokens == 4

    def test_total_is_computed_with_missing_output_tokens(self) -> None:
        response = _make_response(
            choices=[],
            usage=types.SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=None,
                total_tokens=None,
                prompt_tokens_details=None,
                cached_tokens=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens == 10
        assert usage.output_tokens is None
        assert usage.total_tokens == 10

    def test_cached_tokens_are_read_from_details_object(self) -> None:
        response = _make_response(
            choices=[],
            usage=types.SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=14,
                prompt_tokens_details=types.SimpleNamespace(
                    cached_tokens=6,
                ),
            ),
        )

        usage = _extract_usage(response)

        assert usage.cached_input_tokens == 6
        assert usage.cache_read_input_tokens == 6

    def test_cached_tokens_are_read_from_details_mapping(self) -> None:
        response = _make_response(
            choices=[],
            usage=types.SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=14,
                prompt_tokens_details={
                    "cached_tokens": 6,
                },
            ),
        )

        usage = _extract_usage(response)

        assert usage.cached_input_tokens == 6
        assert usage.cache_read_input_tokens == 6

    def test_cached_tokens_fall_back_to_usage_field(self) -> None:
        response = _make_response(
            choices=[],
            usage=types.SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=14,
                prompt_tokens_details=None,
                cached_tokens=6,
            ),
        )

        usage = _extract_usage(response)

        assert usage.cached_input_tokens == 6
        assert usage.cache_read_input_tokens == 6

    def test_missing_usage_returns_empty_usage(self) -> None:
        response = _make_response(
            choices=[],
            usage=None,
        )

        usage = _extract_usage(response)

        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert usage.total_tokens is None
        assert usage.cached_input_tokens is None
        assert usage.cache_read_input_tokens is None


class TestCachedInputTokenExtraction:
    def test_details_object_takes_precedence(self) -> None:
        usage = types.SimpleNamespace(
            prompt_tokens_details=types.SimpleNamespace(
                cached_tokens=7,
            ),
            cached_tokens=3,
        )

        assert _extract_cached_input_tokens(usage) == 7

    def test_details_mapping_is_supported(self) -> None:
        usage = types.SimpleNamespace(
            prompt_tokens_details={
                "cached_tokens": 7,
            },
            cached_tokens=3,
        )

        assert _extract_cached_input_tokens(usage) == 7

    def test_usage_level_fallback_is_supported(self) -> None:
        usage = types.SimpleNamespace(
            prompt_tokens_details=None,
            cached_tokens=3,
        )

        assert _extract_cached_input_tokens(usage) == 3

    def test_missing_cached_token_fields_returns_none(self) -> None:
        usage = types.SimpleNamespace(
            prompt_tokens_details=None,
        )

        assert _extract_cached_input_tokens(usage) is None

    def test_details_without_cached_tokens_returns_none(self) -> None:
        usage = types.SimpleNamespace(
            prompt_tokens_details=types.SimpleNamespace(),
            cached_tokens=3,
        )

        assert _extract_cached_input_tokens(usage) is None


# ---------------------------------------------------------------------------
# Complete adapter behaviour
# ---------------------------------------------------------------------------


class TestOpenAICompatibleChatModel:
    @pytest.mark.parametrize(
        (
            "provider_cls",
            "provider_name",
            "base_url",
        ),
        _PROVIDER_CASES,
    )
    def test_missing_sdk_raises_package_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider_cls: type[BaseOpenAIChatCompletionsModel],
        provider_name: str,
        base_url: str,
    ) -> None:
        del provider_name
        del base_url

        monkeypatch.setitem(sys.modules, "openai", None)

        provider = provider_cls(
            model="test-model",
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
            match="openai",
        ):
            provider.chat(request)

    @pytest.mark.parametrize(
        (
            "provider_cls",
            "provider_name",
            "base_url",
        ),
        _PROVIDER_CASES,
    )
    def test_chat_constructs_request_and_normalises_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider_cls: type[BaseOpenAIChatCompletionsModel],
        provider_name: str,
        base_url: str,
    ) -> None:
        raw_response = _make_response(
            choices=[
                _make_choice(
                    "Hello world",
                    finish_reason="stop",
                )
            ],
            usage=types.SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
                prompt_tokens_details=types.SimpleNamespace(
                    cached_tokens=5,
                ),
            ),
        )

        captured = _install_fake_openai(
            monkeypatch,
            response=raw_response,
        )

        provider = provider_cls(
            model="test-model",
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
            "base_url": base_url,
            "timeout": 12.5,
            "max_retries": 0,
        }

        assert captured["request_kwargs"] == {
            "model": "test-model",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "First system instruction.\n\n"
                        "Second system instruction."
                    ),
                },
                {
                    "role": "user",
                    "content": "Hello?",
                },
                {
                    "role": "assistant",
                    "content": "Previous answer.",
                },
            ],
            "temperature": 0.25,
            "max_tokens": 321,
        }

        assert response.text == "Hello world"
        assert response.provider == provider_name
        assert response.model == "test-model"
        assert response.usage is not None
        assert response.usage.input_tokens == 12
        assert response.usage.output_tokens == 3
        assert response.usage.total_tokens == 15
        assert response.usage.cached_input_tokens == 5
        assert response.usage.cache_read_input_tokens == 5
        assert response.stop_reason == "stop"
        assert response.raw is raw_response

    @pytest.mark.parametrize(
        (
            "provider_cls",
            "provider_name",
            "base_url",
        ),
        _PROVIDER_CASES,
    )
    def test_request_without_system_message_uses_conversation_messages_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider_cls: type[BaseOpenAIChatCompletionsModel],
        provider_name: str,
        base_url: str,
    ) -> None:
        del provider_name
        del base_url

        captured = _install_fake_openai(
            monkeypatch,
            response=_make_response(
                choices=[_make_choice("ok")],
            ),
        )

        provider = provider_cls(
            model="test-model",
            api_key="fake-key",
        )

        provider.chat(
            ChatRequest(
                messages=[
                    Message(
                        role="user",
                        content="hello",
                    ),
                    Message(
                        role="assistant",
                        content="previous response",
                    ),
                ]
            )
        )

        assert captured["request_kwargs"]["messages"] == [
            {
                "role": "user",
                "content": "hello",
            },
            {
                "role": "assistant",
                "content": "previous response",
            },
        ]

    @pytest.mark.parametrize(
        (
            "provider_cls",
            "provider_name",
            "base_url",
        ),
        _PROVIDER_CASES,
    )
    def test_optional_request_fields_are_omitted_when_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider_cls: type[BaseOpenAIChatCompletionsModel],
        provider_name: str,
        base_url: str,
    ) -> None:
        del provider_name
        del base_url

        captured = _install_fake_openai(
            monkeypatch,
            response=_make_response(
                choices=[_make_choice("ok")],
            ),
        )

        provider = provider_cls(
            model="test-model",
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

        request_kwargs = captured["request_kwargs"]

        assert "temperature" not in request_kwargs
        assert "max_tokens" not in request_kwargs

    @pytest.mark.parametrize(
        (
            "provider_cls",
            "provider_name",
            "base_url",
        ),
        _PROVIDER_CASES,
    )
    def test_request_model_overrides_provider_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider_cls: type[BaseOpenAIChatCompletionsModel],
        provider_name: str,
        base_url: str,
    ) -> None:
        del provider_name
        del base_url

        captured = _install_fake_openai(
            monkeypatch,
            response=_make_response(
                choices=[_make_choice("ok")],
            ),
        )

        provider = provider_cls(
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

    @pytest.mark.parametrize(
        (
            "provider_cls",
            "provider_name",
            "base_url",
        ),
        _PROVIDER_CASES,
    )
    def test_structured_text_content_is_converted_for_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider_cls: type[BaseOpenAIChatCompletionsModel],
        provider_name: str,
        base_url: str,
    ) -> None:
        del provider_name
        del base_url

        captured = _install_fake_openai(
            monkeypatch,
            response=_make_response(
                choices=[_make_choice("ok")],
            ),
        )

        provider = provider_cls(
            model="test-model",
            api_key="fake-key",
        )

        provider.chat(
            ChatRequest(
                messages=[
                    Message(
                        role="user",
                        content=[
                            TextContentPart(
                                type="text",
                                text="Part 1",
                            ),
                            TextContentPart(
                                type="text",
                                text="Part 2",
                            ),
                        ],
                    )
                ]
            )
        )

        assert captured["request_kwargs"]["messages"] == [
            {
                "role": "user",
                "content": "Part 1 Part 2",
            }
        ]

    @pytest.mark.parametrize(
        (
            "provider_cls",
            "provider_name",
            "base_url",
        ),
        _PROVIDER_CASES,
    )
    def test_malformed_response_raises_response_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider_cls: type[BaseOpenAIChatCompletionsModel],
        provider_name: str,
        base_url: str,
    ) -> None:
        del provider_name
        del base_url

        _install_fake_openai(
            monkeypatch,
            response=_make_response(choices=[]),
        )

        provider = provider_cls(
            model="test-model",
            api_key="fake-key",
        )

        with pytest.raises(
            ResponseError,
            match="no choices",
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


class TestOpenAICompatibleExceptionMapping:
    @pytest.mark.parametrize(
        (
            "sdk_error",
            "expected_error",
            "match",
        ),
        [
            (
                _OpenAIAuthenticationError("bad key"),
                AuthenticationError,
                "authentication failed",
            ),
            (
                _OpenAIRateLimitError("too many requests"),
                RateLimitError,
                "rate limit exceeded",
            ),
            (
                _OpenAIConnectionError("network unavailable"),
                RequestError,
                "connection error",
            ),
            (
                _OpenAIStatusError(
                    418,
                    "teapot",
                ),
                ProviderError,
                r"API error \(418\): teapot",
            ),
            (
                RuntimeError("unexpected failure"),
                RequestError,
                "Unexpected Moonshot request failure",
            ),
        ],
    )
    def test_shared_sdk_errors_are_mapped_to_package_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sdk_error: Exception,
        expected_error: type[Exception],
        match: str,
    ) -> None:
        _install_fake_openai(
            monkeypatch,
            error=sdk_error,
        )

        provider = MoonshotChatModel(
            model="test-model",
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

    def test_provider_name_is_used_in_deepseek_error_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_openai(
            monkeypatch,
            error=_OpenAIAuthenticationError("bad key"),
        )

        provider = DeepSeekChatModel(
            model="test-model",
            api_key="fake-key",
        )

        with pytest.raises(
            AuthenticationError,
            match="Deepseek authentication failed",
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
