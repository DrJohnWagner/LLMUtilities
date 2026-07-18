"""Tests for the OpenAI chat provider."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from LLMUtilities.exceptions import (
    AuthenticationError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)
from LLMUtilities.providers.openai import (
    OpenAIChatModel,
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
# Fake OpenAI response objects
# ---------------------------------------------------------------------------


def _make_text_part(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        type="output_text",
        text=text,
    )


def _make_non_text_part(
    part_type: str = "tool_call",
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        type=part_type,
        text="ignored",
    )


def _make_message_item(*texts: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        content=[_make_text_part(text) for text in texts],
    )


def _make_response(
    *,
    output: Any,
    usage: Any = None,
    status: str = "completed",
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        output=output,
        usage=usage,
        status=status,
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
    Install a fake ``openai`` module and return captured client and request data.

    The production adapter imports the SDK lazily inside ``chat()``, so replacing
    ``sys.modules["openai"]`` exercises the complete adapter without making a
    network request.
    """
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.responses = types.SimpleNamespace(create=self._create)

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
# Text extraction
# ---------------------------------------------------------------------------


class TestOpenAITextExtraction:
    def test_missing_output_field_raises_response_error(self) -> None:
        response = types.SimpleNamespace(status="completed")

        with pytest.raises(
            ResponseError,
            match="missing the 'output' field",
        ):
            _extract_text(response)

    def test_none_output_raises_response_error(self) -> None:
        response = _make_response(output=None)

        with pytest.raises(
            ResponseError,
            match="missing the 'output' field",
        ):
            _extract_text(response)

    def test_empty_output_raises_response_error(self) -> None:
        response = _make_response(output=[])

        with pytest.raises(
            ResponseError,
            match="no text output",
        ):
            _extract_text(response)

    def test_output_item_without_content_is_skipped(self) -> None:
        response = _make_response(
            output=[types.SimpleNamespace(content=None)],
        )

        with pytest.raises(
            ResponseError,
            match="no text output",
        ):
            _extract_text(response)

    def test_output_with_no_text_parts_raises_response_error(self) -> None:
        response = _make_response(
            output=[
                types.SimpleNamespace(
                    content=[_make_non_text_part()],
                )
            ],
        )

        with pytest.raises(
            ResponseError,
            match="no text output",
        ):
            _extract_text(response)

    def test_single_text_segment(self) -> None:
        response = _make_response(
            output=[_make_message_item("hello")],
        )

        assert _extract_text(response) == "hello"

    def test_multiple_segments_are_concatenated(self) -> None:
        response = _make_response(
            output=[
                _make_message_item(
                    "foo",
                    " bar",
                    " baz",
                )
            ],
        )

        assert _extract_text(response) == "foo bar baz"

    def test_multiple_output_items_are_concatenated(self) -> None:
        response = _make_response(
            output=[
                _make_message_item("part one"),
                _make_message_item(" part two"),
            ],
        )

        assert _extract_text(response) == "part one part two"

    def test_non_text_parts_are_skipped(self) -> None:
        response = _make_response(
            output=[
                types.SimpleNamespace(
                    content=[
                        _make_non_text_part(),
                        _make_text_part("kept"),
                    ],
                )
            ],
        )

        assert _extract_text(response) == "kept"

    def test_error_includes_response_status(self) -> None:
        response = _make_response(
            output=[],
            status="incomplete",
        )

        with pytest.raises(
            ResponseError,
            match="incomplete",
        ):
            _extract_text(response)


# ---------------------------------------------------------------------------
# Content normalisation
# ---------------------------------------------------------------------------


class TestOpenAIContentNormalisation:
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


class TestOpenAIUsageExtraction:
    def test_usage_fields_are_normalised(self) -> None:
        response = _make_response(
            output=[],
            usage=types.SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.total_tokens == 15

    def test_total_is_computed_when_absent(self) -> None:
        response = _make_response(
            output=[],
            usage=types.SimpleNamespace(
                input_tokens=8,
                output_tokens=4,
                total_tokens=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens == 8
        assert usage.output_tokens == 4
        assert usage.total_tokens == 12

    def test_total_is_computed_with_missing_input_tokens(self) -> None:
        response = _make_response(
            output=[],
            usage=types.SimpleNamespace(
                input_tokens=None,
                output_tokens=4,
                total_tokens=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens is None
        assert usage.output_tokens == 4
        assert usage.total_tokens == 4

    def test_total_is_computed_with_missing_output_tokens(self) -> None:
        response = _make_response(
            output=[],
            usage=types.SimpleNamespace(
                input_tokens=8,
                output_tokens=None,
                total_tokens=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens == 8
        assert usage.output_tokens is None
        assert usage.total_tokens == 8

    def test_missing_usage_returns_empty_usage(self) -> None:
        response = _make_response(
            output=[],
            usage=None,
        )

        usage = _extract_usage(response)

        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert usage.total_tokens is None


# ---------------------------------------------------------------------------
# Complete adapter behaviour
# ---------------------------------------------------------------------------


class TestOpenAIChatModel:
    def test_missing_sdk_raises_package_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setitem(sys.modules, "openai", None)

        provider = OpenAIChatModel(
            model="gpt-test",
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

    def test_chat_constructs_request_and_normalises_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        raw_response = _make_response(
            output=[
                _make_message_item(
                    "Hello",
                    " world",
                )
            ],
            usage=types.SimpleNamespace(
                input_tokens=12,
                output_tokens=3,
                total_tokens=15,
            ),
            status="completed",
        )

        captured = _install_fake_openai(
            monkeypatch,
            response=raw_response,
        )

        provider = OpenAIChatModel(
            model="gpt-test",
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
            "model": "gpt-test",
            "input": [
                {
                    "role": "user",
                    "content": "Hello?",
                },
                {
                    "role": "assistant",
                    "content": "Previous answer.",
                },
            ],
            "instructions": (
                "First system instruction.\n\n"
                "Second system instruction."
            ),
            "temperature": 0.25,
            "max_output_tokens": 321,
        }

        assert response.text == "Hello world"
        assert response.provider == "openai"
        assert response.model == "gpt-test"
        assert response.usage is not None
        assert response.usage.input_tokens == 12
        assert response.usage.output_tokens == 3
        assert response.usage.total_tokens == 15
        assert response.stop_reason is None
        assert response.raw is raw_response

    def test_response_status_is_not_used_as_stop_reason(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        raw_response = _make_response(
            output=[_make_message_item("hello")],
            status="completed",
        )

        _install_fake_openai(
            monkeypatch,
            response=raw_response,
        )

        provider = OpenAIChatModel(
            model="gpt-test",
            api_key="fake-key",
        )

        response = provider.chat(
            ChatRequest(
                messages=[
                    Message(
                        role="user",
                        content="hello",
                    )
                ]
            )
        )

        assert raw_response.status == "completed"
        assert response.stop_reason is None
        assert response.stop_reason != raw_response.status

    def test_request_model_overrides_provider_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_openai(
            monkeypatch,
            response=_make_response(
                output=[_make_message_item("ok")],
            ),
        )

        provider = OpenAIChatModel(
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

    def test_optional_request_fields_are_omitted_when_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_openai(
            monkeypatch,
            response=_make_response(
                output=[_make_message_item("ok")],
            ),
        )

        provider = OpenAIChatModel(
            model="gpt-test",
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

        assert "instructions" not in request_kwargs
        assert "temperature" not in request_kwargs
        assert "max_output_tokens" not in request_kwargs

    def test_malformed_response_raises_response_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_openai(
            monkeypatch,
            response=_make_response(output=[]),
        )

        provider = OpenAIChatModel(
            model="gpt-test",
            api_key="fake-key",
        )

        with pytest.raises(
            ResponseError,
            match="no text output",
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


class TestOpenAIExceptionMapping:
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
                "Unexpected OpenAI request failure",
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
        _install_fake_openai(
            monkeypatch,
            error=sdk_error,
        )

        provider = OpenAIChatModel(
            model="gpt-test",
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