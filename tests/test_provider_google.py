"""Tests for the Google Gemini chat provider."""

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
from LLMUtilities.providers.google import (
    GoogleChatModel,
    _extract_stop_reason,
    _extract_text,
    _extract_usage,
)
from LLMUtilities.types import (
    ChatRequest,
    ImageContentPart,
    Message,
    TextContentPart,
)


# ---------------------------------------------------------------------------
# Fake Google response objects
# ---------------------------------------------------------------------------


def _make_text_part(text: str | None) -> types.SimpleNamespace:
    return types.SimpleNamespace(text=text)


def _make_candidate(
    *texts: str | None,
    finish_reason: Any = "STOP",
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        content=types.SimpleNamespace(
            parts=[_make_text_part(text) for text in texts],
        ),
        finish_reason=finish_reason,
    )


def _make_response(
    *,
    candidates: Any,
    usage_metadata: Any = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        candidates=candidates,
        usage_metadata=usage_metadata,
    )


# ---------------------------------------------------------------------------
# Fake Google SDK
# ---------------------------------------------------------------------------


class _GoogleAPIError(Exception):
    def __init__(
        self,
        code: int | None,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _FakePart:
    def __init__(self, *, text: str) -> None:
        self.text = text


class _FakeContent:
    def __init__(
        self,
        *,
        role: str,
        parts: list[_FakePart],
    ) -> None:
        self.role = role
        self.parts = parts


class _FakeGenerateContentConfig:
    def __init__(
        self,
        *,
        system_instruction: str | None,
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> None:
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens


def _fake_genai_types() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        Part=_FakePart,
        Content=_FakeContent,
        GenerateContentConfig=_FakeGenerateContentConfig,
    )


def _install_fake_google(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: Any = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    """
    Install a fake ``google.genai`` package.

    The production adapter imports Google Gen AI lazily inside ``chat()``, so
    replacing the SDK modules exercises the complete adapter without making a
    network request.
    """
    captured: dict[str, Any] = {}

    class _Models:
        @staticmethod
        def generate_content(**kwargs: Any) -> Any:
            captured["request_kwargs"] = kwargs

            if error is not None:
                raise error

            return response

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.models = _Models()

    google_module = types.ModuleType("google")
    google_module.__path__ = []

    genai_module = types.ModuleType("google.genai")
    genai_module.__path__ = []

    errors_module = types.ModuleType("google.genai.errors")
    types_module = types.ModuleType("google.genai.types")

    setattr(genai_module, "Client", _Client)

    setattr(errors_module, "APIError", _GoogleAPIError)

    setattr(types_module, "Part", _FakePart)
    setattr(types_module, "Content", _FakeContent)
    setattr(
        types_module,
        "GenerateContentConfig",
        _FakeGenerateContentConfig,
    )

    setattr(genai_module, "errors", errors_module)
    setattr(genai_module, "types", types_module)
    setattr(google_module, "genai", genai_module)

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(
        sys.modules,
        "google.genai.errors",
        errors_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "google.genai.types",
        types_module,
    )

    return captured


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


class TestGoogleTextExtraction:
    def test_missing_candidates_field_raises_response_error(self) -> None:
        response = types.SimpleNamespace()

        with pytest.raises(
            ResponseError,
            match="no candidates",
        ):
            _extract_text(response)

    def test_none_candidates_raises_response_error(self) -> None:
        response = _make_response(candidates=None)

        with pytest.raises(
            ResponseError,
            match="no candidates",
        ):
            _extract_text(response)

    def test_empty_candidates_raises_response_error(self) -> None:
        response = _make_response(candidates=[])

        with pytest.raises(
            ResponseError,
            match="no candidates",
        ):
            _extract_text(response)

    def test_candidate_without_content_is_skipped(self) -> None:
        response = _make_response(
            candidates=[
                types.SimpleNamespace(
                    content=None,
                    finish_reason="STOP",
                )
            ]
        )

        with pytest.raises(
            ResponseError,
            match="no text parts",
        ):
            _extract_text(response)

    def test_candidate_without_parts_is_skipped(self) -> None:
        response = _make_response(
            candidates=[
                types.SimpleNamespace(
                    content=types.SimpleNamespace(parts=None),
                    finish_reason="STOP",
                )
            ]
        )

        with pytest.raises(
            ResponseError,
            match="no text parts",
        ):
            _extract_text(response)

    def test_empty_text_parts_raise_response_error(self) -> None:
        response = _make_response(
            candidates=[_make_candidate("", None)]
        )

        with pytest.raises(
            ResponseError,
            match="no text parts",
        ):
            _extract_text(response)

    def test_single_text_part(self) -> None:
        response = _make_response(
            candidates=[_make_candidate("hello")]
        )

        assert _extract_text(response) == "hello"

    def test_multiple_parts_are_concatenated(self) -> None:
        response = _make_response(
            candidates=[
                _make_candidate(
                    "foo",
                    " bar",
                    " baz",
                )
            ]
        )

        assert _extract_text(response) == "foo bar baz"

    def test_multiple_candidates_are_concatenated(self) -> None:
        response = _make_response(
            candidates=[
                _make_candidate("first"),
                _make_candidate(" second"),
            ]
        )

        assert _extract_text(response) == "first second"

    def test_empty_parts_are_ignored_when_text_exists(self) -> None:
        response = _make_response(
            candidates=[
                _make_candidate(
                    None,
                    "",
                    "kept",
                )
            ]
        )

        assert _extract_text(response) == "kept"


# ---------------------------------------------------------------------------
# Content normalisation
# ---------------------------------------------------------------------------


class TestGoogleContentNormalisation:
    def test_string_content_is_unchanged(self) -> None:
        assert GoogleChatModel._normalize_content("hello") == "hello"

    def test_dictionary_text_parts_are_joined(self) -> None:
        content = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]

        assert (
            GoogleChatModel._normalize_content(content)
            == "Part 1 Part 2"
        )

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

        assert (
            GoogleChatModel._normalize_content(content)
            == "Part 1 Part 2"
        )

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

        assert (
            GoogleChatModel._normalize_content(content)
            == "before after"
        )

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

        assert (
            GoogleChatModel._normalize_content(content)
            == "before after"
        )

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

        assert GoogleChatModel._normalize_content(content) == ""

    def test_unrecognised_content_is_converted_to_string(self) -> None:
        assert GoogleChatModel._normalize_content(42) == "42"


# ---------------------------------------------------------------------------
# Role mapping
# ---------------------------------------------------------------------------


class TestGoogleRoleMapping:
    def test_user_role_maps_to_user(self) -> None:
        assert GoogleChatModel._map_role("user") == "user"

    def test_assistant_role_maps_to_model(self) -> None:
        assert GoogleChatModel._map_role("assistant") == "model"

    def test_unsupported_role_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError,
            match="Unsupported Google message role",
        ):
            GoogleChatModel._map_role("system")


# ---------------------------------------------------------------------------
# Message splitting
# ---------------------------------------------------------------------------


class TestGoogleMessageSplitting:
    def test_system_messages_are_joined_and_separated(
        self,
    ) -> None:
        request = ChatRequest(
            messages=[
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
                    content="Hello?",
                ),
                Message(
                    role="assistant",
                    content="Previous answer.",
                ),
            ]
        )

        system_instruction, contents = GoogleChatModel._split_messages(
            request,
            _fake_genai_types(),
        )

        assert system_instruction == (
            "First instruction.\n\n"
            "Second instruction."
        )

        assert len(contents) == 2

        assert contents[0].role == "user"
        assert len(contents[0].parts) == 1
        assert contents[0].parts[0].text == "Hello?"

        assert contents[1].role == "model"
        assert len(contents[1].parts) == 1
        assert contents[1].parts[0].text == "Previous answer."

    def test_request_without_system_messages_returns_none(
        self,
    ) -> None:
        request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content="Hello?",
                )
            ]
        )

        system_instruction, contents = GoogleChatModel._split_messages(
            request,
            _fake_genai_types(),
        )

        assert system_instruction is None
        assert len(contents) == 1

    def test_system_messages_are_not_added_to_contents(
        self,
    ) -> None:
        request = ChatRequest(
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

        _, contents = GoogleChatModel._split_messages(
            request,
            _fake_genai_types(),
        )

        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "User message."

    def test_system_only_request_raises_value_error(self) -> None:
        request = ChatRequest(
            messages=[
                Message(
                    role="system",
                    content="System instruction.",
                )
            ]
        )

        with pytest.raises(
            ValueError,
            match="at least one non-system message",
        ):
            GoogleChatModel._split_messages(
                request,
                _fake_genai_types(),
            )


# ---------------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------------


class TestGoogleGenerationConfig:
    def test_config_is_none_when_all_values_are_unset(self) -> None:
        config = GoogleChatModel._build_config(
            system_instruction=None,
            temperature=None,
            max_output_tokens=None,
            genai_types=_fake_genai_types(),
        )

        assert config is None

    def test_config_contains_all_supplied_values(self) -> None:
        config = GoogleChatModel._build_config(
            system_instruction="System instruction.",
            temperature=0.25,
            max_output_tokens=321,
            genai_types=_fake_genai_types(),
        )

        assert config.system_instruction == "System instruction."
        assert config.temperature == 0.25
        assert config.max_output_tokens == 321

    def test_config_is_created_for_system_instruction_only(self) -> None:
        config = GoogleChatModel._build_config(
            system_instruction="System instruction.",
            temperature=None,
            max_output_tokens=None,
            genai_types=_fake_genai_types(),
        )

        assert config.system_instruction == "System instruction."
        assert config.temperature is None
        assert config.max_output_tokens is None

    def test_config_is_created_for_temperature_only(self) -> None:
        config = GoogleChatModel._build_config(
            system_instruction=None,
            temperature=0.5,
            max_output_tokens=None,
            genai_types=_fake_genai_types(),
        )

        assert config.system_instruction is None
        assert config.temperature == 0.5
        assert config.max_output_tokens is None

    def test_config_is_created_for_max_output_tokens_only(self) -> None:
        config = GoogleChatModel._build_config(
            system_instruction=None,
            temperature=None,
            max_output_tokens=123,
            genai_types=_fake_genai_types(),
        )

        assert config.system_instruction is None
        assert config.temperature is None
        assert config.max_output_tokens == 123


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


class TestGoogleUsageExtraction:
    def test_usage_fields_are_normalised(self) -> None:
        response = _make_response(
            candidates=[],
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=30,
                candidates_token_count=12,
                total_token_count=42,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens == 30
        assert usage.output_tokens == 12
        assert usage.total_tokens == 42

    def test_total_is_computed_when_absent(self) -> None:
        response = _make_response(
            candidates=[],
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=30,
                candidates_token_count=12,
                total_token_count=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.total_tokens == 42

    def test_total_is_computed_with_missing_input_tokens(self) -> None:
        response = _make_response(
            candidates=[],
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=None,
                candidates_token_count=12,
                total_token_count=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens is None
        assert usage.output_tokens == 12
        assert usage.total_tokens == 12

    def test_total_is_computed_with_missing_output_tokens(self) -> None:
        response = _make_response(
            candidates=[],
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=30,
                candidates_token_count=None,
                total_token_count=None,
            ),
        )

        usage = _extract_usage(response)

        assert usage.input_tokens == 30
        assert usage.output_tokens is None
        assert usage.total_tokens == 30

    def test_missing_usage_returns_empty_usage(self) -> None:
        response = _make_response(
            candidates=[],
            usage_metadata=None,
        )

        usage = _extract_usage(response)

        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert usage.total_tokens is None


# ---------------------------------------------------------------------------
# Stop-reason extraction
# ---------------------------------------------------------------------------


class TestGoogleStopReasonExtraction:
    def test_missing_candidates_returns_none(self) -> None:
        response = types.SimpleNamespace()

        assert _extract_stop_reason(response) is None

    def test_none_candidates_returns_none(self) -> None:
        response = _make_response(candidates=None)

        assert _extract_stop_reason(response) is None

    def test_empty_candidates_returns_none(self) -> None:
        response = _make_response(candidates=[])

        assert _extract_stop_reason(response) is None

    def test_first_candidate_finish_reason_is_returned(self) -> None:
        response = _make_response(
            candidates=[
                _make_candidate(
                    "first",
                    finish_reason="STOP",
                ),
                _make_candidate(
                    "second",
                    finish_reason="MAX_TOKENS",
                ),
            ]
        )

        assert _extract_stop_reason(response) == "STOP"

    def test_finish_reason_is_converted_to_string(self) -> None:
        finish_reason = types.SimpleNamespace(name="STOP")
        response = _make_response(
            candidates=[
                _make_candidate(
                    "hello",
                    finish_reason=finish_reason,
                )
            ]
        )

        assert _extract_stop_reason(response) == str(finish_reason)

    def test_none_finish_reason_returns_none(self) -> None:
        response = _make_response(
            candidates=[
                _make_candidate(
                    "hello",
                    finish_reason=None,
                )
            ]
        )

        assert _extract_stop_reason(response) is None


# ---------------------------------------------------------------------------
# Complete adapter behaviour
# ---------------------------------------------------------------------------


class TestGoogleChatModel:
    def test_missing_sdk_raises_package_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setitem(sys.modules, "google", None)
        monkeypatch.setitem(sys.modules, "google.genai", None)

        provider = GoogleChatModel(
            model="gemini-test",
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
            match="google-genai",
        ):
            provider.chat(request)

    def test_chat_constructs_request_and_normalises_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        raw_response = _make_response(
            candidates=[
                _make_candidate(
                    "Hello",
                    " world",
                    finish_reason="STOP",
                )
            ],
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=3,
                total_token_count=15,
            ),
        )

        captured = _install_fake_google(
            monkeypatch,
            response=raw_response,
        )

        provider = GoogleChatModel(
            model="gemini-test",
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
        }

        request_kwargs = captured["request_kwargs"]

        assert request_kwargs["model"] == "gemini-test"

        contents = request_kwargs["contents"]
        assert len(contents) == 2

        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello?"

        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Previous answer."

        config = request_kwargs["config"]
        assert config.system_instruction == (
            "First system instruction.\n\n"
            "Second system instruction."
        )
        assert config.temperature == 0.25
        assert config.max_output_tokens == 321

        assert response.text == "Hello world"
        assert response.provider == "google"
        assert response.model == "gemini-test"
        assert response.usage is not None
        assert response.usage.input_tokens == 12
        assert response.usage.output_tokens == 3
        assert response.usage.total_tokens == 15
        assert response.stop_reason == "STOP"
        assert response.raw is raw_response

    def test_request_without_optional_configuration_passes_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_google(
            monkeypatch,
            response=_make_response(
                candidates=[_make_candidate("ok")]
            ),
        )

        provider = GoogleChatModel(
            model="gemini-test",
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

        assert captured["request_kwargs"]["config"] is None

    def test_system_instruction_creates_configuration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_google(
            monkeypatch,
            response=_make_response(
                candidates=[_make_candidate("ok")]
            ),
        )

        provider = GoogleChatModel(
            model="gemini-test",
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
                        content="hello",
                    ),
                ]
            )
        )

        config = captured["request_kwargs"]["config"]

        assert config is not None
        assert config.system_instruction == "System instruction."
        assert config.temperature is None
        assert config.max_output_tokens is None

    def test_request_model_overrides_provider_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_fake_google(
            monkeypatch,
            response=_make_response(
                candidates=[_make_candidate("ok")]
            ),
        )

        provider = GoogleChatModel(
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

    def test_system_only_request_raises_value_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_google(
            monkeypatch,
            response=_make_response(
                candidates=[_make_candidate("unused")]
            ),
        )

        provider = GoogleChatModel(
            model="gemini-test",
            api_key="fake-key",
        )

        with pytest.raises(
            ValueError,
            match="at least one non-system message",
        ):
            provider.chat(
                ChatRequest(
                    messages=[
                        Message(
                            role="system",
                            content="System instruction.",
                        )
                    ]
                )
            )

    def test_malformed_response_raises_response_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_google(
            monkeypatch,
            response=_make_response(candidates=[]),
        )

        provider = GoogleChatModel(
            model="gemini-test",
            api_key="fake-key",
        )

        with pytest.raises(
            ResponseError,
            match="no candidates",
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


class TestGoogleExceptionMapping:
    @pytest.mark.parametrize(
        (
            "status_code",
            "expected_error",
            "match",
        ),
        [
            (
                401,
                AuthenticationError,
                "authentication failed",
            ),
            (
                403,
                AuthenticationError,
                "authentication failed",
            ),
            (
                429,
                RateLimitError,
                "rate limit or quota exceeded",
            ),
            (
                408,
                RequestError,
                "request failed",
            ),
            (
                500,
                RequestError,
                "request failed",
            ),
            (
                502,
                RequestError,
                "request failed",
            ),
            (
                503,
                RequestError,
                "request failed",
            ),
            (
                504,
                RequestError,
                "request failed",
            ),
            (
                400,
                ProviderError,
                r"API error \(400\)",
            ),
        ],
    )
    def test_api_errors_are_mapped_by_status_code(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status_code: int,
        expected_error: type[Exception],
        match: str,
    ) -> None:
        _install_fake_google(
            monkeypatch,
            error=_GoogleAPIError(
                status_code,
                "provider failure",
            ),
        )

        provider = GoogleChatModel(
            model="gemini-test",
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

    def test_api_error_without_status_code_raises_provider_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_google(
            monkeypatch,
            error=_GoogleAPIError(
                None,
                "unknown provider failure",
            ),
        )

        provider = GoogleChatModel(
            model="gemini-test",
            api_key="fake-key",
        )

        with pytest.raises(
            ProviderError,
            match=r"API error \(None\)",
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

    def test_unexpected_exception_raises_request_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_google(
            monkeypatch,
            error=RuntimeError("unexpected failure"),
        )

        provider = GoogleChatModel(
            model="gemini-test",
            api_key="fake-key",
        )

        with pytest.raises(
            RequestError,
            match="Unexpected Google request failure",
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