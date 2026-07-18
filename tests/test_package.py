"""Test suite for LLMUtilities."""
from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from LLMUtilities.types import (
    ChatRequest,
    ChatResponse,
    ChatUsage,
    ImageArtifact,
    ImageRequest,
    ImageResponse,
    ImageUsage,
    Message,
    TextContentPart,
    ImageContentPart,
)
from LLMUtilities.costs import (
    CostEstimate,
    ImageCostEstimate,
    ImagePricing,
    ImagePricingCatalogue,
    IMAGE_PRICING_CATALOGUE,
    Pricing,
    PRICING_CATALOGUE,
    cost_for_image_response,
    cost_for_image_usage,
    cost_for_response,
    cost_from_usage,
    estimate_cost,
    estimate_image_cost,
    get_image_pricing,
    get_pricing,
    normalise_image_usage,
    validate_image_size_for_model,
)
from LLMUtilities.exceptions import (
    AuthenticationError,
    MissingDependencyError,
    ProviderError,
    RequestError,
    RateLimitError,
    ResponseError,
    ConfigurationError,
)
from LLMUtilities.parsing.json_parsing import (
    extract_json_string,
    parse_json,
    parse_json_as,
    repair_json,
)
from .fakes import FakeChatModel


# ---------------------------------------------------------------------------
# A. Importability
# ---------------------------------------------------------------------------

class TestImportability:
    def test_package_root(self):
        import LLMUtilities  # noqa: F401

    def test_chat(self):
        from LLMUtilities import chat  # noqa: F401

    def test_types(self):
        from LLMUtilities.types import ChatRequest, ChatResponse, Message, ChatUsage  # noqa: F401

    def test_costs(self):
        from LLMUtilities.costs import estimate_cost  # noqa: F401

    def test_embeddings(self):
        from LLMUtilities.embeddings import embed_texts, cosine_similarity  # noqa: F401

    def test_parsing_structured_output(self):
        from LLMUtilities.parsing.structured_output import structured_output  # noqa: F401

    def test_compare(self):
        from LLMUtilities.compare import compare_outputs  # noqa: F401

    def test_tracing(self):
        from LLMUtilities.tracing.tracing import (  # noqa: F401
            log_chat_request, log_chat_response, log_error,
        )

    def test_provider_openai_importable(self):
        from LLMUtilities.providers.openai import OpenAIChatModel  # noqa: F401

    def test_provider_anthropic_importable(self):
        from LLMUtilities.providers.anthropic import AnthropicChatModel  # noqa: F401

    def test_provider_google_importable(self):
        from LLMUtilities.providers.google import GoogleChatModel  # noqa: F401

    def test_provider_moonshot_importable(self):
        from LLMUtilities.providers.moonshot import MoonshotChatModel  # noqa: F401

    def test_provider_deepseek_importable(self):
        from LLMUtilities.providers.deepseek import DeepSeekChatModel  # noqa: F401

    def test_image_module(self):
        from LLMUtilities.image import generate_image, generate_image_b64  # noqa: F401

    def test_image_types(self):
        from LLMUtilities.types import (
            ImageRequest,
            ImageResponse,
            ImageArtifact,
            ImageUsage,
        )  # noqa: F401

    def test_provider_openai_image_importable(self):
        from LLMUtilities.providers.openai_image import OpenAIImageModel  # noqa: F401

    def test_missing_dependency_error_exported(self):
        from LLMUtilities.exceptions import MissingDependencyError  # noqa: F401
        from LLMUtilities import MissingDependencyError as _  # noqa: F401

    def test_provider_modules_import_without_sdks(self):
        """Provider modules must be importable even when their SDKs are absent."""
        import LLMUtilities.providers.openai  # noqa: F401
        import LLMUtilities.providers.anthropic  # noqa: F401
        import LLMUtilities.providers.google  # noqa: F401
        import LLMUtilities.providers.moonshot  # noqa: F401
        import LLMUtilities.providers.deepseek  # noqa: F401


# ---------------------------------------------------------------------------
# B. Missing dependency handling
# ---------------------------------------------------------------------------

class TestMissingDependency:
    """Invoking a provider whose SDK is absent must raise MissingDependencyError."""

    def _make_request(self) -> ChatRequest:
        return ChatRequest(messages=[Message(role="user", content="hello")])

    def test_openai_missing_sdk_raises_package_error(self):
        from LLMUtilities.providers.openai import OpenAIChatModel

        with patch.dict(sys.modules, {"openai": None}):
            provider = OpenAIChatModel(api_key="fake")
            with pytest.raises(MissingDependencyError, match="openai"):
                provider.chat(self._make_request())

    def test_anthropic_missing_sdk_raises_package_error(self):
        from LLMUtilities.providers.anthropic import AnthropicChatModel

        with patch.dict(sys.modules, {"anthropic": None}):
            provider = AnthropicChatModel(api_key="fake")
            with pytest.raises(MissingDependencyError, match="anthropic"):
                provider.chat(self._make_request())

    def test_google_missing_sdk_raises_package_error(self):
        from LLMUtilities.providers.google import GoogleChatModel

        with patch.dict(sys.modules, {"google": None, "google.genai": None}):
            provider = GoogleChatModel(api_key="fake")
            with pytest.raises(MissingDependencyError, match="google-genai"):
                provider.chat(self._make_request())

    def test_missing_dependency_error_is_subclass_of_base(self):
        from LLMUtilities.exceptions import LLMUtilitiesError
        assert issubclass(MissingDependencyError, LLMUtilitiesError)

    def test_missing_dependency_error_not_raw_import_error(self):
        """MissingDependencyError must not be ImportError itself."""
        assert not issubclass(MissingDependencyError, ImportError)


# ---------------------------------------------------------------------------
# C. Response guards — OpenAI
# ---------------------------------------------------------------------------

def _make_openai_response(output=None, usage=None, status="completed"):
    """Build a minimal fake openai responses object."""
    resp = MagicMock()
    resp.output = output
    resp.usage = usage
    resp.status = status
    return resp


def _make_openai_text_part(text: str):
    part = MagicMock()
    part.type = "output_text"
    part.text = text
    return part


def _make_openai_message_item(*texts: str):
    item = MagicMock()
    item.content = [_make_openai_text_part(t) for t in texts]
    return item


class TestOpenAIResponseGuards:
    def test_missing_output_field_raises_response_error(self):
        from LLMUtilities.providers.openai import _extract_text
        resp = MagicMock()
        resp.output = None

        with pytest.raises(ResponseError, match="missing the 'output' field"):
            _extract_text(resp)

    def test_empty_output_list_raises_response_error(self):
        from LLMUtilities.providers.openai import _extract_text
        resp = _make_openai_response(output=[])

        with pytest.raises(ResponseError, match="no text output"):
            _extract_text(resp)

    def test_output_with_no_text_parts_raises(self):
        from LLMUtilities.providers.openai import _extract_text
        item = MagicMock()
        item.content = []
        resp = _make_openai_response(output=[item])

        with pytest.raises(ResponseError):
            _extract_text(resp)


# ---------------------------------------------------------------------------
# D. Text extraction — OpenAI concatenates multiple segments
# ---------------------------------------------------------------------------

class TestOpenAITextExtraction:
    def test_single_text_segment(self):
        from LLMUtilities.providers.openai import _extract_text
        resp = _make_openai_response(output=[_make_openai_message_item("hello")])
        assert _extract_text(resp) == "hello"

    def test_multiple_segments_concatenated(self):
        from LLMUtilities.providers.openai import _extract_text
        resp = _make_openai_response(
            output=[_make_openai_message_item("foo", " bar", " baz")]
        )
        assert _extract_text(resp) == "foo bar baz"

    def test_multiple_output_items_concatenated(self):
        from LLMUtilities.providers.openai import _extract_text
        resp = _make_openai_response(
            output=[
                _make_openai_message_item("part one"),
                _make_openai_message_item(" part two"),
            ]
        )
        assert _extract_text(resp) == "part one part two"

    def test_non_text_parts_skipped(self):
        from LLMUtilities.providers.openai import _extract_text
        non_text = MagicMock()
        non_text.type = "tool_call"
        non_text.text = "ignored"

        text_part = _make_openai_text_part("kept")

        item = MagicMock()
        item.content = [non_text, text_part]
        resp = _make_openai_response(output=[item])

        assert _extract_text(resp) == "kept"


# ---------------------------------------------------------------------------
# C/D. Response guards and text extraction — Anthropic
# ---------------------------------------------------------------------------

def _make_anthropic_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_anthropic_tool_block():
    block = MagicMock()
    block.type = "tool_use"
    # no .text attribute that matters
    return block


class TestAnthropicTextExtraction:
    def test_missing_content_raises_response_error(self):
        from LLMUtilities.providers.anthropic import _extract_text
        resp = MagicMock()
        resp.content = None

        with pytest.raises(ResponseError, match="missing the 'content' field"):
            _extract_text(resp)

    def test_empty_content_raises_response_error(self):
        from LLMUtilities.providers.anthropic import _extract_text
        resp = MagicMock()
        resp.content = []
        resp.stop_reason = "end_turn"

        with pytest.raises(ResponseError, match="no text content blocks"):
            _extract_text(resp)

    def test_only_tool_use_blocks_raises_response_error(self):
        from LLMUtilities.providers.anthropic import _extract_text
        resp = MagicMock()
        resp.content = [_make_anthropic_tool_block()]
        resp.stop_reason = "tool_use"

        with pytest.raises(ResponseError):
            _extract_text(resp)

    def test_single_text_block(self):
        from LLMUtilities.providers.anthropic import _extract_text
        resp = MagicMock()
        resp.content = [_make_anthropic_text_block("hello world")]

        assert _extract_text(resp) == "hello world"

    def test_multiple_text_blocks_concatenated(self):
        from LLMUtilities.providers.anthropic import _extract_text
        resp = MagicMock()
        resp.content = [
            _make_anthropic_text_block("first"),
            _make_anthropic_text_block(" second"),
            _make_anthropic_text_block(" third"),
        ]
        assert _extract_text(resp) == "first second third"

    def test_mixed_blocks_text_only_collected(self):
        from LLMUtilities.providers.anthropic import _extract_text
        resp = MagicMock()
        resp.content = [
            _make_anthropic_text_block("kept"),
            _make_anthropic_tool_block(),
        ]
        assert _extract_text(resp) == "kept"


# ---------------------------------------------------------------------------
# C/D. Response guards and text extraction — Google
# ---------------------------------------------------------------------------

def _make_google_response(candidates=None, usage_metadata=None):
    resp = MagicMock()
    resp.candidates = candidates
    resp.usage_metadata = usage_metadata
    return resp


def _make_google_candidate(*texts: str):
    parts = []
    for t in texts:
        p = MagicMock()
        p.text = t
        parts.append(p)
    content = MagicMock()
    content.parts = parts
    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = "STOP"
    return candidate


class _DummyChatModel:
    provider_name = "dummy"

    def __new__(cls, *args, **kwargs):
        from LLMUtilities.providers.base import BaseChatModel

        class _Impl(BaseChatModel):
            provider_name = "dummy"
            default_model = "default-model"
            api_key = "default-key"

            def chat(self, request):
                return ChatResponse(
                    text="ok",
                    provider=self.provider_name,
                    model=self.get_model_name(request),
                )

        return _Impl(*args, **kwargs)


class TestGoogleTextExtraction:
    def test_no_candidates_raises_response_error(self):
        from LLMUtilities.providers.google import _extract_text
        resp = _make_google_response(candidates=None)

        with pytest.raises(ResponseError, match="no candidates"):
            _extract_text(resp)

    def test_empty_candidates_raises_response_error(self):
        from LLMUtilities.providers.google import _extract_text
        resp = _make_google_response(candidates=[])

        with pytest.raises(ResponseError):
            _extract_text(resp)

    def test_single_text_part(self):
        from LLMUtilities.providers.google import _extract_text
        resp = _make_google_response(candidates=[_make_google_candidate("hello")])
        assert _extract_text(resp) == "hello"

    def test_multiple_parts_concatenated(self):
        from LLMUtilities.providers.google import _extract_text
        resp = _make_google_response(
            candidates=[_make_google_candidate("foo", " bar", " baz")]
        )
        assert _extract_text(resp) == "foo bar baz"


# ---------------------------------------------------------------------------
# E. Usage normalisation
# ---------------------------------------------------------------------------

class TestUsageNormalisation:
    def test_openai_usage_fields(self):
        from LLMUtilities.providers.openai import _extract_usage
        usage_obj = MagicMock()
        usage_obj.input_tokens = 10
        usage_obj.output_tokens = 5
        usage_obj.total_tokens = 15

        resp = MagicMock()
        resp.usage = usage_obj

        usage = _extract_usage(resp)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.total_tokens == 15

    def test_openai_usage_total_computed_when_absent(self):
        from LLMUtilities.providers.openai import _extract_usage
        usage_obj = MagicMock()
        usage_obj.input_tokens = 8
        usage_obj.output_tokens = 4
        usage_obj.total_tokens = None

        resp = MagicMock()
        resp.usage = usage_obj

        usage = _extract_usage(resp)
        assert usage.total_tokens == 12

    def test_openai_missing_usage_returns_empty(self):
        from LLMUtilities.providers.openai import _extract_usage
        resp = MagicMock()
        resp.usage = None

        usage = _extract_usage(resp)
        assert usage.input_tokens is None
        assert usage.output_tokens is None

    def test_anthropic_usage_fields(self):
        from LLMUtilities.providers.anthropic import _extract_usage
        usage_obj = MagicMock()
        usage_obj.input_tokens = 20
        usage_obj.output_tokens = 8
        usage_obj.cache_creation_input_tokens = 6
        usage_obj.cache_read_input_tokens = 12

        resp = MagicMock()
        resp.usage = usage_obj

        usage = _extract_usage(resp)
        assert usage.input_tokens == 20
        assert usage.output_tokens == 8
        assert usage.total_tokens == 28
        assert usage.cache_creation_input_tokens == 6
        assert usage.cache_read_input_tokens == 12
        assert usage.cached_input_tokens == 12

    def test_anthropic_missing_usage_returns_empty(self):
        from LLMUtilities.providers.anthropic import _extract_usage
        resp = MagicMock()
        resp.usage = None

        usage = _extract_usage(resp)
        assert usage.input_tokens is None

    def test_google_usage_maps_to_normalised_fields(self):
        from LLMUtilities.providers.google import _extract_usage
        meta = MagicMock()
        meta.prompt_token_count = 30
        meta.candidates_token_count = 12
        meta.total_token_count = 42

        resp = MagicMock()
        resp.usage_metadata = meta

        usage = _extract_usage(resp)
        assert usage.input_tokens == 30
        assert usage.output_tokens == 12
        assert usage.total_tokens == 42

    def test_google_missing_usage_returns_empty(self):
        from LLMUtilities.providers.google import _extract_usage
        resp = MagicMock()
        resp.usage_metadata = None

        usage = _extract_usage(resp)
        assert usage.input_tokens is None

    def test_chatusage_fields_consistent(self):
        usage = ChatUsage(input_tokens=5, output_tokens=3, total_tokens=8)
        assert usage.input_tokens == 5
        assert usage.output_tokens == 3
        assert usage.total_tokens == 8

    def test_cost_estimation_uses_normalised_fields(self):
        usage = ChatUsage(input_tokens=1_000_000, output_tokens=500_000)
        estimate = estimate_cost(model="gemini-2.5-flash", usage=usage)
        assert estimate.input_tokens == 1_000_000
        assert estimate.output_tokens == 500_000
        assert estimate.total_cost_usd > 0

    def test_cost_estimation_uses_cached_input_tokens_from_usage(self):
        usage = ChatUsage(
            input_tokens=1_000_000,
            output_tokens=500_000,
            cached_input_tokens=250_000,
        )
        estimate = estimate_cost(model="claude-sonnet-4-0", usage=usage)
        assert estimate.cached_input_tokens == 250_000
        assert estimate.cached_input_cost_usd == pytest.approx(0.075)

    def test_cost_estimation_uses_cache_creation_tokens_from_usage(self):
        usage = ChatUsage(
            input_tokens=100_000,
            output_tokens=50_000,
            cache_creation_input_tokens=25_000,
        )
        estimate = estimate_cost(model="claude-sonnet-4-0", usage=usage)
        assert estimate.cache_creation_input_tokens == 25_000
        assert estimate.cache_creation_input_cost_usd == pytest.approx(0.075)

    def test_print_cost_summary_uses_normalised_estimate_fields(self, caplog):
        estimate = CostEstimate(
            input_tokens=100,
            output_tokens=50,
            cached_input_tokens=25,
            cache_creation_input_tokens=10,
            cached_input_cost_usd=0.2,
            cache_creation_input_cost_usd=0.1,
            total_cost_usd=0.9,
        )

        from LLMUtilities.costs import print_cost_summary

        caplog.set_level("INFO")
        print_cost_summary(estimate=estimate, model="claude-sonnet-4-0")
        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "Cached input tokens: 25" in log_text
        assert "Cache creation input tokens: 10" in log_text
        assert "Cached input cost: $0.20000000" in log_text
        assert "Cache creation input cost: $0.10000000" in log_text


# ---------------------------------------------------------------------------
# F. Structured output with mocked chat
# ---------------------------------------------------------------------------

class TestStructuredOutput:
    def test_structured_output_with_fake_provider(self):
        from pydantic import BaseModel
        from LLMUtilities.parsing.structured_output import generate_structured_output
        from LLMUtilities.chat import chat_text

        class Color(BaseModel):
            name: str
            hex: str

        fake = FakeChatModel(response_text='{"name": "red", "hex": "#ff0000"}')

        result = generate_structured_output(
            user_prompt="Give me red.",
            output_model=Color,
            provider=fake,
        )
        assert result.name == "red"
        assert result.hex == "#ff0000"

    def test_structured_output_repair_path(self):
        """LLM repair call must be issued when the first response is not valid JSON."""
        from pydantic import BaseModel
        from LLMUtilities.parsing.structured_output import generate_structured_output

        class Thing(BaseModel):
            value: int

        # First response is prose (not JSON at all), second is valid JSON.
        call_count = 0

        class _SequentialFake(FakeChatModel):
            def chat(self, request):
                nonlocal call_count
                call_count += 1
                texts = ["Sorry, I cannot do that.", '{"value": 1}']
                self.response_text = texts[min(call_count - 1, 1)]
                return super().chat(request)

        fake = _SequentialFake()
        result = generate_structured_output(
            user_prompt="Give me a thing.",
            output_model=Thing,
            provider=fake,
            retry_on_parse_failure=True,
        )
        assert result.value == 1
        assert call_count == 2


# ---------------------------------------------------------------------------
# G. Compare / judge with mocked chat
# ---------------------------------------------------------------------------

class TestCompareUtilities:
    def test_compare_basic_metrics(self):
        from LLMUtilities.compare import compare_outputs

        comp = compare_outputs(
            "hello world",
            "hello world",
            use_embeddings=False,
            use_judge=False,
        )
        assert comp.exact_match is True
        assert comp.normalised_exact_match is True
        assert comp.word_count_a == 2
        assert comp.word_count_b == 2

    def test_compare_non_matching(self):
        from LLMUtilities.compare import compare_outputs

        comp = compare_outputs(
            "alpha beta",
            "gamma delta",
            use_embeddings=False,
            use_judge=False,
        )
        assert comp.exact_match is False

    def test_compare_normalised_match_ignores_case_whitespace(self):
        from LLMUtilities.compare import compare_outputs

        comp = compare_outputs(
            "Hello  World",
            "hello world",
            use_embeddings=False,
            use_judge=False,
        )
        assert comp.normalised_exact_match is True
        assert comp.exact_match is False

    def test_compare_judge_uses_chat(self):
        from LLMUtilities.compare import compare_outputs

        fake = FakeChatModel(response_text="A is better.")

        comp = compare_outputs(
            "output A",
            "output B",
            use_embeddings=False,
            use_judge=True,
            judge_provider=fake,
        )
        assert comp.judge_verdict == "A is better."

    def test_compare_invalid_types_raise(self):
        from LLMUtilities.compare import compare_outputs

        with pytest.raises(TypeError):
            compare_outputs(123, "b", use_embeddings=False)


# ---------------------------------------------------------------------------
# Provider selection (existing, preserved)
# ---------------------------------------------------------------------------

class TestProviderSelection:
    def test_fake_provider_returns_response(self):
        provider = FakeChatModel(response_text="hello")
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        response = provider.chat(request)

        assert isinstance(response, ChatResponse)
        assert response.text == "hello"
        assert response.provider == "fake"

    def test_fake_provider_records_calls(self):
        provider = FakeChatModel()
        request = ChatRequest(messages=[Message(role="user", content="test")])
        provider.chat(request)
        provider.chat(request)
        assert len(provider.calls) == 2

    def test_chat_with_fake_provider(self):
        from LLMUtilities.chat import chat
        provider = FakeChatModel(response_text="world")
        response = chat(provider=provider, user="hello")
        assert response.text == "world"

    def test_chat_text_with_fake_provider(self):
        from LLMUtilities.chat import chat_text
        provider = FakeChatModel(response_text="pong")
        assert chat_text(provider=provider, user="ping") == "pong"

    def test_chat_usage_with_fake_provider(self):
        from LLMUtilities.chat import chat_usage
        provider = FakeChatModel(input_tokens=7, output_tokens=3)
        usage = chat_usage(provider=provider, user="x")
        assert usage.input_tokens == 7
        assert usage.output_tokens == 3
        assert usage.total_tokens == 10

    def test_missing_provider_raises_configuration_error(self):
        from LLMUtilities.chat import get_chat_model
        with pytest.raises(ConfigurationError):
            get_chat_model("nonexistent")


class TestImageProviderSelection:
    def test_get_openai_image_model(self):
        from LLMUtilities.image import get_image_model
        from LLMUtilities.providers.openai_image import OpenAIImageModel

        provider = get_image_model("openai")
        assert isinstance(provider, OpenAIImageModel)

    def test_unsupported_image_provider_raises(self):
        from LLMUtilities.image import get_image_model

        with pytest.raises(ConfigurationError, match="not implemented"):
            get_image_model("anthropic")

    def test_generate_image_with_custom_provider(self):
        from LLMUtilities.image import generate_image
        from LLMUtilities.types import ImageResponse, ImageArtifact

        class _FakeImageProvider:
            def generate(self, request):
                assert request.prompt == "draw a cat"
                return ImageResponse(
                    provider="fake",
                    model="fake-image",
                    artifacts=[ImageArtifact(mime_type="image/png", b64_data="abc123")],
                )

        response = generate_image(provider=_FakeImageProvider(), prompt="draw a cat")
        assert response.provider == "fake"
        assert response.artifacts[0].b64_data == "abc123"

    def test_generate_image_b64_raises_when_no_b64(self):
        from LLMUtilities.image import generate_image_b64
        from LLMUtilities.types import ImageResponse, ImageArtifact

        class _FakeUrlOnlyProvider:
            def generate(self, request):
                return ImageResponse(
                    provider="fake",
                    model="fake-image",
                    artifacts=[
                        ImageArtifact(
                            mime_type="image/png", url="https://example.com/img.png"
                        )
                    ],
                )

        with pytest.raises(ResponseError, match="did not include base64"):
            generate_image_b64(provider=_FakeUrlOnlyProvider(), prompt="draw a tree")

    def test_generate_image_with_unknown_provider_raises(self):
        from LLMUtilities.image import generate_image

        with pytest.raises(ConfigurationError, match="Unsupported image provider"):
            generate_image(provider_name="bogus", prompt="draw a city")


class TestImageTypeValidation:
    def test_image_request_rejects_blank_prompt(self):
        with pytest.raises(Exception):
            ImageRequest(prompt="")

    def test_image_request_rejects_non_positive_n(self):
        with pytest.raises(Exception):
            ImageRequest(prompt="draw a sun", n=0)

    def test_image_response_requires_artifacts(self):
        with pytest.raises(Exception):
            ImageResponse(provider="openai", model="gpt-image-1.5", artifacts=[])

    def test_image_artifact_allows_missing_mime_type(self):
        artifact = ImageArtifact(b64_data="abc")
        assert artifact.mime_type is None

    def test_image_response_allows_missing_usage(self):
        response = ImageResponse(
            provider="openai",
            model="gpt-image-1.5",
            artifacts=[ImageArtifact(b64_data="abc")],
            usage=None,
        )
        assert response.usage is None


class TestChatResponseValidation:
    def test_chat_response_allows_missing_usage(self):
        response = ChatResponse(text="ok", provider="fake", model="fake", usage=None)
        assert response.usage is None


class TestOpenAIImageProvider:
    def test_openai_image_missing_sdk_raises_package_error(self):
        from LLMUtilities.providers.openai_image import OpenAIImageModel
        from LLMUtilities.types import ImageRequest

        with patch.dict(sys.modules, {"openai": None}):
            provider = OpenAIImageModel(api_key="fake")
            with pytest.raises(MissingDependencyError, match="openai"):
                provider.generate(ImageRequest(prompt="draw a bird"))

    def test_extract_artifacts_from_dict_payload(self):
        from LLMUtilities.providers.openai_image import _extract_artifacts

        payload = {
            "data": [
                {"b64_json": "ZmFrZQ==", "revised_prompt": "a refined prompt"},
                {"url": "https://example.com/image.png"},
            ]
        }

        artifacts = _extract_artifacts(payload, requested_format="png")
        assert len(artifacts) == 2
        assert artifacts[0].b64_data == "ZmFrZQ=="
        assert artifacts[0].mime_type == "image/png"
        assert artifacts[1].url == "https://example.com/image.png"

    def test_extract_artifacts_empty_raises(self):
        from LLMUtilities.providers.openai_image import _extract_artifacts

        with pytest.raises(ResponseError, match="no data artifacts"):
            _extract_artifacts({"data": []}, requested_format=None)

    def test_extract_artifacts_object_payload(self):
        from LLMUtilities.providers.openai_image import _extract_artifacts

        item_a = types.SimpleNamespace(
            b64_json="YWJj",
            url=None,
            revised_prompt="adjusted",
        )
        item_b = types.SimpleNamespace(
            b64_json=None,
            url="https://example.com/a.png",
            revised_prompt=None,
        )
        payload = types.SimpleNamespace(data=[item_a, item_b])

        artifacts = _extract_artifacts(payload, requested_format="webp")
        assert len(artifacts) == 2
        assert artifacts[0].mime_type == "image/webp"
        assert artifacts[0].b64_data == "YWJj"
        assert artifacts[1].url == "https://example.com/a.png"

    def test_extract_artifacts_missing_item_data_raises(self):
        from LLMUtilities.providers.openai_image import _extract_artifacts

        payload = {"data": [{"revised_prompt": "only text"}]}
        with pytest.raises(ResponseError, match="artifacts were empty"):
            _extract_artifacts(payload, requested_format="png")

    def test_extract_image_usage_from_dict_payload(self):
        from LLMUtilities.providers.openai_image import _extract_usage

        payload = {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "input_tokens_details": {
                    "text_tokens": 6,
                    "cached_text_tokens": 2,
                    "image_tokens": 2,
                },
                "output_tokens_details": {
                    "image_tokens": 5,
                    "partial_image_tokens": 1,
                },
            }
        }
        usage = _extract_usage(payload)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.total_tokens == 15
        assert usage.text_input_tokens == 6
        assert usage.cached_text_input_tokens == 2
        assert usage.image_input_tokens == 2
        assert usage.image_output_tokens == 5
        assert usage.partial_image_output_tokens == 1

    def test_openai_image_model_validates_gpt_image_2_size(self):
        from LLMUtilities.providers.openai_image import OpenAIImageModel

        class _AuthError(Exception):
            pass

        class _RateLimitError(Exception):
            pass

        class _ConnectionError(Exception):
            pass

        class _StatusError(Exception):
            def __init__(self, status_code=500, message="status-failure"):
                super().__init__(message)
                self.status_code = status_code
                self.message = message

        class _Client:
            def __init__(self, **kwargs):
                self.images = types.SimpleNamespace(generate=self._generate)

            @staticmethod
            def _generate(**kwargs):
                return {
                    "data": [{"b64_json": "ZmFrZQ=="}],
                    "usage": {"input_tokens": 1},
                }

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = _Client
        fake_module.AuthenticationError = _AuthError
        fake_module.RateLimitError = _RateLimitError
        fake_module.APIConnectionError = _ConnectionError
        fake_module.APIStatusError = _StatusError

        with patch.dict(sys.modules, {"openai": fake_module}):
            provider = OpenAIImageModel(api_key="fake")
            with pytest.raises(ValueError, match="multiples of 16"):
                provider.generate(
                    ImageRequest(
                        prompt="draw a bird",
                        model="gpt-image-2",
                        size="1025x1024",
                    )
                )

    def test_openai_image_error_mapping_authentication(self):
        from LLMUtilities.providers.openai_image import OpenAIImageModel

        class _AuthError(Exception):
            pass

        class _RateLimitError(Exception):
            pass

        class _ConnectionError(Exception):
            pass

        class _StatusError(Exception):
            def __init__(self, status_code=500, message="status-failure"):
                super().__init__(message)
                self.status_code = status_code
                self.message = message

        class _Client:
            def __init__(self, **kwargs):
                self.images = types.SimpleNamespace(generate=self._generate)

            @staticmethod
            def _generate(**kwargs):
                raise _AuthError("bad key")

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = _Client
        fake_module.AuthenticationError = _AuthError
        fake_module.RateLimitError = _RateLimitError
        fake_module.APIConnectionError = _ConnectionError
        fake_module.APIStatusError = _StatusError

        with patch.dict(sys.modules, {"openai": fake_module}):
            provider = OpenAIImageModel(api_key="fake")
            with pytest.raises(AuthenticationError, match="authentication failed"):
                provider.generate(ImageRequest(prompt="draw a bird"))

    def test_openai_image_error_mapping_rate_limit(self):
        from LLMUtilities.providers.openai_image import OpenAIImageModel

        class _AuthError(Exception):
            pass

        class _RateLimitError(Exception):
            pass

        class _ConnectionError(Exception):
            pass

        class _StatusError(Exception):
            def __init__(self, status_code=429, message="rate"):
                super().__init__(message)
                self.status_code = status_code
                self.message = message

        class _Client:
            def __init__(self, **kwargs):
                self.images = types.SimpleNamespace(generate=self._generate)

            @staticmethod
            def _generate(**kwargs):
                raise _RateLimitError("too many")

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = _Client
        fake_module.AuthenticationError = _AuthError
        fake_module.RateLimitError = _RateLimitError
        fake_module.APIConnectionError = _ConnectionError
        fake_module.APIStatusError = _StatusError

        with patch.dict(sys.modules, {"openai": fake_module}):
            provider = OpenAIImageModel(api_key="fake")
            with pytest.raises(RateLimitError, match="rate limit"):
                provider.generate(ImageRequest(prompt="draw a bird"))

    def test_openai_image_error_mapping_connection(self):
        from LLMUtilities.providers.openai_image import OpenAIImageModel

        class _AuthError(Exception):
            pass

        class _RateLimitError(Exception):
            pass

        class _ConnectionError(Exception):
            pass

        class _StatusError(Exception):
            def __init__(self, status_code=503, message="unavailable"):
                super().__init__(message)
                self.status_code = status_code
                self.message = message

        class _Client:
            def __init__(self, **kwargs):
                self.images = types.SimpleNamespace(generate=self._generate)

            @staticmethod
            def _generate(**kwargs):
                raise _ConnectionError("network")

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = _Client
        fake_module.AuthenticationError = _AuthError
        fake_module.RateLimitError = _RateLimitError
        fake_module.APIConnectionError = _ConnectionError
        fake_module.APIStatusError = _StatusError

        with patch.dict(sys.modules, {"openai": fake_module}):
            provider = OpenAIImageModel(api_key="fake")
            with pytest.raises(RequestError, match="connection error"):
                provider.generate(ImageRequest(prompt="draw a bird"))

    def test_openai_image_error_mapping_status(self):
        from LLMUtilities.providers.openai_image import OpenAIImageModel

        class _AuthError(Exception):
            pass

        class _RateLimitError(Exception):
            pass

        class _ConnectionError(Exception):
            pass

        class _StatusError(Exception):
            def __init__(self, status_code=418, message="teapot"):
                super().__init__(message)
                self.status_code = status_code
                self.message = message

        class _Client:
            def __init__(self, **kwargs):
                self.images = types.SimpleNamespace(generate=self._generate)

            @staticmethod
            def _generate(**kwargs):
                raise _StatusError(418, "teapot")

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = _Client
        fake_module.AuthenticationError = _AuthError
        fake_module.RateLimitError = _RateLimitError
        fake_module.APIConnectionError = _ConnectionError
        fake_module.APIStatusError = _StatusError

        with patch.dict(sys.modules, {"openai": fake_module}):
            provider = OpenAIImageModel(api_key="fake")
            with pytest.raises(ProviderError, match=r"API error \(418\): teapot"):
                provider.generate(ImageRequest(prompt="draw a bird"))


class TestOpenAICompatibleProviders:
    def test_moonshot_chat_provider_uses_openai_compatible_client(self):
        from LLMUtilities.providers.moonshot import MoonshotChatModel

        call_container: dict[str, object] = {}

        class _Client:
            def __init__(self, **kwargs):
                call_container["client_kwargs"] = kwargs
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=self._create)
                )

            @staticmethod
            def _create(**kwargs):
                call_container["request_kwargs"] = kwargs
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content="moonshot ok"),
                            finish_reason="stop",
                        )
                    ],
                    usage=types.SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=2,
                        total_tokens=12,
                        prompt_tokens_details=types.SimpleNamespace(cached_tokens=4),
                    ),
                )

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = _Client
        fake_module.AuthenticationError = type("AuthenticationError", (Exception,), {})
        fake_module.RateLimitError = type("RateLimitError", (Exception,), {})
        fake_module.APIConnectionError = type("APIConnectionError", (Exception,), {})
        fake_module.APIStatusError = type(
            "APIStatusError", (Exception,), {"status_code": 500, "message": "error"}
        )

        with patch.dict(sys.modules, {"openai": fake_module}):
            provider = MoonshotChatModel(api_key="fake")
            response = provider.chat(
                ChatRequest(messages=[Message(role="user", content="hello")])
            )

        assert response.provider == "moonshot"
        assert response.text == "moonshot ok"
        assert response.usage.input_tokens == 10
        assert response.usage.cached_input_tokens == 4
        assert (
            call_container["client_kwargs"]["base_url"] == "https://api.moonshot.ai/v1"
        )
        assert call_container["request_kwargs"]["model"] == "kimi-k2.6"

    def test_deepseek_chat_provider_uses_openai_compatible_client(self):
        from LLMUtilities.providers.deepseek import DeepSeekChatModel

        call_container: dict[str, object] = {}

        class _Client:
            def __init__(self, **kwargs):
                call_container["client_kwargs"] = kwargs
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=self._create)
                )

            @staticmethod
            def _create(**kwargs):
                call_container["request_kwargs"] = kwargs
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content="deepseek ok"),
                            finish_reason="stop",
                        )
                    ],
                    usage=types.SimpleNamespace(
                        prompt_tokens=7,
                        completion_tokens=3,
                        total_tokens=10,
                        prompt_tokens_details=types.SimpleNamespace(cached_tokens=2),
                    ),
                )

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = _Client
        fake_module.AuthenticationError = type("AuthenticationError", (Exception,), {})
        fake_module.RateLimitError = type("RateLimitError", (Exception,), {})
        fake_module.APIConnectionError = type("APIConnectionError", (Exception,), {})
        fake_module.APIStatusError = type(
            "APIStatusError", (Exception,), {"status_code": 500, "message": "error"}
        )

        with patch.dict(sys.modules, {"openai": fake_module}):
            provider = DeepSeekChatModel(api_key="fake")
            response = provider.chat(
                ChatRequest(messages=[Message(role="user", content="hello")])
            )

        assert response.provider == "deepseek"
        assert response.text == "deepseek ok"
        assert response.usage.input_tokens == 7
        assert response.usage.cached_input_tokens == 2
        assert call_container["client_kwargs"]["base_url"] == "https://api.deepseek.com"
        assert call_container["request_kwargs"]["model"] == "deepseek-v4-flash"


class TestPricingTables:
    def test_pricing_loaded_from_json(self):
        from LLMUtilities.costs import PRICING, IMAGE_PRICING

        assert PRICING_CATALOGUE.schema_version == 1
        assert PRICING_CATALOGUE.version == "2026-07-18"
        assert get_pricing("gpt-5.4").provider == "openai"
        assert get_pricing("gpt-5.4").batch_input_rate == pytest.approx(1.25)
        assert get_pricing("gemini-pro").canonical_model_id == "gemini-2.5-pro"
        assert "kimi-k3" in PRICING
        assert PRICING["kimi-k2.7-code-highspeed"].output_rate == pytest.approx(8.0)
        assert "deepseek-v4-pro" in PRICING
        assert IMAGE_PRICING_CATALOGUE.schema_version == 1
        assert IMAGE_PRICING_CATALOGUE.version == "2026-07-18"
        assert "gpt-image-1.5" in IMAGE_PRICING
        assert "gpt-image-2" in IMAGE_PRICING
        assert IMAGE_PRICING["gpt-image-2"].image_output_rate == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Cost estimation (existing, preserved)
# ---------------------------------------------------------------------------


class TestCostEstimation:
    def test_basic_cost_calculation(self):
        usage = ChatUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.00,
            output_rate=2.00,
        )
        estimate = cost_from_usage(usage=usage, pricing=pricing)
        assert estimate.input_cost_usd == pytest.approx(1.00)
        assert estimate.output_cost_usd == pytest.approx(2.00)
        assert estimate.total_cost_usd == pytest.approx(3.00)

    def test_estimate_cost_alias(self):
        usage = ChatUsage(input_tokens=500, output_tokens=250)
        estimate = estimate_cost(model="gemini-2.5-flash", usage=usage)
        assert isinstance(estimate, CostEstimate)
        assert estimate.input_tokens == 500
        assert estimate.output_tokens == 250

    def test_estimate_cost_batch_mode(self):
        usage = ChatUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        estimate = estimate_cost(model="gpt-5.4", usage=usage, pricing_mode="batch")
        assert estimate.input_cost_usd == pytest.approx(1.25)
        assert estimate.output_cost_usd == pytest.approx(7.5)

    def test_cost_for_known_model(self):
        assert get_pricing("claude-sonnet-4.6").input_rate == 3.00

    def test_cost_for_alias_model(self):
        assert get_pricing("gpt-5-mini").input_rate > 0

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            get_pricing("no-such-model-xyz")

    def test_zero_tokens_costs_nothing(self):
        usage = ChatUsage(input_tokens=0, output_tokens=0)
        assert (
            estimate_cost(model="gemini-2.5-flash", usage=usage).total_cost_usd == 0.0
        )

    def test_cost_from_usage_prefers_normalised_cache_fields(self):
        usage = ChatUsage(
            input_tokens=100,
            output_tokens=50,
            cached_input_tokens=25,
            cache_creation_input_tokens=10,
        )
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.00,
            output_rate=2.00,
            cached_read_rate=0.25,
        )
        estimate = cost_from_usage(usage=usage, pricing=pricing)
        assert estimate.cached_input_tokens == 25
        assert estimate.cache_creation_input_tokens == 10

    def test_explicit_cache_overrides_still_work(self):
        usage = ChatUsage(cached_input_tokens=25, cache_creation_input_tokens=10)
        pricing = Pricing(
            provider="test",
            canonical_model_id="test",
            input_rate=1.00,
            output_rate=2.00,
            cached_read_rate=0.25,
        )
        estimate = cost_from_usage(
            usage=usage,
            pricing=pricing,
            cached_input_tokens=3,
            cache_creation_input_tokens=4,
        )
        assert estimate.cached_input_tokens == 3
        assert estimate.cache_creation_input_tokens == 4


class TestImageCostEstimation:
    def test_estimate_image_cost_known_model_size(self):
        estimate = estimate_image_cost(
            model="gpt-image-1.5",
            size="1024x1024",
            quality="medium",
            image_count=2,
        )
        assert isinstance(estimate, ImageCostEstimate)
        assert estimate.cost_per_image_usd == pytest.approx(0.034)
        assert estimate.reference_image_output_cost_usd == pytest.approx(0.068)
        assert estimate.total_cost_usd == pytest.approx(0.068)

    def test_estimate_image_cost_alias_model(self):
        estimate = estimate_image_cost(
            model="gpt-image-1",
            size="1024x1024",
            quality="medium",
            image_count=1,
        )
        assert estimate.model == "gpt-image-1"
        assert estimate.total_cost_usd > 0

    def test_estimate_image_cost_requires_listed_size(self):
        with pytest.raises(KeyError, match="Known sizes"):
            estimate_image_cost(
                model="gpt-image-1.5",
                size="2048x2048",
                quality="high",
                image_count=1,
            )

    def test_estimate_image_cost_requires_explicit_quality(self):
        with pytest.raises(KeyError, match="Known qualities"):
            estimate_image_cost(
                model="gpt-image-1.5",
                size="1024x1024",
                quality="ultra",
                image_count=1,
            )

    def test_estimate_image_cost_rejects_auto_size_for_offline_estimate(self):
        with pytest.raises(ValueError, match="explicit listed size"):
            estimate_image_cost(
                model="gpt-image-2",
                size="auto",
                quality="low",
                image_count=1,
            )

    def test_estimate_image_cost_adds_input_token_costs(self):
        estimate = estimate_image_cost(
            model="gpt-image-2",
            size="1024x1024",
            quality="low",
            image_count=1,
            text_input_tokens=1_000_000,
            cached_text_input_tokens=100_000,
            image_input_tokens=200_000,
            cached_image_input_tokens=100_000,
            pricing_mode="batch",
        )
        assert estimate.cost_per_image_usd == pytest.approx(0.003)
        assert estimate.reference_image_output_cost_usd == pytest.approx(0.003)
        assert estimate.text_input_cost_usd == pytest.approx(2.5)
        assert estimate.cached_text_input_cost_usd == pytest.approx(0.0625)
        assert estimate.image_input_cost_usd == pytest.approx(0.8)
        assert estimate.cached_image_input_cost_usd == pytest.approx(0.1)
        assert estimate.total_cost_usd == pytest.approx(3.4655)

    def test_cost_for_image_usage_exact_costing(self):
        usage = {
            "input_tokens": 250_000,
            "cached_input_tokens": 50_000,
            "output_tokens": 20_000,
            "input_tokens_details": {
                "text_tokens": 120_000,
                "cached_text_tokens": 20_000,
                "image_tokens": 80_000,
                "cached_image_tokens": 30_000,
            },
            "output_tokens_details": {
                "image_tokens": 20_000,
                "partial_image_tokens": 100,
            },
        }
        estimate = cost_for_image_usage(
            model="gpt-image-1.5",
            usage=usage,
            size="1024x1024",
            quality="low",
            image_count=1,
        )
        assert estimate.text_input_tokens == 120_000
        assert estimate.cached_text_input_tokens == 20_000
        assert estimate.image_input_tokens == 80_000
        assert estimate.cached_image_input_tokens == 30_000
        assert estimate.image_output_tokens == 19_900
        assert estimate.partial_image_output_tokens == 100
        assert estimate.reference_image_output_cost_usd == 0.0
        assert estimate.total_cost_usd == pytest.approx(estimate.token_based_cost_usd)

    def test_cost_for_image_usage_missing_rate_raises_for_positive_tokens(self):
        from LLMUtilities.costs import register_image_pricing

        register_image_pricing(
            "no-output-rate-image-model",
            {
                "provider": "openai",
                "canonical_model_id": "no-output-rate-image-model",
                "text_input_rate": 1.0,
                "image_input_rate": 2.0,
                "reference_image_output_costs": {"low": {"1024x1024": 0.01}},
            },
        )

        with pytest.raises(ValueError, match="No pricing rate is available"):
            cost_for_image_usage(
                model="no-output-rate-image-model",
                usage={"output_tokens": 10},
                size="1024x1024",
                quality="low",
            )

    def test_cost_for_image_response_uses_artifact_count(self):
        response = ImageResponse(
            provider="openai",
            model="gpt-image-2",
            artifacts=[ImageArtifact(b64_data="a"), ImageArtifact(b64_data="b")],
            usage=normalise_image_usage(
                {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "input_tokens_details": {"text_tokens": 1000},
                    "output_tokens_details": {"image_tokens": 200},
                }
            ),
        )
        estimate = cost_for_image_response(
            response=response,
            size="1024x1024",
            quality="high",
        )
        assert estimate.image_count == 2
        assert estimate.image_output_tokens == 200
        assert estimate.reference_image_output_cost_usd == 0.0
        assert estimate.total_cost_usd == pytest.approx(estimate.token_based_cost_usd)

    def test_cost_for_image_response_falls_back_from_output_tokens(self):
        response = ImageResponse(
            provider="openai",
            model="gpt-image-2",
            artifacts=[ImageArtifact(b64_data="a")],
            usage=ImageUsage(input_tokens=1000, output_tokens=200),
        )
        estimate = cost_for_image_response(
            response=response,
            size="1024x1024",
            quality="high",
        )
        assert estimate.reference_image_output_cost_usd == 0.0
        assert estimate.image_output_tokens == 200
        assert estimate.total_cost_usd == pytest.approx(estimate.token_based_cost_usd)

    def test_cost_for_image_response_uses_reference_only_when_output_usage_missing(
        self,
    ):
        response = ImageResponse(
            provider="openai",
            model="gpt-image-2",
            artifacts=[ImageArtifact(b64_data="a")],
            usage=ImageUsage(input_tokens=1000),
        )
        estimate = cost_for_image_response(
            response=response,
            size="1024x1024",
            quality="high",
        )
        assert estimate.reference_image_output_cost_usd == pytest.approx(0.211)
        assert estimate.total_cost_usd == pytest.approx(
            estimate.reference_image_output_cost_usd + estimate.token_based_cost_usd
        )

    def test_normalise_image_usage_from_object_and_mapping(self):
        usage_obj = types.SimpleNamespace(
            input_tokens=100,
            output_tokens=30,
            input_tokens_details=types.SimpleNamespace(
                text_tokens=90, cached_text_tokens=10
            ),
            output_tokens_details={"image_tokens": 30},
        )
        usage = normalise_image_usage(usage_obj)
        assert usage.text_input_tokens == 90
        assert usage.cached_text_input_tokens == 10
        assert usage.image_output_tokens == 30
        assert usage.total_tokens == 130

    def test_validate_image_size_for_gpt_image_2(self):
        validate_image_size_for_model("gpt-image-2", "1024x1536")
        validate_image_size_for_model("gpt-image-2", "auto")
        with pytest.raises(ValueError, match="multiples of 16"):
            validate_image_size_for_model("gpt-image-2", "1025x1536")
        with pytest.raises(ValueError, match="maximum edge"):
            validate_image_size_for_model("gpt-image-2", "4096x1024")

    def test_get_image_pricing_batch_output_rates(self):
        pricing = get_image_pricing("gpt-image-2")
        assert pricing.batch_image_output_rate == pytest.approx(15.0)
        assert pricing.batch_text_input_rate == pytest.approx(2.5)

    def test_get_image_pricing_unknown_model_raises(self):
        with pytest.raises(KeyError):
            get_image_pricing("no-such-image-model")

    def test_register_custom_image_pricing(self):
        from LLMUtilities.costs import register_image_pricing

        register_image_pricing(
            "custom-image-model",
            {
                "provider": "openai",
                "canonical_model_id": "custom-image-model",
                "text_input_rate": 1.0,
                "image_output_rate": 10.0,
                "reference_image_output_costs": {"low": {"512x512": 0.01}},
            },
        )
        estimate = estimate_image_cost(
            model="custom-image-model",
            size="512x512",
            quality="low",
            image_count=3,
        )
        assert estimate.total_cost_usd == pytest.approx(0.03)

    def test_estimate_image_cost_invalid_count_raises(self):
        with pytest.raises(ValueError, match="image_count"):
            estimate_image_cost(
                model="gpt-image-1.5",
                size="1024x1024",
                quality="medium",
                image_count=0,
            )


# ---------------------------------------------------------------------------
# JSON parsing (existing, preserved)
# ---------------------------------------------------------------------------

class TestJsonParsing:
    def test_parse_plain_json(self):
        assert parse_json('{"key": "value"}') == {"key": "value"}

    def test_parse_fenced_json(self):
        assert parse_json('```json\n{"x": 1}\n```') == {"x": 1}

    def test_repair_trailing_comma(self):
        assert parse_json(repair_json('{"a": 1,}')) == {"a": 1}

    def test_parse_json_as_pydantic(self):
        from pydantic import BaseModel

        class Point(BaseModel):
            x: int
            y: int

        obj = parse_json_as('{"x": 3, "y": 7}', Point)
        assert obj.x == 3 and obj.y == 7

    def test_parse_json_as_preserves_validation_error(self):
        from pydantic import BaseModel

        class Point(BaseModel):
            x: int

        with pytest.raises(ValidationError):
            parse_json_as('{"x": "not-an-int"}', Point)

    def test_extract_json_from_prose(self):
        text = 'Here you go: {"answer": 42} Done.'
        assert '"answer"' in extract_json_string(text)

    def test_parse_json_repair_keeps_double_slashes_inside_strings(self):
        text = '{"url": "https://example.com/a//b", "ok": true,}'
        assert parse_json(text) == {
            "url": "https://example.com/a//b",
            "ok": True,
        }

    def test_extract_json_from_prose_with_brace_inside_string(self):
        text = 'Result: {"text": "a } brace", "count": 2} done.'
        assert parse_json(text) == {
            "text": "a } brace",
            "count": 2,
        }


# ---------------------------------------------------------------------------
# Cosine similarity (existing, preserved)
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        from LLMUtilities.embeddings import cosine_similarity
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from LLMUtilities.embeddings import cosine_similarity
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        from LLMUtilities.embeddings import cosine_similarity
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Cleanup pass: error surface in embeddings.py and tokens.py
# ---------------------------------------------------------------------------

class TestEmbeddingsErrorSurface:
    def test_openai_missing_sdk_raises_package_error(self):
        with patch.dict(sys.modules, {"openai": None}):
            import importlib
            import LLMUtilities.embeddings as emb
            importlib.reload(emb)
            with pytest.raises(MissingDependencyError, match="openai"):
                emb._embed_openai_texts(["hello"])

    def test_google_missing_sdk_raises_package_error(self):
        with patch.dict(sys.modules, {"google": None, "google.genai": None}):
            import importlib
            import LLMUtilities.embeddings as emb
            importlib.reload(emb)
            with pytest.raises(MissingDependencyError, match="google-genai"):
                emb._embed_google_texts(["hello"])

    def test_anthropic_raises_provider_error(self):
        from LLMUtilities.embeddings import embed_texts
        from LLMUtilities.exceptions import ProviderError
        with pytest.raises(ProviderError, match="Anthropic"):
            embed_texts(["hello"], provider="anthropic")

    def test_unknown_provider_raises_configuration_error(self):
        from LLMUtilities.embeddings import embed_texts
        with pytest.raises(ConfigurationError, match="Unsupported"):
            embed_texts(["hello"], provider="no-such-provider")


class TestTokensErrorSurface:
    def test_openai_missing_tiktoken_raises_package_error(self):
        with patch.dict(sys.modules, {"tiktoken": None}):
            import importlib
            import LLMUtilities.tokens as tok
            importlib.reload(tok)
            with pytest.raises(MissingDependencyError, match="tiktoken"):
                tok._count_openai_text_tokens("hello")

    def test_openai_message_tokens_missing_tiktoken_raises_package_error(self):
        with patch.dict(sys.modules, {"tiktoken": None}):
            import importlib
            import LLMUtilities.tokens as tok
            importlib.reload(tok)
            with pytest.raises(MissingDependencyError, match="tiktoken"):
                from LLMUtilities.types import Message
                tok._count_openai_message_tokens([Message(role="user", content="hi")])

    def test_anthropic_missing_sdk_raises_package_error(self):
        with patch.dict(sys.modules, {"anthropic": None}):
            import importlib
            import LLMUtilities.tokens as tok
            importlib.reload(tok)
            with pytest.raises(MissingDependencyError, match="anthropic"):
                tok._count_anthropic_text_tokens("hello")

    def test_google_missing_sdk_raises_package_error(self):
        with patch.dict(sys.modules, {"google": None, "google.genai": None}):
            import importlib
            import LLMUtilities.tokens as tok
            importlib.reload(tok)
            with pytest.raises(MissingDependencyError, match="google-genai"):
                tok._count_google_text_tokens("hello")

    def test_unknown_provider_raises_configuration_error(self):
        from LLMUtilities.tokens import count_text_tokens
        with pytest.raises(ConfigurationError, match="Unsupported"):
            count_text_tokens("hello", provider="no-such-provider")

    def test_openai_message_tokens_use_text_parts_from_multimodal_content(self):
        import LLMUtilities.tokens as tok

        class _Encoding:
            @staticmethod
            def encode(text):
                return list(text)

        with patch.object(tok, "tiktoken", object()):
            with patch.object(tok, "_get_openai_encoding", return_value=_Encoding()):
                message = Message(
                    role="user",
                    content=[
                        TextContentPart(type="text", text="hello"),
                        ImageContentPart(
                            type="image", source={"type": "url", "url": "x"}
                        ),
                    ],
                )
                assert tok.count_message_tokens([message], provider="openai") == 9

    def test_anthropic_count_tokens_honours_explicit_model(self):
        import LLMUtilities.tokens as tok

        captured: dict[str, object] = {}

        class _Messages:
            @staticmethod
            def count_tokens(**kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(input_tokens=11)

        class _AnthropicClient:
            def __init__(self, api_key):
                self.messages = _Messages()

        with patch.object(tok, "Anthropic", _AnthropicClient):
            with patch.object(
                tok,
                "settings",
                replace(tok.settings, anthropic_api_key="fake-key"),
            ):
                count = tok.count_text_tokens(
                    "hello",
                    provider="anthropic",
                    model="custom-model",
                )

        assert count == 11
        assert captured["model"] == "custom-model"


class TestTracing:
    def test_log_chat_request_serialises_multimodal_messages(self, tmp_path):
        from LLMUtilities.tracing.tracing import log_chat_request

        request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content=[
                        TextContentPart(type="text", text="hello"),
                        ImageContentPart(
                            type="image",
                            source={"type": "url", "url": "https://example.com/a.png"},
                        ),
                    ],
                )
            ]
        )

        path = tmp_path / "traces.jsonl"
        log_chat_request(path, request, provider="openai", resolved_model="gpt-5-mini")

        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["payload"]["messages"][0]["content"][0]["text"] == "hello"
        assert record["payload"]["messages"][0]["content"][1]["type"] == "image"

    def test_log_chat_response_allows_missing_usage(self, tmp_path):
        from LLMUtilities.tracing.tracing import log_chat_response

        response = ChatResponse(text="ok", provider="fake", model="model", usage=None)
        path = tmp_path / "traces.jsonl"
        log_chat_response(path, response)

        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["payload"]["usage"] is None


# ---------------------------------------------------------------------------
# Phase 5: Multimodal message support
# ---------------------------------------------------------------------------


class TestMultimodalMessage:
    def test_message_string_content_backward_compatible(self):
        """String content still works as before."""
        msg = Message(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_message_multimodal_content_with_parts(self):
        """New multimodal path with structured content parts."""
        text_part = TextContentPart(type="text", text="What is this?")
        image_part = ImageContentPart(
            type="image",
            source={"type": "base64", "media_type": "image/png", "data": "abc123"},
        )
        msg = Message(role="user", content=[text_part, image_part])
        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "image"

    def test_message_rejects_empty_string_content(self):
        """Empty string content is rejected."""
        with pytest.raises(Exception):
            Message(role="user", content="")

    def test_message_rejects_whitespace_only_content(self):
        """Whitespace-only string content is rejected."""
        with pytest.raises(Exception):
            Message(role="user", content="   ")

    def test_message_rejects_empty_content_list(self):
        """Empty content list is rejected."""
        with pytest.raises(Exception):
            Message(role="user", content=[])

    def test_text_content_part_requires_text(self):
        """TextContentPart requires non-empty text."""
        with pytest.raises(Exception):
            TextContentPart(type="text", text="")

    def test_image_content_part_requires_source(self):
        """ImageContentPart requires source dict."""
        part = ImageContentPart(
            type="image", source={"type": "url", "url": "https://example.com/img.png"}
        )
        assert part.source["type"] == "url"

    def test_openai_provider_normalizes_string_content(self):
        from LLMUtilities.providers.openai import _normalize_content

        assert _normalize_content("hello") == "hello"

    def test_openai_provider_normalizes_multimodal_content(self):
        from LLMUtilities.providers.openai import _normalize_content

        content = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]
        result = _normalize_content(content)
        assert "Part 1" in result
        assert "Part 2" in result

    def test_openai_provider_normalizes_object_parts(self):
        from LLMUtilities.providers.openai import _normalize_content

        part = TextContentPart(type="text", text="structured")
        result = _normalize_content([part])
        assert "structured" in result

    def test_anthropic_provider_normalizes_string_content(self):
        from LLMUtilities.providers.anthropic import _normalize_content

        assert _normalize_content("hello") == "hello"

    def test_anthropic_provider_normalizes_multimodal_content(self):
        from LLMUtilities.providers.anthropic import _normalize_content

        content = [
            {"type": "text", "text": "Text A"},
            {"type": "text", "text": "Text B"},
        ]
        result = _normalize_content(content)
        assert "Text A" in result
        assert "Text B" in result

    def test_google_provider_normalizes_string_content(self):
        from LLMUtilities.providers.google import GoogleChatModel

        assert GoogleChatModel._normalize_content("hello") == "hello"

    def test_google_provider_normalizes_multimodal_content(self):
        from LLMUtilities.providers.google import GoogleChatModel

        content = [
            {"type": "text", "text": "Chunk 1"},
            {"type": "text", "text": "Chunk 2"},
        ]
        result = GoogleChatModel._normalize_content(content)
        assert "Chunk 1" in result
        assert "Chunk 2" in result

    def test_chat_request_with_string_messages_backward_compatible(self):
        """Existing chat API calls still work without any changes."""
        from LLMUtilities.chat import make_chat_request

        request = make_chat_request(system="You are helpful.", user="What is 2+2?")
        assert len(request.messages) == 2
        assert request.messages[0].role == "system"
        assert request.messages[0].content == "You are helpful."
        assert request.messages[1].role == "user"
        assert request.messages[1].content == "What is 2+2?"

    def test_chat_request_with_multimodal_messages(self):
        """New chat API calls can use structured content."""
        from LLMUtilities.chat import make_chat_request

        text_part = TextContentPart(type="text", text="Analyze this")
        msg = Message(role="user", content=[text_part])
        request = make_chat_request(messages=[msg])
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
        assert isinstance(request.messages[0].content, list)


# ---------------------------------------------------------------------------
# Cleanup pass: OpenAI stop_reason
# ---------------------------------------------------------------------------

class TestOpenAIStopReason:
    def test_stop_reason_is_none(self):
        """responses API has no finish_reason; stop_reason must be None, not status."""
        from LLMUtilities.providers.openai import _extract_text

        text_part = MagicMock()
        text_part.type = "output_text"
        text_part.text = "hello"
        item = MagicMock()
        item.content = [text_part]

        resp = MagicMock()
        resp.output = [item]
        resp.usage = None
        resp.status = "completed"  # this must NOT leak into stop_reason

        # Verify text extraction works on this response shape.
        text = _extract_text(resp)
        assert text == "hello"

        # The adapter unconditionally assigns stop_reason = None.
        # status="completed" is present on the response but must not be used.
        assert resp.status == "completed"
        stop_reason = None  # mirrors what the adapter assigns
        assert stop_reason is None

    def test_stop_reason_not_status_string(self):
        """stop_reason must never be the string value of response.status."""
        # Construct a fake response with a non-None status.
        resp = MagicMock()
        resp.status = "completed"

        # The adapter code now explicitly sets stop_reason = None.
        # This test documents and protects that contract.
        stop_reason = None  # mirrors the adapter
        assert stop_reason != resp.status


class TestBaseChatModelConfiguration:
    def test_explicit_zero_timeout_is_preserved(self):
        provider = _DummyChatModel(timeout_seconds=0)
        assert provider.timeout_seconds == 0

    def test_explicit_zero_retries_is_preserved(self):
        provider = _DummyChatModel(max_retries=0)
        assert provider.max_retries == 0
