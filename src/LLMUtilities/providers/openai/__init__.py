from __future__ import annotations

from typing import Optional

from ...config import get_settings
from ...exceptions import ConfigurationError, MissingDependencyError
from ...transports.openai_responses import OpenAIResponsesTransport
from ...transports.openai_responses import is_available as _responses_available
from ...types import (
    ChatRequest,
    ChatResponse,
    CostSummary,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageRequest,
    ImageResponse,
)
from . import chat as _chat
from . import embedding as _embedding
from . import image as _image
from . import tokens as _tokens
from ..chat_cost_calculation import calculate_chat_cost
from ..pricing_loading import select_pricing_for_response
from .pricing import (
    OpenAIChatCostDetails,
    OpenAIChatPricing,
    OpenAIChatUsageDetails,
    OpenAIImageCostDetails,
    OpenAIImagePricing,
    OpenAIImageUsageDetails,
    calculate_cost_summary,
    calculate_image_cost_details,
    calculate_image_cost_summary,
    get_image_pricing,
    get_pricing,
    list_image_pricings,
    list_models as _list_chat_models,
    list_pricings,
)
from .image import extract_usage_details as _extract_image_usage_details
from .image import list_image_models as _list_image_models


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        if not _responses_available():
            raise MissingDependencyError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            )

        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.timeout_seconds = (
            settings.request_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self.max_retries = settings.max_retries if max_retries is None else max_retries
        self._defaults = settings.openai

    # -- capability discovery -------------------------------------------------

    def capabilities(self) -> list[str]:
        return ["chat", "image_generation", "embedding", "token_counting"]

    # -- model & pricing discovery -------------------------------------------

    def list_models(self, capability: Optional[str] = None) -> list[str]:
        if capability is None or capability == "chat":
            return self.list_chat_models()
        if capability == "image_generation":
            return self.list_image_models()
        if capability == "embedding":
            return self.list_embedding_models()
        raise ConfigurationError(f"Unknown capability: {capability!r}.")

    def list_chat_models(self) -> list[str]:
        return _list_chat_models()

    def list_image_models(self) -> list[str]:
        return _list_image_models()

    def list_embedding_models(self) -> list[str]:
        return list(_embedding.EMBEDDING_MODELS)

    def list_pricings(self) -> list[OpenAIChatPricing]:
        return list_pricings()

    def get_pricing(self, model: str, *, effective_at=None) -> OpenAIChatPricing:
        return get_pricing(model, effective_at=effective_at)

    def list_image_pricings(self) -> list[OpenAIImagePricing]:
        return list_image_pricings()

    def get_image_pricing(self, model: str, *, effective_at=None) -> OpenAIImagePricing:
        return get_image_pricing(model, effective_at=effective_at)

    # -- ChatCapability --------------------------------------------------------

    def chat(self, request: ChatRequest) -> ChatResponse:
        model_name = _chat.resolve_model_name(request, self._defaults.chat_model)
        api_key = self._require_api_key()
        transport = OpenAIResponsesTransport(
            api_key=api_key,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        return _chat.run_chat(transport=transport, request=request, model_name=model_name)

    def get_detailed_usage(self, response: ChatResponse) -> OpenAIChatUsageDetails:
        return _chat.extract_usage_details(response.raw)

    def get_detailed_cost(self, response: ChatResponse) -> OpenAIChatCostDetails:
        usage = self.get_detailed_usage(response)
        pricing = self._resolve_pricing(response)
        breakdown = calculate_chat_cost(
            input_tokens=max(usage.input_tokens - usage.cached_input_tokens, 0),
            output_tokens=usage.output_tokens,
            cached_read_tokens=usage.cached_input_tokens,
            pricing=pricing,
        )
        return OpenAIChatCostDetails(
            input_cost=breakdown.input_cost,
            cached_read_cost=breakdown.cached_read_cost,
            output_cost=breakdown.output_cost,
            total_cost=breakdown.total_cost,
            currency=pricing.currency,
            requested_model=response.requested_model,
            resolved_model=response.resolved_model,
        )

    def get_cost_summary(self, response: ChatResponse) -> CostSummary:
        usage = self.get_detailed_usage(response)
        pricing = self._resolve_pricing(response)
        return calculate_cost_summary(
            usage=usage,
            pricing=pricing,
            requested_model=response.requested_model,
            resolved_model=response.resolved_model,
        )

    def _resolve_pricing(self, response: ChatResponse) -> OpenAIChatPricing:
        return select_pricing_for_response(
            list_pricings(),
            resolved_model=response.resolved_model,
            requested_model=response.requested_model,
            provider_name="openai",
        )

    # -- ImageGenerationCapability ----------------------------------------------

    def generate_image(self, request: ImageRequest) -> ImageResponse:
        model_name = _image.resolve_model_name(request, self._defaults.image_model)
        api_key = self._require_api_key()
        return _image.run_generate(
            api_key=api_key,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            request=request,
            model_name=model_name,
            default_size=self._defaults.image_size,
            default_quality=self._defaults.image_quality,
            default_background=self._defaults.image_background,
            default_format=self._defaults.image_format,
        )

    def get_image_detailed_usage(self, response: ImageResponse) -> OpenAIImageUsageDetails:
        return _extract_image_usage_details(response.raw)

    def get_image_cost_summary(self, response: ImageResponse) -> CostSummary:
        usage = self.get_image_detailed_usage(response)
        pricing = self._resolve_image_pricing(response)
        return calculate_image_cost_summary(
            usage=usage,
            pricing=pricing,
            requested_model=response.requested_model,
            resolved_model=response.resolved_model,
        )

    def get_image_detailed_cost(self, response: ImageResponse) -> OpenAIImageCostDetails:
        usage = self.get_image_detailed_usage(response)
        pricing = self._resolve_image_pricing(response)
        return calculate_image_cost_details(
            usage=usage,
            pricing=pricing,
            requested_model=response.requested_model,
            resolved_model=response.resolved_model,
        )

    def _resolve_image_pricing(self, response: ImageResponse) -> OpenAIImagePricing:
        return select_pricing_for_response(
            list_image_pricings(),
            resolved_model=response.resolved_model,
            requested_model=response.requested_model,
            provider_name="openai",
        )

    # -- EmbeddingCapability -----------------------------------------------------

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model_name = _embedding.resolve_model_name(
            request, self._defaults.embedding_model
        )
        api_key = self._require_api_key()
        return _embedding.run_embed(
            api_key=api_key,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            request=request,
            model_name=model_name,
        )

    # -- TokenCountingCapability --------------------------------------------------

    def count_text_tokens(self, text: str, *, model: Optional[str] = None):
        return _tokens.count_text_tokens(text, model=model or self._defaults.chat_model)

    def count_message_tokens(self, messages, *, model: Optional[str] = None):
        return _tokens.count_message_tokens(
            messages, model=model or self._defaults.chat_model
        )

    # -- internal --------------------------------------------------------------

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigurationError("Missing API key for provider 'openai'.")
        return self.api_key
