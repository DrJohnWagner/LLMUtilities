from __future__ import annotations

from typing import Optional, Sequence

from ...config import get_settings
from ...exceptions import ConfigurationError, MissingDependencyError
from ...transports.google_generate_content import GoogleGenerateContentTransport
from ...transports.google_generate_content import is_available as _genai_available
from ...types import (
    ChatRequest,
    ChatResponse,
    CostSummary,
    EmbeddingRequest,
    EmbeddingResponse,
    Message,
)
from ..pricing_loading import select_pricing_for_response
from . import chat as _chat
from . import embedding as _embedding
from . import tokens as _tokens
from .pricing import (
    GoogleChatCostDetails,
    GoogleChatPricing,
    GoogleChatUsageDetails,
    calculate_cost_details,
    calculate_cost_summary,
    get_pricing,
    list_models as _list_models,
    list_pricings,
)


class GoogleProvider:
    name = "google"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        if not _genai_available():
            raise MissingDependencyError(
                "The 'google-genai' package is required for the Google provider. "
                "Install it with: pip install google-genai"
            )

        settings = get_settings()
        self.api_key = api_key or settings.google_api_key
        self.timeout_seconds = (
            settings.request_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self.max_retries = settings.max_retries if max_retries is None else max_retries
        self._defaults = settings.google

    def capabilities(self) -> list[str]:
        return ["chat", "embedding", "token_counting"]

    def list_models(self, capability: Optional[str] = None) -> list[str]:
        if capability is None or capability == "chat":
            return self.list_chat_models()
        if capability == "embedding":
            return self.list_embedding_models()
        raise ConfigurationError(f"Unknown capability: {capability!r}.")

    def list_chat_models(self) -> list[str]:
        return _list_models()

    def list_embedding_models(self) -> list[str]:
        return list(_embedding.EMBEDDING_MODELS)

    def list_pricings(self) -> list[GoogleChatPricing]:
        return list_pricings()

    def get_pricing(self, model: str, *, effective_at=None) -> GoogleChatPricing:
        return get_pricing(model, effective_at=effective_at)

    def chat(self, request: ChatRequest) -> ChatResponse:
        model_name = _chat.resolve_model_name(request, self._defaults.chat_model)
        transport = self._transport()
        return _chat.run_chat(transport=transport, request=request, model_name=model_name)

    def get_detailed_usage(self, response: ChatResponse) -> GoogleChatUsageDetails:
        return _chat.extract_usage_details(response.raw)

    def get_detailed_cost(self, response: ChatResponse) -> GoogleChatCostDetails:
        usage = self.get_detailed_usage(response)
        pricing = self._resolve_pricing(response)
        return calculate_cost_details(
            usage=usage,
            pricing=pricing,
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

    def _resolve_pricing(self, response: ChatResponse) -> GoogleChatPricing:
        return select_pricing_for_response(
            list_pricings(),
            resolved_model=response.resolved_model,
            requested_model=response.requested_model,
            provider_name="google",
        )

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model_name = _embedding.resolve_model_name(
            request, self._defaults.embedding_model
        )
        return _embedding.run_embed(
            transport=self._transport(), request=request, model_name=model_name
        )

    def count_text_tokens(self, text: str, *, model: Optional[str] = None):
        resolved_model = self._require_chat_model(model)
        return _tokens.count_text_tokens(
            transport=self._transport(), text=text, model=resolved_model
        )

    def count_message_tokens(
        self, messages: Sequence[Message], *, model: Optional[str] = None
    ):
        resolved_model = self._require_chat_model(model)
        return _tokens.count_message_tokens(
            transport=self._transport(), messages=messages, model=resolved_model
        )

    def _require_chat_model(self, model: Optional[str]) -> str:
        resolved_model = model or self._defaults.chat_model
        if not resolved_model:
            raise ConfigurationError("No chat model configured for provider 'google'.")
        return resolved_model

    def _transport(self) -> GoogleGenerateContentTransport:
        return GoogleGenerateContentTransport(api_key=self._require_api_key())

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigurationError("Missing API key for provider 'google'.")
        return self.api_key
