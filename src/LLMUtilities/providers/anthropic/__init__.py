from __future__ import annotations

from typing import Optional, Sequence

from ...config import get_settings
from ...exceptions import ConfigurationError, MissingDependencyError
from ...transports.anthropic_messages import AnthropicMessagesTransport
from ...transports.anthropic_messages import is_available as _messages_available
from ...types import ChatRequest, ChatResponse, CostSummary, Message
from ..pricing_loading import select_pricing_for_response
from . import chat as _chat
from . import tokens as _tokens
from .pricing import (
    AnthropicChatCostDetails,
    AnthropicChatPricing,
    AnthropicChatUsageDetails,
    calculate_cost_details,
    calculate_cost_summary,
    get_pricing,
    list_models as _list_models,
    list_pricings,
)


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        if not _messages_available():
            raise MissingDependencyError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "Install it with: pip install anthropic"
            )

        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key
        self.timeout_seconds = (
            settings.request_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self.max_retries = settings.max_retries if max_retries is None else max_retries
        self._defaults = settings.anthropic

    def capabilities(self) -> list[str]:
        return ["chat", "token_counting"]

    def list_models(self, capability: Optional[str] = None) -> list[str]:
        if capability is None or capability == "chat":
            return self.list_chat_models()
        raise ConfigurationError(f"Unknown capability: {capability!r}.")

    def list_chat_models(self) -> list[str]:
        return _list_models()

    def list_pricings(self) -> list[AnthropicChatPricing]:
        return list_pricings()

    def get_pricing(self, model: str, *, effective_at=None) -> AnthropicChatPricing:
        return get_pricing(model, effective_at=effective_at)

    def chat(self, request: ChatRequest) -> ChatResponse:
        model_name = _chat.resolve_model_name(request, self._defaults.chat_model)
        transport = self._transport()
        return _chat.run_chat(
            transport=transport,
            request=request,
            model_name=model_name,
            default_max_output_tokens=self._defaults.max_output_tokens,
        )

    def get_detailed_usage(self, response: ChatResponse) -> AnthropicChatUsageDetails:
        return _chat.extract_usage_details(response.raw)

    def get_detailed_cost(self, response: ChatResponse) -> AnthropicChatCostDetails:
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

    def _resolve_pricing(self, response: ChatResponse) -> AnthropicChatPricing:
        return select_pricing_for_response(
            list_pricings(),
            resolved_model=response.resolved_model,
            requested_model=response.requested_model,
            provider_name="anthropic",
        )

    def count_text_tokens(self, text: str, *, model: Optional[str] = None):
        resolved_model = model or self._defaults.chat_model
        if not resolved_model:
            raise ConfigurationError(
                "No chat model configured for provider 'anthropic'."
            )
        return _tokens.count_text_tokens(
            transport=self._transport(), text=text, model=resolved_model
        )

    def count_message_tokens(
        self, messages: Sequence[Message], *, model: Optional[str] = None
    ):
        resolved_model = model or self._defaults.chat_model
        if not resolved_model:
            raise ConfigurationError(
                "No chat model configured for provider 'anthropic'."
            )
        return _tokens.count_message_tokens(
            transport=self._transport(), messages=messages, model=resolved_model
        )

    def _transport(self) -> AnthropicMessagesTransport:
        return AnthropicMessagesTransport(
            api_key=self._require_api_key(),
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigurationError("Missing API key for provider 'anthropic'.")
        return self.api_key
