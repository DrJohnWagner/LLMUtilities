from __future__ import annotations

from typing import Optional

from ...config import get_settings
from ...exceptions import ConfigurationError, MissingDependencyError
from ...transports.openai_chat_completions import OpenAIChatCompletionsTransport
from ...transports.openai_chat_completions import is_available as _chat_completions_available
from ...types import ChatRequest, ChatResponse, CostSummary
from .. import openai_compatible_chat as _chat
from ..pricing_loading import select_pricing_for_response
from .pricing import (
    DeepSeekChatCostDetails,
    DeepSeekChatPricing,
    DeepSeekChatUsageDetails,
    calculate_cost_details,
    calculate_cost_summary,
    get_pricing,
    list_models as _list_models,
    list_pricings,
)

BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider:
    name = "deepseek"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        if not _chat_completions_available():
            raise MissingDependencyError(
                "The 'openai' package is required for the DeepSeek provider "
                "(DeepSeek speaks the OpenAI-compatible Chat Completions "
                "protocol). Install it with: pip install openai"
            )

        settings = get_settings()
        self.api_key = api_key or settings.deepseek_api_key
        self.timeout_seconds = (
            settings.request_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self.max_retries = settings.max_retries if max_retries is None else max_retries
        self._defaults = settings.deepseek

    def capabilities(self) -> list[str]:
        return ["chat"]

    def list_models(self, capability: Optional[str] = None) -> list[str]:
        if capability is None or capability == "chat":
            return self.list_chat_models()
        raise ConfigurationError(f"Unknown capability: {capability!r}.")

    def list_chat_models(self) -> list[str]:
        return _list_models()

    def list_pricings(self) -> list[DeepSeekChatPricing]:
        return list_pricings()

    def get_pricing(self, model: str, *, effective_at=None) -> DeepSeekChatPricing:
        return get_pricing(model, effective_at=effective_at)

    def chat(self, request: ChatRequest) -> ChatResponse:
        model_name = _chat.resolve_model_name(
            request, self._defaults.chat_model, "deepseek"
        )
        transport = self._transport()
        return _chat.run_chat(
            transport=transport,
            request=request,
            model_name=model_name,
            provider_name="deepseek",
        )

    def get_detailed_usage(self, response: ChatResponse) -> DeepSeekChatUsageDetails:
        input_tokens, output_tokens, total_tokens, cached_input_tokens = (
            _chat.extract_usage_totals(response.raw)
        )
        return DeepSeekChatUsageDetails(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def get_detailed_cost(self, response: ChatResponse) -> DeepSeekChatCostDetails:
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

    def _resolve_pricing(self, response: ChatResponse) -> DeepSeekChatPricing:
        return select_pricing_for_response(
            list_pricings(),
            resolved_model=response.resolved_model,
            requested_model=response.requested_model,
            provider_name="deepseek",
        )

    def _transport(self) -> OpenAIChatCompletionsTransport:
        return OpenAIChatCompletionsTransport(
            api_key=self._require_api_key(),
            base_url=BASE_URL,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            display_name="DeepSeek",
        )

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigurationError("Missing API key for provider 'deepseek'.")
        return self.api_key
