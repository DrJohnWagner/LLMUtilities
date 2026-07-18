from __future__ import annotations

from typing import Optional

from LLMUtilities.providers.base import BaseChatModel
from LLMUtilities.types import ChatRequest, ChatResponse, ChatUsage


class FakeChatModel(BaseChatModel):
    """
    In-memory stub provider for unit tests.

    Returns a canned response without making any network calls.
    Records all requests in ``self.calls`` for inspection.
    """

    provider_name = "fake"
    default_model = "fake-model"

    def __init__(
        self,
        response_text: str = "fake response",
        model: Optional[str] = None,
        input_tokens: int = 10,
        output_tokens: int = 5,
        stop_reason: str = "stop",
    ) -> None:
        # Skip api_key requirement in base by providing a placeholder.
        super().__init__(model=model, api_key="fake-key")
        self.response_text = response_text
        self.stop_reason = stop_reason
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Return canned response; record request for later inspection."""
        request = self.prepare_request(request)
        self.calls.append(request)

        model_name = request.model or self.default_model

        usage = ChatUsage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=self._input_tokens + self._output_tokens,
        )

        return ChatResponse(
            text=self.response_text,
            provider=self.provider_name,
            model=model_name,
            usage=usage,
            stop_reason=self.stop_reason,
            raw=None,
        )
