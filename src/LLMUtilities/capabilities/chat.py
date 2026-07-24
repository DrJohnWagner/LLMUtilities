from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import ChatRequest, ChatResponse


@runtime_checkable
class ChatCapability(Protocol):
    def chat(self, request: ChatRequest) -> ChatResponse: ...
