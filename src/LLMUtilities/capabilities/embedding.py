from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import EmbeddingRequest, EmbeddingResponse


@runtime_checkable
class EmbeddingCapability(Protocol):
    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...
