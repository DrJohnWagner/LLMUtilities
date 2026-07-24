from __future__ import annotations

from typing import Literal, Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ..types import Message

TokenCountMethod = Literal["exact", "provider_reported", "local_estimate"]


class TokenCountResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    provider: str
    model: Optional[str] = None
    method: TokenCountMethod


@runtime_checkable
class TokenCountingCapability(Protocol):
    def count_text_tokens(
        self, text: str, *, model: Optional[str] = None
    ) -> TokenCountResult: ...

    def count_message_tokens(
        self, messages: Sequence[Message], *, model: Optional[str] = None
    ) -> TokenCountResult: ...
