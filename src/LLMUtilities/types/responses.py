from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class CommonUsage(BaseModel):
    """
    Deliberately limited, provider-independent usage summary.

    Only concepts that are stable across every provider belong here. Cache
    reads/writes, reasoning tokens and modality breakdowns are provider-specific
    and are exposed through each provider's detailed usage types instead.
    """

    model_config = ConfigDict(extra="forbid")

    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    text: str
    provider: str
    requested_model: Optional[str] = None
    resolved_model: str
    usage: Optional[CommonUsage] = None
    stop_reason: Optional[str] = None
    raw: Any = None


class ImageArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mime_type: Optional[str] = None
    b64_data: Optional[str] = None
    url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    revised_prompt: Optional[str] = None


class ImageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    provider: str
    requested_model: Optional[str] = None
    resolved_model: str
    artifacts: list[ImageArtifact] = Field(min_length=1)
    usage: Optional[CommonUsage] = None
    raw: Any = None


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    provider: str
    requested_model: Optional[str] = None
    resolved_model: str
    vectors: list[list[float]]
    usage: Optional[CommonUsage] = None
    raw: Any = None
