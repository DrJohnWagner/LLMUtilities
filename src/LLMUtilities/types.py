from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

MessageRole = Literal["system", "user", "assistant"]


class TextContentPart(BaseModel):
    """Text-only content part for multimodal messages."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str = Field(min_length=1)


class ImageContentPart(BaseModel):
    """Image content part for multimodal messages."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image"]
    source: dict = Field(description="Provider-agnostic image source spec")


ContentPart = Union[TextContentPart, ImageContentPart]
ContentType = Union[str, list[ContentPart]]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: ContentType = Field(
        description="Either a string (backward compatible) or list of content parts (multimodal)"
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Any) -> Any:
        """Validate that content is either a non-empty string or non-empty list."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("String content must not be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Content list must not be empty")
        else:
            raise ValueError("Content must be a string or list of content parts")
        return v


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1)
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None


class ChatUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    text: str
    provider: str
    model: str
    usage: ChatUsage = Field(default_factory=ChatUsage)
    stop_reason: Optional[str] = None
    raw: Any = None


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    model: Optional[str] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    background: Optional[str] = None
    format: Optional[str] = None
    n: int = Field(default=1, ge=1)
    seed: Optional[int] = None
    user: Optional[str] = None


class ImageArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mime_type: str
    b64_data: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ImageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    provider: str
    model: str
    artifacts: list[ImageArtifact] = Field(min_length=1)
    usage: ImageUsage = Field(default_factory=ImageUsage)
    raw: Any = None
