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
    content: ContentType

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Any) -> Any:
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


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(min_length=1)
    model: Optional[str] = None
    dimensions: Optional[int] = None
    task_type: Optional[str] = None
