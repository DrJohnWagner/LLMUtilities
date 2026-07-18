from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..config import settings
from ..exceptions import ConfigurationError
from ..types import ImageRequest, ImageResponse


class BaseImageModel(ABC):
    provider_name: str = ""
    default_model: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: float
    max_retries: int

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.default_model = self.default_model if model is None else model
        self.api_key = self.api_key if api_key is None else api_key
        self.timeout_seconds = (
            settings.request_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self.max_retries = settings.max_retries if max_retries is None else max_retries

    @abstractmethod
    def generate(self, request: ImageRequest) -> ImageResponse:
        raise NotImplementedError

    def get_model_name(self, request: ImageRequest) -> str:
        model_name = request.model or self.default_model
        if not model_name:
            raise ConfigurationError(
                f"No image model configured for provider {self.provider_name!r}."
            )
        return model_name

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigurationError(
                f"Missing API key for provider {self.provider_name!r}."
            )
        return self.api_key

    def prepare_request(self, request: ImageRequest) -> ImageRequest:
        if not request.prompt.strip():
            raise ValueError("ImageRequest.prompt must not be empty.")
        return request
