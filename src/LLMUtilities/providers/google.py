from __future__ import annotations

from typing import Any, Optional

from ..config import settings
from ..exceptions import (
    AuthenticationError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)
from ..types import ChatRequest, ChatResponse, ChatUsage
from .base import BaseChatModel


class GoogleChatModel(BaseChatModel):
    """Google Gemini chat adapter."""

    provider_name = "google"
    default_model = settings.google.chat_model
    api_key = settings.google_api_key

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to Google Gemini and return a normalised ChatResponse."""
        try:
            from google import genai
            from google.genai import errors as _genai_errors
            from google.genai import types as genai_types
        except ImportError as exc:
            raise MissingDependencyError(
                "The 'google-genai' package is required for the Google provider. "
                "Install it with: pip install google-genai"
            ) from exc

        request = self.prepare_request(request)
        model_name = self.get_model_name(request)
        api_key = self.require_api_key()

        client = genai.Client(api_key=api_key)

        system_instruction, contents = self._split_messages(request, genai_types)
        config = self._build_config(
            system_instruction=system_instruction,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            genai_types=genai_types,
        )

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except _genai_errors.APIError as exc:
            self._raise_provider_exception(exc)
        except Exception as exc:
            raise RequestError(f"Unexpected Google request failure: {exc}") from exc

        try:
            text = _extract_text(response)
            usage = _extract_usage(response)
            stop_reason = _extract_stop_reason(response)

            return ChatResponse(
                text=text,
                provider=self.provider_name,
                model=model_name,
                usage=usage,
                stop_reason=stop_reason,
                raw=response,
            )
        except (ResponseError, ValueError):
            raise
        except Exception as exc:
            raise ResponseError(f"Failed to parse Google response: {exc}") from exc

    @staticmethod
    def _split_messages(
        request: ChatRequest,
        genai_types: Any,
    ) -> tuple[Optional[str], list[Any]]:
        system_parts: list[str] = []
        contents: list[Any] = []

        for message in request.messages:
            content_str = GoogleChatModel._normalize_content(message.content)
            if message.role == "system":
                system_parts.append(content_str)
                continue

            contents.append(
                genai_types.Content(
                    role=GoogleChatModel._map_role(message.role),
                    parts=[genai_types.Part(text=content_str)],
                )
            )

        if not contents:
            raise ValueError(
                "Google chat requests must include at least one non-system message."
            )

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    @staticmethod
    def _normalize_content(content: Any) -> str:
        """
        Normalize message content to a string for Google Gemini API.

        Handles both backward-compatible string content and new multimodal content parts.
        For multimodal content, extracts all text parts and concatenates them.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(part["text"])
                elif (
                    hasattr(part, "type")
                    and part.type == "text"
                    and hasattr(part, "text")
                ):
                    text_parts.append(part.text)
            return " ".join(text_parts) if text_parts else ""

        return str(content)

    @staticmethod
    def _map_role(role: str) -> str:
        if role == "user":
            return "user"
        if role == "assistant":
            return "model"
        raise ValueError(f"Unsupported Google message role: {role!r}")

    @staticmethod
    def _build_config(
        *,
        system_instruction: Optional[str],
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        genai_types: Any,
    ) -> Any:
        if system_instruction is None and temperature is None and max_output_tokens is None:
            return None

        return genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    @staticmethod
    def _raise_provider_exception(exc: Any) -> None:
        code = getattr(exc, "code", None)
        message = getattr(exc, "message", str(exc))

        if code in (401, 403):
            raise AuthenticationError(f"Google authentication failed: {message}") from exc
        if code == 429:
            raise RateLimitError(f"Google rate limit or quota exceeded: {message}") from exc
        if code in (408, 500, 502, 503, 504):
            raise RequestError(f"Google request failed: {message}") from exc

        raise ProviderError(f"Google API error ({code}): {message}") from exc


def _extract_text(response: Any) -> str:
    """
    Concatenate all text parts across all candidates in order.

    Iterates ``response.candidates``, then each candidate's content parts,
    and collects every part that has a non-empty ``text`` attribute.
    Raises ``ResponseError`` if no text is found.
    """
    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise ResponseError("Google response contained no candidates.")

    segments: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text_val = getattr(part, "text", None)
            if text_val:
                segments.append(text_val)

    if not segments:
        raise ResponseError(
            "Google response candidates contained no text parts."
        )

    return "".join(segments)


def _extract_usage(response: Any) -> ChatUsage:
    """
    Normalise Google usage metadata into the package's ``ChatUsage`` schema.

    Google's ``prompt_token_count`` maps to ``input_tokens``; ``candidates_token_count``
    maps to ``output_tokens``.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return ChatUsage()

    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)

    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return ChatUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _extract_stop_reason(response: Any) -> Optional[str]:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    finish_reason = getattr(candidates[0], "finish_reason", None)
    return str(finish_reason) if finish_reason is not None else None
