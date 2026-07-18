from __future__ import annotations

from typing import Any, Optional

from ..exceptions import (
    AuthenticationError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)
from ..types import ChatRequest, ChatResponse, ChatUsage
from ..utils import content_to_text
from .base import BaseChatModel


class BaseOpenAIChatCompletionsModel(BaseChatModel):
    openai_base_url: str

    def chat(self, request: ChatRequest) -> ChatResponse:
        try:
            from openai import OpenAI
            from openai import AuthenticationError as _OAIAuthError
            from openai import RateLimitError as _OAIRateLimitError
            from openai import APIConnectionError as _OAIConnectionError
            from openai import APIStatusError as _OAIStatusError
        except ImportError as exc:
            raise MissingDependencyError(
                "The 'openai' package is required for this provider. "
                "Install it with: pip install openai"
            ) from exc

        request = self.prepare_request(request)
        model_name = self.get_model_name(request)
        api_key = self.require_api_key()

        client = OpenAI(
            api_key=api_key,
            base_url=self.openai_base_url,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

        system_parts: list[str] = []
        input_messages: list[dict[str, str]] = []

        for msg in request.messages:
            content_str = content_to_text(msg.content)
            if msg.role == "system":
                system_parts.append(content_str)
            else:
                input_messages.append({"role": msg.role, "content": content_str})

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": input_messages,
        }
        if system_parts:
            kwargs["messages"] = _prefix_system_message(input_messages, system_parts)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens

        try:
            response = client.chat.completions.create(**kwargs)
        except _OAIAuthError as exc:
            raise AuthenticationError(
                f"{self.provider_name.title()} authentication failed: {exc}"
            ) from exc
        except _OAIRateLimitError as exc:
            raise RateLimitError(
                f"{self.provider_name.title()} rate limit exceeded: {exc}"
            ) from exc
        except _OAIConnectionError as exc:
            raise RequestError(
                f"{self.provider_name.title()} connection error: {exc}"
            ) from exc
        except _OAIStatusError as exc:
            raise ProviderError(
                f"{self.provider_name.title()} API error ({exc.status_code}): {exc.message}"
            ) from exc
        except Exception as exc:
            raise RequestError(
                f"Unexpected {self.provider_name.title()} request failure: {exc}"
            ) from exc

        try:
            choice = _first_choice(response)
            text = _extract_text(choice)
            usage = _extract_usage(response)

            return ChatResponse(
                text=text,
                provider=self.provider_name,
                model=model_name,
                usage=usage,
                stop_reason=getattr(choice, "finish_reason", None),
                raw=response,
            )
        except (ResponseError, ValueError):
            raise
        except Exception as exc:
            raise ResponseError(
                f"Failed to parse {self.provider_name.title()} response: {exc}"
            ) from exc


def _prefix_system_message(
    input_messages: list[dict[str, str]],
    system_parts: list[str],
) -> list[dict[str, str]]:
    messages = list(input_messages)
    messages.insert(
        0,
        {
            "role": "system",
            "content": "\n\n".join(system_parts),
        },
    )
    return messages


def _first_choice(response: Any) -> Any:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ResponseError("Chat completion response contained no choices.")
    return choices[0]


def _extract_text(choice: Any) -> str:
    message = getattr(choice, "message", None)
    if message is None:
        raise ResponseError("Chat completion choice is missing the message field.")

    content = getattr(message, "content", None)
    text = _collect_text(content)
    if not text:
        raise ResponseError("Chat completion choice contained no text output.")
    return text


def _collect_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        segments: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    segments.append(part["text"])
            elif getattr(part, "type", None) == "text" and getattr(part, "text", None):
                segments.append(part.text)
        return "".join(segments)

    if content is None:
        return ""

    return str(content)


def _extract_usage(response: Any) -> ChatUsage:
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return ChatUsage()

    input_tokens = getattr(usage_obj, "prompt_tokens", None)
    output_tokens = getattr(usage_obj, "completion_tokens", None)
    total_tokens = getattr(usage_obj, "total_tokens", None)

    cached_input_tokens = _extract_cached_input_tokens(usage_obj)

    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return ChatUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_read_input_tokens=cached_input_tokens,
    )


def _extract_cached_input_tokens(usage_obj: Any) -> Optional[int]:
    prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
    if prompt_details is None:
        return getattr(usage_obj, "cached_tokens", None)

    cached_tokens = getattr(prompt_details, "cached_tokens", None)
    if cached_tokens is not None:
        return cached_tokens

    if isinstance(prompt_details, dict):
        return prompt_details.get("cached_tokens")

    return None