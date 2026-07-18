from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from ..types import ChatRequest, ChatResponse, Message

# from LLMUtilities.logging.tracing import log_chat_request, log_chat_response
# from LLMUtilities.types import ChatRequest, Message

# request = ChatRequest(
#     messages=[
#         Message(role="system", content="You are a poet."),
#         Message(role="user", content="Write a haiku about recursion."),
#     ]
# )

# log_chat_request(
#     "logs/traces.jsonl",
#     request,
#     provider="openai",
#     resolved_model="gpt-5-mini",
# )

#
# THEN after a response is received from the model, you can log the response:
#

# log_chat_response(
#     "logs/traces.jsonl",
#     response,
# )

# A JSONL log line will look roughly like:
# {
#     "event_type": "chat_request",
#     "timestamp": "2026-04-15T12:34:56.789012+00:00",
#     "provider": "openai",
#     "model": "gpt-5-mini",
#     "payload": {
#         "messages": [
#             {"role": "system", "content": "You are a poet."},
#             {"role": "user", "content": "Write a haiku about recursion."},
#         ],
#         "request_model": null,
#         "resolved_model": "gpt-5-mini",
#         "temperature": null,
#         "max_output_tokens": null,
#     },
# }

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TraceRecord:
    event_type: str
    timestamp: str
    provider: Optional[str] = None
    model: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_trace_record(path: str | Path, record: TraceRecord) -> None:
    trace_path = Path(path)
    ensure_parent_dir(trace_path)

    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False))
        f.write("\n")


def truncate_text(text: str, max_chars: Optional[int] = None) -> str:
    if max_chars is None or max_chars <= 0:
        return text

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "...<truncated>"


def serialise_message(
    message: Message,
    *,
    max_chars: Optional[int] = None,
) -> dict[str, str]:
    return {
        "role": message.role,
        "content": truncate_text(message.content, max_chars=max_chars),
    }


def serialise_messages(
    messages: Sequence[Message],
    *,
    max_chars: Optional[int] = None,
) -> list[dict[str, str]]:
    return [
        serialise_message(message, max_chars=max_chars)
        for message in messages
    ]


def log_chat_request(
    path: str | Path,
    request: ChatRequest,
    *,
    provider: Optional[str] = None,
    resolved_model: Optional[str] = None,
    max_chars: Optional[int] = 2000,
    extra_payload: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "messages": serialise_messages(request.messages, max_chars=max_chars),
        "request_model": request.model,
        "resolved_model": resolved_model,
        "temperature": request.temperature,
        "max_output_tokens": request.max_output_tokens,
    }

    if extra_payload:
        payload.update(extra_payload)

    record = TraceRecord(
        event_type="chat_request",
        timestamp=utc_timestamp(),
        provider=provider,
        model=resolved_model or request.model,
        payload=payload,
    )
    append_trace_record(path, record)


def log_chat_response(
    path: str | Path,
    response: ChatResponse,
    *,
    max_chars: Optional[int] = 4000,
    include_raw: bool = False,
    extra_payload: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "text": truncate_text(response.text, max_chars=max_chars),
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "stop_reason": response.stop_reason,
    }

    if include_raw:
        payload["raw"] = _safe_raw_repr(response.raw, max_chars=max_chars)

    if extra_payload:
        payload.update(extra_payload)

    record = TraceRecord(
        event_type="chat_response",
        timestamp=utc_timestamp(),
        provider=response.provider,
        model=response.model,
        payload=payload,
    )
    append_trace_record(path, record)


def log_error(
    path: str | Path,
    error: Exception,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    extra_payload: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if extra_payload:
        payload.update(extra_payload)

    record = TraceRecord(
        event_type="error",
        timestamp=utc_timestamp(),
        provider=provider,
        model=model,
        payload=payload,
    )
    append_trace_record(path, record)


def _safe_raw_repr(raw: Any, *, max_chars: Optional[int] = None) -> str:
    try:
        text = repr(raw)
    except Exception:
        text = "<unrepresentable raw object>"

    return truncate_text(text, max_chars=max_chars)
