from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from ..types import ChatRequest, ChatResponse, CostSummary, ImageResponse, Message
from ..utils import serialise_content, truncate_text


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TraceRecord:
    event_type: str
    timestamp: str
    provider: Optional[str] = None
    requested_model: Optional[str] = None
    resolved_model: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_trace_record(path: str | Path, record: TraceRecord) -> None:
    trace_path = Path(path)
    ensure_parent_dir(trace_path)

    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False))
        f.write("\n")


def serialise_message(
    message: Message, *, max_chars: Optional[int] = None
) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": serialise_content(message.content, max_chars=max_chars),
    }


def serialise_messages(
    messages: Sequence[Message], *, max_chars: Optional[int] = None
) -> list[dict[str, Any]]:
    return [serialise_message(message, max_chars=max_chars) for message in messages]


def log_chat_request(
    path: str | Path,
    request: ChatRequest,
    *,
    provider: Optional[str] = None,
    max_chars: Optional[int] = 2000,
    extra_payload: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "messages": serialise_messages(request.messages, max_chars=max_chars),
        "requested_model": request.model,
        "temperature": request.temperature,
        "max_output_tokens": request.max_output_tokens,
    }
    if extra_payload:
        payload.update(extra_payload)

    record = TraceRecord(
        event_type="chat_request",
        timestamp=utc_timestamp(),
        provider=provider,
        requested_model=request.model,
        resolved_model=None,
        payload=payload,
    )
    append_trace_record(path, record)


def log_chat_response(
    path: str | Path,
    response: ChatResponse,
    *,
    cost_summary: Optional[CostSummary] = None,
    max_chars: Optional[int] = 4000,
    include_raw: bool = False,
    extra_payload: Optional[dict[str, Any]] = None,
) -> None:
    payload: dict[str, Any] = {
        "text": truncate_text(response.text, max_chars=max_chars),
        "usage": _serialise_usage(response),
        "stop_reason": response.stop_reason,
    }
    if cost_summary is not None:
        payload["cost_summary"] = cost_summary.model_dump()
    if include_raw:
        payload["raw"] = _safe_raw_repr(response.raw, max_chars=max_chars)
    if extra_payload:
        payload.update(extra_payload)

    record = TraceRecord(
        event_type="chat_response",
        timestamp=utc_timestamp(),
        provider=response.provider,
        requested_model=response.requested_model,
        resolved_model=response.resolved_model,
        payload=payload,
    )
    append_trace_record(path, record)


def log_image_response(
    path: str | Path,
    response: ImageResponse,
    *,
    cost_summary: Optional[CostSummary] = None,
    include_raw: bool = False,
    extra_payload: Optional[dict[str, Any]] = None,
) -> None:
    payload: dict[str, Any] = {
        "artifact_count": len(response.artifacts),
    }
    if cost_summary is not None:
        payload["cost_summary"] = cost_summary.model_dump()
    if include_raw:
        payload["raw"] = _safe_raw_repr(response.raw)
    if extra_payload:
        payload.update(extra_payload)

    record = TraceRecord(
        event_type="image_response",
        timestamp=utc_timestamp(),
        provider=response.provider,
        requested_model=response.requested_model,
        resolved_model=response.resolved_model,
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
    payload = {"error_type": type(error).__name__, "error_message": str(error)}
    if extra_payload:
        payload.update(extra_payload)

    record = TraceRecord(
        event_type="error",
        timestamp=utc_timestamp(),
        provider=provider,
        requested_model=model,
        resolved_model=None,
        payload=payload,
    )
    append_trace_record(path, record)


def _safe_raw_repr(raw: Any, *, max_chars: Optional[int] = None) -> str:
    try:
        text = repr(raw)
    except Exception:
        text = "<unrepresentable raw object>"
    return truncate_text(text, max_chars=max_chars)


def _serialise_usage(response: ChatResponse) -> Optional[dict[str, Any]]:
    if response.usage is None:
        return None
    return {
        "total_input_tokens": response.usage.total_input_tokens,
        "total_output_tokens": response.usage.total_output_tokens,
        "total_tokens": response.usage.total_tokens,
    }
