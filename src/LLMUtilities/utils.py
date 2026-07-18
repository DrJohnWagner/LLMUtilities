from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    text_parts.append(part["text"])
                continue

            if isinstance(part, BaseModel):
                if getattr(part, "type", None) == "text" and getattr(
                    part, "text", None
                ):
                    text_parts.append(part.text)
                continue

            if getattr(part, "type", None) == "text" and getattr(part, "text", None):
                text_parts.append(part.text)

        return " ".join(text_parts)

    return str(content)


def serialise_content(content: Any, *, max_chars: int | None = None) -> Any:
    if isinstance(content, str):
        return truncate_text(content, max_chars=max_chars)

    if not isinstance(content, list):
        return content

    serialised: list[Any] = []
    for part in content:
        if isinstance(part, BaseModel):
            item = part.model_dump(mode="python")
        elif isinstance(part, dict):
            item = dict(part)
        else:
            item = part

        if isinstance(item, dict) and isinstance(item.get("text"), str):
            item["text"] = truncate_text(item["text"], max_chars=max_chars)

        serialised.append(item)

    return serialised


def truncate_text(text: str, max_chars: int | None = None) -> str:
    if max_chars is None or max_chars <= 0:
        return text

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "...<truncated>"
