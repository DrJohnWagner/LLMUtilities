from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, TypeAdapter

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def extract_json_string(text: str) -> str:
    """
    Extract a JSON string from LLM output.

    Tries:
    1. ```json ... ``` blocks
    2. first {...} or [...] span
    3. fallback to full text
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")

    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    brace_start = text.find("{")
    bracket_start = text.find("[")

    starts = [i for i in (brace_start, bracket_start) if i != -1]
    if starts:
        start = min(starts)
        return _extract_balanced_json(text[start:])

    return text.strip()


def parse_json(text: str, *, strict: bool = False) -> Any:
    """
    Parse JSON from LLM output.

    If strict=False, attempts light repair before failing.
    """
    json_str = extract_json_string(text)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        if strict:
            raise

    repaired = repair_json(json_str)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON after repair:\n{json_str}") from exc


def parse_json_as(text: str, model: Any, *, strict: bool = False) -> Any:
    """Parse JSON and validate against a Pydantic model or type annotation."""
    data = parse_json(text, strict=strict)

    if isinstance(model, type) and issubclass(model, BaseModel):
        return model.model_validate(data)
    return TypeAdapter(model).validate_python(data)


def repair_json(text: str) -> str:
    """
    Apply simple heuristics to fix common LLM JSON mistakes.

    This is intentionally conservative.
    """
    s = text.strip()

    if "'" in s and '"' not in s:
        s = s.replace("'", '"')

    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = _strip_to_outer_json(s)
    s = _remove_json_line_comments(s)

    return s.strip()


def safe_parse_json(text: str, *, default: Any = None, strict: bool = False) -> Any:
    """Parse JSON but return a default instead of raising."""
    try:
        return parse_json(text, strict=strict)
    except Exception:
        return default


def _extract_balanced_json(text: str) -> str:
    """Extract the first balanced JSON object/array from text."""
    stack: list[str] = []
    start = None
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue

        if ch == "\\" and in_string:
            escaped = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch in "{[":
            if start is None:
                start = i
            stack.append(ch)

        elif ch in "}]":
            if not stack:
                continue

            opening = stack.pop()
            if not _matches(opening, ch):
                continue

            if not stack and start is not None:
                return text[start : i + 1]

    return text


def _matches(opening: str, closing: str) -> bool:
    return (opening == "{" and closing == "}") or (opening == "[" and closing == "]")


def _remove_json_line_comments(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    i = 0

    while i < len(text):
        ch = text[i]

        if escaped:
            result.append(ch)
            escaped = False
            i += 1
            continue

        if ch == "\\" and in_string:
            result.append(ch)
            escaped = True
            i += 1
            continue

        if ch == '"':
            result.append(ch)
            in_string = not in_string
            i += 1
            continue

        if not in_string and ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _strip_to_outer_json(text: str) -> str:
    """Strip text to the outermost JSON object or array."""
    brace_start = text.find("{")
    bracket_start = text.find("[")

    starts = [i for i in (brace_start, bracket_start) if i != -1]
    if not starts:
        return text

    start = min(starts)
    return text[start:]
