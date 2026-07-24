from __future__ import annotations

import json
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from ..chat import chat_text
from .json_parsing import parse_json_as

T = TypeVar("T", bound=BaseModel)


def build_json_schema_prompt(
    model: Type[T],
    *,
    include_schema: bool = True,
    include_rules: bool = True,
) -> str:
    parts: list[str] = []

    if include_rules:
        parts.append(
            "Return exactly one valid JSON object and nothing else.\n"
            "Do not wrap the JSON in markdown fences.\n"
            "Do not add commentary, explanation or notes.\n"
            "Do not omit required fields.\n"
            "Do not add extra fields."
        )

    if include_schema:
        schema = model.model_json_schema()
        schema_json = json.dumps(schema, indent=4, ensure_ascii=False)
        parts.append(f"JSON schema:\n{schema_json}")

    return "\n\n".join(parts)


def build_structured_output_prompt(
    user_prompt: str,
    model: Type[T],
    *,
    include_schema: bool = True,
    include_rules: bool = True,
) -> str:
    schema_prompt = build_json_schema_prompt(
        model, include_schema=include_schema, include_rules=include_rules
    )

    return f"{user_prompt.strip()}\n\n{schema_prompt}"


def generate_structured_output(
    *,
    user_prompt: str,
    output_model: Type[T],
    provider: Any = None,
    provider_name: str = "openai",
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    include_schema: bool = True,
    include_rules: bool = True,
    retry_on_parse_failure: bool = True,
) -> T:
    prompt = build_structured_output_prompt(
        user_prompt,
        output_model,
        include_schema=include_schema,
        include_rules=include_rules,
    )

    response_text = chat_text(
        provider=provider,
        provider_name=provider_name,
        model=model,
        system=system,
        user=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    try:
        return parse_json_as(response_text, output_model)
    except Exception:
        if not retry_on_parse_failure:
            raise

        retry_prompt = build_repair_prompt(
            bad_output=response_text, output_model=output_model
        )

        repaired_response_text = chat_text(
            provider=provider,
            provider_name=provider_name,
            model=model,
            system=system,
            user=retry_prompt,
            temperature=0,
            max_output_tokens=max_output_tokens,
        )

        try:
            return parse_json_as(repaired_response_text, output_model)
        except Exception as second_exc:
            raise ValueError(
                "Failed to generate valid structured output.\n\n"
                f"First response:\n{response_text}\n\n"
                f"Repair response:\n{repaired_response_text}"
            ) from second_exc


def build_repair_prompt(*, bad_output: str, output_model: Type[T]) -> str:
    schema = json.dumps(output_model.model_json_schema(), indent=4, ensure_ascii=False)

    return (
        "The following output was supposed to be a single valid JSON object "
        "matching the schema below, but it is malformed or invalid.\n\n"
        "Your task:\n"
        "1. Repair it\n"
        "2. Return exactly one valid JSON object\n"
        "3. Return nothing except the JSON object\n"
        "4. Do not add commentary\n\n"
        f"Schema:\n{schema}\n\n"
        f"Bad output:\n{bad_output}"
    )


def validate_structured_output(data: object, output_model: Type[T]) -> T:
    if isinstance(data, output_model):
        return data
    return output_model.model_validate(data)


structured_output = generate_structured_output
