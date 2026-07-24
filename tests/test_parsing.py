from __future__ import annotations

import pytest
from pydantic import BaseModel

from LLMUtilities.parsing import (
    build_structured_output_prompt,
    generate_structured_output,
    parse_json,
    parse_json_as,
    safe_parse_json,
)
from LLMUtilities.types import ChatRequest, ChatResponse


class _Result(BaseModel):
    name: str
    score: int


def test_parse_json_extracts_fenced_block():
    text = 'Here you go:\n```json\n{"name": "JoJo", "score": 42}\n```\n'
    assert parse_json(text) == {"name": "JoJo", "score": 42}


def test_parse_json_repairs_trailing_comma_and_single_quotes():
    text = "{'name': 'JoJo', 'score': 42,}"
    assert parse_json(text) == {"name": "JoJo", "score": 42}


def test_parse_json_as_validates_pydantic_model():
    text = '{"name": "JoJo", "score": 42}'
    result = parse_json_as(text, _Result)
    assert result == _Result(name="JoJo", score=42)


def test_safe_parse_json_returns_default_on_failure():
    assert safe_parse_json("not json at all {{{", default="fallback") == "fallback"


def test_build_structured_output_prompt_includes_schema_and_rules():
    prompt = build_structured_output_prompt("Summarise this.", _Result)
    assert "Summarise this." in prompt
    assert "JSON schema" in prompt
    assert '"score"' in prompt


class _FakeStructuredProvider:
    name = "fake"

    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, request: ChatRequest) -> ChatResponse:
        text = self._responses.pop(0)
        return ChatResponse(
            text=text, provider="fake", requested_model=request.model,
            resolved_model=request.model or "fake-model", usage=None, raw=None,
        )


def test_generate_structured_output_parses_first_response():
    provider = _FakeStructuredProvider(['{"name": "JoJo", "score": 42}'])
    result = generate_structured_output(
        user_prompt="Describe JoJo.", output_model=_Result, provider=provider
    )
    assert result == _Result(name="JoJo", score=42)


def test_generate_structured_output_retries_once_on_bad_first_response():
    provider = _FakeStructuredProvider(
        ["not json at all", '{"name": "JoJo", "score": 42}']
    )
    result = generate_structured_output(
        user_prompt="Describe JoJo.", output_model=_Result, provider=provider
    )
    assert result == _Result(name="JoJo", score=42)


def test_generate_structured_output_raises_when_repair_also_fails():
    provider = _FakeStructuredProvider(["still not json", "still not json either"])
    with pytest.raises(ValueError):
        generate_structured_output(
            user_prompt="Describe JoJo.", output_model=_Result, provider=provider
        )


def test_generate_structured_output_can_disable_retry():
    provider = _FakeStructuredProvider(["not json at all"])
    with pytest.raises(Exception):
        generate_structured_output(
            user_prompt="Describe JoJo.",
            output_model=_Result,
            provider=provider,
            retry_on_parse_failure=False,
        )
