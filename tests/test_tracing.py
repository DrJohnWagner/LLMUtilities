from __future__ import annotations

import json

from LLMUtilities.tracing import log_chat_request, log_chat_response, log_error
from LLMUtilities.types import ChatRequest, ChatResponse, CommonUsage, CostSummary, Message


def _read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_log_chat_request_writes_expected_shape(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    request = ChatRequest(
        messages=[Message(role="user", content="hello")], model="gpt-5.6-terra"
    )

    log_chat_request(trace_path, request, provider="openai")

    records = _read_records(trace_path)
    assert len(records) == 1
    record = records[0]
    assert record["event_type"] == "chat_request"
    assert record["provider"] == "openai"
    assert record["requested_model"] == "gpt-5.6-terra"
    assert record["payload"]["messages"][0]["content"] == "hello"


def test_log_chat_response_includes_cost_summary_when_given(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    response = ChatResponse(
        text="hi there",
        provider="openai",
        requested_model="gpt-5.6-terra",
        resolved_model="gpt-5.6-terra-2026-07-18",
        usage=CommonUsage(total_input_tokens=10, total_output_tokens=5, total_tokens=15),
        raw=None,
    )
    cost_summary = CostSummary(
        input_cost=0.001,
        output_cost=0.002,
        other_cost=0.0,
        total_cost=0.003,
        currency="USD",
        provider="openai",
        requested_model="gpt-5.6-terra",
        resolved_model="gpt-5.6-terra-2026-07-18",
    )

    log_chat_response(trace_path, response, cost_summary=cost_summary)

    records = _read_records(trace_path)
    record = records[0]
    assert record["event_type"] == "chat_response"
    assert record["resolved_model"] == "gpt-5.6-terra-2026-07-18"
    assert record["payload"]["usage"]["total_tokens"] == 15
    assert record["payload"]["cost_summary"]["total_cost"] == 0.003


def test_log_chat_response_truncates_long_text(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    response = ChatResponse(
        text="x" * 100,
        provider="openai",
        requested_model=None,
        resolved_model="gpt-5.6-terra",
        raw=None,
    )

    log_chat_response(trace_path, response, max_chars=10)

    records = _read_records(trace_path)
    assert records[0]["payload"]["text"].endswith("...<truncated>")


def test_log_error_records_error_type_and_message(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    log_error(trace_path, ValueError("bad input"), provider="openai")

    records = _read_records(trace_path)
    assert records[0]["event_type"] == "error"
    assert records[0]["payload"]["error_type"] == "ValueError"
    assert records[0]["payload"]["error_message"] == "bad input"
