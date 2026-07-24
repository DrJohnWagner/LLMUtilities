from __future__ import annotations

import pytest

import LLMUtilities.compare as compare_module
from LLMUtilities.compare import OutputComparison, compare_outputs


def test_compare_outputs_basic_text_stats():
    comparison = compare_outputs("Hello world", "hello   world", use_embeddings=False)

    assert isinstance(comparison, OutputComparison)
    assert comparison.exact_match is False
    assert comparison.normalised_exact_match is True
    assert comparison.word_count_a == 2
    assert comparison.word_count_b == 2


def test_compare_outputs_rejects_non_string_inputs():
    with pytest.raises(TypeError):
        compare_outputs(123, "hello", use_embeddings=False)


def test_compare_outputs_computes_embedding_similarity(monkeypatch):
    def fake_embed_text(text, *, provider_name=None, model=None):
        return [1.0, 0.0]

    monkeypatch.setattr(compare_module, "embed_text", fake_embed_text)

    comparison = compare_outputs("abc", "xyz", use_embeddings=True)
    assert comparison.embedding_similarity == pytest.approx(1.0)


def test_compare_outputs_runs_judge_with_provider_name_string(monkeypatch):
    captured = {}

    def fake_chat_text(*, provider=None, provider_name=None, model=None, system=None, user=None, **kwargs):
        captured["provider_name"] = provider_name
        captured["user"] = user
        return "Output A wins"

    monkeypatch.setattr(compare_module, "chat_text", fake_chat_text)

    comparison = compare_outputs(
        "one", "two",
        use_embeddings=False,
        use_judge=True,
        judge_provider="anthropic",
        judge_model="claude-sonnet-5",
    )

    assert comparison.judge_verdict == "Output A wins"
    assert comparison.judge_provider == "anthropic"
    assert captured["provider_name"] == "anthropic"
    assert "one" in captured["user"] and "two" in captured["user"]


def test_compare_outputs_runs_judge_with_provider_object(monkeypatch):
    class _FakeProvider:
        name = "fake"

    def fake_chat_text(*, provider=None, provider_name=None, model=None, system=None, user=None, **kwargs):
        return "Output B wins"

    monkeypatch.setattr(compare_module, "chat_text", fake_chat_text)

    comparison = compare_outputs(
        "one", "two",
        use_embeddings=False,
        use_judge=True,
        judge_provider=_FakeProvider(),
    )

    assert comparison.judge_verdict == "Output B wins"
    assert comparison.judge_provider == "fake"


def test_compare_outputs_skips_judge_by_default():
    comparison = compare_outputs("one", "two", use_embeddings=False)
    assert comparison.judge_verdict is None
    assert comparison.judge_provider is None
