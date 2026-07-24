from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .chat import chat_text
from .embeddings import cosine_similarity, embed_text

logger = logging.getLogger(__name__)


def _emit(message: str) -> None:
    logger.info(message)


@dataclass(frozen=True)
class OutputComparison:
    output_a: str
    output_b: str

    length_a: int
    length_b: int

    word_count_a: int
    word_count_b: int

    exact_match: bool
    normalised_exact_match: bool

    embedding_similarity: Optional[float] = None

    judge_provider: Optional[str] = None
    judge_model: Optional[str] = None
    judge_verdict: Optional[str] = None


def compare_outputs(
    output_a: str,
    output_b: str,
    *,
    use_embeddings: bool = True,
    embedding_provider: str = "openai",
    embedding_model: Optional[str] = None,
    use_judge: bool = False,
    judge_provider: Any = None,
    judge_model: Optional[str] = None,
    judge_system: Optional[str] = None,
    judge_prompt: Optional[str] = None,
) -> OutputComparison:
    """
    ``judge_provider`` may be either a provider name string (e.g. ``'openai'``)
    or an instantiated provider object with a ``.chat(...)`` method.
    """
    if not isinstance(output_a, str) or not isinstance(output_b, str):
        raise TypeError("output_a and output_b must both be strings.")

    length_a = len(output_a)
    length_b = len(output_b)

    word_count_a = len(output_a.split())
    word_count_b = len(output_b.split())

    exact_match = output_a == output_b
    normalised_exact_match = _normalise_text(output_a) == _normalise_text(output_b)

    embedding_similarity: Optional[float] = None
    if use_embeddings:
        vector_a = embed_text(output_a, provider_name=embedding_provider, model=embedding_model)
        vector_b = embed_text(output_b, provider_name=embedding_provider, model=embedding_model)
        embedding_similarity = cosine_similarity(vector_a, vector_b)

    judge_verdict: Optional[str] = None
    if use_judge:
        prompt = judge_prompt or _default_judge_prompt(output_a, output_b)
        system = judge_system or _default_judge_system()

        if isinstance(judge_provider, str):
            judge_verdict = chat_text(
                provider_name=judge_provider, model=judge_model, system=system, user=prompt
            )
        else:
            judge_verdict = chat_text(
                provider=judge_provider, model=judge_model, system=system, user=prompt
            )

    judge_provider_name = (
        judge_provider
        if isinstance(judge_provider, str)
        else getattr(judge_provider, "name", "custom")
    )

    return OutputComparison(
        output_a=output_a,
        output_b=output_b,
        length_a=length_a,
        length_b=length_b,
        word_count_a=word_count_a,
        word_count_b=word_count_b,
        exact_match=exact_match,
        normalised_exact_match=normalised_exact_match,
        embedding_similarity=embedding_similarity,
        judge_provider=judge_provider_name if use_judge else None,
        judge_model=judge_model if use_judge else None,
        judge_verdict=judge_verdict,
    )


def print_comparison(comparison: OutputComparison) -> None:
    _emit(f"Length A: {comparison.length_a}")
    _emit(f"Length B: {comparison.length_b}")
    _emit(f"Word count A: {comparison.word_count_a}")
    _emit(f"Word count B: {comparison.word_count_b}")
    _emit(f"Exact match: {comparison.exact_match}")
    _emit(f"Normalised exact match: {comparison.normalised_exact_match}")

    if comparison.embedding_similarity is not None:
        _emit(f"Embedding similarity: {comparison.embedding_similarity:.6f}")

    if comparison.judge_verdict:
        _emit("Judge verdict:")
        _emit(comparison.judge_verdict)


def _normalise_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _default_judge_system() -> str:
    return (
        "You are a careful evaluator. Compare two outputs fairly. "
        "Do not assume longer means better. "
        "Do not assume more elaborate means better. "
        "State concrete differences, strengths, weaknesses and an overall verdict."
    )


def _default_judge_prompt(output_a: str, output_b: str) -> str:
    return (
        "Compare the following two outputs.\n\n"
        "Give:\n"
        "1. A short summary of the main differences\n"
        "2. Strengths of Output A\n"
        "3. Strengths of Output B\n"
        "4. Weaknesses of Output A\n"
        "5. Weaknesses of Output B\n"
        "6. A final verdict\n\n"
        "Output A:\n"
        f"{output_a}\n\n"
        "Output B:\n"
        f"{output_b}"
    )
