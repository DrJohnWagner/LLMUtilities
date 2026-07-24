from __future__ import annotations

from typing import Any, Optional, Sequence

from .capabilities import EmbeddingCapability, require_capability
from .chat import resolve_provider
from .types import EmbeddingRequest, EmbeddingResponse


def embed_texts(
    texts: Sequence[str],
    *,
    provider: Any = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    task_type: Optional[str] = None,
) -> EmbeddingResponse:
    request = EmbeddingRequest(
        texts=list(texts), model=model, dimensions=dimensions, task_type=task_type
    )

    resolved_provider = resolve_provider(provider, provider_name)
    require_capability(resolved_provider, EmbeddingCapability, "EmbeddingCapability")
    return resolved_provider.embed(request)


def embed_text(
    text: str,
    *,
    provider: Any = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    task_type: Optional[str] = None,
) -> list[float]:
    """Convenience wrapper - returns only the single embedding vector."""
    response = embed_texts(
        [text],
        provider=provider,
        provider_name=provider_name,
        model=model,
        dimensions=dimensions,
        task_type=task_type,
    )
    return response.vectors[0]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cannot compute cosine similarity for a zero vector.")

    return dot / ((norm_a**0.5) * (norm_b**0.5))
