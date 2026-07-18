from __future__ import annotations

import logging
from typing import Optional, Sequence

from .config import settings
from .exceptions import ConfigurationError, MissingDependencyError, ProviderError

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
except ImportError:
    genai = None


logger = logging.getLogger(__name__)


def _emit(message: str) -> None:
    logger.info(message)


def embed_text(
    text: str,
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    task_type: Optional[str] = None,
) -> list[float]:
    embeddings = embed_texts(
        [text],
        provider=provider,
        model=model,
        dimensions=dimensions,
        task_type=task_type,
    )
    return embeddings[0]


def embed_texts(
    texts: Sequence[str],
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    task_type: Optional[str] = None,
) -> list[list[float]]:
    if not texts:
        return []

    provider_name = provider.strip().lower()

    if provider_name == "openai":
        return _embed_openai_texts(
            texts,
            model=model,
            dimensions=dimensions,
        )

    if provider_name == "google":
        return _embed_google_texts(
            texts,
            model=model,
            dimensions=dimensions,
            task_type=task_type,
        )

    if provider_name == "anthropic":
        raise ProviderError("Anthropic does not support embeddings in LLMUtilities.")

    raise ConfigurationError(
        f"Unsupported embedding provider: {provider_name!r}. "
        f"Expected one of: 'openai', 'google'."
    )


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


def print_embedding_summary(
    embedding: Sequence[float],
    *,
    label: Optional[str] = None,
    preview_size: int = 8,
) -> None:
    if label:
        _emit(f"{label}:")

    _emit(f"Length: {len(embedding)}")
    _emit(f"Preview: {list(embedding[:preview_size])}")


def _embed_openai_texts(
    texts: Sequence[str],
    *,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> list[list[float]]:
    if OpenAI is None:
        raise MissingDependencyError(
            "The 'openai' package is required for OpenAI embeddings. "
            "Install it with: pip install openai"
        )

    api_key = settings.openai_api_key
    if not api_key:
        raise ConfigurationError("OPENAI_API_KEY is not configured.")

    embedding_model = model or settings.openai.embedding_model
    if not embedding_model:
        raise ConfigurationError("No OpenAI embedding model is configured.")

    client = OpenAI(api_key=api_key)

    kwargs = {
        "input": list(texts),
        "model": embedding_model,
    }

    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    response = client.embeddings.create(**kwargs)

    return [item.embedding for item in response.data]


def _embed_google_texts(
    texts: Sequence[str],
    *,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    task_type: Optional[str] = None,
) -> list[list[float]]:
    if genai is None:
        raise MissingDependencyError(
            "The 'google-genai' package is required for Google embeddings. "
            "Install it with: pip install google-genai"
        )

    api_key = settings.google_api_key
    if not api_key:
        raise ConfigurationError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not configured.")

    embedding_model = model or settings.google.embedding_model
    if not embedding_model:
        raise ConfigurationError("No Google embedding model is configured.")

    client = genai.Client(api_key=api_key)

    config: dict[str, object] = {}
    if dimensions is not None:
        config["output_dimensionality"] = dimensions
    if task_type is not None:
        config["task_type"] = task_type

    response = client.models.embed_content(
        model=embedding_model,
        contents=list(texts),
        config=config or None,
    )

    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        raise ValueError("Google embedding response did not contain embeddings.")

    vectors: list[list[float]] = []
    for item in embeddings:
        values = getattr(item, "values", None)
        if values is None:
            raise ValueError("Google embedding item did not contain values.")
        vectors.append(list(values))

    return vectors
