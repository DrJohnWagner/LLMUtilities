from __future__ import annotations

import pytest

from LLMUtilities.embeddings import cosine_similarity, embed_texts
from LLMUtilities.exceptions import UnsupportedCapabilityError
from LLMUtilities.types import EmbeddingRequest, EmbeddingResponse


class _FakeEmbeddingProvider:
    name = "fake"

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(
            provider="fake",
            requested_model=request.model,
            resolved_model=request.model or "fake-embedding-model",
            vectors=[[1.0, 0.0] for _ in request.texts],
            usage=None,
            raw=None,
        )


class _NotAnEmbeddingProvider:
    name = "not-embedding"


def test_embed_texts_delegates_to_provider():
    response = embed_texts(["a", "b"], provider=_FakeEmbeddingProvider())
    assert response.vectors == [[1.0, 0.0], [1.0, 0.0]]


def test_embed_texts_rejects_provider_without_capability():
    with pytest.raises(UnsupportedCapabilityError):
        embed_texts(["a"], provider=_NotAnEmbeddingProvider())


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cosine_similarity([1, 0], [1, 0, 0])


def test_cosine_similarity_rejects_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 0])
