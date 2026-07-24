from __future__ import annotations

from typing import Optional

from ...exceptions import ConfigurationError, ResponseError
from ...transports.google_generate_content import GoogleGenerateContentTransport
from ...types import EmbeddingRequest, EmbeddingResponse

EMBEDDING_MODELS = ["text-embedding-004"]


def resolve_model_name(request: EmbeddingRequest, default_model: Optional[str]) -> str:
    model_name = request.model or default_model
    if not model_name:
        raise ConfigurationError("No embedding model configured for provider 'google'.")
    return model_name


def run_embed(
    *,
    transport: GoogleGenerateContentTransport,
    request: EmbeddingRequest,
    model_name: str,
) -> EmbeddingResponse:
    config: dict = {}
    if request.dimensions is not None:
        config["output_dimensionality"] = request.dimensions
    if request.task_type is not None:
        config["task_type"] = request.task_type

    response = transport.embed_content(
        model=model_name,
        contents=list(request.texts),
        config=config or None,
    )

    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        raise ResponseError("Google embedding response did not contain embeddings.")

    vectors: list[list[float]] = []
    for item in embeddings:
        values = getattr(item, "values", None)
        if values is None:
            raise ResponseError("Google embedding item did not contain values.")
        vectors.append(list(values))

    return EmbeddingResponse(
        provider="google",
        requested_model=request.model,
        resolved_model=model_name,
        vectors=vectors,
        usage=None,
        raw=response,
    )
