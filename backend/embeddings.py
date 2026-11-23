"""Shared Google GenAI embedding configuration and helpers."""

import hashlib
import math
import os
from typing import Any

from google.genai import types

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_MODEL_ENV = "GEMINI_EMBEDDING_MODEL"
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"


def embedding_model() -> str:
    return os.getenv(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)


def note_embedding_text(title: str, content: str) -> str:
    return f"{title}\n\n{content}"


def note_content_hash(title: str, content: str) -> str:
    value = note_embedding_text(title, content).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _embedding_values(response: Any) -> list[float]:
    embeddings = response.embeddings
    if not embeddings or embeddings[0].values is None:
        raise ValueError("The embedding model returned no vector.")

    try:
        values = [float(value) for value in embeddings[0].values]
    except (TypeError, ValueError) as exc:
        raise ValueError("The embedding model returned an invalid vector.") from exc

    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("The embedding model returned an invalid vector.")
    return values


def embed_text(
    client,
    text: str,
    *,
    task_type: str,
    model: str | None = None,
) -> list[float]:
    response = client.models.embed_content(
        model=model or embedding_model(),
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return _embedding_values(response)


async def embed_text_async(
    client,
    text: str,
    *,
    task_type: str,
    model: str | None = None,
) -> list[float]:
    response = await client.models.embed_content(
        model=model or embedding_model(),
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return _embedding_values(response)
