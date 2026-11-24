"""Shared Qdrant storage and retrieval for note, memory, and conversation vectors."""

import hashlib
import os
import threading

from qdrant_client import QdrantClient, models


_client: QdrantClient | None = None
_client_config: tuple[str, str | None] | None = None
_client_lock = threading.RLock()


def _client_for_config() -> QdrantClient:
    # Reuse one thread-safe client per Qdrant configuration. Tests use
    # ":memory:", while normal deployments use QDRANT_URL and optionally
    # QDRANT_API_KEY.
    global _client, _client_config

    with _client_lock:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        config = (url, api_key)
        if config != _client_config:
            if _client is not None:
                _client.close()
            if url == ":memory:":
                _client = QdrantClient(":memory:")
            else:
                _client = QdrantClient(url=url, api_key=api_key)
            _client_config = config
        return _client


def collection_name(model: str, dimensions: int, entity: str = "notes") -> str:
    # Embedding dimensions and models define incompatible vector spaces, so
    # keep each combination and entity in a separate collection.
    notes_prefix = os.getenv("QDRANT_COLLECTION_PREFIX", "note_embeddings")
    if entity == "notes":
        prefix = notes_prefix
    elif entity == "memories":
        prefix = os.getenv(
            "QDRANT_MEMORY_COLLECTION_PREFIX",
            f"{notes_prefix}_memories",
        )
        if prefix == notes_prefix:
            raise ValueError("Notes and memories require separate Qdrant collections.")
    elif entity == "conversations":
        prefix = os.getenv("QDRANT_CONVERSATION_COLLECTION_PREFIX") or (
            f"{notes_prefix}_conversations"
        )
        memory_prefix = os.getenv(
            "QDRANT_MEMORY_COLLECTION_PREFIX",
            f"{notes_prefix}_memories",
        )
        if prefix in {notes_prefix, memory_prefix}:
            raise ValueError("Conversations require a separate Qdrant collection.")
    else:
        raise ValueError(f"Unknown embedding entity: {entity!r}.")
    suffix = hashlib.sha256(f"{model}:{dimensions}".encode()).hexdigest()[:12]
    return f"{prefix}_{suffix}"


def _ensure_collection(name: str, dimensions: int) -> QdrantClient:
    with _client_lock:
        client = _client_for_config()
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
        return client


def upsert(
    *,
    record_id: int | str,
    user_id: int,
    vector: list[float],
    model: str,
    content_hash: str,
    entity: str = "notes",
) -> str:
    # Qdrant owns the vector and cosine index. SQLite stores only a reference
    # to the collection and the content hash used for stale-result checks.
    name = collection_name(model, len(vector), entity)
    client = _ensure_collection(name, len(vector))
    client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=record_id,
                vector=vector,
                payload={
                    "user_id": user_id,
                    "model": model,
                    "content_hash": content_hash,
                },
            )
        ],
        wait=True,
    )
    return name


def search(
    *,
    user_id: int,
    vector: list[float],
    model: str,
    limit: int,
    entity: str = "notes",
) -> list[tuple[int | str, float]]:
    # Apply the owner filter inside Qdrant so a user never retrieves another
    # user's vector results.
    name = collection_name(model, len(vector), entity)
    client = _client_for_config()
    if not client.collection_exists(name):
        return []
    results = client.query_points(
        collection_name=name,
        query=vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                ),
                models.FieldCondition(
                    key="model",
                    match=models.MatchValue(value=model),
                ),
            ]
        ),
        limit=limit,
        with_payload=False,
    )
    return [(point.id, point.score) for point in results.points]


def delete_points(collection: str, point_ids: list[int | str]) -> None:
    if not point_ids:
        return
    client = _client_for_config()
    if client.collection_exists(collection):
        client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )
