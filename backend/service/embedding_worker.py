"""Background generation and reconciliation of note, memory, and chat embeddings."""

import logging
import os
import threading
import uuid

from google import genai
from google.genai import types

from backend.conversation_chunks import chunk_conversation
from backend.embeddings import (
    DOCUMENT_TASK_TYPE,
    embedding_model,
    embed_text,
    note_content_hash,
    note_embedding_text,
)
from backend.service import db, memory_store

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    def __init__(self, *, batch_size: int = 20, poll_interval: float = 5):
        self._batch_size = batch_size
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lease_owner = uuid.uuid4().hex

    def start(self) -> None:
        if self._thread is not None:
            return
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "Embedding worker is disabled; set GEMINI_API_KEY to index notes, memories, and conversations."
            )
            return
        self._thread = threading.Thread(
            target=self._run,
            args=(api_key,),
            name="embedding-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self, api_key: str) -> None:
        try:
            options = types.HttpOptions(timeout=60_000)  # milliseconds
            with genai.Client(api_key=api_key, http_options=options) as client:
                self._process_jobs(client)
        except Exception:
            logger.exception("Embedding worker stopped unexpectedly.")

    def _process_jobs(self, client) -> None:
        while not self._stop_event.is_set():
            queues = [
                (
                    "note",
                    db.get_pending_note_embeddings(
                        self._batch_size,
                        lease_owner=self._lease_owner,
                    ),
                    db.get_legacy_note_embedding,
                    db.upsert_note_embedding,
                    db.record_note_embedding_failure,
                ),
                (
                    "memory",
                    memory_store.get_pending_memory_embeddings(
                        self._batch_size,
                        lease_owner=self._lease_owner,
                    ),
                    None,
                    memory_store.upsert_memory_embedding,
                    memory_store.record_memory_embedding_failure,
                ),
            ]
            conversation_jobs = db.get_pending_conversation_embeddings(
                self._batch_size,
                lease_owner=self._lease_owner,
            )
            if not any(jobs for _, jobs, _, _, _ in queues) and not conversation_jobs:
                self._stop_event.wait(self._poll_interval)
                continue
            for entity, jobs, get_legacy, upsert, record_failure in queues:
                for job in jobs:
                    if self._stop_event.is_set():
                        break
                    model = embedding_model()
                    content_hash = note_content_hash(job["title"], job["content"])
                    try:
                        legacy = get_legacy(job["id"]) if get_legacy else None
                        if (
                            legacy is not None
                            and legacy["model"] == model
                            and legacy["content_hash"] == content_hash
                        ):
                            vector = legacy["embedding"]
                        else:
                            vector = embed_text(
                                client,
                                note_embedding_text(job["title"], job["content"]),
                                task_type=DOCUMENT_TASK_TYPE,
                                model=model,
                            )
                        indexed = upsert(
                            job["user_id"],
                            job["id"],
                            vector,
                            model,
                            content_hash,
                        )
                        if not indexed:
                            logger.info(
                                "Skipped stale embedding result for %s %s.",
                                entity,
                                job["id"],
                            )
                    except Exception as exc:
                        record_failure(job["id"], str(exc))
                        logger.warning(
                            "Could not index %s %s; it will be retried.",
                            entity,
                            job["id"],
                            exc_info=True,
                        )

            for job in conversation_jobs:
                if self._stop_event.is_set():
                    break
                try:
                    model = embedding_model()
                    chunks = chunk_conversation(job["conversation_id"], job["messages"])
                    existing = {
                        item["chunk_id"]: item
                        for item in db.get_conversation_embedding_index(
                            job["user_id"], job["conversation_id"]
                        )
                    }
                    vectors = {}
                    for chunk in chunks:
                        indexed = existing.get(chunk["chunk_id"])
                        if (
                            indexed is not None
                            and indexed["content_hash"] == chunk["content_hash"]
                            and indexed["model"] == model
                        ):
                            continue
                        vectors[chunk["chunk_id"]] = embed_text(
                            client,
                            chunk["content"],
                            task_type=DOCUMENT_TASK_TYPE,
                            model=model,
                        )
                    indexed = db.upsert_conversation_embeddings(
                        job["user_id"],
                        job["conversation_id"],
                        chunks,
                        vectors,
                        model,
                        job["snapshot_hash"],
                    )
                    if not indexed:
                        logger.info(
                            "Skipped stale conversation embedding result for %s.",
                            job["conversation_id"],
                        )
                except Exception as exc:
                    db.record_conversation_embedding_failure(
                        job["conversation_id"], str(exc)
                    )
                    logger.warning(
                        "Could not index conversation %s; it will be retried.",
                        job["conversation_id"],
                        exc_info=True,
                    )


NoteEmbeddingWorker = EmbeddingWorker
