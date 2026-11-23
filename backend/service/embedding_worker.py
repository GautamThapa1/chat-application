"""Background generation and reconciliation of note embeddings."""

import logging
import os
import threading

from google import genai

from backend.embeddings import (
    DOCUMENT_TASK_TYPE,
    embedding_model,
    embed_text,
    note_content_hash,
    note_embedding_text,
)
from backend.service import db

logger = logging.getLogger(__name__)


class NoteEmbeddingWorker:
    def __init__(self, *, batch_size: int = 20, poll_interval: float = 5):
        self._batch_size = batch_size
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "Note embedding worker is disabled; set GEMINI_API_KEY to index notes."
            )
            return
        self._thread = threading.Thread(
            target=self._run,
            args=(api_key,),
            name="note-embedding-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self, api_key: str) -> None:
        try:
            with genai.Client(api_key=api_key) as client:
                self._process_jobs(client)
        except Exception:
            logger.exception("Note embedding worker stopped unexpectedly.")

    def _process_jobs(self, client) -> None:
        while not self._stop_event.is_set():
            jobs = db.get_pending_note_embeddings(self._batch_size)
            if not jobs:
                self._stop_event.wait(self._poll_interval)
                continue
            for job in jobs:
                if self._stop_event.is_set():
                    break
                model = embedding_model()
                content_hash = note_content_hash(job["title"], job["content"])
                try:
                    # During migration, reuse a matching legacy SQLite vector
                    # else use Qdrant
                    # instead of spending an embedding API call unnecessarily.
                    legacy = db.get_legacy_note_embedding(job["id"])
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
                    indexed = db.upsert_note_embedding(
                        job["user_id"],
                        job["id"],
                        vector,
                        model,
                        content_hash,
                    )
                    if not indexed:
                        logger.info(
                            "Skipped stale embedding result for note %s.",
                            job["id"],
                        )
                except Exception as exc:
                    db.record_note_embedding_failure(job["id"], str(exc))
                    logger.warning(
                        "Could not index note %s; it will be retried.",
                        job["id"],
                        exc_info=True,
                    )
