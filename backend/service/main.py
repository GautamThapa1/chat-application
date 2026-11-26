"""Entry point: start the gRPC server."""

import logging
import os
import signal
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.service import db_backup
from backend.service.chat_service import create_server
from backend.service.db import init_db
from backend.service.embedding_worker import EmbeddingWorker
from backend.service.task_reminder_worker import TaskReminderWorker


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    from shared_common.common.log import setup_logging

    setup_logging("chat-service")
    # Optional: restore/keep the DB in object storage (hosts without a disk).
    backup = db_backup.from_env()
    if backup is not None:
        backup.restore_and_claim()
    init_db()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before starting ChatService.")
    retry_options = types.HttpRetryOptions(
        attempts=3,
        initial_delay=1,
        max_delay=5,
        http_status_codes=[503],
    )
    http_options = types.HttpOptions(retry_options=retry_options)

    with genai.Client(api_key=api_key, http_options=http_options) as gemini_client:
        server = create_server(
            port=50051,
            gemini_client=gemini_client,
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        )
        server.start()
        # Hosting platforms stop containers with SIGTERM: shut down cleanly so
        # the workers stop and the final database backup runs.
        def _on_sigterm(*_):
            logging.info("SIGTERM received; shutting down...")
            server.stop(grace=5)

        signal.signal(signal.SIGTERM, _on_sigterm)
        embedding_worker = EmbeddingWorker()
        task_reminder_worker = TaskReminderWorker()
        embedding_worker.start()
        task_reminder_worker.start()
        if backup is not None:
            backup.start()
        logging.info("ChatService listening on :50051")

        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        finally:
            task_reminder_worker.stop()
            embedding_worker.stop()
            server.stop(grace=1)
            if backup is not None:
                backup.stop()  # final snapshot after all writers stopped
            logging.info("ChatService stopped.")

if __name__== "__main__":
    main()
