"""Entry point: start the gRPC server."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.service.chat_service import create_server
from backend.service.db import init_db
from backend.service.embedding_worker import EmbeddingWorker
from backend.service.task_reminder_worker import TaskReminderWorker


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
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
        embedding_worker = EmbeddingWorker()
        task_reminder_worker = TaskReminderWorker()
        embedding_worker.start()
        task_reminder_worker.start()
        logging.info("ChatService listening on :50051")

        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        finally:
            task_reminder_worker.stop()
            embedding_worker.stop()
            server.stop(grace=1)

if __name__== "__main__":
    main()
