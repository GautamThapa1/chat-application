"""Entry point: start the gRPC server."""

import logging

from backend.service.chat_service import create_server
from backend.service.db import init_db
from backend.service.embedding_worker import EmbeddingWorker
from backend.service.task_reminder_worker import TaskReminderWorker


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    init_db()

    server = create_server(port=50051)
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
