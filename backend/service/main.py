"""Entry point: start the gRPC server."""

import logging

from backend.service.chat_service import create_server
from backend.service.db import init_db
from backend.service.embedding_worker import EmbeddingWorker


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    init_db()

    server = create_server(port=50051)
    server.start()
    embedding_worker = EmbeddingWorker()
    embedding_worker.start()
    logging.info("ChatService listening on :50051")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        embedding_worker.stop()
        server.stop(grace=1)

if __name__== "__main__":
    main()
