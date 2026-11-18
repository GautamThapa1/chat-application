"""WebSocket endpoint for chat messages."""

import logging

from fastapi import WebSocket, WebSocketDisconnect

from backend.orchestrator.intent_handler import handle_message
from backend.service.db import save_message

log = logging.getLogger("orchestrator")


async def websocket_endpoint(websocket: WebSocket, stub) -> None:
    await websocket.accept()
    log.info("WebSocket connected")

    try:
        while True:
            message = await websocket.receive_text()
            save_message("user", message)
            response = await handle_message(message, stub)
            save_message("assistant", response)
            await websocket.send_json({"role": "assistant", "content": response})
    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
