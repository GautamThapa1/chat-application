"""WebSocket endpoint for chat messages."""

import logging

from fastapi import WebSocket, WebSocketDisconnect

from backend.orchestrator.intent_handler import handle_message
from backend.orchestrator.response_generator import generate_response_chunks

log = logging.getLogger("orchestrator")


async def websocket_endpoint(websocket: WebSocket, stub) -> None:
    await websocket.accept()
    log.info("WebSocket connected")

    try:
        while True:
            message = await websocket.receive_text()
            response = await handle_message(message, stub) # my function
            await websocket.send_json({"type": "start", "role": "assistant"})
            async for chunk in generate_response_chunks(response):
                await websocket.send_json({"type": "chunk", "content": chunk})
            await websocket.send_json({"type": "complete"})
    except WebSocketDisconnect: # send response to the user
        log.info("WebSocket disconnected")

# ochestrator was accessing the database directly here