"""WebSocket endpoint for chat messages."""

import json
import logging

import grpc
from google.genai import errors
from fastapi import WebSocket, WebSocketDisconnect

from backend.orchestrator.intent_handler import handle_message
from backend.orchestrator.response_generator import generate_response_chunks

log = logging.getLogger("orchestrator")


async def websocket_endpoint(
    websocket: WebSocket,
    stub,
    gemini_client,
    model: str,
) -> None:
    await websocket.accept()
    log.info("WebSocket connected")

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                payload = {}

            if isinstance(payload, dict) and payload.get("type") == "message":
                message = payload.get("content")
                request_id = payload.get("request_id")
                if not isinstance(message, str) or not isinstance(request_id, str):
                    await websocket.send_json({
                        "type": "error",
                        "request_id": request_id,
                        "error": "Invalid message payload.",
                    })
                    continue
            else:
                message = raw_message
                request_id = None

            try:
                response = await handle_message(message, stub, gemini_client, model)
            except (errors.APIError, grpc.aio.AioRpcError, ValueError) as exc:
                log.exception("Failed to process chat message")
                await websocket.send_json({
                    "type": "error",
                    "request_id": request_id,
                    "error": f"Message failed: {exc}",
                })
                continue
            await websocket.send_json({
                "type": "start",
                "request_id": request_id,
                "role": "assistant",
            })
            async for chunk in generate_response_chunks(response):
                await websocket.send_json({
                    "type": "chunk",
                    "request_id": request_id,
                    "content": chunk,
                })
            await websocket.send_json({
                "type": "complete",
                "request_id": request_id,
            })
    except WebSocketDisconnect: # send response to the user
        log.info("WebSocket disconnected")

# ochestrator was accessing the database directly here