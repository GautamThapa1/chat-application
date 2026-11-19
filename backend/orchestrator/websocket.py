
"""WebSocket endpoint for chat messages."""

import json
import logging

import grpc
from fastapi import WebSocket, WebSocketDisconnect
from google.genai import errors

from backend.generated import chat_pb2
from backend.orchestrator.gemini_service import stream_chat_response
from backend.orchestrator.intent_handler import (
    classify_message,
    execute_action,
    save_conversation,
)

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

            # Parse the incoming message.
            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                payload = {}

            # Support structured JSON and plain-text messages.
            if isinstance(payload, dict) and payload.get("type") == "message":
                message = payload.get("content")
                request_id = payload.get("request_id")
                conversation_id = payload.get("conversation_id")

                if (
                    not isinstance(message, str)
                    or not isinstance(request_id, str)
                    or isinstance(conversation_id, bool)
                    or not isinstance(conversation_id, int)
                    or conversation_id < 1
                ):
                    await websocket.send_json({
                        "type": "error",
                        "request_id": request_id,
                        "error": "Invalid message payload.",
                    })
                    continue
            else:
                await websocket.send_json({
                    "type": "error",
                    "request_id": None,
                    "error": "Invalid message payload.",
                })
                continue

            try:
                # The gRPC service filters by the authenticated user and returns
                # NOT_FOUND for both missing and foreign conversation IDs.
                conversation = await stub.GetConversation(
                    chat_pb2.GetConversationRequest(
                        conversation_id=conversation_id,
                    )
                )
                history = [
                    {"role": item.role, "content": item.content}
                    for item in conversation.messages
                ]

                # Step 1: Classify the user's message.
                decision = await classify_message(
                    message,
                    gemini_client,
                    model,
                )

                # Step 2: Handle general chat with true Gemini streaming.
                if decision["intent"] == "CHAT":
                    await websocket.send_json({
                        "type": "start",
                        "request_id": request_id,
                        "role": "assistant",
                    })

                    response_chunks = []

                    async for chunk in stream_chat_response(
                        message,
                        gemini_client,
                        model,
                        history,
                    ):
                        response_chunks.append(chunk)

                        # Immediately forward each Gemini chunk.
                        await websocket.send_json({
                            "type": "chunk",
                            "request_id": request_id,
                            "content": chunk,
                        })

                    response = "".join(response_chunks)

                    if not response:
                        raise ValueError("Gemini returned an empty response.")

                # Step 3: Handle actions through gRPC.
                else:
                    response = await execute_action(decision, stub)

                    await websocket.send_json({
                        "type": "start",
                        "request_id": request_id,
                        "role": "assistant",
                    })

                    # Action results are already complete, so send them once.
                    await websocket.send_json({
                        "type": "chunk",
                        "request_id": request_id,
                        "content": response,
                    })

                # Step 4: Save the completed exchange through gRPC.
                await save_conversation(
                    message, response, stub, conversation_id
                )

                # Step 5: Tell the frontend the response is complete.
                await websocket.send_json({
                    "type": "complete",
                    "request_id": request_id,
                })

            except (errors.APIError, grpc.aio.AioRpcError, ValueError) as exc:
                log.exception("Failed to process chat message")

                await websocket.send_json({
                    "type": "error",
                    "request_id": request_id,
                    "error": f"Message failed: {exc}",
                })

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
