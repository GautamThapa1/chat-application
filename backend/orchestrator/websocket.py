
"""WebSocket endpoint for chat messages."""

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable

import grpc
from fastapi import WebSocket, WebSocketDisconnect
from google.genai import errors

from backend.generated import chat_pb2
from backend.orchestrator.gemini_service import stream_chat_response
from backend.orchestrator.intent_handler import (
    handle_function_calls,
    save_conversation,
)
from backend.orchestrator.rate_limit import message_limit_retry_after

log = logging.getLogger("orchestrator")


async def _process_message(
    websocket: WebSocket,
    stub,
    gemini_client,
    model: str,
    message: str,
    request_id: str,
    conversation_id: int,
    *,
    mode: str = "message",
    message_id: int | None = None,
    assistant_message_id: int | None = None,
) -> None:
    """Answer one chat message while the receive loop remains responsive."""
    response_chunks: list[str] = []
    try:
        history = []
        replace_assistant_id = None

        if mode == "edit" and message_id is not None:
            # Editing a branch updates the original row and deletes every later
            # message before Gemini sees the edited prompt.
            edited = await stub.EditConversationMessage(
                chat_pb2.EditConversationMessageRequest(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    content=message,
                    request_id=request_id,
                )
            )
            message = edited.user_content
            conversation = await stub.GetConversation(
                chat_pb2.GetConversationRequest(conversation_id=conversation_id)
            )
            history = [
                {"role": item.role, "content": item.content}
                for item in conversation.messages
                if item.message_id != message_id
            ]
        elif mode == "regenerate" and assistant_message_id is not None:
            context = await stub.RegenerateConversationMessage(
                chat_pb2.RegenerateConversationMessageRequest(
                    conversation_id=conversation_id,
                    assistant_message_id=assistant_message_id,
                    request_id=request_id,
                )
            )
            message = context.user_content
            history = [
                {"role": item.role, "content": item.content}
                for item in context.history
            ]
            replace_assistant_id = assistant_message_id
        else:
            conversation = await stub.GetConversation(
                chat_pb2.GetConversationRequest(conversation_id=conversation_id)
            )
            history = [
                {"role": item.role, "content": item.content}
                for item in conversation.messages
            ]
        # this sends to gemini
        await websocket.send_json({
            "type": "start",
            "request_id": request_id,
            "role": "assistant",
        })
        tool_response = await handle_function_calls(
            message,
            gemini_client,
            model,
            history,
            stub,
        )

        if tool_response is None:
            async for chunk in stream_chat_response(
                message, gemini_client, model, history,
            ):
                response_chunks.append(chunk)
                await websocket.send_json({
                    "type": "chunk",
                    "request_id": request_id,
                    "content": chunk,
                })

            response = "".join(response_chunks)
            if not response:
                raise ValueError("Gemini returned an empty response.")
        else:
            response = tool_response
            response_chunks.append(response)
            await websocket.send_json({
                "type": "chunk",
                "request_id": request_id,
                "content": response,
            })

        if replace_assistant_id is not None:
            replaced = await stub.ReplaceConversationMessage(
                chat_pb2.ReplaceConversationMessageRequest(
                    conversation_id=conversation_id,
                    message_id=replace_assistant_id,
                    content=response,
                    request_id=request_id,
                )
            )
            if not replaced.success:
                raise ValueError("Could not replace the assistant response.")
        else:
            if mode == "edit":
                await save_conversation(
                    message,
                    response,
                    stub,
                    conversation_id,
                    request_id,
                )
            else:
                await save_conversation(
                    message,
                    response,
                    stub,
                    conversation_id,
                    request_id,
                )
        await websocket.send_json({
            "type": "complete",
            "request_id": request_id,
        })

    except asyncio.CancelledError:
        partial = "".join(response_chunks)
        if partial:
            try:
                await save_conversation(
                    message,
                    partial,
                    stub,
                    conversation_id,
                    request_id,
                )
            except grpc.aio.AioRpcError:
                log.exception("Could not save the partial reply")
        with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError):
            await websocket.send_json({
                "type": "cancelled",
                "request_id": request_id,
            })
        raise

    except (errors.APIError, grpc.aio.AioRpcError, ValueError) as exc:
        log.exception("Failed to process chat message")
        await websocket.send_json({
            "type": "error",
            "request_id": request_id,
            "error": f"Message failed: {exc}",
        })


def _log_task_failure(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("Message task crashed", exc_info=task.exception())


async def websocket_endpoint(
    websocket: WebSocket,
    stub,
    gemini_client,
    model: str,
    *,
    user_id: int,
    revalidate_session: Callable[[], Awaitable[int | None]] | None = None,
) -> None:
    await websocket.accept()
    log.info("WebSocket connected")
    current: asyncio.Task | None = None
    current_id: str | None = None

    try:
        while True:
            raw_message = await websocket.receive_text()

            # re-validate session on every message so expired websocket doesn't let it through
            if revalidate_session is not None:
                try:
                    current_user_id = await revalidate_session()
                except Exception:
                    log.exception("Session revalidation failed")
                    current_user_id = None
                if current_user_id is None or current_user_id != user_id:
                    log.warning("WebSocket session is invalid; closing connection")
                    await websocket.close(code=1008)
                    return

            # Parse the incoming message.
            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                payload = {}

            # Heartbeat: the client asks "are you alive?", we answer at once.
            # Handled before the rate limit, so a throttled user stays connected.
            if isinstance(payload, dict) and payload.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # stop/cancel 
            busy = current is not None and not current.done()
            if isinstance(payload, dict) and payload.get("type") == "cancel":
                if busy and payload.get("request_id") == current_id:
                    current.cancel()
                continue

            # Support structured JSON and plain-text messages.
            if isinstance(payload, dict) and payload.get("type") == "message":
                message = payload.get("content")
                request_id = payload.get("request_id")
                conversation_id = payload.get("conversation_id")
                mode = payload.get("mode", "message")
                message_id = payload.get("message_id")
                assistant_message_id = payload.get("assistant_message_id")

                if (
                    not isinstance(message, str)
                    or not isinstance(request_id, str)
                    or isinstance(conversation_id, bool)
                    or not isinstance(conversation_id, int)
                    or conversation_id < 1
                    or mode not in {"message", "edit", "regenerate"}
                ):
                    await websocket.send_json({
                        "type": "error",
                        "request_id": request_id,
                        "error": "Invalid message payload.",
                    })
                    continue
                if mode == "edit" and (
                    isinstance(message_id, bool)
                    or not isinstance(message_id, int)
                    or message_id < 1
                ):
                    await websocket.send_json({
                        "type": "error",
                        "request_id": request_id,
                        "error": "Invalid message payload.",
                    })
                    continue
                if mode == "regenerate" and (
                    isinstance(assistant_message_id, bool)
                    or not isinstance(assistant_message_id, int)
                    or assistant_message_id < 1
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

            # One at a time; refused messages do not use up the user's quota.
            if busy:
                await websocket.send_json({
                    "type": "error",
                    "request_id": request_id,
                    "error": "Still answering the previous message.",
                })
                continue

            # Rate limit: only valid messages count, since only they cost money.
            retry_after = message_limit_retry_after(user_id)
            if retry_after is not None:
                await websocket.send_json({
                    "type": "error",
                    "request_id": request_id,
                    "error": f"Too many messages. Try again in {retry_after}s.",
                })
                continue

            current_id = request_id
            current = asyncio.create_task(
                _process_message(
                    websocket, stub, gemini_client, model,
                    message, request_id, conversation_id,
                    mode=mode,
                    message_id=message_id,
                    assistant_message_id=assistant_message_id,
                )
            )
            current.add_done_callback(_log_task_failure)

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    finally:
        if current is not None and not current.done():
            current.cancel()
        if current is not None:
            await asyncio.gather(current, return_exceptions=True)

