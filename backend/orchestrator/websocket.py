"""WebSocket endpoint for chat messages, with background generation.

How it works (the important part):

* A generation (one service reply) is an asyncio task owned by the
  `GenerationHub`, NOT by the WebSocket that started it.
* The hub is keyed by (user_id, conversation_id), so every conversation has
  its own generation and its own status. Different conversations can generate
  at the same time.
* WebSockets are only "viewers": they subscribe to their user's events.
  Closing a socket (switching chat, refreshing the page, a network drop) just
  unsubscribes it. The task keeps consuming the service stream.
* Every event carries conversation_id and request_id so the browser can route
  it to the right conversation and ignore stale ones.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import grpc
from fastapi import WebSocket, WebSocketDisconnect

from backend.generated import chat_pb2
from backend.orchestrator.rate_limit import message_limit_retry_after
from shared_common.common.log import trace_context

log = logging.getLogger("orchestrator")
_GENERIC_CHAT_ERROR = "Something went wrong. Please try again."
_SAVE_PARTIAL_TIMEOUT_SECONDS = 5


def _friendly_grpc_error(code: grpc.StatusCode) -> str:
    if code == grpc.StatusCode.UNAVAILABLE:
        return "The assistant is temporarily unavailable. Please try again."
    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
        return "The assistant is busy right now. Please try again shortly."
    return _GENERIC_CHAT_ERROR


def bad_id(value) -> bool:
    """True unless `value` is a positive integer (bools are not ids)."""
    return isinstance(value, bool) or not isinstance(value, int) or value < 1


class Connection:
    """One open WebSocket. Sends are serialised so events keep their order."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._lock = asyncio.Lock()  # FIFO: first caller sends first
        self.alive = True

    async def send(self, data: dict) -> bool:
        if not self.alive:
            return False
        async with self._lock:
            try:
                await self.websocket.send_json(data)
                return True
            except Exception:  # a dead viewer must never break a generation
                self.alive = False
                return False


@dataclass
class Generation:
    """Everything the server knows about one in-flight reply."""

    user_id: int
    conversation_id: int
    request_id: str
    mode: str  # "message" | "edit" | "regenerate"
    prompt: str  # what the user asked (normal messages are saved only at the end)
    assistant_message_id: int | None = None  # set when regenerating
    chunks: list[str] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    task: asyncio.Task | None = None

    @property
    def text(self) -> str:
        return "".join(self.chunks)


class GenerationHub:
    """Registry of running generations and of connected viewers."""

    def __init__(self):
        # stores users generations and connections
        self._generations: dict[tuple[int, int], Generation] = {}
        self._connections: dict[int, set[Connection]] = {}

    def reset(self) -> None:
        """Forget everything (used by tests)."""
        self._generations.clear()
        self._connections.clear()

    # ---- viewers ---------------------------------------------------- #
    def add_connection(self, user_id: int, conn: Connection) -> None:
        self._connections.setdefault(user_id, set()).add(conn)

    def remove_connection(self, user_id: int, conn: Connection) -> None:
        conns = self._connections.get(user_id)
        if conns is not None:
            conns.discard(conn)
            if not conns:
                del self._connections[user_id]

    async def broadcast(self, user_id: int, event: dict) -> None:
        conns = list(self._connections.get(user_id, ()))
        if conns:
            await asyncio.gather(*(conn.send(event) for conn in conns))

    # ---- generations ------------------------------------------------ #
    def get(self, user_id: int, conversation_id: int) -> Generation | None:
        return self._generations.get((user_id, conversation_id))

    def active_event(self, user_id: int) -> dict:
        """Snapshot of this user's running generations, for (re)connecting."""
        return {
            "type": "active",
            "generations": [
                {
                    "conversation_id": gen.conversation_id,
                    "request_id": gen.request_id,
                    "mode": gen.mode,
                    "prompt": gen.prompt,
                    "assistant_message_id": gen.assistant_message_id,
                    "content": gen.text,
                    "tool_results": gen.tool_results,
                }
                for (uid, _), gen in self._generations.items()
                if uid == user_id
            ],
        }

    def start(self, gen: Generation, coroutine_factory) -> None:
        """Register `gen` and run it as a task that outlives any socket."""
        self._generations[(gen.user_id, gen.conversation_id)] = gen
        gen.task = asyncio.create_task(coroutine_factory(gen))
        gen.task.add_done_callback(lambda task: self._on_done(gen, task))

    def finish(self, gen: Generation) -> None:
        """Remove `gen` from the registry (safe to call twice)."""
        key = (gen.user_id, gen.conversation_id)
        if self._generations.get(key) is gen:
            del self._generations[key]

    def cancel(
        self, user_id: int, request_id: object, conversation_id: object = None
    ) -> bool:
        for (uid, conv_id), gen in list(self._generations.items()):
            if uid != user_id or gen.request_id != request_id:
                continue
            if conversation_id is not None and conversation_id != conv_id:
                continue
            if gen.task is not None and not gen.task.done():
                gen.task.cancel()
                return True
        return False

    def _on_done(self, gen: Generation, task: asyncio.Task) -> None:
        self.finish(gen)  # safety net: never leave a "busy" ghost behind
        if not task.cancelled() and task.exception() is not None:
            log.error("Message task crashed", exc_info=task.exception())


# One hub for the whole process. (With several worker processes each would
# have its own hub; run a single uvicorn worker, or move this state to Redis.)
hub = GenerationHub()


async def _save_stopped_reply(gen: Generation, stub) -> None:
    """Save what a stopped reply had produced, before the browser reloads.

    The service saves it too, but only once its blocking model read returns,
    which can be after the browser has already reloaded. The shared request_id
    makes the second save a no-op. A stopped edit or regeneration saves nothing,
    so the earlier messages stay.
    """
    if not gen.text or gen.mode != "message":
        return
    try:
        await stub.SaveConversation(
            chat_pb2.SaveConversationRequest(
                conversation_id=gen.conversation_id,
                user_content=gen.prompt,
                assistant_content=gen.text,
                request_id=gen.request_id,
            ),
            timeout=_SAVE_PARTIAL_TIMEOUT_SECONDS,
        )
    except Exception:
        log.warning("Could not save the stopped reply", exc_info=True)


async def _run_generation(
    gen: Generation,
    stub,
    message: str,
    *,
    message_id: int | None = None,
) -> None:
    with trace_context():
        log.info("Starting chat generation")
        await _run_generation_with_trace(
            gen,
            stub,
            message,
            message_id=message_id,
        )


async def _run_generation_with_trace(
    gen: Generation,
    stub,
    message: str,
    *,
    message_id: int | None = None,
) -> None:
    """Bridge the service's gRPC stream to the existing WebSocket events."""
    conversation_id = gen.conversation_id
    request_id = gen.request_id

    async def emit(event_type: str, **fields) -> None:
        await hub.broadcast(gen.user_id, {
            "type": event_type,
            "conversation_id": conversation_id,
            "request_id": request_id,
            **fields,
        })

    async def emit_text(text: str) -> None:
        offset = len(gen.text)
        gen.chunks.append(text)
        await emit("chunk", content=text, offset=offset)

    async def emit_tool_result(tool: str, result: dict) -> None:
        index = len(gen.tool_results)
        tool_result = {"tool": tool, "result": result}
        gen.tool_results.append(tool_result)
        await emit("tool_result", index=index, **tool_result)

    terminal: tuple[str, dict] | None = None
    cancelled = False
    try:
        await emit("start", role="assistant")
        stream = stub.StreamMessage(chat_pb2.ChatTurnRequest(
            conversation_id=conversation_id,
            message=message,
            request_id=request_id,
            mode=gen.mode,
            message_id=message_id or 0,
            assistant_message_id=gen.assistant_message_id or 0,
        ))
        saw_terminal = False
        try:
            async for event in stream:
                if saw_terminal:
                    raise ValueError("Chat stream returned events after termination.")
                payload = event.WhichOneof("payload")
                if payload == "started":
                    continue
                elif payload == "chunk":
                    await emit_text(event.chunk.content)
                elif payload == "tool_result":
                    result = json.loads(event.tool_result.result_json)
                    await emit_tool_result(event.tool_result.tool, result)
                elif payload == "completed":
                    full_text = event.completed.full_text
                    if not full_text:
                        raise ValueError("Gemini returned an empty response.")
                    if not full_text.startswith(gen.text):
                        raise ValueError(
                            "Completed response did not match streamed text."
                        )
                    suffix = full_text[len(gen.text):]
                    if suffix:
                        await emit_text(suffix)
                    terminal = ("complete", {})
                    saw_terminal = True
                elif payload == "cancelled":
                    terminal = ("cancelled", {})
                    saw_terminal = True
                elif payload == "error":
                    log.error(
                        "ChatService reported an error for request %s",
                        request_id,
                    )
                    terminal = ("error", {"error": _GENERIC_CHAT_ERROR})
                    saw_terminal = True
                else:
                    raise ValueError("Chat stream returned an unknown event.")
        except asyncio.CancelledError:
            stream.cancel()
            raise
        except grpc.aio.AioRpcError as exc:
            log.error("gRPC stream failed: %s", exc.code())
            terminal = ("error", {"error": _friendly_grpc_error(exc.code())})
            saw_terminal = True

        if not saw_terminal:
            raise ValueError("Chat stream ended without a terminal event.")

    except asyncio.CancelledError:
        cancelled = True
        terminal = ("cancelled", {})
        await _save_stopped_reply(gen, stub)

    except Exception:
        log.exception("Failed to process chat message")
        terminal = ("error", {"error": _GENERIC_CHAT_ERROR})

    finally:
        # Free the conversation first, so a client that reacts to the final
        # event by sending again is not told "still answering".
        hub.finish(gen)
        if terminal is not None:
            with contextlib.suppress(Exception):
                await emit(terminal[0], **terminal[1])

    if cancelled:
        raise asyncio.CancelledError


async def websocket_endpoint(
    websocket: WebSocket,
    stub,
    *,
    user_id: int,
    revalidate_session: Callable[[], Awaitable[int | None]] | None = None,
) -> None:
    await websocket.accept()
    log.info("WebSocket connected")
    conn = Connection(websocket)
    hub.add_connection(user_id, conn)

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
            kind = payload.get("type") if isinstance(payload, dict) else None

            # Heartbeat: the client asks "are you alive?", we answer at once.
            # Handled before the rate limit, so a throttled user stays connected.
            if kind == "ping":
                await conn.send({"type": "pong"})
                continue

            # "sync": the browser (re)connected and asks which generations are
            # still running, so it can show the right status per conversation.
            if kind == "sync":
                await conn.send(hub.active_event(user_id))
                continue

            # stop/cancel one generation of this user
            if kind == "cancel":
                hub.cancel(
                    user_id,
                    payload.get("request_id"),
                    payload.get("conversation_id"),
                )
                continue

            # Support structured JSON messages.
            if kind != "message":
                await conn.send({
                    "type": "error",
                    "request_id": None,
                    "error": "Invalid message payload.",
                })
                continue

            message = payload.get("content")
            request_id = payload.get("request_id")
            conversation_id = payload.get("conversation_id")
            mode = payload.get("mode", "message")
            message_id = payload.get("message_id")
            assistant_message_id = payload.get("assistant_message_id")

            if (
                not isinstance(message, str)
                or not isinstance(request_id, str)
                or bad_id(conversation_id)
                or mode not in {"message", "edit", "regenerate"}
                or (mode == "edit" and bad_id(message_id))
                or (mode == "regenerate" and bad_id(assistant_message_id))
            ):
                await conn.send({
                    "type": "error",
                    "request_id": request_id,
                    "error": "Invalid message payload.",
                })
                continue

            # One at a time PER CONVERSATION; other conversations are free.
            # Refused messages do not use up the user's quota.
            if hub.get(user_id, conversation_id) is not None:
                await conn.send({
                    "type": "error",
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                    "error": "Still answering the previous message.",
                })
                continue

            # Rate limit: only valid messages count, since only they cost money.
            retry_after = message_limit_retry_after(user_id)
            if retry_after is not None:
                await conn.send({
                    "type": "error",
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                    "error": f"Too many messages. Try again in {retry_after}s.",
                })
                continue

            gen = Generation(
                user_id=user_id,
                conversation_id=conversation_id,
                request_id=request_id,
                mode=mode,
                prompt=message,
                assistant_message_id=(
                    assistant_message_id if mode == "regenerate" else None
                ),
            )
            hub.start(
                gen,
                lambda g, message=message, message_id=message_id: _run_generation(
                    g, stub, message,
                    message_id=message_id,
                ),
            )

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    finally:
        # Only the viewer leaves. Running generations are NOT cancelled: they
        # finish, save to the database, and the browser re-syncs on return.
        conn.alive = False
        hub.remove_connection(user_id, conn)
