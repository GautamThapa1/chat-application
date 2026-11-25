"""WebSocket endpoint for chat messages, with background generation.

How it works (the important part):

* A generation (one Gemini reply) is an asyncio task owned by the
  `GenerationHub`, NOT by the WebSocket that started it.
* The hub is keyed by (user_id, conversation_id), so every conversation has
  its own generation and its own status. Different conversations can generate
  at the same time.
* WebSockets are only "viewers": they subscribe to their user's events.
  Closing a socket (switching chat, refreshing the page, a network drop) just
  unsubscribes it. The task keeps running and saves its answer to the database.
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
from google.genai import errors

from backend.generated import chat_pb2
from backend.orchestrator.intent_handler import (
    handle_function_calls,
    save_conversation,
)
from backend.orchestrator.rate_limit import message_limit_retry_after

log = logging.getLogger("orchestrator")


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


async def _run_generation(
    gen: Generation,
    stub,
    gemini_client,
    model: str,
    message: str,
    *,
    message_id: int | None = None,
) -> None:
    """Produce one reply, save it, and tell whoever is watching."""
    conversation_id = gen.conversation_id
    request_id = gen.request_id

    async def emit(event_type: str, **fields) -> None:
        await hub.broadcast(gen.user_id, {
            "type": event_type,
            "conversation_id": conversation_id,
            "request_id": request_id,
            **fields,
        })

    async def emit_tool_result(tool: str, result: dict) -> None:
        tool_result = {"tool": tool, "result": result}
        index = len(gen.tool_results)
        gen.tool_results.append(tool_result)
        await emit("tool_result", index=index, **tool_result)

    async def emit_text(text: str) -> None:
        offset = len(gen.text)
        gen.chunks.append(text)
        await emit("chunk", content=text, offset=offset)

    terminal: tuple[str, dict] | None = None
    cancelled = False
    try:
        history = []
        replace_assistant_id = None

        if gen.mode == "edit" and message_id is not None:
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
        elif gen.mode == "regenerate" and gen.assistant_message_id is not None:
            context = await stub.RegenerateConversationMessage(
                chat_pb2.RegenerateConversationMessageRequest(
                    conversation_id=conversation_id,
                    assistant_message_id=gen.assistant_message_id,
                    request_id=request_id,
                )
            )
            message = context.user_content
            history = [
                {"role": item.role, "content": item.content}
                for item in context.history
            ]
            replace_assistant_id = gen.assistant_message_id
        else:
            conversation = await stub.GetConversation(
                chat_pb2.GetConversationRequest(conversation_id=conversation_id)
            )
            history = [
                {"role": item.role, "content": item.content}
                for item in conversation.messages
            ]

        await emit("start", role="assistant")
        tool_response = await handle_function_calls(
            message,
            gemini_client,
            model,
            history,
            stub,
            on_tool_result=emit_tool_result,
            on_text=emit_text,
        )

        if tool_response is not None:
            await emit_text(("\n\n" if gen.text else "") + tool_response)

        response = gen.text
        if not response:
            raise ValueError("Gemini returned an empty response.")

        # Save BEFORE announcing "complete": the database is the source of
        # truth, and it is written even if nobody is watching any more.
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
            # For "edit" the user row already exists; the database layer
            # notices that and only appends the assistant reply.
            await save_conversation(
                message, response, stub, conversation_id, request_id,
            )
        terminal = ("complete", {})

    except asyncio.CancelledError:
        cancelled = True
        partial = gen.text
        # Keep what was streamed so far, except when regenerating: there the
        # old answer is still in the database and must not be duplicated.
        if partial and gen.mode != "regenerate":
            try:
                await save_conversation(
                    message, partial, stub, conversation_id, request_id,
                )
            except grpc.aio.AioRpcError:
                log.exception("Could not save the partial reply")
        terminal = ("cancelled", {})

    except Exception as exc:  # noqa: BLE001 - a background task must report, not vanish
        if isinstance(exc, (errors.APIError, grpc.aio.AioRpcError, ValueError)):
            log.exception("Failed to process chat message")
        else:
            log.exception("Unexpected error while generating a reply")
        terminal = ("error", {"error": f"Message failed: {exc}"})

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
    gemini_client,
    model: str,
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
                    g, stub, gemini_client, model, message,
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
