"""gRPC servicer for ChatService — each RPC delegates to db.py."""

import json
import logging
import os
import threading
from concurrent import futures
from contextlib import closing

import grpc

from backend.generated import chat_pb2, chat_pb2_grpc
from backend.service import db, memory_store
from backend.service.gemini import runner as gemini_chat
from backend.service.auth_interceptor import AuthInterceptor, current_user_id

logger = logging.getLogger(__name__)


class _ConversationMessageNotFound(LookupError):
    pass


_CHAT_STREAM_ERROR_MESSAGE = "Something went wrong. Please try again."


def _load_history_for_mode(
    request,
    user_id: int,
    mode: str,
) -> tuple[str, list[dict[str, str]], int | None]:
    if db.get_conversation(user_id, request.conversation_id) is None:
        raise db.ConversationNotFound

    if mode == "edit":
        success, message = db.edit_conversation_message(
            user_id,
            request.conversation_id,
            request.message_id,
            request.message,
        )
        if not success:
            raise _ConversationMessageNotFound("Message not found.")
        history = [
            {"role": row["role"], "content": row["content"]}
            for row in db.get_conversation_messages(
                user_id, request.conversation_id
            )
            if row["id"] != request.message_id
        ]
        return message, history, None

    if mode == "regenerate":
        success, message, history = db.get_regeneration_context(
            user_id,
            request.conversation_id,
            request.assistant_message_id,
        )
        if not success:
            raise _ConversationMessageNotFound("Message not found.")
        return message, history, request.assistant_message_id

    history = [
        {"role": row["role"], "content": row["content"]}
        for row in db.get_conversation_messages(user_id, request.conversation_id)
    ]
    return request.message, history, None


class ChatServicer(chat_pb2_grpc.ChatServiceServicer):
    def __init__(
        self,
        gemini_client=None,
        model: str | None = None,
        max_concurrent_streams: int = 40,
    ):
        if max_concurrent_streams < 1:
            raise ValueError("max_concurrent_streams must be positive.")
        self._gemini_client = gemini_client
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self._stream_slots = threading.BoundedSemaphore(max_concurrent_streams)

    def SendMessage(self, request, context):
        user_id = current_user_id()
        if request.role not in {"", "user"}:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Only user messages are accepted.")
        reply = f"Echo: {request.content}"
        try:
            db.save_conversation(
                user_id,
                request.conversation_id,
                request.content,
                reply,
            )
        except db.ConversationNotFound:
            context.abort(grpc.StatusCode.NOT_FOUND, "Conversation not found.")
        return chat_pb2.ChatResponse(role="assistant", content=reply)

    def StreamMessage(self, request, context):
        user_id = current_user_id()
        if not self._stream_slots.acquire(blocking=False):
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Too many active chat streams.",
            )
        try:
            yield from self._stream_message_events(request, context, user_id)
        finally:
            self._stream_slots.release()

    def _stream_message_events(self, request, context, user_id: int):
        cancelled = threading.Event()
        context.add_callback(cancelled.set)
        mode = request.mode or "message"

        if request.conversation_id < 1:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "conversation_id must be a positive integer.",
            )
        if mode not in {"message", "edit", "regenerate"}:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid message mode.")
        if mode == "edit" and request.message_id < 1:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "message_id must be a positive integer.",
            )
        if mode == "regenerate" and request.assistant_message_id < 1:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "assistant_message_id must be a positive integer.",
            )

        logger.info("StreamMessage started")

        try:
            message, history, replace_assistant_id = _load_history_for_mode(
                request, user_id, mode
            )
        except db.ConversationNotFound:
            context.abort(grpc.StatusCode.NOT_FOUND, "Conversation not found.")
        except _ConversationMessageNotFound as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception:
            logging.exception("StreamMessage failed before generation")
            yield chat_pb2.ChatEvent(
                request_id=request.request_id,
                error=chat_pb2.ChatError(message=_CHAT_STREAM_ERROR_MESSAGE),
            )
            return

        chunks = []
        offset = 0
        tool_result_index = 0
        try:
            yield chat_pb2.ChatEvent(
                request_id=request.request_id,
                started=chat_pb2.Started(),
            )
            if cancelled.is_set():
                raise GeneratorExit
            # this is runner.py as gemini_chat, where all tool calls happens
            turn = gemini_chat.run_turn(
                message,
                self._gemini_client,
                self._model,
                history,
                user_id,
                request.request_id,
            )
            with closing(turn):
                for kind, *payload in turn:
                    # The synchronous Gemini iterator can block inside next();
                    # cancellation is observed after that network read returns.
                    if cancelled.is_set():
                        raise GeneratorExit
                    if kind == "chunk":
                        text = payload[0]
                        chunks.append(text)
                        yield chat_pb2.ChatEvent(
                            request_id=request.request_id,
                            chunk=chat_pb2.TextChunk(
                                content=text,
                                offset=offset,
                            ),
                        )
                        offset += len(text)
                    elif kind == "tool_result":
                        tool, result = payload
                        yield chat_pb2.ChatEvent(
                            request_id=request.request_id,
                            tool_result=chat_pb2.ToolResult(
                                index=tool_result_index,
                                tool=tool,
                                result_json=json.dumps(result),
                            ),
                        )
                        tool_result_index += 1
                    elif kind == "final":
                        chunks.append(payload[0])
                    else:
                        raise ValueError(f"Unknown Gemini turn event: {kind!r}.")

            full_text = "".join(chunks)
            if not full_text:
                raise ValueError("Gemini returned an empty response.")

            if replace_assistant_id is not None:
                replaced = db.replace_assistant_response(
                    user_id,
                    request.conversation_id,
                    replace_assistant_id,
                    full_text,
                    request.request_id,
                )
                if not replaced:
                    raise ValueError("Could not replace the assistant response.")
            else:
                db.save_conversation(
                    user_id,
                    request.conversation_id,
                    message,
                    full_text,
                    request.request_id,
                )

            yield chat_pb2.ChatEvent(
                request_id=request.request_id,
                completed=chat_pb2.Completed(full_text=full_text),
            )
        except GeneratorExit:
            partial = "".join(chunks)
            if partial and mode != "regenerate":
                try:
                    db.save_conversation(
                        user_id,
                        request.conversation_id,
                        message,
                        partial,
                        request.request_id,
                    )
                except Exception:
                    logging.exception("Could not save the partial response")
            raise
        except Exception:
            logging.exception("StreamMessage failed")
            yield chat_pb2.ChatEvent(
                request_id=request.request_id,
                error=chat_pb2.ChatError(message=_CHAT_STREAM_ERROR_MESSAGE),
            )

    # ---------------------------------------------------------------- #
    # Auth
    # ---------------------------------------------------------------- #
    def Register(self, request, context):
        try:
            user_id = db.register_user(request.username, request.password)
        except db.UsernameTaken as exc:  # must come first: it is a ValueError too
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.RegisterResponse(user_id=user_id)

    def Login(self, request, context):
        user_id = db.authenticate(request.username, request.password)
        if user_id is None:
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "Invalid username or password."
            )
        token = db.create_session(user_id)
        return chat_pb2.LoginResponse(session_token=token, user_id=user_id)

    def Logout(self, request, context):
        db.delete_session(request.session_token)
        return chat_pb2.LogoutResponse()

    def ValidateSession(self, request, context):
        user_id = db.get_session_user(request.session_token)
        return chat_pb2.ValidateSessionResponse(
            valid=user_id is not None,
            user_id=user_id or 0,
            username=db.get_session_username(request.session_token) or "",
        )

    def SaveConversation(self, request, context):
        try:
            db.save_conversation(
                current_user_id(),
                request.conversation_id,
                request.user_content,
                request.assistant_content,
                request.request_id,
            )
        except db.ConversationNotFound:
            context.abort(grpc.StatusCode.NOT_FOUND, "Conversation not found.")
        return chat_pb2.SaveConversationResponse(success=True)

    def CreateConversation(self, request, context):
        conversation = db.create_conversation(current_user_id(), request.title)
        return chat_pb2.CreateConversationResponse(
            conversation=_conversation_message(conversation)
        )


    def ListConversations(self, request, context):
        user_id = current_user_id()

        # Use defaults when the caller omits pagination values.
        limit = request.limit or 20
        offset = request.offset

        try:
            conversations, total_count = db.list_conversations(
                user_id,
                limit=limit,
                offset=offset,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        return chat_pb2.ListConversationsResponse(
            conversations=[
                _conversation_message(row)
                for row in conversations
            ],
            total_count=total_count,
        )


    def GetConversation(self, request, context):
        user_id = current_user_id()
        conversation = db.get_conversation(user_id, request.conversation_id)
        if conversation is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "Conversation not found.")
        return chat_pb2.GetConversationResponse(
            conversation=_conversation_message(conversation),
            messages=[
                chat_pb2.ConversationMessage(
                    message_id=row["id"],
                    conversation_id=row["conversation_id"],
                    user_id=row["user_id"],
                    role=row["role"],
                    content=row["content"],
                    created_at=str(row["created_at"] or ""),
                )
                for row in db.get_conversation_messages(
                    user_id, request.conversation_id
                )
            ],
        )

    def DeleteConversation(self, request, context):
        deleted = db.delete_conversation(
            current_user_id(), request.conversation_id
        )
        if not deleted:
            context.abort(grpc.StatusCode.NOT_FOUND, "Conversation not found.")
        return chat_pb2.DeleteConversationResponse(success=True)

    def SearchConversations(self, request, context):
        user_id = current_user_id()
        try:
            limit = request.limit or 20
            if request.embedding and request.embedding_model:
                rows = db.search_conversations_hybrid(
                    user_id,
                    request.keyword,
                    list(request.embedding),
                    request.embedding_model,
                    limit=limit,
                )
            else:
                rows = db.search_conversations_keyword(
                    user_id,
                    request.keyword,
                    limit=limit,
                )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.SearchConversationsResponse(
            results=[
                chat_pb2.ConversationSearchResult(
                    conversation_id=row["conversation_id"],
                    title=row["title"],
                    excerpt=row["excerpt"],
                    updated_at=str(row["updated_at"] or ""),
                    score=row["score"],
                    message_start_id=row.get("message_start_id", 0),
                    message_end_id=row.get("message_end_id", 0),
                )
                for row in rows
            ]
        )

    def EditConversationMessage(self, request, context):
        try:
            success, content = db.edit_conversation_message(
                current_user_id(),
                request.conversation_id,
                request.message_id,
                request.content,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        if not success:
            context.abort(grpc.StatusCode.NOT_FOUND, "Message not found.")
        return chat_pb2.EditConversationMessageResponse(
            success=True,
            user_content=content,
        )

    def RegenerateConversationMessage(self, request, context):
        success, user_content, history = db.get_regeneration_context(
            current_user_id(),
            request.conversation_id,
            request.assistant_message_id,
        )
        if not success:
            context.abort(grpc.StatusCode.NOT_FOUND, "Message not found.")
        return chat_pb2.RegenerateConversationMessageResponse(
            success=True,
            user_content=user_content,
            history=[
                chat_pb2.ConversationMessage(
                    conversation_id=request.conversation_id,
                    user_id=current_user_id(),
                    role=row["role"],
                    content=row["content"],
                )
                for row in history
            ],
        )

    def ReplaceConversationMessage(self, request, context):
        success = db.replace_assistant_response(
            current_user_id(),
            request.conversation_id,
            request.message_id,
            request.content,
            request.request_id,
        )
        if not success:
            context.abort(grpc.StatusCode.NOT_FOUND, "Message not found.")
        return chat_pb2.ReplaceConversationMessageResponse(success=True)

    # ---------------------------------------------------------------- #
    # Notes  (db.save_note / db.search_notes)
    # ---------------------------------------------------------------- #
    def SaveNote(self, request, context):
        try:
            result = db.save_note_idempotent(
                current_user_id(),
                request.title,
                request.content,
                request.idempotency_key,
            )
        except db.IdempotencyConflict as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except db.IdempotencyInProgress as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.SaveNoteResponse(
            note=chat_pb2.Note(
                id=result["id"],
                title=result["title"],
                content=result["content"],
                created_at=result["created_at"],
            )
        )

    def UpdateNote(self, request, context):
        user_id = current_user_id()
        try:
            result = db.update_note_idempotent(
                user_id,
                request.note_id,
                request.title,
                request.content,
                request.idempotency_key,
            )
        except db.IdempotencyConflict as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except db.IdempotencyInProgress as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        if not result["found"]:
            context.abort(grpc.StatusCode.NOT_FOUND, "Note not found.")
        return chat_pb2.UpdateNoteResponse(
            note=chat_pb2.Note(
                id=result["id"],
                title=result["title"],
                content=result["content"],
                created_at=result["created_at"],
            )
        )

    def DeleteNote(self, request, context):
        return chat_pb2.DeleteNoteResponse(
            success=db.delete_note(current_user_id(), request.note_id)
        )

    def SearchNotes(self, request, context):
        rows = db.search_notes(current_user_id(), request.keyword)
        return chat_pb2.SearchNotesResponse(
            notes=[
                chat_pb2.Note(
                    id=r["id"],
                    title=r["title"],
                    content=r["content"],
                    created_at=str(r["created_at"] or ""),
                )
                for r in rows
            ]
        )

    def SearchNotesHybrid(self, request, context):
        try:
            rows = db.search_notes_hybrid(
                current_user_id(),
                request.keyword,
                list(request.embedding),
                request.embedding_model,
                limit=request.limit or 20,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.SearchNotesResponse(
            notes=[
                chat_pb2.Note(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"],
                    created_at=str(row["created_at"] or ""),
                )
                for row in rows
            ]
        )

    def UpsertNoteEmbedding(self, request, context):
        try:
            indexed = db.upsert_note_embedding(
                current_user_id(),
                request.note_id,
                list(request.embedding),
                request.embedding_model,
                request.content_hash,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.UpsertNoteEmbeddingResponse(indexed=indexed)

    def ClearNotes(self, request, context):
        try:
            result = db.clear_notes_idempotent(
                current_user_id(),
                request.idempotency_key,
            )
        except db.IdempotencyConflict as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except db.IdempotencyInProgress as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.ClearNotesResponse(
            deleted_count=result["deleted_count"]
        )

    # ---------------------------------------------------------------- #
    # Memories
    # ---------------------------------------------------------------- #
    def SaveMemory(self, request, context):
        try:
            result = memory_store.save_memory_idempotent(
                current_user_id(),
                request.title,
                request.content,
                request.idempotency_key,
            )
        except db.IdempotencyConflict as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except db.IdempotencyInProgress as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.SaveMemoryResponse(
            memory=chat_pb2.Memory(
                id=result["id"],
                title=result["title"],
                content=result["content"],
                created_at=str(result["created_at"] or ""),
            )
        )

    def UpdateMemory(self, request, context):
        try:
            result = memory_store.update_memory_idempotent(
                current_user_id(),
                request.memory_id,
                request.title,
                request.content,
                request.idempotency_key,
            )
        except db.IdempotencyConflict as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except db.IdempotencyInProgress as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        if not result["found"]:
            context.abort(grpc.StatusCode.NOT_FOUND, "Memory not found.")
        return chat_pb2.UpdateMemoryResponse(
            memory=chat_pb2.Memory(
                id=result["id"],
                title=result["title"],
                content=result["content"],
                created_at=str(result["created_at"] or ""),
            )
        )

    def DeleteMemory(self, request, context):
        return chat_pb2.DeleteMemoryResponse(
            success=memory_store.delete_memory(
                current_user_id(), request.memory_id
            )
        )

    def ListMemories(self, request, context):
        try:
            rows, total_count = memory_store.list_memories(
                current_user_id(),
                limit=request.limit or 20,
                offset=request.offset,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        # this passes memory to memory_store.py    
        return chat_pb2.ListMemoriesResponse(
            memories=[
                chat_pb2.Memory(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"],
                    created_at=str(row["created_at"] or ""),
                )
                for row in rows
            ],
            total_count=total_count,
        )

    def SearchMemories(self, request, context):
        try:
            rows = memory_store.search_memories(
                current_user_id(), request.keyword
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.SearchMemoriesResponse(
            memories=[
                chat_pb2.Memory(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"],
                    created_at=str(row["created_at"] or ""),
                )
                for row in rows
            ]
        )

    def SearchMemoriesHybrid(self, request, context):
        try:
            rows = memory_store.search_memories_hybrid(
                current_user_id(),
                request.keyword,
                list(request.embedding),
                request.embedding_model,
                limit=request.limit or 20,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.SearchMemoriesResponse(
            memories=[
                chat_pb2.Memory(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"],
                    created_at=str(row["created_at"] or ""),
                )
                for row in rows
            ]
        )

    def UpsertMemoryEmbedding(self, request, context):
        try:
            indexed = memory_store.upsert_memory_embedding(
                current_user_id(),
                request.memory_id,
                list(request.embedding),
                request.embedding_model,
                request.content_hash,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.UpsertMemoryEmbeddingResponse(indexed=indexed)

    # ---------------------------------------------------------------- #
    # Tasks  (db.create_task / db.get_tasks / db.complete_task)
    # ---------------------------------------------------------------- #
    def CreateTask(self, request, context):
        try:
            result = db.create_task_idempotent(
                current_user_id(),
                request.title,
                request.due_at_local,
                request.idempotency_key,
            )
        except db.IdempotencyConflict as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except db.IdempotencyInProgress as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.CreateTaskResponse(
            task=chat_pb2.Task(
                id=result["id"],
                title=result["title"],
                due_at_utc=result["due_at_utc"],
                timezone=result["timezone"],
                legacy_due_date=result["legacy_due_date"],
                completed=result["completed"],
                created_at=result["created_at"],
            )
        )

    def UpdateTask(self, request, context):
        try:
            result = db.update_task_deadline_idempotent(
                current_user_id(),
                request.task_id,
                request.due_at_local,
                request.idempotency_key,
            )
        except db.IdempotencyConflict as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except db.IdempotencyInProgress as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        if not result.get("found"):
            context.abort(grpc.StatusCode.NOT_FOUND, "Task not found.")
        return chat_pb2.UpdateTaskResponse(
            task=chat_pb2.Task(
                id=result["id"],
                title=result["title"],
                due_at_utc=result["due_at_utc"],
                timezone=result["timezone"],
                legacy_due_date=result["legacy_due_date"],
                completed=result["completed"],
                created_at=result["created_at"],
            )
        )

    def ListTasks(self, request, context):
        rows = db.get_tasks(current_user_id())
        return chat_pb2.ListTasksResponse(
            tasks=[
                chat_pb2.Task(
                    id=r["id"],
                    title=r["title"],
                    due_at_utc=r["due_at_utc"],
                    timezone=r["timezone"],
                    legacy_due_date=r["legacy_due_date"],
                    completed=bool(r["completed"]),
                    created_at=str(r["created_at"] or ""),
                )
                for r in rows
            ]
        )

    def CompleteTask(self, request, context):
        try:
            result = db.complete_task_idempotent(
                current_user_id(),
                request.task_id,
                request.idempotency_key,
            )
        except db.IdempotencyConflict as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except db.IdempotencyInProgress as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.CompleteTaskResponse(success=result["success"])

    def DeleteTask(self, request, context):
        try:
            result = db.delete_task_idempotent(
                current_user_id(),
                request.task_id,
                request.idempotency_key,
            )
        except db.IdempotencyConflict as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except db.IdempotencyInProgress as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.DeleteTaskResponse(success=result["success"])

    def ClearTasks(self, request, context):
        try:
            result = db.clear_tasks_idempotent(
                current_user_id(),
                request.idempotency_key,
            )
        except db.IdempotencyConflict as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except db.IdempotencyInProgress as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.ClearTasksResponse(deleted_count=result["deleted_count"])

    def SetUserTimezone(self, request, context):
        try:
            if not db.set_user_timezone(current_user_id(), request.timezone):
                context.abort(grpc.StatusCode.NOT_FOUND, "User not found.")
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.SetUserTimezoneResponse()

    def SavePushSubscription(self, request, context):
        if not request.endpoint or not request.p256dh or not request.auth:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Push endpoint and keys are required.",
            )
        try:
            db.save_push_subscription(
                current_user_id(),
                request.endpoint,
                request.p256dh,
                request.auth,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return chat_pb2.SavePushSubscriptionResponse()

    def DeletePushSubscription(self, request, context):
        success = db.delete_push_subscription(
            current_user_id(), request.endpoint
        )
        return chat_pb2.DeletePushSubscriptionResponse(success=success)


def _conversation_message(row: dict) -> chat_pb2.Conversation:
    return chat_pb2.Conversation(
        conversation_id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
    )


def create_server(
    port: int = 50051,
    host: str | None = None,
    gemini_client=None,
    model: str | None = None,
    max_workers: int = 50,
    max_concurrent_streams: int = 40,
) -> grpc.Server:
    # Loopback by default (safe for local runs). In Docker the orchestrator is a
    # different container, so compose sets GRPC_BIND=0.0.0.0 and the port stays
    # on the private compose network (it is never published to the host).
    host = host or os.getenv("GRPC_BIND", "127.0.0.1")
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=[AuthInterceptor()],
    )
    chat_pb2_grpc.add_ChatServiceServicer_to_server(
        ChatServicer(
            gemini_client=gemini_client,
            model=model,
            max_concurrent_streams=max_concurrent_streams,
        ),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    return server