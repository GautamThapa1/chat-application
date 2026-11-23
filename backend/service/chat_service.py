"""gRPC servicer for ChatService — each RPC delegates to db.py."""

from concurrent import futures

import grpc

from backend.generated import chat_pb2, chat_pb2_grpc
from backend.service import db
from backend.service.auth_interceptor import AuthInterceptor, current_user_id


class ChatServicer(chat_pb2_grpc.ChatServiceServicer):

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
            valid=user_id is not None, user_id=user_id or 0
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
    # Tasks  (db.create_task / db.get_tasks / db.complete_task)
    # ---------------------------------------------------------------- #
    def CreateTask(self, request, context):
        try:
            result = db.create_task_idempotent(
                current_user_id(),
                request.title,
                request.due_date or None,
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
                due_date=result["due_date"],
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
                    due_date=r["due_date"] or "",
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


def _conversation_message(row: dict) -> chat_pb2.Conversation:
    return chat_pb2.Conversation(
        conversation_id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
    )


def create_server(port: int = 50051) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[AuthInterceptor()],
    )
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatServicer(), server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    return server