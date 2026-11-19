"""gRPC servicer for ChatService — each RPC delegates to db.py."""

from concurrent import futures

import grpc

from backend.generated import chat_pb2, chat_pb2_grpc
from backend.service import db
from backend.service.auth_interceptor import AuthInterceptor


class ChatServicer(chat_pb2_grpc.ChatServiceServicer):

    def SendMessage(self, request, context):
        db.save_message(request.role or "user", request.content)
        reply = f"Echo: {request.content}"
        db.save_message("assistant", reply)
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
        db.save_conversation(request.user_content, request.assistant_content)
        return chat_pb2.SaveConversationResponse(success=True)

    # ---------------------------------------------------------------- #
    # Notes  (db.save_note / db.search_notes)
    # ---------------------------------------------------------------- #
    def SaveNote(self, request, context):
        note_id = db.save_note(request.title, request.content)
        return chat_pb2.SaveNoteResponse(
            note=chat_pb2.Note(
                id=note_id,
                title=request.title,
                content=request.content,
            )
        )

    def SearchNotes(self, request, context):
        rows = db.search_notes(request.keyword)
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

    # ---------------------------------------------------------------- #
    # Tasks  (db.create_task / db.get_tasks / db.complete_task)
    # ---------------------------------------------------------------- #
    def CreateTask(self, request, context):
        task_id = db.create_task(request.title, due_date=request.due_date or None)
        return chat_pb2.CreateTaskResponse(
            task=chat_pb2.Task(
                id=task_id,
                title=request.title,
                due_date=request.due_date,
                completed=False,
            )
        )

    def ListTasks(self, request, context):
        rows = db.get_tasks()
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
        ok = db.complete_task(request.task_id)
        return chat_pb2.CompleteTaskResponse(success=ok)


def create_server(port: int = 50051) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[AuthInterceptor()],
    )
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    return server