"""gRPC servicer for ChatService — each RPC delegates to db.py."""

from concurrent import futures

import grpc

from backend.generated import chat_pb2, chat_pb2_grpc
from backend.service import db


class ChatServicer(chat_pb2_grpc.ChatServiceServicer):

    def SendMessage(self, request, context):
        db.save_message(request.role or "user", request.content)
        reply = f"Echo: {request.content}"
        db.save_message("assistant", reply)
        return chat_pb2.ChatResponse(role="assistant", content=reply)

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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    return server