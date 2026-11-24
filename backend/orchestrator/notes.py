"""HTTP endpoints for the signed-in user's notes."""

import grpc
from fastapi import APIRouter, HTTPException, Path, Request

from backend.generated import chat_pb2
from backend.orchestrator.auth import SESSION_COOKIE
from backend.orchestrator.grpc_session import SessionStub

router = APIRouter(prefix="/api/notes", tags=["notes"])


@router.delete("/{note_id}")
async def delete_note(request: Request, note_id: int = Path(gt=0)):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in.")

    stub = SessionStub(request.app.state.grpc_stub, token)
    try:
        response = await stub.DeleteNote(
            chat_pb2.DeleteNoteRequest(note_id=note_id)
        )
    except grpc.aio.AioRpcError as exc:
        status = {
            grpc.StatusCode.INVALID_ARGUMENT: 400,
            grpc.StatusCode.UNAUTHENTICATED: 401,
            grpc.StatusCode.NOT_FOUND: 404,
        }.get(exc.code(), 503)
        detail = exc.details() if status != 503 else "Service temporarily unavailable."
        raise HTTPException(status_code=status, detail=detail) from exc

    if not response.success:
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"ok": True}
