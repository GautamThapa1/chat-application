"""HTTP API for managing the signed-in user's long-term memories."""

import grpc
from fastapi import APIRouter, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from backend.generated import chat_pb2
from backend.orchestrator.auth import SESSION_COOKIE
from backend.orchestrator.grpc_session import SessionStub

router = APIRouter(prefix="/api/memories", tags=["memories"])


class MemoryBody(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=10000)


def _session_stub(request: Request) -> SessionStub:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return SessionStub(request.app.state.grpc_stub, token)


async def _call(request: Request, method: str, message):
    stub = _session_stub(request)
    try:
        return await getattr(stub, method)(message)
    except grpc.aio.AioRpcError as exc:
        status = {
            grpc.StatusCode.INVALID_ARGUMENT: 400,
            grpc.StatusCode.UNAUTHENTICATED: 401,
            grpc.StatusCode.NOT_FOUND: 404,
        }.get(exc.code(), 503)
        detail = exc.details() if status != 503 else "Service temporarily unavailable."
        raise HTTPException(status_code=status, detail=detail) from exc


def _memory_json(memory) -> dict:
    return {
        "memory_id": memory.id,
        "title": memory.title,
        "content": memory.content,
        "created_at": memory.created_at,
    }


def _validated_body(body: MemoryBody) -> tuple[str, str]:
    title, content = body.title.strip(), body.content.strip()
    if not title or not content:
        raise HTTPException(status_code=422, detail="Title and content cannot be blank.")
    return title, content


@router.get("")
async def list_memories(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    result = await _call(
        request,
        "ListMemories",
        chat_pb2.ListMemoriesRequest(limit=limit, offset=offset),
    )
    return {
        "memories": [_memory_json(memory) for memory in result.memories],
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total_count": result.total_count,
        },
    }


@router.get("/search")
async def search_memories(
    request: Request,
    keyword: str = Query(min_length=1, max_length=500),
):
    keyword = keyword.strip()
    if not keyword:
        raise HTTPException(status_code=422, detail="Keyword cannot be blank.")
    result = await _call(
        request,
        "SearchMemories",
        chat_pb2.SearchMemoriesRequest(keyword=keyword),
    )
    return {"memories": [_memory_json(memory) for memory in result.memories]}


@router.post("", status_code=201)
async def create_memory(body: MemoryBody, request: Request):
    title, content = _validated_body(body)
    result = await _call(
        request,
        "SaveMemory",
        chat_pb2.SaveMemoryRequest(title=title, content=content),
    )
    return {"memory": _memory_json(result.memory)}


@router.put("/{memory_id}")
async def update_memory(
    request: Request,
    body: MemoryBody,
    memory_id: int = Path(gt=0),
):
    title, content = _validated_body(body)
    result = await _call(
        request,
        "UpdateMemory",
        chat_pb2.UpdateMemoryRequest(
            memory_id=memory_id,
            title=title,
            content=content,
        ),
    )
    return {"memory": _memory_json(result.memory)}


@router.delete("/{memory_id}")
async def delete_memory(request: Request, memory_id: int = Path(gt=0)):
    result = await _call(
        request,
        "DeleteMemory",
        chat_pb2.DeleteMemoryRequest(memory_id=memory_id),
    )
    if not result.success:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"ok": True}
