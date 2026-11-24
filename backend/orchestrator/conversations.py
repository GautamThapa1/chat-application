"""HTTP endpoints for the signed-in user's conversations."""

import logging

import grpc
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.embeddings import QUERY_TASK_TYPE, embed_text_async, embedding_model
from backend.generated import chat_pb2
from backend.orchestrator.auth import SESSION_COOKIE
from backend.orchestrator.grpc_session import SessionStub

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class CreateConversationBody(BaseModel):
    title: str = Field(default="New chat", max_length=120)


class SearchConversationsBody(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=20, ge=1, le=50)


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


def _conversation_json(conversation) -> dict:
    return {
        "conversation_id": conversation.conversation_id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }



@router.get("")
async def list_conversations(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    result = await _call(
        request,
        "ListConversations",
        chat_pb2.ListConversationsRequest(
            limit=limit,
            offset=offset,
        ),
    )

    return {
        "conversations": [
            _conversation_json(row)
            for row in result.conversations
        ],
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total_count": result.total_count,
        },
    }


@router.post("", status_code=201)
async def create_conversation(body: CreateConversationBody, request: Request):
    result = await _call(
        request,
        "CreateConversation",
        chat_pb2.CreateConversationRequest(title=body.title),
    )
    return {"conversation": _conversation_json(result.conversation)}


@router.post("/search")
async def search_conversations(body: SearchConversationsBody, request: Request):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Search query must not be empty.")

    query_embedding = []
    model = ""
    gemini_client = getattr(request.app.state, "gemini_client", None)
    if gemini_client is not None:
        model = embedding_model()
        try:
            query_embedding = await embed_text_async(
                gemini_client,
                query,
                task_type=QUERY_TASK_TYPE,
                model=model,
            )
        except Exception:
            logger.warning(
                "Conversation search embedding failed; using keyword search.",
                exc_info=True,
            )
            model = ""

    result = await _call(
        request,
        "SearchConversations",
        chat_pb2.SearchConversationsRequest(
            keyword=query,
            embedding=query_embedding,
            embedding_model=model,
            limit=body.limit,
        ),
    )
    return {
        "results": [
            {
                "conversation_id": row.conversation_id,
                "title": row.title,
                "excerpt": row.excerpt,
                "updated_at": row.updated_at,
                "score": row.score,
                "message_start_id": row.message_start_id,
                "message_end_id": row.message_end_id,
            }
            for row in result.results
        ]
    }


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: int, request: Request):
    if conversation_id < 1:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    result = await _call(
        request,
        "GetConversation",
        chat_pb2.GetConversationRequest(conversation_id=conversation_id),
    )
    return {
        "conversation": _conversation_json(result.conversation),
        "messages": [
            {
                "message_id": message.message_id,
                "conversation_id": message.conversation_id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at,
            }
            for message in result.messages
        ],
    }


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: int, request: Request):
    if conversation_id < 1:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    await _call(
        request,
        "DeleteConversation",
        chat_pb2.DeleteConversationRequest(conversation_id=conversation_id),
    )
    return {"ok": True}
