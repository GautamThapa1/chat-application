"""Authenticated endpoints for task-reminder settings and Web Push."""

import grpc
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.generated import chat_pb2
from backend.orchestrator.auth import SESSION_COOKIE
from backend.orchestrator.grpc_session import SessionStub

router = APIRouter(prefix="/api", tags=["task reminders"])


class TimezoneRequest(BaseModel):
    timezone: str = Field(min_length=1, max_length=100)


class PushKeys(BaseModel):
    p256dh: str = Field(min_length=1, max_length=512)
    auth: str = Field(min_length=1, max_length=512)


class PushSubscriptionRequest(BaseModel):
    endpoint: str = Field(min_length=1, max_length=4096)
    keys: PushKeys


class PushSubscriptionDeleteRequest(BaseModel):
    endpoint: str = Field(min_length=1, max_length=4096)


def _session_stub(request: Request) -> SessionStub:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return SessionStub(request.app.state.grpc_stub, token)


def _raise_rpc_error(exc: grpc.aio.AioRpcError) -> None:
    status = {
        grpc.StatusCode.INVALID_ARGUMENT: 400,
        grpc.StatusCode.UNAUTHENTICATED: 401,
        grpc.StatusCode.NOT_FOUND: 404,
    }.get(exc.code(), 503)
    detail = (
        exc.details()
        if status != 503
        else "Service temporarily unavailable."
    )
    raise HTTPException(status_code=status, detail=detail) from exc


@router.get("/push/config")
async def push_config(request: Request):
    return {
        "enabled": bool(request.app.state.vapid_public_key),
        "publicKey": request.app.state.vapid_public_key,
    }


@router.post("/timezone")
async def set_timezone(body: TimezoneRequest, request: Request):
    stub = _session_stub(request)
    try:
        await stub.SetUserTimezone(
            chat_pb2.SetUserTimezoneRequest(timezone=body.timezone)
        )
    except grpc.aio.AioRpcError as exc:
        _raise_rpc_error(exc)
    return {"ok": True}


@router.post("/push/subscription")
async def save_subscription(
    body: PushSubscriptionRequest,
    request: Request,
):
    stub = _session_stub(request)
    try:
        await stub.SavePushSubscription(
            chat_pb2.SavePushSubscriptionRequest(
                endpoint=body.endpoint,
                p256dh=body.keys.p256dh,
                auth=body.keys.auth,
            )
        )
    except grpc.aio.AioRpcError as exc:
        _raise_rpc_error(exc)
    return {"ok": True}


@router.delete("/push/subscription")
async def delete_subscription(
    body: PushSubscriptionDeleteRequest,
    request: Request,
):
    stub = _session_stub(request)
    try:
        await stub.DeletePushSubscription(
            chat_pb2.DeletePushSubscriptionRequest(endpoint=body.endpoint)
        )
    except grpc.aio.AioRpcError as exc:
        _raise_rpc_error(exc)
    return {"ok": True}
