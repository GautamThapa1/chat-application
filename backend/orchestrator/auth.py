"""HTTP auth endpoints: register, login, logout, me.

These translate HTTP <-> the gRPC auth methods. The session token lives in an
HttpOnly cookie, so browser JavaScript never sees it.
"""

import logging
import os

import grpc
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket
from pydantic import BaseModel, Field

from backend.generated import chat_pb2

log = logging.getLogger("orchestrator")

router = APIRouter(prefix="/api")

SESSION_COOKIE = "session"
SESSION_MAX_AGE = 7 * 24 * 3600  # keep in step with the service's session TTL
# Secure cookies are only sent over HTTPS. Keep False for local http://,
# set COOKIE_SECURE=true when you serve over TLS.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"

# Browsers attach an Origin header to WebSocket handshakes. Only pages served
# from these origins may open a socket (blocks cross-site WebSocket hijacking).
ALLOWED_ORIGINS = {
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if o.strip()
}

# gRPC status code -> HTTP status code
_ERROR_MAP = {
    grpc.StatusCode.INVALID_ARGUMENT: 400,
    grpc.StatusCode.UNAUTHENTICATED: 401,
    grpc.StatusCode.ALREADY_EXISTS: 409,
}


class RegisterBody(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_]+$")
    password: str = Field(min_length=8, max_length=128)


class LoginBody(BaseModel):
    username: str = Field(max_length=32)
    password: str = Field(max_length=128)


def _http_error(exc: grpc.aio.AioRpcError) -> HTTPException:
    """Turn a gRPC failure into an HTTP one without leaking internals."""
    status = _ERROR_MAP.get(exc.code())
    if status is not None:
        # These messages were written by our own service for the client.
        return HTTPException(status_code=status, detail=exc.details())
    # Anything else (UNAVAILABLE, INTERNAL, ...) is our problem, not the user's.
    return HTTPException(status_code=503, detail="Service temporarily unavailable.")


@router.post("/register", status_code=201)
async def register(body: RegisterBody, request: Request):
    try:
        result = await request.app.state.grpc_stub.Register(
            chat_pb2.RegisterRequest(username=body.username, password=body.password)
        )
    except grpc.aio.AioRpcError as exc:
        raise _http_error(exc) from exc
    return {"user_id": result.user_id}


@router.post("/login")
async def login(body: LoginBody, request: Request, response: Response):
    try:
        result = await request.app.state.grpc_stub.Login(
            chat_pb2.LoginRequest(username=body.username, password=body.password)
        )
    except grpc.aio.AioRpcError as exc:
        raise _http_error(exc) from exc

    response.set_cookie(
        key=SESSION_COOKIE,
        value=result.session_token,
        max_age=SESSION_MAX_AGE,
        httponly=True,     # JavaScript cannot read it (blunts XSS token theft)
        samesite="lax",    # not sent on cross-site POSTs (blunts CSRF)
        secure=COOKIE_SECURE,
        path="/",
    )
    return {"user_id": result.user_id}


@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        try:
            await request.app.state.grpc_stub.Logout(
                chat_pb2.LogoutRequest(session_token=token)
            )
        except grpc.aio.AioRpcError as exc:
            raise _http_error(exc) from exc
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/me")
async def me(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in.")
    try:
        result = await request.app.state.grpc_stub.ValidateSession(
            chat_pb2.ValidateSessionRequest(session_token=token)
        )
    except grpc.aio.AioRpcError as exc:
        raise _http_error(exc) from exc
    if not result.valid:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return {"user_id": result.user_id}

async def authenticate_websocket(ws: WebSocket) -> int | None:
    """Return the user_id for a handshake that may proceed, else None."""
    origin = ws.headers.get("origin")
    # No Origin header = not a browser (curl, scripts). Those can't ride on a
    # victim's cookie, and still need a valid session below.
    if origin is not None and origin not in ALLOWED_ORIGINS:
        log.warning("WebSocket rejected: origin %r not allowed", origin)
        return None

    token = ws.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        result = await ws.app.state.grpc_stub.ValidateSession(
            chat_pb2.ValidateSessionRequest(session_token=token)
        )
    except grpc.aio.AioRpcError as exc:
        log.error("ValidateSession failed: %s - %s", exc.code().name, exc.details())
        return None
    return result.user_id if result.valid else None