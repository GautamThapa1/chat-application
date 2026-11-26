"""HTTP auth endpoints: register, login, logout, me.

These translate HTTP <-> the gRPC auth methods. The session token lives in an
HttpOnly cookie, so browser JavaScript never sees it.
"""

import logging
import os
import secrets
import time

import grpc
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket
from pydantic import BaseModel, Field

from backend.generated import chat_pb2
from backend.orchestrator.rate_limit import limiter

log = logging.getLogger("orchestrator")

router = APIRouter(prefix="/api")

SESSION_COOKIE = "session"
SESSION_MAX_AGE = 7 * 24 * 3600  # keep in step with the service's session TTL
# Secure cookies are only sent over HTTPS. Keep False for local http://,
# set COOKIE_SECURE=true when you serve over TLS.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"

# check which is trying to establish web connection before allowing it
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
@limiter.limit("5 per 15 minutes")
async def register(body: RegisterBody, request: Request):
    try:
        result = await request.app.state.grpc_stub.Register(
            chat_pb2.RegisterRequest(
                username=body.username,
                password=body.password
            )
        )
    except grpc.aio.AioRpcError as exc:
        raise _http_error(exc) from exc

    return {"user_id": result.user_id}

@router.post("/login")
@limiter.limit("10 per 15 minutes")  # little more because legit user may attemp to sign in more
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
    return {"user_id": result.user_id, "username": result.username}

# ---------- WebSocket tickets ----------
# When the frontend is on another site (e.g. *.vercel.app -> *.onrender.com),
# the browser will not send the session cookie on the WebSocket handshake.
# The page instead asks (through the same-site /api proxy, cookie included) for
# a short-lived, single-use ticket and puts it in the socket URL. Tickets live
# in this process's memory, which is fine because the app runs ONE worker.
WS_TICKET_TTL_SECONDS = 30
_ws_tickets: dict[str, tuple[str, float]] = {}  # ticket -> (session token, expiry)


def _prune_ws_tickets(now: float) -> None:
    for ticket, (_, expires) in list(_ws_tickets.items()):
        if expires <= now:
            _ws_tickets.pop(ticket, None)


def issue_ws_ticket(session_token: str) -> str:
    now = time.monotonic()
    _prune_ws_tickets(now)
    ticket = secrets.token_urlsafe(32)
    _ws_tickets[ticket] = (session_token, now + WS_TICKET_TTL_SECONDS)
    return ticket


def redeem_ws_ticket(ticket: str) -> str | None:
    """Return the session token for a valid ticket and consume it."""
    entry = _ws_tickets.pop(ticket, None)
    if entry is None:
        return None
    session_token, expires = entry
    return session_token if expires > time.monotonic() else None


@router.post("/ws-ticket")
async def ws_ticket(request: Request):
    await me(request)  # 401 unless the cookie holds a valid session
    return {"ticket": issue_ws_ticket(request.cookies[SESSION_COOKIE])}


def websocket_session_token(ws: WebSocket) -> str | None:
    """Session token for a handshake: a redeemed ?ticket=, else the cookie."""
    query = getattr(ws, "query_params", None) or {}
    ticket = query.get("ticket")
    if ticket:
        return redeem_ws_ticket(ticket)
    return (getattr(ws, "cookies", None) or {}).get(SESSION_COOKIE)


# this authenticate the websocket to see if it is allowed from the ALLOWED ORIGINS
async def authenticate_websocket(ws: WebSocket, token: str | None = None) -> int | None:
    """Return the user_id for a handshake that may proceed, else None.

    `token` is the session token to check; by default the cookie's.
    """
    origin = ws.headers.get("origin")
    # No Origin header = not a browser (curl, scripts). Those can't ride on a
    # victim's cookie, and still need a valid session below.
    if origin is not None and origin not in ALLOWED_ORIGINS:
        log.warning("WebSocket rejected: origin %r not allowed", origin)
        return None

    if token is None:
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