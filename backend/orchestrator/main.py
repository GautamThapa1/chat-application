"""FastAPI orchestrator: WebSocket bridge to the internal gRPC ChatService."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import grpc
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Before the backend imports: some of them read settings when imported.
load_dotenv(Path(__file__).resolve().parents[2] / ".env") # parents move upto 3 levels above

from backend.generated import chat_pb2_grpc
from backend.orchestrator import auth, conversations, memories, notes, reminders
from backend.orchestrator.grpc_session import SessionStub
from backend.orchestrator.rate_limit import default_limit_retry_after, limiter
from backend.orchestrator.websocket import websocket_endpoint
from shared_common.common.log import setup_logging

DEFAULT_GRPC_TARGET = "127.0.0.1:50051"


@asynccontextmanager # before yield setup and yield cleanup
async def lifespan(app: FastAPI): # lifespan: manages the gRPC object and channel
    setup_logging("orchestrator")
    grpc_target = os.getenv("GRPC_TARGET", DEFAULT_GRPC_TARGET)
    channel = grpc.aio.insecure_channel(grpc_target) # TODO: make it secure later for production
    app.state.grpc_stub = chat_pb2_grpc.ChatServiceStub(channel) # stub object of orchestrator
    try:
        yield # pasue point to give handle to FastAPI
    finally:
        await channel.close()


app = FastAPI(title="chat-orchestrator", lifespan=lifespan)
app.state.vapid_public_key = os.getenv("VAPID_PUBLIC_KEY", "")

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

@app.middleware("http")  # HTTP only: WebSockets are not touched
async def default_rate_limit(request: Request, call_next):
    retry_after = default_limit_retry_after(get_remote_address(request))
    if retry_after is not None:
        return JSONResponse(
            {"detail": "Too many requests. Please slow down."},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    return await call_next(request)


# Added last = outermost, so even 429s carry CORS headers. Lets the frontend on
# another origin (e.g. app.example.com -> api.example.com) call the API with the
# session cookie. Same allow-list as the WebSocket origin check.
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(auth.ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.websocket("/ws") # frontend connects here
async def ws_chat(ws: WebSocket):
    # Cookie (same site) or single-use ?ticket= (frontend on another site).
    token = auth.websocket_session_token(ws)
    user_id = await auth.authenticate_websocket(ws, token)
    if user_id is None or not token:
        # Closing BEFORE accept() makes the handshake fail with HTTP 403.
        await ws.close(code=1008)  # 1008 = policy violation
        return
    stub = SessionStub(ws.app.state.grpc_stub, token)
    await websocket_endpoint(
        ws,
        stub,
        user_id=user_id,
        revalidate_session=lambda: auth.authenticate_websocket(ws, token),
    )

@app.get("/api/health", include_in_schema=False)
async def health():
    """Liveness probe for the hosting platform (no auth, no gRPC call)."""
    return {"ok": True}


app.include_router(auth.router)  # must come BEFORE the "/" static mount
app.include_router(conversations.router)
app.include_router(memories.router)
app.include_router(notes.router)
app.include_router(reminders.router)

# connect the frontend: the Vite build output (run `npm run build` in frontend/)
app.mount(
    "/",
    StaticFiles(directory="frontend/dist", html=True, check_dir=False),
    name="frontend",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.orchestrator.main:app", host="0.0.0.0", port=8000, reload=True)


# got error about orchestrator.main about the imports
# another critical error was error about orchestrator directly accessing the database
# gives structure in such a way that it is convinent for the AI rather than me. eg: orchestrator tests instead of root test 