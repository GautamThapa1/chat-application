"""FastAPI orchestrator: WebSocket bridge to the internal gRPC ChatService."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import grpc
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from slowapi import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


from backend.generated import chat_pb2_grpc
from backend.orchestrator import auth, conversations
from backend.orchestrator.grpc_session import SessionStub
from backend.orchestrator.rate_limit import default_limit_retry_after, limiter
from backend.orchestrator.websocket import websocket_endpoint

load_dotenv(Path(__file__).resolve().parents[2] / ".env") # parents move upto 3 levels above

DEFAULT_GRPC_TARGET = "127.0.0.1:50051"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


@asynccontextmanager # before yield setup and yield cleanup
async def lifespan(app: FastAPI): # lifespan: manages the gRPC object and channel
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before starting the orchestrator.")

    grpc_target = os.getenv("GRPC_TARGET", DEFAULT_GRPC_TARGET)
    channel = grpc.aio.insecure_channel(grpc_target) # TODO: make it secure later for production
    app.state.grpc_stub = chat_pb2_grpc.ChatServiceStub(channel) # stub object of orchestrator

    # for gemini cleint
    try:
        retry_options = types.HttpRetryOptions(
            attempts=3,
            initial_delay=1,
            max_delay=5,
            http_status_codes=[503],
        )
        http_options = types.HttpOptions(retry_options=retry_options)
        async with genai.Client(
            api_key=api_key,
            http_options=http_options,
        ).aio as gemini_client:
            app.state.gemini_client = gemini_client
            app.state.gemini_model = GEMINI_MODEL
            yield # pasue point to give handle to FastAPI
    finally:
        await channel.close()


app = FastAPI(title="chat-orchestrator", lifespan=lifespan)

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


@app.websocket("/ws") # frontend connects here
async def ws_chat(ws: WebSocket):
    user_id = await auth.authenticate_websocket(ws)
    if user_id is None:
        # Closing BEFORE accept() makes the handshake fail with HTTP 403.
        await ws.close(code=1008)  # 1008 = policy violation
        return
    stub = SessionStub(
        ws.app.state.grpc_stub,
        ws.cookies[auth.SESSION_COOKIE],
    )
    await websocket_endpoint(
        ws,
        stub,
        ws.app.state.gemini_client,
        ws.app.state.gemini_model,
        user_id=user_id,
        revalidate_session=lambda: auth.authenticate_websocket(ws),
    )

app.include_router(auth.router)  # must come BEFORE the "/" static mount
app.include_router(conversations.router)

# connect the frontend
app.mount(
    "/",
    StaticFiles(directory="frontend", html=True),
    name="frontend",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.orchestrator.main:app", host="0.0.0.0", port=8000, reload=True)


# got error about orchestrator.main about the imports
# another critical error was error about orchestrator directly accessing the database
# gives structure in such a way that it is convinent for the AI rather than me. eg: orchestrator tests instead of root test 