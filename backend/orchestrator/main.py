"""FastAPI orchestrator: WebSocket bridge to the internal gRPC ChatService."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import grpc
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

from backend.generated import chat_pb2_grpc
from backend.orchestrator import auth
from backend.orchestrator.grpc_session import SessionStub
from backend.orchestrator.websocket import websocket_endpoint

load_dotenv(Path(__file__).resolve().parents[2] / ".env") # parents move upto 3 levels above

GRPC_TARGET = "localhost:50051"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


@asynccontextmanager # before yield setup and yield cleanup
async def lifespan(app: FastAPI): # lifespan: manages the gRPC object and channel
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before starting the orchestrator.")

    channel = grpc.aio.insecure_channel(GRPC_TARGET) # TODO: make it secure later for production
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
    )

app.include_router(auth.router)  # must come BEFORE the "/" static mount

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