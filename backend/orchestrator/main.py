"""FastAPI orchestrator: WebSocket bridge to the internal gRPC ChatService."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import grpc
from google import genai
from google.genai import types
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from backend.generated import chat_pb2_grpc
from backend.orchestrator.websocket import websocket_endpoint

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

GRPC_TARGET = "localhost:50051"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before starting the orchestrator.")

    channel = grpc.aio.insecure_channel(GRPC_TARGET)
    app.state.grpc_stub = chat_pb2_grpc.ChatServiceStub(channel) # stub object
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
            yield
    finally:
        await channel.close()


app = FastAPI(title="chat-orchestrator", lifespan=lifespan)

@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await websocket_endpoint(
        ws,
        ws.app.state.grpc_stub,
        ws.app.state.gemini_client,
        ws.app.state.gemini_model,
    )

# connect the frontend
app.mount(
    "/",
    StaticFiles(directory="frontend", html=True),
    name="frontend",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.orchestrator.main:app", host="0.0.0.0", port=8000, reload=True)


#     File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
#     return _bootstrap._gcd_import(name[level:], package, level)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
#   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
#   File "<frozen importlib._bootstrap>", line 1310, in _find_and_load_unlocked
#   File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
#   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
#   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
#   File "<frozen importlib._bootstrap>", line 1324, in _find_and_load_unlocked
# ModuleNotFoundError: No module named 'orchestrator'

# used backend.orches....