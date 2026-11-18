"""FastAPI orchestrator: WebSocket bridge to the internal gRPC ChatService."""

from contextlib import asynccontextmanager

import grpc
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from backend.generated import chat_pb2_grpc
from backend.orchestrator.websocket import websocket_endpoint

GRPC_TARGET = "localhost:50051"



@asynccontextmanager
async def lifespan(app: FastAPI):
    channel = grpc.aio.insecure_channel(GRPC_TARGET)
    app.state.grpc_stub = chat_pb2_grpc.ChatServiceStub(channel)
    try:
        yield # code before runs during startup and code after runs after startup
    finally:
        await channel.close()


app = FastAPI(title="chat-orchestrator", lifespan=lifespan)

@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await websocket_endpoint(ws, ws.app.state.grpc_stub)


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