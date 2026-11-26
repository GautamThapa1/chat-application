
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from backend.orchestrator import main


@pytest.mark.asyncio
async def test_lifespan_initializes_and_cleans_up(monkeypatch):
    monkeypatch.setenv("GRPC_TARGET", "service:50051")

    fake_channel = SimpleNamespace(close=AsyncMock())
    fake_stub = object()

    fake_channel_factory = Mock(return_value=fake_channel)
    fake_stub_factory = Mock(return_value=fake_stub)

    monkeypatch.setattr(
        main.grpc.aio,
        "insecure_channel",
        fake_channel_factory,
    )
    monkeypatch.setattr(
        main.chat_pb2_grpc,
        "ChatServiceStub",
        fake_stub_factory,
    )
    setup_logging = Mock()
    monkeypatch.setattr(main, "setup_logging", setup_logging)

    app = FastAPI()

    async with main.lifespan(app):
        setup_logging.assert_called_once_with("orchestrator")
        assert app.state.grpc_stub is fake_stub

        fake_channel_factory.assert_called_once_with("service:50051")
        fake_stub_factory.assert_called_once_with(fake_channel)

        assert fake_channel.close.await_count == 0

    fake_channel.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_does_not_require_gemini_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    monkeypatch.setattr(
        main.grpc.aio,
        "insecure_channel",
        Mock(return_value=SimpleNamespace(close=AsyncMock())),
    )
    monkeypatch.setattr(
        main.chat_pb2_grpc,
        "ChatServiceStub",
        Mock(return_value=object()),
    )

    app = FastAPI()

    async with main.lifespan(app):
        assert hasattr(app.state, "grpc_stub")


@pytest.mark.asyncio
async def test_websocket_route_passes_dependencies(monkeypatch):
    fake_stub = object()

    fake_websocket = SimpleNamespace(
        cookies={"session": "tok-123"},
        app=SimpleNamespace(
            state=SimpleNamespace(
                grpc_stub=fake_stub,
            )
        )
    )

    endpoint = AsyncMock()

    monkeypatch.setattr(
        main,
        "websocket_endpoint",
        endpoint,
    )

    monkeypatch.setattr(
        main.auth,
        "authenticate_websocket",
        AsyncMock(return_value=1),  # pretend the handshake is authenticated
    )

    await main.ws_chat(fake_websocket)
    assert endpoint.await_args.kwargs["user_id"] == 1

    endpoint.assert_awaited_once()
    ws_arg, stub_arg = endpoint.await_args.args
    assert ws_arg is fake_websocket
    assert isinstance(stub_arg, main.SessionStub)
    assert stub_arg.stub is fake_stub

@pytest.mark.asyncio
async def test_websocket_route_rejects_unauthenticated(monkeypatch):
    fake_websocket = SimpleNamespace(close=AsyncMock())
    endpoint = AsyncMock()

    monkeypatch.setattr(main, "websocket_endpoint", endpoint)
    monkeypatch.setattr(
        main.auth, "authenticate_websocket", AsyncMock(return_value=None)
    )

    await main.ws_chat(fake_websocket)

    fake_websocket.close.assert_awaited_once_with(code=1008)
    endpoint.assert_not_awaited()


def test_websocket_route_is_registered():
    assert any(
        route.path == "/ws"
        for route in main.app.routes
    )
