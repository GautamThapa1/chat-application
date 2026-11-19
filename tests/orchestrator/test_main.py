
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from backend.orchestrator import main


@pytest.mark.asyncio
async def test_lifespan_initializes_and_cleans_up(monkeypatch):
    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "fake-api-key")

    fake_channel = SimpleNamespace(close=AsyncMock())
    fake_stub = object()
    fake_gemini_client = object()

    fake_channel_factory = Mock(return_value=fake_channel)
    fake_stub_factory = Mock(return_value=fake_stub)

    class FakeGeminiContext:
        def __init__(self):
            self.aio = self
            self.exited = False

        async def __aenter__(self):
            return fake_gemini_client

        async def __aexit__(self, exc_type, exc_value, traceback):
            self.exited = True

    fake_gemini_context = FakeGeminiContext()
    fake_gemini_factory = Mock(return_value=fake_gemini_context)

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
    monkeypatch.setattr(
        main.genai,
        "Client",
        fake_gemini_factory,
    )

    app = FastAPI()

    # Act
    async with main.lifespan(app):
        # Assert startup
        assert app.state.grpc_stub is fake_stub
        assert app.state.gemini_client is fake_gemini_client
        assert app.state.gemini_model == main.GEMINI_MODEL

        fake_channel_factory.assert_called_once_with(main.GRPC_TARGET)
        fake_stub_factory.assert_called_once_with(fake_channel)

        fake_gemini_factory.assert_called_once()

        assert fake_channel.close.await_count == 0

    # Assert shutdown
    fake_channel.close.assert_awaited_once()
    assert fake_gemini_context.exited is True


@pytest.mark.asyncio
async def test_lifespan_requires_gemini_api_key(monkeypatch):
    # Arrange
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    channel_factory = Mock()

    monkeypatch.setattr(
        main.grpc.aio,
        "insecure_channel",
        channel_factory,
    )

    app = FastAPI()

    # Act and Assert
    with pytest.raises(
        RuntimeError,
        match="Set GEMINI_API_KEY",
    ):
        async with main.lifespan(app):
            pass

    # No connection should be created.
    channel_factory.assert_not_called()


@pytest.mark.asyncio
async def test_websocket_route_passes_dependencies(monkeypatch):
    # Arrange
    fake_stub = object()
    fake_gemini_client = object()
    fake_model = "test-model"

    fake_websocket = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                grpc_stub=fake_stub,
                gemini_client=fake_gemini_client,
                gemini_model=fake_model,
            )
        )
    )

    endpoint = AsyncMock()

    monkeypatch.setattr(
        main,
        "websocket_endpoint",
        endpoint,
    )

    # Act
    await main.ws_chat(fake_websocket)

    # Assert
    endpoint.assert_awaited_once_with(
        fake_websocket,
        fake_stub,
        fake_gemini_client,
        fake_model,
    )


def test_websocket_route_is_registered():
    # Assert
    assert any(
        route.path == "/ws"
        for route in main.app.routes
    )
