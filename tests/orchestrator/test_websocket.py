
import json
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocketDisconnect

import backend.orchestrator.websocket as websocket_module


class FakeWebSocket:
    """Simulate a WebSocket connection for unit testing."""

    def __init__(self, messages):
        self.messages = list(messages)
        self.sent_messages = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        if self.messages:
            return self.messages.pop(0)

        raise WebSocketDisconnect(code=1000)

    async def send_json(self, data):
        self.sent_messages.append(data)


def create_message(content, request_id="request-1"):
    return json.dumps({
        "type": "message",
        "content": content,
        "request_id": request_id,
    })


@pytest.mark.asyncio
async def test_websocket_streams_chat_and_saves_response(monkeypatch):
    # Arrange
    stub = object()
    gemini_client = object()

    websocket = FakeWebSocket([
        create_message("Hello")
    ])

    decision = {"intent": "CHAT"}

    classify = AsyncMock(return_value=decision)
    execute = AsyncMock()
    save = AsyncMock()

    async def fake_stream(message, client, model):
        yield "Hello"
        yield " there!"

    monkeypatch.setattr(
        websocket_module,
        "classify_message",
        classify,
    )
    monkeypatch.setattr(
        websocket_module,
        "execute_action",
        execute,
    )
    monkeypatch.setattr(
        websocket_module,
        "save_conversation",
        save,
    )
    monkeypatch.setattr(
        websocket_module,
        "stream_chat_response",
        fake_stream,
    )

    # Act
    await websocket_module.websocket_endpoint(
        websocket,
        stub,
        gemini_client,
        "test-model",
    )

    # Assert
    assert websocket.accepted is True

    events = websocket.sent_messages

    assert [event["type"] for event in events] == [
        "start",
        "chunk",
        "chunk",
        "complete",
    ]

    chunks = [
        event["content"]
        for event in events
        if event["type"] == "chunk"
    ]

    assert chunks == ["Hello", " there!"]

    classify.assert_awaited_once_with(
        "Hello",
        gemini_client,
        "test-model",
    )

    execute.assert_not_awaited()

    save.assert_awaited_once_with(
        "Hello",
        "Hello there!",
        stub,
    )


@pytest.mark.asyncio
async def test_websocket_handles_action_request(monkeypatch):
    # Arrange
    stub = object()
    gemini_client = object()

    websocket = FakeWebSocket([
        create_message("Save a note about my meeting")
    ])

    decision = {"intent": "SAVE_NOTE"}

    classify = AsyncMock(return_value=decision)
    execute = AsyncMock(return_value="Note saved successfully")
    save = AsyncMock()

    stream_calls = []

    async def fake_stream(message, client, model):
        stream_calls.append(message)
        yield "Unexpected chat response"

    monkeypatch.setattr(websocket_module, "classify_message", classify)
    monkeypatch.setattr(websocket_module, "execute_action", execute)
    monkeypatch.setattr(websocket_module, "save_conversation", save)
    monkeypatch.setattr(websocket_module, "stream_chat_response", fake_stream)

    # Act
    await websocket_module.websocket_endpoint(
        websocket,
        stub,
        gemini_client,
        "test-model",
    )

    # Assert
    events = websocket.sent_messages

    assert [event["type"] for event in events] == [
        "start",
        "chunk",
        "complete",
    ]

    assert events[1]["content"] == "Note saved successfully"

    execute.assert_awaited_once_with(decision, stub)

    assert stream_calls == []

    save.assert_awaited_once_with(
        "Save a note about my meeting",
        "Note saved successfully",
        stub,
    )


@pytest.mark.asyncio
async def test_websocket_rejects_invalid_payload(monkeypatch):
    # Arrange
    websocket = FakeWebSocket([
        json.dumps({
            "type": "message",
            "content": 123,
            "request_id": "request-1",
        })
    ])

    classify = AsyncMock()

    monkeypatch.setattr(
        websocket_module,
        "classify_message",
        classify,
    )

    # Act
    await websocket_module.websocket_endpoint(
        websocket,
        object(),
        object(),
        "test-model",
    )

    # Assert
    assert len(websocket.sent_messages) == 1

    event = websocket.sent_messages[0]

    assert event["type"] == "error"
    assert event["request_id"] == "request-1"
    assert event["error"] == "Invalid message payload."

    classify.assert_not_awaited()


@pytest.mark.asyncio
async def test_websocket_does_not_save_incomplete_stream(monkeypatch):
    # Arrange
    stub = object()
    gemini_client = object()

    websocket = FakeWebSocket([
        create_message("Hello")
    ])

    classify = AsyncMock(return_value={"intent": "CHAT"})
    save = AsyncMock()

    async def broken_stream(message, client, model):
        yield "Partial response"
        raise ValueError("Streaming failed")

    monkeypatch.setattr(websocket_module, "classify_message", classify)
    monkeypatch.setattr(websocket_module, "save_conversation", save)
    monkeypatch.setattr(websocket_module, "stream_chat_response", broken_stream)

    # Act
    await websocket_module.websocket_endpoint(
        websocket,
        stub,
        gemini_client,
        "test-model",
    )

    # Assert
    events = websocket.sent_messages

    assert [event["type"] for event in events] == [
        "start",
        "chunk",
        "error",
    ]

    assert events[1]["content"] == "Partial response"
    assert "Streaming failed" in events[2]["error"]

    save.assert_not_awaited()
