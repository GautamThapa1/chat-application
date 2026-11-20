
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import grpc
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


def create_message(content, request_id="request-1", conversation_id=3):
    return json.dumps({
        "type": "message",
        "content": content,
        "request_id": request_id,
        "conversation_id": conversation_id,
    })


def fake_stub(messages=()):
    return SimpleNamespace(
        GetConversation=AsyncMock(return_value=SimpleNamespace(messages=messages))
    )


@pytest.mark.asyncio
async def test_websocket_streams_chat_and_saves_response(monkeypatch):
    # Arrange
    history = [SimpleNamespace(role="user", content="Earlier question")]
    stub = fake_stub(history)
    gemini_client = object()

    websocket = FakeWebSocket([
        create_message("Hello")
    ])

    decision = {"intent": "CHAT"}

    classify = AsyncMock(return_value=decision)
    execute = AsyncMock()
    save = AsyncMock()

    async def fake_stream(message, client, model, prior_history):
        assert prior_history == [
            {"role": "user", "content": "Earlier question"}
        ]
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
        user_id=1,
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
        3,
    )
    stub.GetConversation.assert_awaited_once()
    assert stub.GetConversation.await_args.args[0].conversation_id == 3


@pytest.mark.asyncio
async def test_websocket_handles_action_request(monkeypatch):
    # Arrange
    stub = fake_stub()
    gemini_client = object()

    websocket = FakeWebSocket([
        create_message("Save a note about my meeting")
    ])

    decision = {"intent": "SAVE_NOTE"}

    classify = AsyncMock(return_value=decision)
    execute = AsyncMock(return_value="Note saved successfully")
    save = AsyncMock()

    stream_calls = []

    async def fake_stream(message, client, model, history):
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
        user_id=1,
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
        3,
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
        fake_stub(),
        object(),
        "test-model",
        user_id=1,
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
    stub = fake_stub()
    gemini_client = object()

    websocket = FakeWebSocket([
        create_message("Hello")
    ])

    classify = AsyncMock(return_value={"intent": "CHAT"})
    save = AsyncMock()

    async def broken_stream(message, client, model, history):
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
        user_id=1,
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


@pytest.mark.asyncio
async def test_websocket_does_not_process_a_foreign_conversation(monkeypatch):
    stub = SimpleNamespace(
        GetConversation=AsyncMock(
            side_effect=grpc.aio.AioRpcError(
                grpc.StatusCode.NOT_FOUND,
                grpc.aio.Metadata(),
                grpc.aio.Metadata(),
            )
        )
    )
    websocket = FakeWebSocket([create_message("Hello", conversation_id=999)])
    classify = AsyncMock()
    monkeypatch.setattr(websocket_module, "classify_message", classify)

    await websocket_module.websocket_endpoint(
        websocket, stub, object(), "test-model", user_id=1
    )

    assert websocket.sent_messages[-1]["type"] == "error"
    assert "NOT_FOUND" in websocket.sent_messages[-1]["error"]
    classify.assert_not_awaited()

def patch_chat(monkeypatch):
    """Replace Gemini and gRPC helpers; return the classify mock."""
    classify = AsyncMock(return_value={"intent": "CHAT"})

    async def fake_stream(message, client, model, prior_history):
        yield "ok"

    monkeypatch.setattr(websocket_module, "classify_message", classify)
    monkeypatch.setattr(websocket_module, "save_conversation", AsyncMock())
    monkeypatch.setattr(websocket_module, "stream_chat_response", fake_stream)
    return classify


@pytest.mark.asyncio
async def test_websocket_blocks_the_eleventh_message(monkeypatch):
    classify = patch_chat(monkeypatch)
    websocket = FakeWebSocket([
        create_message("hi", request_id=f"r{i}") for i in range(11)
    ])

    await websocket_module.websocket_endpoint(
        websocket, fake_stub(), object(), "test-model", user_id=1,
    )

    errors = [e for e in websocket.sent_messages if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["request_id"] == "r10"          # the 11th message
    assert "Too many messages" in errors[0]["error"]
    assert classify.await_count == 10                # Gemini never saw the 11th


@pytest.mark.asyncio
async def test_websocket_limit_is_per_user(monkeypatch):
    patch_chat(monkeypatch)
    for _ in range(10):  # user 1 uses up their quota
        websocket = FakeWebSocket([create_message("hi")])
        await websocket_module.websocket_endpoint(
            websocket, fake_stub(), object(), "test-model", user_id=1,
        )

    other = FakeWebSocket([create_message("hi")])
    await websocket_module.websocket_endpoint(
        other, fake_stub(), object(), "test-model", user_id=2,
    )

    assert not [e for e in other.sent_messages if e["type"] == "error"]


@pytest.mark.asyncio
async def test_invalid_payloads_do_not_use_up_the_quota(monkeypatch):
    classify = patch_chat(monkeypatch)
    junk = ["not json"] * 20
    websocket = FakeWebSocket(junk + [create_message("hi")])

    await websocket_module.websocket_endpoint(
        websocket, fake_stub(), object(), "test-model", user_id=1,
    )

    assert classify.await_count == 1                 # the real message got through