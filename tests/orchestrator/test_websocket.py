
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import grpc
import pytest
from fastapi import WebSocketDisconnect
from google.genai import types

import backend.orchestrator.websocket as websocket_module


class FakeWebSocket:
    """Simulate a WebSocket connection for unit testing."""

    def __init__(self, messages):
        self.messages = list(messages)
        self.sent_messages = []
        self.close_codes = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        # Like a real client, send the next frame after the server can process
        # the previous reply running in the background.
        await asyncio.sleep(0.01)
        if self.messages:
            return self.messages.pop(0)

        raise WebSocketDisconnect(code=1000)

    async def send_json(self, data):
        self.sent_messages.append(data)

    async def close(self, code=1000):
        self.close_codes.append(code)


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
async def test_websocket_revalidates_session_on_each_message():
    websocket = FakeWebSocket([create_message("Hello")])
    revalidations = 0

    async def revalidate_session():
        nonlocal revalidations
        revalidations += 1
        return 1

    await websocket_module.websocket_endpoint(
        websocket,
        fake_stub(),
        object(),
        "test-model",
        user_id=1,
        revalidate_session=revalidate_session,
    )

    assert revalidations == 1


@pytest.mark.asyncio
async def test_expired_session_closes_websocket():
    websocket = FakeWebSocket([create_message("Hello")])

    async def revalidate_session():
        return None

    await websocket_module.websocket_endpoint(
        websocket,
        fake_stub(),
        object(),
        "test-model",
        user_id=1,
        revalidate_session=revalidate_session,
    )

    assert websocket.close_codes == [1008]


@pytest.mark.asyncio
async def test_websocket_streams_chat_and_saves_response(monkeypatch):
    # Arrange
    history = [SimpleNamespace(role="user", content="Earlier question")]
    stub = fake_stub(history)
    gemini_client = object()

    websocket = FakeWebSocket([
        create_message("Hello")
    ])

    handle_tools = AsyncMock(return_value=None)
    save = AsyncMock()

    async def fake_stream(message, client, model, prior_history):
        assert prior_history == [
            {"role": "user", "content": "Earlier question"}
        ]
        yield "Hello"
        yield " there!"

    monkeypatch.setattr(
        websocket_module,
        "handle_function_calls",
        handle_tools,
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

    handle_tools.assert_awaited_once_with(
        "Hello",
        gemini_client,
        "test-model",
        [{"role": "user", "content": "Earlier question"}],
        stub,
    )

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

    handle_tools = AsyncMock(return_value="Note saved successfully")
    save = AsyncMock()

    stream_calls = []

    async def fake_stream(message, client, model, history):
        stream_calls.append(message)
        yield "Unexpected chat response"

    monkeypatch.setattr(
        websocket_module, "handle_function_calls", handle_tools
    )
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

    handle_tools.assert_awaited_once_with(
        "Save a note about my meeting",
        gemini_client,
        "test-model",
        [],
        stub,
    )

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

    monkeypatch.setattr(
        websocket_module,
        "handle_function_calls",
        AsyncMock(),
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

    websocket_module.handle_function_calls.assert_not_awaited()


@pytest.mark.asyncio
async def test_websocket_does_not_save_incomplete_stream(monkeypatch):
    # Arrange
    stub = fake_stub()
    gemini_client = object()

    websocket = FakeWebSocket([
        create_message("Hello")
    ])

    handle_tools = AsyncMock(return_value=None)
    save = AsyncMock()

    async def broken_stream(message, client, model, history):
        yield "Partial response"
        raise ValueError("Streaming failed")

    monkeypatch.setattr(
        websocket_module, "handle_function_calls", handle_tools
    )
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
    handle_tools = AsyncMock()
    monkeypatch.setattr(
        websocket_module, "handle_function_calls", handle_tools
    )

    await websocket_module.websocket_endpoint(
        websocket, stub, object(), "test-model", user_id=1
    )

    assert websocket.sent_messages[-1]["type"] == "error"
    assert "NOT_FOUND" in websocket.sent_messages[-1]["error"]
    handle_tools.assert_not_awaited()

def patch_chat(monkeypatch):
    """Replace tool selection and streaming; return the tool-selection mock."""
    handle_tools = AsyncMock(return_value=None)

    async def fake_stream(message, client, model, prior_history):
        yield "ok"

    monkeypatch.setattr(
        websocket_module, "handle_function_calls", handle_tools
    )
    monkeypatch.setattr(websocket_module, "save_conversation", AsyncMock())
    monkeypatch.setattr(websocket_module, "stream_chat_response", fake_stream)
    return handle_tools


@pytest.mark.asyncio
async def test_websocket_blocks_the_eleventh_message(monkeypatch):
    handle_tools = patch_chat(monkeypatch)
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
    assert handle_tools.await_count == 10             # Gemini never saw the 11th


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
    handle_tools = patch_chat(monkeypatch)
    junk = ["not json"] * 20
    websocket = FakeWebSocket(junk + [create_message("hi")])

    await websocket_module.websocket_endpoint(
        websocket, fake_stub(), object(), "test-model", user_id=1,
    )

    assert handle_tools.await_count == 1              # the real message got through

@pytest.mark.asyncio
async def test_ping_gets_a_pong(monkeypatch):
    websocket = FakeWebSocket([json.dumps({"type": "ping"})])

    await websocket_module.websocket_endpoint(
        websocket, fake_stub(), object(), "test-model", user_id=1,
    )

    assert websocket.sent_messages == [{"type": "pong"}]


@pytest.mark.asyncio
async def test_pings_do_not_use_up_the_message_quota(monkeypatch):
    handle_tools = patch_chat(monkeypatch)
    pings = [json.dumps({"type": "ping"})] * 50
    websocket = FakeWebSocket(pings + [create_message("hi")])

    await websocket_module.websocket_endpoint(
        websocket, fake_stub(), object(), "test-model", user_id=1,
    )

    assert handle_tools.await_count == 1
    assert not [e for e in websocket.sent_messages if e["type"] == "error"]


@pytest.mark.asyncio
async def test_rate_limited_user_still_gets_pongs(monkeypatch):
    patch_chat(monkeypatch)
    messages = [create_message("hi", request_id=f"r{i}") for i in range(11)]
    websocket = FakeWebSocket(messages + [json.dumps({"type": "ping"})])

    await websocket_module.websocket_endpoint(
        websocket, fake_stub(), object(), "test-model", user_id=1,
    )

    assert websocket.sent_messages[-1] == {"type": "pong"}


class LiveWebSocket:
    """A socket where the test decides when each frame arrives."""

    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()

    async def accept(self):
        pass

    async def receive_text(self):
        frame = await self.incoming.get()
        if frame is None:
            raise WebSocketDisconnect(code=1000)
        return frame

    async def send_json(self, data):
        await self.outgoing.put(data)

    def push(self, payload):
        self.incoming.put_nowait(json.dumps(payload))

    def disconnect(self):
        self.incoming.put_nowait(None)

    async def next_event(self, wanted=None):
        """Return the next event, or fail after one second."""
        while True:
            event = await asyncio.wait_for(self.outgoing.get(), timeout=1)
            if wanted is None or event["type"] == wanted:
                return event


def start_endpoint(websocket, user_id=1):
    return asyncio.create_task(
        websocket_module.websocket_endpoint(
            websocket, fake_stub(), object(), "test-model", user_id=user_id,
        )
    )


def patch_blocking_stream(monkeypatch, *, first_chunk="Hel"):
    """Stream one chunk, then wait until the test releases the reply."""
    release = asyncio.Event()
    state = {"cancelled": False}
    save = AsyncMock()

    async def stream(message, client, model, history):
        try:
            yield first_chunk
            await release.wait()
            yield "lo"
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    monkeypatch.setattr(
        websocket_module,
        "handle_function_calls",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(websocket_module, "stream_chat_response", stream)
    monkeypatch.setattr(websocket_module, "save_conversation", save)
    return release, state, save


def message_payload(request_id="r1", content="hi"):
    return {
        "type": "message",
        "content": content,
        "request_id": request_id,
        "conversation_id": 3,
    }


@pytest.mark.asyncio
async def test_cancel_stops_the_reply_and_keeps_the_partial_text(monkeypatch):
    release, state, save = patch_blocking_stream(monkeypatch)
    ws = LiveWebSocket()
    task = start_endpoint(ws)

    ws.push(message_payload("r1"))
    assert (await ws.next_event("chunk"))["content"] == "Hel"

    ws.push({"type": "cancel", "request_id": "r1"})
    assert await ws.next_event("cancelled") == {
        "type": "cancelled",
        "request_id": "r1",
    }
    assert state["cancelled"] is True
    save.assert_awaited_once()
    assert save.await_args.args[:2] == ("hi", "Hel")

    ws.disconnect()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_cancel_before_any_text_saves_nothing(monkeypatch):
    save = AsyncMock()
    gate = asyncio.Event()

    async def slow_tool_selection(message, client, model, history, stub):
        await gate.wait()

    monkeypatch.setattr(
        websocket_module, "handle_function_calls", slow_tool_selection
    )
    monkeypatch.setattr(websocket_module, "save_conversation", save)
    ws = LiveWebSocket()
    task = start_endpoint(ws)

    ws.push(message_payload("r1"))
    await asyncio.sleep(0.01)
    ws.push({"type": "cancel", "request_id": "r1"})

    assert (await ws.next_event("cancelled"))["request_id"] == "r1"
    save.assert_not_awaited()

    ws.disconnect()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_cancel_for_an_unknown_request_is_ignored(monkeypatch):
    release, state, save = patch_blocking_stream(monkeypatch)
    ws = LiveWebSocket()
    task = start_endpoint(ws)

    ws.push(message_payload("r1"))
    await ws.next_event("chunk")

    ws.push({"type": "cancel", "request_id": "nope"})
    ws.push({"type": "cancel"})
    ws.push({"type": "ping"})
    assert (await ws.next_event())["type"] == "pong"
    assert state["cancelled"] is False

    release.set()
    assert (await ws.next_event("complete"))["request_id"] == "r1"
    ws.disconnect()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_an_action_is_not_cut_off_by_cancel(monkeypatch):
    gate = asyncio.Event()
    rpc_started = asyncio.Event()
    save = AsyncMock()

    async def slow_create_task(request):
        rpc_started.set()
        await gate.wait()
        return SimpleNamespace(
            task=SimpleNamespace(
                id=1,
                title=request.title,
                due_date=request.due_date,
            )
        )

    stub = fake_stub()
    stub.CreateTask = AsyncMock(side_effect=slow_create_task)
    client = SimpleNamespace(
        models=SimpleNamespace(
            count_tokens=AsyncMock(
                return_value=SimpleNamespace(total_tokens=1)
            ),
            generate_content=AsyncMock(side_effect=[
                SimpleNamespace(
                    candidates=[
                        SimpleNamespace(
                            content=types.Content(
                                role="model",
                                parts=[
                                    types.Part(
                                        function_call=types.FunctionCall(
                                            name="create_task",
                                            args={"title": "Task"},
                                        )
                                    )
                                ],
                            )
                        )
                    ],
                    text=None,
                ),
                SimpleNamespace(
                    candidates=[
                        SimpleNamespace(
                            content=types.Content(
                                role="model",
                                parts=[types.Part(text="Task created.")],
                            )
                        )
                    ],
                    text="Task created.",
                ),
            ]),
        )
    )
    monkeypatch.setattr(websocket_module, "save_conversation", save)
    ws = LiveWebSocket()
    task = asyncio.create_task(
        websocket_module.websocket_endpoint(
            ws, stub, client, "test-model", user_id=1,
        )
    )

    ws.push(message_payload("r1", "create a task"))
    await asyncio.wait_for(rpc_started.wait(), timeout=1)
    ws.push({"type": "cancel", "request_id": "r1"})
    await asyncio.sleep(0.01)
    gate.set()

    chunk = await ws.next_event("chunk")
    assert chunk["content"] == "Task created."
    assert (await ws.next_event("complete"))["request_id"] == "r1"
    save.assert_awaited_once()
    assert ws.outgoing.empty()

    ws.disconnect()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_second_message_is_refused_while_one_is_in_flight(monkeypatch):
    release, state, save = patch_blocking_stream(monkeypatch)
    ws = LiveWebSocket()
    task = start_endpoint(ws)

    ws.push(message_payload("r1"))
    await ws.next_event("chunk")

    ws.push(message_payload("r2"))
    refused = await ws.next_event("error")
    assert refused["request_id"] == "r2"
    assert "previous message" in refused["error"]

    release.set()
    assert (await ws.next_event("complete"))["request_id"] == "r1"
    save.assert_awaited_once()

    ws.disconnect()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_refused_message_does_not_use_up_the_quota(monkeypatch):
    release, state, save = patch_blocking_stream(monkeypatch)
    ws = LiveWebSocket()
    task = start_endpoint(ws)

    ws.push(message_payload("r1"))
    await ws.next_event("chunk")
    for i in range(30):
        ws.push(message_payload(f"x{i}"))
    for _ in range(30):
        await ws.next_event("error")

    release.set()
    await ws.next_event("complete")
    ws.push(message_payload("r2"))
    assert (await ws.next_event())["type"] != "error"

    ws.disconnect()
    release.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_disconnect_cancels_the_reply_in_flight(monkeypatch):
    release, state, save = patch_blocking_stream(monkeypatch)
    ws = LiveWebSocket()
    task = start_endpoint(ws)

    ws.push(message_payload("r1"))
    await ws.next_event("chunk")

    ws.disconnect()
    await asyncio.wait_for(task, timeout=1)
    assert state["cancelled"] is True


@pytest.mark.asyncio
async def test_pings_are_answered_while_a_reply_is_streaming(monkeypatch):
    release, state, save = patch_blocking_stream(monkeypatch)
    ws = LiveWebSocket()
    task = start_endpoint(ws)

    ws.push(message_payload("r1"))
    await ws.next_event("chunk")

    ws.push({"type": "ping"})
    assert await ws.next_event("pong") == {"type": "pong"}

    ws.disconnect()
    await asyncio.wait_for(task, timeout=1)