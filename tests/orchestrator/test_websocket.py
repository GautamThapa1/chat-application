import asyncio
import json
from unittest.mock import Mock

import grpc
import pytest
from fastapi import WebSocketDisconnect

from backend.generated import chat_pb2
from backend.orchestrator import websocket as websocket_module
from backend.orchestrator.websocket import hub
from shared_common.common.log import get_trace_id


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent_messages = []
        self.close_codes = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        await asyncio.sleep(0.01)
        if self.messages:
            return self.messages.pop(0)
        raise WebSocketDisconnect(code=1000)

    async def send_json(self, data):
        self.sent_messages.append(data)

    async def close(self, code=1000):
        self.close_codes.append(code)


class LiveWebSocket:
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
        while True:
            event = await asyncio.wait_for(self.outgoing.get(), timeout=1)
            if wanted is None or event["type"] == wanted:
                return event


class FakeCall:
    def __init__(self, events):
        self._events = events
        self.cancelled = False

    def __aiter__(self):
        return self._events.__aiter__()

    def cancel(self):
        self.cancelled = True


class FakeStub:
    def __init__(self, stream_factory=None):
        self.requests = []
        self.calls = []
        self.trace_ids = []
        if stream_factory is None:
            stream_factory = lambda request: events(
                started(), chunk("ok", 0), completed("ok")
            )

        def stream_message(request):
            self.requests.append(request)
            self.trace_ids.append(get_trace_id())
            call = FakeCall(stream_factory(request))
            self.calls.append(call)
            return call

        self.StreamMessage = Mock(side_effect=stream_message)
        self.saved = []
        self.save_error = None

    async def SaveConversation(self, request, timeout=None):
        if self.save_error is not None:
            raise self.save_error
        self.saved.append(request)


def started():
    return chat_pb2.ChatEvent(started=chat_pb2.Started())


def chunk(content, offset=0):
    return chat_pb2.ChatEvent(
        chunk=chat_pb2.TextChunk(content=content, offset=offset)
    )


def tool_result(index, tool, result):
    return chat_pb2.ChatEvent(
        tool_result=chat_pb2.ToolResult(
            index=index,
            tool=tool,
            result_json=json.dumps(result),
        )
    )


def completed(text):
    return chat_pb2.ChatEvent(
        completed=chat_pb2.Completed(full_text=text)
    )


def chat_error(message):
    return chat_pb2.ChatEvent(error=chat_pb2.ChatError(message=message))


def events(*items):
    async def generate():
        for item in items:
            yield item

    return generate()


def create_message(content, request_id="request-1", conversation_id=3, **fields):
    return json.dumps({
        "type": "message",
        "content": content,
        "request_id": request_id,
        "conversation_id": conversation_id,
        **fields,
    })


async def wait_until(condition, timeout=1):
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


def start_endpoint(websocket, stub, user_id=1):
    return asyncio.create_task(
        websocket_module.websocket_endpoint(websocket, stub, user_id=user_id)
    )


@pytest.fixture(autouse=True)
def clear_hub():
    hub.reset()
    yield
    hub.reset()


@pytest.mark.asyncio
async def test_websocket_maps_stream_events_and_completed_suffix():
    result = {"note_id": 5, "status": "saved"}
    stub = FakeStub(lambda request: events(
        started(),
        chunk("Hello ", 0),
        tool_result(0, "save_note", result),
        chunk("world", 6),
        completed("Hello world!"),
    ))
    websocket = FakeWebSocket([create_message("Hello")])

    await websocket_module.websocket_endpoint(websocket, stub, user_id=1)
    await wait_until(lambda: hub.get(1, 3) is None)

    assert websocket.accepted
    assert [event["type"] for event in websocket.sent_messages] == [
        "start", "chunk", "tool_result", "chunk", "chunk", "complete",
    ]
    assert websocket.sent_messages[2] == {
        "type": "tool_result",
        "conversation_id": 3,
        "request_id": "request-1",
        "index": 0,
        "tool": "save_note",
        "result": result,
    }
    assert [
        (event["content"], event["offset"])
        for event in websocket.sent_messages
        if event["type"] == "chunk"
    ] == [("Hello ", 0), ("world", 6), ("!", 11)]
    request = stub.requests[0]
    assert request.conversation_id == 3
    assert request.message == "Hello"
    assert request.request_id == "request-1"
    assert request.mode == "message"


@pytest.mark.asyncio
async def test_in_band_stream_error_is_redacted_after_partial_text(caplog):
    stub = FakeStub(lambda request: events(
        started(), chunk("partial", 0), chat_error("generation failed")
    ))
    websocket = FakeWebSocket([create_message("Hello")])

    await websocket_module.websocket_endpoint(websocket, stub, user_id=1)
    await wait_until(lambda: hub.get(1, 3) is None)

    assert [event["type"] for event in websocket.sent_messages] == [
        "start", "chunk", "error",
    ]
    assert websocket.sent_messages[-1]["error"] == (
        "Something went wrong. Please try again."
    )
    assert "generation failed" not in caplog.text


@pytest.mark.asyncio
async def test_grpc_stream_error_redacts_rpc_details(caplog):
    failure = grpc.aio.AioRpcError(
        grpc.StatusCode.NOT_FOUND,
        grpc.aio.Metadata(),
        grpc.aio.Metadata(),
        details="Conversation not found.",
    )

    async def stream(request):
        yield started()
        raise failure

    stub = FakeStub(lambda request: stream(request))
    websocket = FakeWebSocket([create_message("Hello")])

    await websocket_module.websocket_endpoint(websocket, stub, user_id=1)
    await wait_until(lambda: hub.get(1, 3) is None)

    assert [event["type"] for event in websocket.sent_messages] == [
        "start", "error",
    ]
    assert websocket.sent_messages[-1]["error"] == (
        "Something went wrong. Please try again."
    )
    assert "Conversation not found." not in caplog.text


@pytest.mark.asyncio
async def test_stream_without_terminal_event_reports_error():
    stub = FakeStub(lambda request: events(started(), chunk("partial", 0)))
    websocket = FakeWebSocket([create_message("Hello")])

    await websocket_module.websocket_endpoint(websocket, stub, user_id=1)
    await wait_until(lambda: hub.get(1, 3) is None)

    assert websocket.sent_messages[-1]["type"] == "error"
    assert websocket.sent_messages[-1]["error"] == (
        "Something went wrong. Please try again."
    )


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            grpc.StatusCode.UNAVAILABLE,
            "The assistant is temporarily unavailable. Please try again.",
        ),
        (
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "The assistant is busy right now. Please try again shortly.",
        ),
        (
            grpc.StatusCode.INTERNAL,
            "Something went wrong. Please try again.",
        ),
    ],
)
def test_grpc_status_uses_a_friendly_error_message(code, message):
    assert websocket_module._friendly_grpc_error(code) == message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "message", "fields"),
    [
        ("edit", "edited prompt", {"message_id": 8}),
        ("regenerate", "", {"assistant_message_id": 12}),
    ],
)
async def test_websocket_passes_edit_and_regenerate_fields(mode, message, fields):
    stub = FakeStub()
    websocket = FakeWebSocket([
        create_message(message, mode=mode, **fields),
    ])

    await websocket_module.websocket_endpoint(websocket, stub, user_id=1)
    await wait_until(lambda: hub.get(1, 3) is None)

    request = stub.requests[0]
    assert request.mode == mode
    assert request.message == message
    assert request.message_id == fields.get("message_id", 0)
    assert request.assistant_message_id == fields.get("assistant_message_id", 0)


@pytest.mark.asyncio
async def test_explicit_cancel_cancels_the_grpc_stream():
    parked = asyncio.Event()
    release = asyncio.Event()

    async def stream(request):
        yield started()
        yield chunk("partial", 0)
        parked.set()
        await release.wait()
        yield completed("partial response")

    stub = FakeStub(lambda request: stream(request))
    websocket = LiveWebSocket()
    task = start_endpoint(websocket, stub)
    websocket.push(json.loads(create_message("Hello")))

    assert (await websocket.next_event("chunk"))["content"] == "partial"
    await asyncio.wait_for(parked.wait(), timeout=1)
    websocket.push({"type": "cancel", "request_id": "request-1"})

    cancelled = await websocket.next_event("cancelled")
    assert cancelled["request_id"] == "request-1"
    assert stub.calls[0].cancelled
    assert hub.get(1, 3) is None

    websocket.disconnect()
    await asyncio.wait_for(task, timeout=1)


async def stop_reply(stub, websocket, *, wait_for_chunk=True, **fields):
    """Send a message, stop it, and return what was saved when "cancelled" arrived."""
    task = start_endpoint(websocket, stub)
    websocket.push(json.loads(create_message("Hello", **fields)))
    await websocket.next_event("chunk" if wait_for_chunk else "start")
    websocket.push({"type": "cancel", "request_id": "request-1"})
    await websocket.next_event("cancelled")
    saved = list(stub.saved)
    websocket.disconnect()
    await asyncio.wait_for(task, timeout=1)
    return saved


def endless_stream(*first_events):
    async def stream(request):
        for event in first_events:
            yield event
        await asyncio.Event().wait()

    return FakeStub(lambda request: stream(request))


@pytest.mark.asyncio
async def test_cancel_saves_the_partial_reply_before_announcing_it():
    stub = endless_stream(started(), chunk("partial", 0))

    saved = await stop_reply(stub, LiveWebSocket())

    assert [
        (r.conversation_id, r.user_content, r.assistant_content, r.request_id)
        for r in saved
    ] == [(3, "Hello", "partial", "request-1")]


@pytest.mark.asyncio
async def test_cancel_before_any_text_saves_nothing():
    stub = endless_stream(started())

    assert await stop_reply(stub, LiveWebSocket(), wait_for_chunk=False) == []


@pytest.mark.asyncio
async def test_cancelled_regeneration_keeps_the_old_answer():
    stub = endless_stream(started(), chunk("partial", 0))

    saved = await stop_reply(
        stub, LiveWebSocket(), mode="regenerate", assistant_message_id=12
    )

    assert saved == []


@pytest.mark.asyncio
async def test_cancel_is_announced_even_if_saving_fails():
    stub = endless_stream(started(), chunk("partial", 0))
    stub.save_error = RuntimeError("database is down")

    assert await stop_reply(stub, LiveWebSocket()) == []
    assert hub.get(1, 3) is None


@pytest.mark.asyncio
async def test_disconnect_does_not_cancel_generation_and_sync_resumes_events():
    parked = asyncio.Event()
    release = asyncio.Event()

    async def stream(request):
        yield started()
        yield chunk("Hel", 0)
        parked.set()
        await release.wait()
        yield chunk("lo", 3)
        yield completed("Hello")

    stub = FakeStub(lambda request: stream(request))
    first = LiveWebSocket()
    first_task = start_endpoint(first, stub)
    first.push(json.loads(create_message("Hello")))
    await first.next_event("chunk")
    await asyncio.wait_for(parked.wait(), timeout=1)
    first.disconnect()
    await asyncio.wait_for(first_task, timeout=1)

    assert not stub.calls[0].cancelled
    generation = hub.get(1, 3)
    assert generation is not None
    assert generation.text == "Hel"

    second = LiveWebSocket()
    second_task = start_endpoint(second, stub)
    second.push({"type": "sync"})
    active = await second.next_event("active")
    assert active["generations"][0]["content"] == "Hel"

    release.set()
    assert (await second.next_event("chunk"))["content"] == "lo"
    assert (await second.next_event("complete"))["request_id"] == "request-1"
    assert hub.get(1, 3) is None

    second.disconnect()
    await asyncio.wait_for(second_task, timeout=1)


@pytest.mark.asyncio
async def test_invalid_payload_does_not_call_stream():
    websocket = FakeWebSocket([json.dumps({
        "type": "message",
        "content": 123,
        "request_id": "bad",
        "conversation_id": 3,
    })])
    stub = FakeStub()

    await websocket_module.websocket_endpoint(websocket, stub, user_id=1)

    assert websocket.sent_messages == [{
        "type": "error",
        "request_id": "bad",
        "error": "Invalid message payload.",
    }]
    stub.StreamMessage.assert_not_called()


@pytest.mark.asyncio
async def test_session_is_revalidated_for_each_message():
    websocket = FakeWebSocket([create_message("Hello")])
    revalidations = 0

    async def revalidate():
        nonlocal revalidations
        revalidations += 1
        return 1

    await websocket_module.websocket_endpoint(
        websocket,
        FakeStub(),
        user_id=1,
        revalidate_session=revalidate,
    )
    await wait_until(lambda: hub.get(1, 3) is None)

    assert revalidations == 1


@pytest.mark.asyncio
async def test_expired_session_closes_websocket_before_stream_call():
    async def revalidate():
        return None

    websocket = FakeWebSocket([create_message("Hello")])
    stub = FakeStub()
    await websocket_module.websocket_endpoint(
        websocket,
        stub,
        user_id=1,
        revalidate_session=revalidate,
    )

    assert websocket.close_codes == [1008]
    stub.StreamMessage.assert_not_called()


@pytest.mark.asyncio
async def test_conversation_is_released_after_completion():
    gate = asyncio.Event()

    async def stream(request):
        yield started()
        await gate.wait()
        yield completed("done")

    stub = FakeStub(lambda request: stream(request))
    websocket = LiveWebSocket()
    task = start_endpoint(websocket, stub)
    websocket.push(json.loads(create_message("first", "r1")))
    await websocket.next_event("start")

    websocket.push(json.loads(create_message("second", "r2")))
    refused = await websocket.next_event("error")
    assert refused["request_id"] == "r2"
    assert "previous message" in refused["error"]
    assert len(stub.requests) == 1

    gate.set()
    assert (await websocket.next_event("complete"))["request_id"] == "r1"
    websocket.push(json.loads(create_message("second", "r2")))
    assert (await websocket.next_event("start"))["request_id"] == "r2"
    await wait_until(lambda: len(stub.requests) == 2)

    websocket.disconnect()
    await asyncio.wait_for(task, timeout=1)
    await wait_until(lambda: hub.get(1, 3) is None)


@pytest.mark.asyncio
async def test_concurrent_generations_have_independent_trace_ids():
    stub = FakeStub()
    websocket = LiveWebSocket()
    task = start_endpoint(websocket, stub)
    websocket.push(json.loads(create_message(
        "first", "r1", conversation_id=3,
    )))
    websocket.push(json.loads(create_message(
        "second", "r2", conversation_id=4,
    )))

    await wait_until(lambda: len(stub.trace_ids) == 2)
    assert len(set(stub.trace_ids)) == 2
    assert "-" not in stub.trace_ids

    websocket.disconnect()
    await asyncio.wait_for(task, timeout=1)
    await wait_until(lambda: hub.get(1, 3) is None and hub.get(1, 4) is None)
    assert get_trace_id() == "-"
