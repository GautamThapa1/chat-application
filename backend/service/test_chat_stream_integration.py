import json
import socket
import threading
import time
from types import SimpleNamespace

import grpc
import pytest
from google.genai import types

from backend.generated import chat_pb2, chat_pb2_grpc
from backend.service import db
from backend.service.auth_interceptor import SESSION_METADATA_KEY
from backend.service.chat_service import create_server


def _chunk(text):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=text)],
                )
            )
        ]
    )


def _tool_call(name, arguments):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name=name,
                                args=arguments,
                            )
                        )
                    ],
                )
            )
        ]
    )


def _text_response(text):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=text)],
                )
            )
        ],
        text=text,
    )


class _Models:
    def __init__(self, chunks=(), *, generated=(), count_tokens=None):
        self.chunks = chunks
        self.generated = iter(generated)
        self._count_tokens = count_tokens
        self.inputs = []

    def count_tokens(self, *, model, contents):
        self.inputs.append(
            [
                part.text
                for content in contents
                for part in content.parts or []
                if part.text is not None
            ]
        )
        if self._count_tokens is not None:
            return self._count_tokens(model=model, contents=contents)
        return SimpleNamespace(total_tokens=1)

    def generate_content_stream(self, *, model, contents, config):
        def stream():
            yield from self.chunks

        return stream()

    def generate_content(self, *, model, contents, config):
        return next(self.generated)


class _Client:
    def __init__(self, chunks=(), *, generated=(), count_tokens=None):
        self.models = _Models(
            chunks,
            generated=generated,
            count_tokens=count_tokens,
        )


@pytest.fixture
def grpc_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "chat.db"))
    monkeypatch.setenv("DATABASE_KEY", "integration-test-key")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv(
        "QDRANT_COLLECTION_PREFIX",
        f"integration_{tmp_path.name.replace('-', '_')}",
    )
    db.init_db()
    user_id = db.register_user("grpc_user", "integration-password")
    token = db.create_session(user_id)
    conversation = db.create_conversation(user_id, "Integration chat")
    metadata = ((SESSION_METADATA_KEY, token),)
    resources = []

    def start_server(client, *, max_workers=4, max_concurrent_streams=40):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server = create_server(
            port=port,
            host="127.0.0.1",
            gemini_client=client,
            model="integration-model",
            max_workers=max_workers,
            max_concurrent_streams=max_concurrent_streams,
        )
        server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        grpc.channel_ready_future(channel).result(timeout=5)
        resources.append((server, channel))
        return chat_pb2_grpc.ChatServiceStub(channel)

    yield user_id, token, conversation["id"], metadata, start_server

    for server, channel in reversed(resources):
        channel.close()
        server.stop(grace=0).wait(timeout=5)


def _request(conversation_id, *, message="Hello", request_id="grpc-request", **kwargs):
    return chat_pb2.ChatTurnRequest(
        conversation_id=conversation_id,
        message=message,
        request_id=request_id,
        **kwargs,
    )


def _wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for the gRPC operation.")


def test_real_grpc_stream_delivers_chunks_and_persists_response(grpc_runtime):
    user_id, _, conversation_id, metadata, start_server = grpc_runtime
    client = _Client([_chunk("Hello "), _chunk("from gRPC.")])
    stub = start_server(client)

    events = list(stub.StreamMessage(
        _request(conversation_id),
        metadata=metadata,
    ))

    assert [event.WhichOneof("payload") for event in events] == [
        "started",
        "chunk",
        "chunk",
        "completed",
    ]
    assert [(event.chunk.content, event.chunk.offset) for event in events[1:3]] == [
        ("Hello ", 0),
        ("from gRPC.", 6),
    ]
    assert events[-1].completed.full_text == "Hello from gRPC."
    assert [
        (row["role"], row["content"])
        for row in db.get_conversation_messages(user_id, conversation_id)
    ] == [
        ("user", "Hello"),
        ("assistant", "Hello from gRPC."),
    ]


def test_real_grpc_stream_executes_save_note_tool_in_database(grpc_runtime):
    user_id, _, conversation_id, metadata, start_server = grpc_runtime
    client = _Client(
        [_tool_call("save_note", {"title": "Plan", "content": "Ship chat"})],
        generated=[_text_response("I saved the plan.")],
    )
    stub = start_server(client)

    events = list(stub.StreamMessage(
        _request(conversation_id, request_id="save-note-request"),
        metadata=metadata,
        timeout=5,
    ))

    tool_events = [
        event.tool_result
        for event in events
        if event.WhichOneof("payload") == "tool_result"
    ]
    notes = db.search_notes(user_id, "")
    assert len(tool_events) == 1
    assert tool_events[0].tool == "save_note"
    assert json.loads(tool_events[0].result_json)["title"] == "Plan"
    assert [(note["title"], note["content"]) for note in notes] == [
        ("Plan", "Ship chat")
    ]
    assert events[-1].completed.full_text == "I saved the plan."


def test_real_grpc_stream_executes_multiple_tools_sequentially(grpc_runtime):
    user_id, _, conversation_id, metadata, start_server = grpc_runtime
    client = _Client(
        [
            _tool_call("save_note", {"title": "First", "content": "One"}),
            _tool_call("save_note", {"title": "Second", "content": "Two"}),
        ],
        generated=[_text_response("Both notes are saved.")],
    )
    stub = start_server(client)

    events = list(stub.StreamMessage(
        _request(conversation_id, request_id="multiple-tools-request"),
        metadata=metadata,
        timeout=5,
    ))

    tool_results = [
        json.loads(event.tool_result.result_json)
        for event in events
        if event.WhichOneof("payload") == "tool_result"
    ]
    assert [result["title"] for result in tool_results] == ["First", "Second"]
    assert [
        (note["title"], note["content"])
        for note in db.search_notes(user_id, "")
    ] == [("First", "One"), ("Second", "Two")]
    assert events[-1].completed.full_text == "Both notes are saved."


def test_invalid_api_key_is_reported_as_an_in_band_grpc_error(grpc_runtime):
    _, _, conversation_id, metadata, start_server = grpc_runtime

    def invalid_key(**kwargs):
        raise RuntimeError("API key not valid. Please pass a valid API key.")

    stub = start_server(_Client(count_tokens=invalid_key))
    events = list(stub.StreamMessage(
        _request(conversation_id),
        metadata=metadata,
    ))

    assert [event.WhichOneof("payload") for event in events] == [
        "started",
        "error",
    ]
    assert events[-1].error.message == "Something went wrong. Please try again."
    assert "API key not valid" not in events[-1].error.message


def test_real_grpc_edit_updates_message_and_discards_later_turns(grpc_runtime):
    user_id, _, conversation_id, metadata, start_server = grpc_runtime
    db.save_conversation(
        user_id, conversation_id, "Original question", "Old answer", "seed-1"
    )
    db.save_conversation(
        user_id, conversation_id, "Later question", "Later answer", "seed-2"
    )
    message_id = db.get_conversation_messages(user_id, conversation_id)[0]["id"]
    client = _Client([_chunk("Edited answer")])
    stub = start_server(client)

    events = list(stub.StreamMessage(
        _request(
            conversation_id,
            message="Edited question",
            request_id="edit-request",
            mode="edit",
            message_id=message_id,
        ),
        metadata=metadata,
    ))

    assert events[-1].completed.full_text == "Edited answer"
    assert client.models.inputs[0] == ["Edited question"]
    assert [
        (row["role"], row["content"])
        for row in db.get_conversation_messages(user_id, conversation_id)
    ] == [
        ("user", "Edited question"),
        ("assistant", "Edited answer"),
    ]


def test_real_grpc_regenerate_replaces_target_and_removes_later_turns(grpc_runtime):
    user_id, _, conversation_id, metadata, start_server = grpc_runtime
    db.save_conversation(
        user_id, conversation_id, "First question", "First answer", "seed-1"
    )
    db.save_conversation(
        user_id, conversation_id, "Second question", "Second answer", "seed-2"
    )
    messages = db.get_conversation_messages(user_id, conversation_id)
    assistant_id = messages[1]["id"]
    client = _Client([_chunk("Regenerated answer")])
    stub = start_server(client)

    events = list(stub.StreamMessage(
        _request(
            conversation_id,
            message="",
            request_id="regenerate-request",
            mode="regenerate",
            assistant_message_id=assistant_id,
        ),
        metadata=metadata,
    ))

    assert events[-1].completed.full_text == "Regenerated answer"
    assert client.models.inputs[0] == ["First question"]
    assert [
        (row["role"], row["content"])
        for row in db.get_conversation_messages(user_id, conversation_id)
    ] == [
        ("user", "First question"),
        ("assistant", "Regenerated answer"),
    ]


def test_real_grpc_cancel_saves_partial_after_blocking_gemini_read_returns(
    grpc_runtime,
):
    user_id, _, conversation_id, metadata, start_server = grpc_runtime
    read_blocked = threading.Event()
    release_read = threading.Event()
    stream_closed = threading.Event()

    def generate_chunks():
        try:
            yield _chunk("Partial response")
            read_blocked.set()
            release_read.wait(timeout=5)
            yield _chunk("This chunk is discarded")
        finally:
            stream_closed.set()

    client = _Client(generate_chunks())
    stub = start_server(client)
    call = stub.StreamMessage(
        _request(conversation_id, request_id="cancel-request"),
        metadata=metadata,
    )

    assert next(call).WhichOneof("payload") == "started"
    first_chunk = next(call)
    assert first_chunk.chunk.content == "Partial response"
    assert read_blocked.wait(timeout=2)

    assert call.cancel()
    assert not stream_closed.wait(timeout=0.1)
    release_read.set()
    assert stream_closed.wait(timeout=5)
    _wait_for(lambda: len(db.get_conversation_messages(user_id, conversation_id)) == 2)

    assert [
        (row["role"], row["content"])
        for row in db.get_conversation_messages(user_id, conversation_id)
    ] == [
        ("user", "Hello"),
        ("assistant", "Partial response"),
    ]


def test_stream_limit_fails_fast_with_resource_exhausted(grpc_runtime):
    _, _, conversation_id, metadata, start_server = grpc_runtime
    read_blocked = threading.Event()
    release_read = threading.Event()
    stream_closed = threading.Event()

    def generate_chunks():
        try:
            yield _chunk("Holding the stream slot")
            read_blocked.set()
            release_read.wait(timeout=5)
            yield _chunk("Final")
        finally:
            stream_closed.set()

    stub = start_server(
        _Client(generate_chunks()),
        max_workers=2,
        max_concurrent_streams=1,
    )
    first_call = stub.StreamMessage(
        _request(conversation_id, request_id="stream-1"),
        metadata=metadata,
    )
    assert next(first_call).WhichOneof("payload") == "started"
    assert next(first_call).chunk.content == "Holding the stream slot"
    assert read_blocked.wait(timeout=2)

    second_call = stub.StreamMessage(
        _request(conversation_id, request_id="stream-2"),
        metadata=metadata,
    )
    with pytest.raises(grpc.RpcError) as error:
        list(second_call)
    assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED

    first_call.cancel()
    release_read.set()
    assert stream_closed.wait(timeout=5)


def test_real_grpc_regenerate_changes_no_data(grpc_runtime):
    user_id, _, conversation_id, metadata, start_server = grpc_runtime
    db.save_conversation(user_id, conversation_id, "add milk", "Added.", "seed")
    assistant_id = db.get_conversation_messages(user_id, conversation_id)[-1]["id"]
    client = _Client(
        [_tool_call("create_task", {"title": "milk"})],
        generated=[_text_response("Send a new message to add it.")],
    )
    stub = start_server(client)

    events = list(stub.StreamMessage(
        _request(conversation_id, request_id="regen", mode="regenerate",
                 assistant_message_id=assistant_id),
        metadata=metadata,
    ))

    assert events[-1].WhichOneof("payload") == "completed"
    assert db.get_tasks(user_id) == []


def test_real_grpc_edit_cannot_reuse_or_forge_a_confirmation(grpc_runtime):
    user_id, _, conversation_id, metadata, start_server = grpc_runtime
    prompt = "To confirm deleting all notes, reply exactly `confirm clear notes abc`."
    db.save_conversation(user_id, conversation_id, "clear notes", prompt, "seed")
    db.save_note(user_id, "keep", "body")

    def send(request_id, **kwargs):
        client = _Client(
            [_tool_call("clear_notes", {})],
            generated=[_text_response("Please confirm.")],
        )
        return list(start_server(client).StreamMessage(
            _request(conversation_id, request_id=request_id, **kwargs),
            metadata=metadata,
        ))

    # A prompt with no server record (saved before this change).
    send("old", message="confirm clear notes abc")
    assert len(db.search_notes(user_id, "")) == 1

    # An edit may not confirm, even with a valid record.
    db.create_pending_confirmation(user_id, conversation_id, "clear_notes", "abc")
    edited_id = db.get_conversation_messages(user_id, conversation_id)[2]["id"]
    send("edit", message="confirm clear notes abc", mode="edit", message_id=edited_id)
    assert len(db.search_notes(user_id, "")) == 1
