import json
from types import SimpleNamespace

import grpc
import pytest
from google.genai import types

from backend.generated import chat_pb2
from backend.service import db
from backend.service.auth_interceptor import AuthInterceptor, SESSION_METADATA_KEY
from backend.service.chat_service import ChatServicer


class _Abort(Exception):
    def __init__(self, code, details):
        self.code = code
        super().__init__(details)


class _Context:
    def __init__(self, token):
        self._metadata = ((SESSION_METADATA_KEY, token),)
        self._callbacks = []

    def invocation_metadata(self):
        return self._metadata

    def add_callback(self, callback):
        self._callbacks.append(callback)
        return True

    def abort(self, code, details):
        raise _Abort(code, details)

    def cancel(self):
        for callback in self._callbacks:
            callback()


class _Models:
    def __init__(self, responses, *, failure=None):
        self.responses = responses
        self.failure = failure
        self.inputs = []
        self.stream_calls = []

    def count_tokens(self, *, model, contents):
        self.inputs.append(
            (
                model,
                [
                    part.text
                    for content in contents
                    for part in content.parts or []
                ],
            )
        )
        return SimpleNamespace(total_tokens=1)

    def generate_content_stream(self, *, model, contents, config):
        self.stream_calls.append((model, contents, config))

        def stream():
            yield from self.responses
            if self.failure is not None:
                raise self.failure

        return stream()

    def generate_content(self, **kwargs):
        raise AssertionError(f"Unexpected non-streaming Gemini call: {kwargs!r}")


class _GeminiClient:
    def __init__(self, responses, *, failure=None):
        self.models = _Models(responses, failure=failure)


@pytest.fixture
def chat_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "chat.db"))
    monkeypatch.setenv("DATABASE_KEY", "test-database-key")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv(
        "QDRANT_COLLECTION_PREFIX",
        f"test_{tmp_path.name.replace('-', '_')}",
    )
    db.init_db()
    user_id = db.register_user("stream_user", "correct-horse")
    token = db.create_session(user_id)
    conversation = db.create_conversation(user_id, "New chat")

    def handler_for(client):
        servicer = ChatServicer(gemini_client=client, model="test-model")
        details = type("Details", (), {"method": "/chat.ChatService/StreamMessage"})()
        return AuthInterceptor().intercept_service(
            lambda _: grpc.unary_stream_rpc_method_handler(
                servicer.StreamMessage
            ),
            details,
        )

    return user_id, token, conversation["id"], handler_for


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


def _request(conversation_id, **kwargs):
    return chat_pb2.ChatTurnRequest(
        conversation_id=conversation_id,
        message=kwargs.pop("message", "Hello"),
        request_id=kwargs.pop("request_id", "request-1"),
        **kwargs,
    )


def test_stream_message_emits_chunks_and_persists_the_completed_turn(chat_setup):
    user_id, token, conversation_id, handler_for = chat_setup
    client = _GeminiClient([_chunk("Hello "), _chunk("there.")])
    handler = handler_for(client)

    events = list(
        handler.unary_stream(
            _request(conversation_id, mode="message"),
            _Context(token),
        )
    )

    assert [event.WhichOneof("payload") for event in events] == [
        "started",
        "chunk",
        "chunk",
        "completed",
    ]
    assert [(event.chunk.content, event.chunk.offset) for event in events[1:3]] == [
        ("Hello ", 0),
        ("there.", 6),
    ]
    assert events[-1].completed.full_text == "Hello there."
    assert [
        (row["role"], row["content"])
        for row in db.get_conversation_messages(user_id, conversation_id)
    ] == [
        ("user", "Hello"),
        ("assistant", "Hello there."),
    ]


def test_stream_message_aborts_before_started_for_missing_conversation(chat_setup):
    _, token, conversation_id, handler_for = chat_setup
    handler = handler_for(_GeminiClient([_chunk("unused")]))

    with pytest.raises(_Abort) as error:
        list(
            handler.unary_stream(
                _request(conversation_id + 100, mode="message"),
                _Context(token),
            )
        )

    assert error.value.code == grpc.StatusCode.NOT_FOUND


def test_stream_message_redacts_pre_generation_errors(
    chat_setup, monkeypatch, caplog
):
    _, token, conversation_id, handler_for = chat_setup

    def fail_history_load(*args):
        raise RuntimeError("private database detail")

    monkeypatch.setattr(db, "get_conversation", fail_history_load)
    handler = handler_for(_GeminiClient([]))
    events = list(
        handler.unary_stream(
            _request(conversation_id, mode="message"),
            _Context(token),
        )
    )

    assert len(events) == 1
    assert events[0].WhichOneof("payload") == "error"
    assert events[0].error.message == "Something went wrong. Please try again."
    assert "private database detail" in caplog.text


def test_stream_message_reports_generation_errors_in_band(chat_setup, caplog):
    _, token, conversation_id, handler_for = chat_setup
    client = _GeminiClient(
        [_chunk("Partial text")],
        failure=RuntimeError("model unavailable"),
    )
    handler = handler_for(client)

    events = list(
        handler.unary_stream(
            _request(conversation_id, mode="message"),
            _Context(token),
        )
    )

    assert [event.WhichOneof("payload") for event in events] == [
        "started",
        "chunk",
        "error",
    ]
    assert events[-1].error.message == "Something went wrong. Please try again."
    assert "model unavailable" in caplog.text


def test_stream_message_serializes_tool_results_with_sequential_indexes(
    chat_setup, monkeypatch
):
    user_id, token, conversation_id, handler_for = chat_setup

    def run_turn(message, client, model, history, actual_user_id, request_id, **kwargs):
        assert message == "Use tools"
        assert isinstance(client, _GeminiClient)
        assert model == "test-model"
        assert history == []
        assert actual_user_id == user_id
        assert request_id == "tool-request"
        yield "tool_result", "save_note", {"note_id": 4}
        yield "tool_result", "list_notes", {"notes": []}
        yield "final", "Done."

    monkeypatch.setattr("backend.service.chat_service.gemini_chat.run_turn", run_turn)
    handler = handler_for(_GeminiClient([]))

    events = list(
        handler.unary_stream(
            _request(
                conversation_id,
                message="Use tools",
                request_id="tool-request",
                mode="message",
            ),
            _Context(token),
        )
    )
    tool_events = [
        event.tool_result
        for event in events
        if event.WhichOneof("payload") == "tool_result"
    ]

    assert [(event.index, event.tool) for event in tool_events] == [
        (0, "save_note"),
        (1, "list_notes"),
    ]
    assert [json.loads(event.result_json) for event in tool_events] == [
        {"note_id": 4},
        {"notes": []},
    ]
    assert events[-1].completed.full_text == "Done."


def test_stream_message_saves_partial_text_when_cancelled(chat_setup):
    user_id, token, conversation_id, handler_for = chat_setup
    handler = handler_for(_GeminiClient([_chunk("Partial text"), _chunk("ignored")]))
    context = _Context(token)
    events = handler.unary_stream(
        _request(conversation_id, mode="message"),
        context,
    )

    assert next(events).WhichOneof("payload") == "started"
    assert next(events).chunk.content == "Partial text"
    context.cancel()
    with pytest.raises(GeneratorExit):
        next(events)

    assert [
        (row["role"], row["content"])
        for row in db.get_conversation_messages(user_id, conversation_id)
    ] == [
        ("user", "Hello"),
        ("assistant", "Partial text"),
    ]


def test_stream_message_edit_reuses_the_existing_user_row(chat_setup):
    user_id, token, conversation_id, handler_for = chat_setup
    db.save_conversation(
        user_id, conversation_id, "Original question", "Old answer", "seed"
    )
    user_message_id = db.get_conversation_messages(user_id, conversation_id)[0]["id"]
    client = _GeminiClient([_chunk("Edited answer")])
    handler = handler_for(client)

    events = list(
        handler.unary_stream(
            _request(
                conversation_id,
                message="Edited question",
                request_id="edit-request",
                mode="edit",
                message_id=user_message_id,
            ),
            _Context(token),
        )
    )

    assert events[-1].completed.full_text == "Edited answer"
    assert client.models.inputs[0][1] == ["Edited question"]
    assert [
        (row["role"], row["content"])
        for row in db.get_conversation_messages(user_id, conversation_id)
    ] == [
        ("user", "Edited question"),
        ("assistant", "Edited answer"),
    ]


def test_stream_message_regenerate_replaces_the_target_response(chat_setup):
    user_id, token, conversation_id, handler_for = chat_setup
    db.save_conversation(
        user_id, conversation_id, "First question", "First answer", "first"
    )
    db.save_conversation(
        user_id, conversation_id, "Second question", "Second answer", "second"
    )
    messages = db.get_conversation_messages(user_id, conversation_id)
    first_assistant_id = messages[1]["id"]
    client = _GeminiClient([_chunk("Regenerated answer")])
    handler = handler_for(client)

    events = list(
        handler.unary_stream(
            _request(
                conversation_id,
                message="ignored replacement text",
                request_id="regenerate-request",
                mode="regenerate",
                assistant_message_id=first_assistant_id,
            ),
            _Context(token),
        )
    )

    assert events[-1].completed.full_text == "Regenerated answer"
    assert client.models.inputs[0][1] == ["First question"]
    assert [
        (row["role"], row["content"])
        for row in db.get_conversation_messages(user_id, conversation_id)
    ] == [
        ("user", "First question"),
        ("assistant", "Regenerated answer"),
    ]


# backend/service/test_chat_stream.py  (Review Focus 3)
def test_stop_before_saving_keeps_the_old_answer(chat_setup):
    user_id, token, conversation_id, handler_for = chat_setup
    db.save_conversation(user_id, conversation_id, "Question", "Old answer", "seed")
    assistant_id = db.get_conversation_messages(user_id, conversation_id)[-1]["id"]
    context = _Context(token)
    events = handler_for(_GeminiClient([_chunk("New answer")])).unary_stream(
        _request(conversation_id, request_id="regen", mode="regenerate",
                 assistant_message_id=assistant_id),
        context,
    )

    assert next(events).WhichOneof("payload") == "started"
    assert next(events).chunk.content == "New answer"
    context.cancel()
    with pytest.raises(GeneratorExit):
        next(events)
    assert db.get_conversation_messages(user_id, conversation_id)[-1]["content"] == "Old answer"
