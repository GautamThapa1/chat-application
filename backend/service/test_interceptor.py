"""The service must reject unauthenticated calls and know who is calling."""

import socket

import grpc
import pytest

import backend.service.db as db
from backend.generated import chat_pb2, chat_pb2_grpc
from backend.service import auth_interceptor, chat_service
from backend.service.auth_interceptor import (
    SESSION_METADATA_KEY,
    AuthInterceptor,
    current_user_id,
)


@pytest.fixture
def stub(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    server = chat_service.create_server(port=port)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield chat_pb2_grpc.ChatServiceStub(channel)
    finally:
        channel.close()
        server.stop(grace=None)


def _token():
    user_id = db.register_user("alice", "correct-horse")
    return db.create_session(user_id)


def _note_request():
    return chat_pb2.SaveNoteRequest(title="t", content="c")


def test_protected_call_without_token_is_rejected(stub):
    with pytest.raises(grpc.RpcError) as err:
        stub.SaveNote(_note_request())
    assert err.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_protected_call_with_garbage_token_is_rejected(stub):
    with pytest.raises(grpc.RpcError) as err:
        stub.SaveNote(
            _note_request(), metadata=((SESSION_METADATA_KEY, "garbage"),)
        )
    assert err.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_protected_call_with_valid_token_succeeds(stub):
    response = stub.SaveNote(
        _note_request(), metadata=((SESSION_METADATA_KEY, _token()),)
    )
    assert response.note.id > 0


def test_public_methods_work_without_a_token(stub):
    registered = stub.Register(
        chat_pb2.RegisterRequest(username="bob", password="correct-horse")
    )
    login = stub.Login(
        chat_pb2.LoginRequest(username="bob", password="correct-horse")
    )
    assert registered.user_id == login.user_id
    assert login.session_token


class _FakeContext:
    def __init__(self, token):
        self._metadata = ((SESSION_METADATA_KEY, token),)

    def invocation_metadata(self):
        return self._metadata

    def abort(self, code, details):
        raise RuntimeError(f"aborted: {code}")


def test_handler_can_see_authenticated_user_id(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    user_id = db.register_user("alice", "correct-horse")
    token = db.create_session(user_id)

    seen = []

    def inner(request, context):
        seen.append(current_user_id())
        return "ok"

    details = type("D", (), {"method": "/chat.ChatService/SaveNote"})()
    handler = AuthInterceptor().intercept_service(
        lambda _: grpc.unary_unary_rpc_method_handler(inner), details
    )

    assert handler.unary_unary(object(), _FakeContext(token)) == "ok"
    assert seen == [user_id]
    with pytest.raises(RuntimeError):
        current_user_id()


def test_unknown_methods_are_protected_by_default():
    assert "/chat.ChatService/SaveNote" not in auth_interceptor.PUBLIC_METHODS
    assert "/chat.ChatService/ListTasks" not in auth_interceptor.PUBLIC_METHODS
