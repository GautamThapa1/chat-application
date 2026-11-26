"""The service must reject unauthenticated calls and know who is calling."""

import socket
from concurrent import futures

import grpc
import pytest

from backend.embeddings import note_content_hash
from backend.orchestrator.grpc_session import SessionStub
import backend.service.db as db
from backend.generated import chat_pb2, chat_pb2_grpc
from backend.service import auth_interceptor, chat_service
from backend.service.auth_interceptor import (
    SESSION_METADATA_KEY,
    AuthInterceptor,
    current_user_id,
)
from shared_common.common.log import get_trace_id, trace_context


@pytest.fixture
def stub(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DATABASE_KEY", "test-database-key")
    # RPC tests need Qdrant too, but should remain independent of a developer's
    # local Docker service or persisted vector data.
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv(
        "QDRANT_COLLECTION_PREFIX",
        f"test_{tmp_path.name.replace('-', '_')}",
    )
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


def test_conversations_notes_and_tasks_are_isolated(stub):
    stub.Register(
        chat_pb2.RegisterRequest(username="alice", password="correct-horse")
    )
    stub.Register(
        chat_pb2.RegisterRequest(username="bob", password="another-horse")
    )
    token_a = stub.Login(
        chat_pb2.LoginRequest(username="alice", password="correct-horse")
    ).session_token
    token_b = stub.Login(
        chat_pb2.LoginRequest(username="bob", password="another-horse")
    ).session_token
    metadata_a = ((SESSION_METADATA_KEY, token_a),)
    metadata_b = ((SESSION_METADATA_KEY, token_b),)

    conversation_a1 = stub.CreateConversation(
        chat_pb2.CreateConversationRequest(title="Alice one"),
        metadata=metadata_a,
    ).conversation
    conversation_a2 = stub.CreateConversation(
        chat_pb2.CreateConversationRequest(title="Alice two"),
        metadata=metadata_a,
    ).conversation
    stub.SaveConversation(
        chat_pb2.SaveConversationRequest(
            user_content="A's first question",
            assistant_content="A's first answer",
            conversation_id=conversation_a1.conversation_id,
        ),
        metadata=metadata_a,
    )
    stub.SaveConversation(
        chat_pb2.SaveConversationRequest(
            user_content="A's second question",
            assistant_content="A's second answer",
            conversation_id=conversation_a2.conversation_id,
        ),
        metadata=metadata_a,
    )
    note = stub.SaveNote(
        chat_pb2.SaveNoteRequest(title="Alice private", content="secret"),
        metadata=metadata_a,
    )
    task = stub.CreateTask(
        chat_pb2.CreateTaskRequest(title="Alice task"),
        metadata=metadata_a,
    )
    stub.Logout(chat_pb2.LogoutRequest(session_token=token_a))

    assert not stub.ListConversations(
        chat_pb2.ListConversationsRequest(), metadata=metadata_b
    ).conversations
    with pytest.raises(grpc.RpcError) as get_error:
        stub.GetConversation(
            chat_pb2.GetConversationRequest(
                conversation_id=conversation_a1.conversation_id
            ),
            metadata=metadata_b,
        )
    assert get_error.value.code() == grpc.StatusCode.NOT_FOUND
    with pytest.raises(grpc.RpcError) as delete_error:
        stub.DeleteConversation(
            chat_pb2.DeleteConversationRequest(
                conversation_id=conversation_a1.conversation_id
            ),
            metadata=metadata_b,
        )
    assert delete_error.value.code() == grpc.StatusCode.NOT_FOUND
    with pytest.raises(grpc.RpcError) as save_error:
        stub.SaveConversation(
            chat_pb2.SaveConversationRequest(
                user_content="intrusion",
                assistant_content="intrusion",
                conversation_id=conversation_a1.conversation_id,
            ),
            metadata=metadata_b,
        )
    assert save_error.value.code() == grpc.StatusCode.NOT_FOUND

    conversation_b = stub.CreateConversation(
        chat_pb2.CreateConversationRequest(title="Bob's chat"),
        metadata=metadata_b,
    ).conversation
    stub.SaveConversation(
        chat_pb2.SaveConversationRequest(
            user_content="B's question",
            assistant_content="B's answer",
            conversation_id=conversation_b.conversation_id,
        ),
        metadata=metadata_b,
    )
    assert not stub.SearchNotes(
        chat_pb2.SearchNotesRequest(keyword="secret"), metadata=metadata_b
    ).notes
    assert stub.ClearNotes(
        chat_pb2.ClearNotesRequest(), metadata=metadata_b
    ).deleted_count == 0
    assert not stub.ListTasks(
        chat_pb2.ListTasksRequest(), metadata=metadata_b
    ).tasks
    assert not stub.CompleteTask(
        chat_pb2.CompleteTaskRequest(task_id=task.task.id),
        metadata=metadata_b,
    ).success
    assert not stub.DeleteTask(
        chat_pb2.DeleteTaskRequest(task_id=task.task.id),
        metadata=metadata_b,
    ).success
    assert stub.ClearTasks(
        chat_pb2.ClearTasksRequest(), metadata=metadata_b
    ).deleted_count == 0

    token_a = stub.Login(
        chat_pb2.LoginRequest(username="alice", password="correct-horse")
    ).session_token
    metadata_a = ((SESSION_METADATA_KEY, token_a),)
    alice_history = stub.GetConversation(
        chat_pb2.GetConversationRequest(
            conversation_id=conversation_a1.conversation_id
        ),
        metadata=metadata_a,
    )
    assert [message.content for message in alice_history.messages] == [
        "A's first question",
        "A's first answer",
    ]
    assert [item.title for item in stub.ListConversations(
        chat_pb2.ListConversationsRequest(), metadata=metadata_a
    ).conversations] == ["Alice two", "Alice one"]
    assert stub.SearchNotes(
        chat_pb2.SearchNotesRequest(keyword="secret"), metadata=metadata_a
    ).notes[0].id == note.note.id
    assert stub.ClearNotes(
        chat_pb2.ClearNotesRequest(), metadata=metadata_a
    ).deleted_count == 1
    assert not stub.SearchNotes(
        chat_pb2.SearchNotesRequest(keyword="secret"), metadata=metadata_a
    ).notes
    assert stub.ListTasks(
        chat_pb2.ListTasksRequest(), metadata=metadata_a
    ).tasks[0].id == task.task.id
    # Bob's attempts above did not touch Alice's task; she can delete it herself.
    assert stub.DeleteTask(
        chat_pb2.DeleteTaskRequest(task_id=task.task.id), metadata=metadata_a
    ).success
    assert not stub.ListTasks(
        chat_pb2.ListTasksRequest(), metadata=metadata_a
    ).tasks


def test_note_embedding_rpcs_enforce_owner_and_support_semantic_search(stub):
    stub.Register(
        chat_pb2.RegisterRequest(username="alice", password="correct-horse")
    )
    stub.Register(
        chat_pb2.RegisterRequest(username="bob", password="another-horse")
    )
    token_a = stub.Login(
        chat_pb2.LoginRequest(username="alice", password="correct-horse")
    ).session_token
    token_b = stub.Login(
        chat_pb2.LoginRequest(username="bob", password="another-horse")
    ).session_token
    metadata_a = ((SESSION_METADATA_KEY, token_a),)
    metadata_b = ((SESSION_METADATA_KEY, token_b),)
    note = stub.SaveNote(
        chat_pb2.SaveNoteRequest(title="Garden", content="Grow tomatoes"),
        metadata=metadata_a,
    ).note

    with pytest.raises(grpc.RpcError) as update_error:
        stub.UpdateNote(
            chat_pb2.UpdateNoteRequest(
                note_id=note.id,
                title="Stolen",
                content="Private",
            ),
            metadata=metadata_b,
        )
    assert update_error.value.code() == grpc.StatusCode.NOT_FOUND

    request = chat_pb2.UpsertNoteEmbeddingRequest(
        note_id=note.id,
        embedding=[1.0, 0.0],
        embedding_model="test-model",
        content_hash=note_content_hash("Garden", "Grow tomatoes"),
    )
    assert not stub.UpsertNoteEmbedding(request, metadata=metadata_b).indexed
    assert not stub.SearchNotesHybrid(
        chat_pb2.SearchNotesHybridRequest(
            embedding=[1.0, 0.0],
            embedding_model="test-model",
        ),
        metadata=metadata_b,
    ).notes
    assert stub.UpsertNoteEmbedding(request, metadata=metadata_a).indexed
    assert stub.SearchNotesHybrid(
        chat_pb2.SearchNotesHybridRequest(
            embedding=[1.0, 0.0],
            embedding_model="test-model",
        ),
        metadata=metadata_a,
    ).notes[0].id == note.id

    assert not stub.DeleteNote(
        chat_pb2.DeleteNoteRequest(note_id=note.id),
        metadata=metadata_b,
    ).success
    assert stub.SearchNotes(chat_pb2.SearchNotesRequest(), metadata=metadata_a).notes
    assert stub.DeleteNote(
        chat_pb2.DeleteNoteRequest(note_id=note.id),
        metadata=metadata_a,
    ).success
    assert not stub.SearchNotesHybrid(
        chat_pb2.SearchNotesHybridRequest(
            embedding=[1.0, 0.0],
            embedding_model="test-model",
        ),
        metadata=metadata_a,
    ).notes


def test_memory_rpcs_are_separate_owner_scoped_and_editable(stub):
    stub.Register(
        chat_pb2.RegisterRequest(username="alice", password="correct-horse")
    )
    stub.Register(
        chat_pb2.RegisterRequest(username="bob", password="another-horse")
    )
    token_a = stub.Login(
        chat_pb2.LoginRequest(username="alice", password="correct-horse")
    ).session_token
    token_b = stub.Login(
        chat_pb2.LoginRequest(username="bob", password="another-horse")
    ).session_token
    metadata_a = ((SESSION_METADATA_KEY, token_a),)
    metadata_b = ((SESSION_METADATA_KEY, token_b),)
    memory = stub.SaveMemory(
        chat_pb2.SaveMemoryRequest(
            title="Answer style",
            content="Prefers concise answers",
        ),
        metadata=metadata_a,
    ).memory
    note = stub.SaveNote(
        chat_pb2.SaveNoteRequest(
            title="Answer style",
            content="Meeting notes about concise writing",
        ),
        metadata=metadata_a,
    ).note

    assert stub.ListMemories(
        chat_pb2.ListMemoriesRequest(), metadata=metadata_a
    ).memories[0].id == memory.id
    assert stub.SearchMemories(
        chat_pb2.SearchMemoriesRequest(keyword="concise"),
        metadata=metadata_a,
    ).memories[0].id == memory.id
    assert stub.SearchNotes(
        chat_pb2.SearchNotesRequest(keyword="concise"),
        metadata=metadata_a,
    ).notes[0].id == note.id
    assert not stub.SearchMemories(
        chat_pb2.SearchMemoriesRequest(keyword="meeting"),
        metadata=metadata_a,
    ).memories

    embedding = [1.0, 0.0]
    request = chat_pb2.UpsertMemoryEmbeddingRequest(
        memory_id=memory.id,
        embedding=embedding,
        embedding_model="test-model",
        content_hash=note_content_hash("Answer style", "Prefers concise answers"),
    )
    assert not stub.UpsertMemoryEmbedding(
        request, metadata=metadata_b
    ).indexed
    assert stub.UpsertMemoryEmbedding(request, metadata=metadata_a).indexed
    assert stub.SearchMemoriesHybrid(
        chat_pb2.SearchMemoriesHybridRequest(
            embedding=embedding,
            embedding_model="test-model",
        ),
        metadata=metadata_a,
    ).memories[0].id == memory.id

    with pytest.raises(grpc.RpcError) as update_error:
        stub.UpdateMemory(
            chat_pb2.UpdateMemoryRequest(
                memory_id=memory.id,
                title="Stolen",
                content="Private",
            ),
            metadata=metadata_b,
        )
    assert update_error.value.code() == grpc.StatusCode.NOT_FOUND
    updated = stub.UpdateMemory(
        chat_pb2.UpdateMemoryRequest(
            memory_id=memory.id,
            title="Answer style",
            content="Now prefers detailed answers",
        ),
        metadata=metadata_a,
    ).memory
    assert updated.content == "Now prefers detailed answers"
    assert stub.UpsertMemoryEmbedding(
        chat_pb2.UpsertMemoryEmbeddingRequest(
            memory_id=memory.id,
            embedding=embedding,
            embedding_model="test-model",
            content_hash=note_content_hash(
                "Answer style", "Now prefers detailed answers"
            ),
        ),
        metadata=metadata_a,
    ).indexed
    assert not stub.DeleteMemory(
        chat_pb2.DeleteMemoryRequest(memory_id=memory.id),
        metadata=metadata_b,
    ).success
    assert stub.DeleteMemory(
        chat_pb2.DeleteMemoryRequest(memory_id=memory.id),
        metadata=metadata_a,
    ).success
    assert not stub.SearchMemories(
        chat_pb2.SearchMemoriesRequest(keyword="concise"),
        metadata=metadata_a,
    ).memories


def test_conversation_endpoints_require_a_valid_session(stub):
    with pytest.raises(grpc.RpcError) as missing:
        stub.ListConversations(chat_pb2.ListConversationsRequest())
    assert missing.value.code() == grpc.StatusCode.UNAUTHENTICATED

    with pytest.raises(grpc.RpcError) as clear_notes:
        stub.ClearNotes(chat_pb2.ClearNotesRequest())
    assert clear_notes.value.code() == grpc.StatusCode.UNAUTHENTICATED

    with pytest.raises(grpc.RpcError) as clear_tasks:
        stub.ClearTasks(chat_pb2.ClearTasksRequest())
    assert clear_tasks.value.code() == grpc.StatusCode.UNAUTHENTICATED

    with pytest.raises(grpc.RpcError) as delete_task:
        stub.DeleteTask(chat_pb2.DeleteTaskRequest(task_id=1))
    assert delete_task.value.code() == grpc.StatusCode.UNAUTHENTICATED

    with pytest.raises(grpc.RpcError) as delete_note:
        stub.DeleteNote(chat_pb2.DeleteNoteRequest(note_id=1))
    assert delete_note.value.code() == grpc.StatusCode.UNAUTHENTICATED

    with pytest.raises(grpc.RpcError) as list_memories:
        stub.ListMemories(chat_pb2.ListMemoriesRequest())
    assert list_memories.value.code() == grpc.StatusCode.UNAUTHENTICATED

    with pytest.raises(grpc.RpcError) as invalid:
        stub.GetConversation(
            chat_pb2.GetConversationRequest(conversation_id=1),
            metadata=((SESSION_METADATA_KEY, "invalid"),),
        )
    assert invalid.value.code() == grpc.StatusCode.UNAUTHENTICATED


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
    def __init__(self, token, trace_id=None):
        self._metadata = ((SESSION_METADATA_KEY, token),)
        if trace_id is not None:
            self._metadata += (("x-trace-id", trace_id),)

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


def test_unary_stream_handler_keeps_authenticated_user_for_generator_lifetime(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    user_id = db.register_user("alice", "correct-horse")
    token = db.create_session(user_id)

    seen = []

    def inner(request, context):
        seen.append((current_user_id(), get_trace_id()))
        yield "first"
        seen.append((current_user_id(), get_trace_id()))
        yield "second"

    details = type("D", (), {"method": "/chat.ChatService/StreamMessage"})()
    handler = AuthInterceptor().intercept_service(
        lambda _: grpc.unary_stream_rpc_method_handler(inner), details
    )

    assert list(handler.unary_stream(
        object(), _FakeContext(token, "trace-stream")
    )) == [
        "first",
        "second",
    ]
    assert seen == [(user_id, "trace-stream"), (user_id, "trace-stream")]
    with pytest.raises(RuntimeError):
        current_user_id()
    assert get_trace_id() == "-"


def test_trace_context_resets_after_exit():
    assert get_trace_id() == "-"
    with trace_context("trace-test"):
        assert get_trace_id() == "trace-test"
    assert get_trace_id() == "-"


def test_trace_id_crosses_grpc_stream_and_resets(monkeypatch):
    observed = []
    monkeypatch.setattr(db, "get_session_user", lambda token: 1)

    class TraceServicer(chat_pb2_grpc.ChatServiceServicer):
        def StreamMessage(self, request, context):
            observed.append(get_trace_id())
            yield chat_pb2.ChatEvent(started=chat_pb2.Started())
            observed.append(get_trace_id())

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        interceptors=[AuthInterceptor()],
    )
    chat_pb2_grpc.add_ChatServiceServicer_to_server(TraceServicer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = SessionStub(chat_pb2_grpc.ChatServiceStub(channel), "session")
    try:
        with trace_context("trace-roundtrip"):
            list(stub.StreamMessage(chat_pb2.ChatTurnRequest()))
        list(stub.StreamMessage(chat_pb2.ChatTurnRequest()))
    finally:
        channel.close()
        server.stop(grace=None)

    assert observed == [
        "trace-roundtrip",
        "trace-roundtrip",
        "-",
        "-",
    ]


def test_unknown_methods_are_protected_by_default():
    assert "/chat.ChatService/SaveNote" not in auth_interceptor.PUBLIC_METHODS
    assert "/chat.ChatService/SearchNotesHybrid" not in auth_interceptor.PUBLIC_METHODS
    assert "/chat.ChatService/SearchConversations" not in auth_interceptor.PUBLIC_METHODS
    assert "/chat.ChatService/ListTasks" not in auth_interceptor.PUBLIC_METHODS
