from types import SimpleNamespace
from unittest.mock import AsyncMock

import grpc
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.orchestrator.conversations import router


def conversation_stub():
    conversation = SimpleNamespace(
        conversation_id=11,
        title="Private chat",
        created_at="created",
        updated_at="updated",
    )
    message = SimpleNamespace(
        message_id=5,
        conversation_id=11,
        role="user",
        content="Hello",
        created_at="created",
    )
    stub = SimpleNamespace(
        ListConversations=AsyncMock(
            return_value=SimpleNamespace(conversations=[conversation])
        ),
        CreateConversation=AsyncMock(
            return_value=SimpleNamespace(conversation=conversation)
        ),
        GetConversation=AsyncMock(
            return_value=SimpleNamespace(
                conversation=conversation,
                messages=[message],
            )
        ),
        DeleteConversation=AsyncMock(return_value=SimpleNamespace(success=True)),
    )
    app = FastAPI()
    app.state.grpc_stub = stub
    app.include_router(router)
    return TestClient(app), stub


def test_conversation_api_requires_session_cookie():
    client, stub = conversation_stub()

    response = client.get("/api/conversations")

    assert response.status_code == 401
    stub.ListConversations.assert_not_awaited()


def test_conversation_api_forwards_session_and_supports_crud():
    client, stub = conversation_stub()
    client.cookies.set("session", "user-session")

    created = client.post(
        "/api/conversations",
        json={"title": "Private chat", "user_id": 999},
    )
    listed = client.get("/api/conversations")
    history = client.get("/api/conversations/11")
    deleted = client.delete("/api/conversations/11")

    assert created.status_code == 201
    assert created.json()["conversation"]["conversation_id"] == 11
    assert listed.json()["conversations"][0]["title"] == "Private chat"
    assert history.json()["messages"][0]["content"] == "Hello"
    assert deleted.json() == {"ok": True}
    for rpc in (
        stub.CreateConversation,
        stub.ListConversations,
        stub.GetConversation,
        stub.DeleteConversation,
    ):
        assert rpc.await_args.kwargs["metadata"] == (
            ("session-token", "user-session"),
        )


def test_invalid_or_foreign_conversation_id_is_not_found():
    client, stub = conversation_stub()
    client.cookies.set("session", "user-session")
    stub.GetConversation.side_effect = grpc.aio.AioRpcError(
        grpc.StatusCode.NOT_FOUND,
        grpc.aio.Metadata(),
        grpc.aio.Metadata(),
    )

    invalid_id = client.get("/api/conversations/0")
    foreign_id = client.get("/api/conversations/12")

    assert invalid_id.status_code == 404
    assert foreign_id.status_code == 404
