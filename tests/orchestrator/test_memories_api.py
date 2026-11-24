from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.orchestrator.memories import router


def memories_client():
    memory = SimpleNamespace(
        id=42,
        title="Answer style",
        content="Prefers concise answers.",
        created_at="created",
    )
    stub = SimpleNamespace(
        ListMemories=AsyncMock(
            return_value=SimpleNamespace(memories=[memory], total_count=6)
        ),
        SearchMemories=AsyncMock(
            return_value=SimpleNamespace(memories=[memory])
        ),
        SaveMemory=AsyncMock(return_value=SimpleNamespace(memory=memory)),
        UpdateMemory=AsyncMock(return_value=SimpleNamespace(memory=memory)),
        DeleteMemory=AsyncMock(return_value=SimpleNamespace(success=True)),
    )
    app = FastAPI()
    app.state.grpc_stub = stub
    app.include_router(router)
    return TestClient(app), stub


def test_memory_settings_api_requires_a_session():
    client, stub = memories_client()

    response = client.get("/api/memories")

    assert response.status_code == 401
    stub.ListMemories.assert_not_awaited()


def test_memory_settings_api_supports_list_search_create_edit_and_delete():
    client, stub = memories_client()
    client.cookies.set("session", "user-session")

    listed = client.get("/api/memories?limit=10&offset=2")
    searched = client.get("/api/memories/search?keyword=style")
    created = client.post(
        "/api/memories",
        json={"title": " Answer style ", "content": " Prefers concise answers. "},
    )
    updated = client.put(
        "/api/memories/42",
        json={"title": "Updated style", "content": "Still concise."},
    )
    deleted = client.delete("/api/memories/42")

    assert listed.status_code == 200
    assert listed.json()["pagination"] == {
        "limit": 10,
        "offset": 2,
        "total_count": 6,
    }
    assert listed.json()["memories"][0]["memory_id"] == 42
    assert searched.json()["memories"][0]["title"] == "Answer style"
    assert created.status_code == 201
    assert created.json()["memory"]["memory_id"] == 42
    assert updated.json()["memory"]["content"] == "Prefers concise answers."
    assert deleted.json() == {"ok": True}

    save_request = stub.SaveMemory.await_args.args[0]
    update_request = stub.UpdateMemory.await_args.args[0]
    assert (save_request.title, save_request.content) == (
        "Answer style",
        "Prefers concise answers.",
    )
    assert (update_request.memory_id, update_request.title) == (
        42,
        "Updated style",
    )
    for rpc in vars(stub).values():
        assert rpc.await_args.kwargs["metadata"] == (
            ("session-token", "user-session"),
        )


def test_memory_settings_api_rejects_blank_and_missing_memories():
    client, stub = memories_client()
    client.cookies.set("session", "user-session")

    blank = client.post(
        "/api/memories",
        json={"title": "  ", "content": "something"},
    )
    blank_search = client.get("/api/memories/search?keyword=%20")
    stub.DeleteMemory.return_value = SimpleNamespace(success=False)
    missing = client.delete("/api/memories/42")

    assert blank.status_code == 422
    assert blank_search.status_code == 422
    stub.SearchMemories.assert_not_awaited()
    assert missing.status_code == 404
    stub.SaveMemory.assert_not_awaited()
