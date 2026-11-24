from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.orchestrator.notes import router


def notes_client():
    stub = SimpleNamespace(
        DeleteNote=AsyncMock(return_value=SimpleNamespace(success=True))
    )
    app = FastAPI()
    app.state.grpc_stub = stub
    app.include_router(router)
    return TestClient(app), stub


def test_delete_note_requires_session_cookie():
    client, stub = notes_client()

    response = client.delete("/api/notes/42")

    assert response.status_code == 401
    stub.DeleteNote.assert_not_awaited()


def test_delete_note_forwards_authenticated_request():
    client, stub = notes_client()
    client.cookies.set("session", "user-session")

    response = client.delete("/api/notes/42")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert stub.DeleteNote.await_args.args[0].note_id == 42
    assert stub.DeleteNote.await_args.kwargs["metadata"] == (
        ("session-token", "user-session"),
    )


def test_delete_note_rejects_invalid_id_and_missing_note():
    client, stub = notes_client()
    client.cookies.set("session", "user-session")

    invalid = client.delete("/api/notes/0")
    stub.DeleteNote.return_value = SimpleNamespace(success=False)
    missing = client.delete("/api/notes/42")

    assert invalid.status_code == 422
    assert missing.status_code == 404
    assert stub.DeleteNote.await_count == 1
