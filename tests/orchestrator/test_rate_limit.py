from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.orchestrator import auth
from backend.orchestrator.rate_limit import limiter, reset_limits

@pytest.fixture
def client():
    reset_limits()
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.state.grpc_stub = MagicMock()
    app.state.grpc_stub.Login = AsyncMock(
        return_value=MagicMock(session_token="t", user_id=1)
    )
    app.state.grpc_stub.Register = AsyncMock(return_value=MagicMock(user_id=1))
    app.include_router(auth.router)
    return TestClient(app)

def test_login_allows_ten_attempts(client):
    body = {"username": "alice", "password": "pw"}
    for _ in range(10):
        assert client.post("/api/login", json=body).status_code == 200


def test_login_blocks_the_eleventh_attempt(client):
    body = {"username": "alice", "password": "pw"}
    for _ in range(10):
        client.post("/api/login", json=body)
    assert client.post("/api/login", json=body).status_code == 429


def test_blocked_request_never_reaches_grpc(client):
    body = {"username": "alice", "password": "pw"}
    for _ in range(11):
        client.post("/api/login", json=body)
    assert client.app.state.grpc_stub.Login.await_count == 10


def test_register_blocks_the_sixth_attempt(client):
    body = {"username": "alice", "password": "password123"}
    for _ in range(5):
        assert client.post("/api/register", json=body).status_code == 201
    assert client.post("/api/register", json=body).status_code == 429

@pytest.fixture
def app_client():
    """The real app from main.py, so we test its actual wiring."""
    from backend.orchestrator import main

    reset_limits()
    main.app.state.grpc_stub = MagicMock()
    main.app.state.grpc_stub.Login = AsyncMock(
        return_value=MagicMock(session_token="t", user_id=1)
    )
    return TestClient(main.app)


def test_default_limit_blocks_the_sixty_first_request(app_client):
    for _ in range(60):
        assert app_client.get("/api/me").status_code == 401  # not logged in, fine
    assert app_client.get("/api/me").status_code == 429


def test_login_keeps_its_own_stricter_limit(app_client):
    body = {"username": "alice", "password": "pw"}
    for _ in range(10):
        assert app_client.post("/api/login", json=body).status_code == 200
    # Far below 60 requests in total, yet login is already blocked.
    assert app_client.post("/api/login", json=body).status_code == 429


def test_blocked_response_tells_client_when_to_retry(app_client):
    for _ in range(60):
        app_client.get("/api/me")
    blocked = app_client.get("/api/me")
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["Retry-After"]) <= 60