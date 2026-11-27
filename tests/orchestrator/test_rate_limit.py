from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

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


def test_me_returns_the_authenticated_username(client):
    client.cookies.set("session", "valid-session")
    client.app.state.grpc_stub.ValidateSession = AsyncMock(
        return_value=MagicMock(valid=True, user_id=7, username="alice")
    )

    response = client.get("/api/me")

    assert response.status_code == 200
    assert response.json() == {"user_id": 7, "username": "alice"}

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

# ---- per-username limit: holds even when the client address is forged ---- #
@pytest.fixture
def proxied(client):
    """The app as start.sh runs it: forwarded addresses from anyone are believed."""
    wrong_password = grpc.aio.AioRpcError(
        grpc.StatusCode.UNAUTHENTICATED, grpc.aio.Metadata(), grpc.aio.Metadata(),
        details="Invalid username or password.",
    )
    client.app.state.grpc_stub.Login = AsyncMock(side_effect=wrong_password)
    return TestClient(ProxyHeadersMiddleware(client.app, trusted_hosts="*")), client.app.state.grpc_stub


def guess(http, number, username="victim"):
    return http.post(
        "/api/login",
        json={"username": username, "password": "wrong"},
        headers={"X-Forwarded-For": f"10.0.{number}.1"},
    )


def test_forged_addresses_cannot_guess_one_account_forever(proxied):
    http, stub = proxied
    codes = [guess(http, n).status_code for n in range(10)]
    blocked = guess(http, 11, username="VICTIM")  # usernames ignore case

    assert codes == [401] * 10
    assert blocked.status_code == 429
    assert "account" in blocked.json()["detail"]
    assert int(blocked.headers["Retry-After"]) >= 1
    assert stub.Login.await_count == 10
    assert guess(http, 12, username="someone_else").status_code == 401


def test_successful_login_resets_the_account_limit(proxied):
    http, stub = proxied
    for n in range(9):
        guess(http, n)
    wrong_password = stub.Login.side_effect
    stub.Login.side_effect = None
    stub.Login.return_value = MagicMock(session_token="t", user_id=1)
    assert guess(http, 20).status_code == 200

    stub.Login.side_effect = wrong_password
    assert [guess(http, n).status_code for n in range(30, 40)] == [401] * 10


# ---- caps on the address Cloudflare reports, which cannot be forged ------ #
def forged_requests(app_client, count, method, path, connector="203.0.113.50", **kwargs):
    """Send requests that claim a new X-Forwarded-For address every time."""
    http = TestClient(ProxyHeadersMiddleware(app_client.app, trusted_hosts="*"))
    codes = []
    for n in range(count):
        headers = {"X-Forwarded-For": f"10.{n // 250}.{n % 250}.1"}
        if connector:
            headers["CF-Connecting-IP"] = connector
        body = kwargs["json"](n) if "json" in kwargs else None
        codes.append(http.request(method, path, json=body, headers=headers).status_code)
    return codes


def test_forged_addresses_stop_at_the_connector_cap(app_client):
    codes = forged_requests(app_client, 602, "GET", "/api/me")

    assert codes[:600] == [401] * 600
    assert codes[600:] == [429, 429]
    assert forged_requests(app_client, 1, "GET", "/api/me", connector="203.0.113.51") == [401]


def test_forged_addresses_cannot_spray_logins_across_accounts(app_client):
    codes = forged_requests(
        app_client, 101, "POST", "/api/login",
        json=lambda n: {"username": f"user{n}", "password": "guess"},
    )

    assert codes == [200] * 100 + [429]


def test_forged_addresses_cannot_mass_register(app_client):
    app_client.app.state.grpc_stub.Register = AsyncMock(return_value=MagicMock(user_id=1))
    codes = forged_requests(
        app_client, 31, "POST", "/api/register",
        json=lambda n: {"username": f"user{n}", "password": "password123"},
    )

    assert codes == [201] * 30 + [429]


def test_no_connector_cap_without_the_cloudflare_header(app_client):
    codes = forged_requests(app_client, 602, "GET", "/api/me", connector=None)

    assert codes == [401] * 602
