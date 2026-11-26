from types import SimpleNamespace
from unittest.mock import AsyncMock

import grpc
import pytest

from backend.orchestrator import auth


def fake_ws(origin=None, cookie=None, valid=True, user_id=7, error=None):
    validate = AsyncMock(
        return_value=SimpleNamespace(valid=valid, user_id=user_id),
        side_effect=error,
    )
    ws = SimpleNamespace(
        headers={"origin": origin} if origin else {},
        cookies={"session": cookie} if cookie else {},
        app=SimpleNamespace(state=SimpleNamespace(
            grpc_stub=SimpleNamespace(ValidateSession=validate)
        )),
    )
    return ws, validate


@pytest.mark.asyncio
async def test_no_cookie_is_rejected():
    ws, validate = fake_ws()
    assert await auth.authenticate_websocket(ws) is None
    validate.assert_not_awaited()


@pytest.mark.asyncio
async def test_bad_origin_is_rejected_before_any_lookup():
    ws, validate = fake_ws(origin="http://evil.example", cookie="tok")
    assert await auth.authenticate_websocket(ws) is None
    validate.assert_not_awaited()


@pytest.mark.asyncio
async def test_allowed_origin_with_valid_session_passes():
    ws, _ = fake_ws(origin="http://localhost:8000", cookie="tok")
    assert await auth.authenticate_websocket(ws) == 7


@pytest.mark.asyncio
async def test_no_origin_header_with_valid_session_passes():
    ws, _ = fake_ws(cookie="tok")
    assert await auth.authenticate_websocket(ws) == 7


@pytest.mark.asyncio
async def test_invalid_session_is_rejected():
    ws, _ = fake_ws(cookie="tok", valid=False)
    assert await auth.authenticate_websocket(ws) is None


@pytest.mark.asyncio
async def test_grpc_failure_rejects_instead_of_crashing():
    err = grpc.aio.AioRpcError(
        grpc.StatusCode.UNAVAILABLE, grpc.aio.Metadata(), grpc.aio.Metadata()
    )
    ws, _ = fake_ws(cookie="tok", error=err)
    assert await auth.authenticate_websocket(ws) is None

# ---------- single-use WebSocket tickets (frontend on another site) ----------

def test_ws_ticket_is_single_use():
    ticket = auth.issue_ws_ticket("tok-abc")
    ws = SimpleNamespace(query_params={"ticket": ticket}, cookies={})
    assert auth.websocket_session_token(ws) == "tok-abc"
    assert auth.websocket_session_token(ws) is None


def test_ws_ticket_expires(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(auth.time, "monotonic", lambda: now[0])
    ticket = auth.issue_ws_ticket("tok-abc")
    now[0] += auth.WS_TICKET_TTL_SECONDS + 1
    assert auth.redeem_ws_ticket(ticket) is None


def test_unknown_ticket_does_not_fall_back_to_cookie():
    ws = SimpleNamespace(query_params={"ticket": "bogus"}, cookies={"session": "tok"})
    assert auth.websocket_session_token(ws) is None


def test_cookie_used_without_ticket():
    ws = SimpleNamespace(query_params={}, cookies={"session": "tok"})
    assert auth.websocket_session_token(ws) == "tok"


@pytest.mark.asyncio
async def test_authenticate_websocket_checks_given_token():
    ws, validate = fake_ws(origin="http://localhost:8000")
    assert await auth.authenticate_websocket(ws, "tok-from-ticket") == 7
    assert validate.await_args.args[0].session_token == "tok-from-ticket"


@pytest.mark.asyncio
async def test_ws_ticket_endpoint_requires_session():
    from fastapi import HTTPException
    request = SimpleNamespace(cookies={})
    with pytest.raises(HTTPException) as exc:
        await auth.ws_ticket(request)
    assert exc.value.status_code == 401
