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