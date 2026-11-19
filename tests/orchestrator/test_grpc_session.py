from unittest.mock import AsyncMock

import pytest

from backend.orchestrator.grpc_session import SessionStub


@pytest.mark.asyncio
async def test_every_call_carries_the_session_token():
    raw_stub = type("Stub", (), {})()
    raw_stub.SaveNote = AsyncMock(return_value="result")

    stub = SessionStub(raw_stub, "tok-123")
    result = await stub.SaveNote("request")

    assert result == "result"
    raw_stub.SaveNote.assert_awaited_once_with(
        "request", metadata=(("session-token", "tok-123"),)
    )


@pytest.mark.asyncio
async def test_extra_keyword_arguments_are_passed_through():
    raw_stub = type("Stub", (), {})()
    raw_stub.ListTasks = AsyncMock()

    await SessionStub(raw_stub, "tok").ListTasks("req", timeout=5)

    raw_stub.ListTasks.assert_awaited_once_with(
        "req", metadata=(("session-token", "tok"),), timeout=5
    )
