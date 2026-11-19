from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.orchestrator.gemini_service import stream_chat_response

@pytest.mark.asyncio
async def test_stream_chat_response_yields_chunks():

    # Create fake gemini stream
    async def fake_stream():
        yield SimpleNamespace(text="Hello") # later can be accessed by obj.text
        yield SimpleNamespace(text=" there")
        yield SimpleNamespace(text="!")

    generate_stream = AsyncMock(return_value=fake_stream())

    # faked here
    gemini_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content_stream=generate_stream
        )
    )

    # Act: collect the chunks from our function
    chunks = [
        chunk
        async for chunk in stream_chat_response(
            "Hi", 
            gemini_client, 
            "test-model",
        )
    ]
    assert chunks == ["Hello", " there", "!"]

    generate_stream.assert_awaited_once_with(
        model="test-model",
        contents="Hi",
    ) # checks if it matches

@pytest.mark.asyncio
async def test_stream_chat_response_skips_empty_chunks():

    # Arrange
    async def fake_stream():
        yield SimpleNamespace(text="Hello")
        yield SimpleNamespace(text=None)
        yield SimpleNamespace(text="")
        yield SimpleNamespace(text="!")

    gemini_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content_stream=AsyncMock(
                return_value=fake_stream()
            )
        )
    )

    # Act
    chunks = [
        chunk
        async for chunk in stream_chat_response(
            "Hi",
            gemini_client,
            "test-model",
        )
    ]

    # Assert
    assert chunks == ["Hello", "!"]
