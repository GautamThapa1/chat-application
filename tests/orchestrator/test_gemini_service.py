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
            generate_content_stream=generate_stream,
            count_tokens=AsyncMock(return_value=SimpleNamespace(total_tokens=1)),
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

    generate_stream.assert_awaited_once()
    call = generate_stream.await_args.kwargs
    assert call["model"] == "test-model"
    assert call["contents"][0].role == "user"
    assert call["contents"][0].parts[0].text == "Hi"

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
            generate_content_stream=AsyncMock(return_value=fake_stream()),
            count_tokens=AsyncMock(return_value=SimpleNamespace(total_tokens=1)),
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


@pytest.mark.asyncio
async def test_stream_chat_response_includes_only_the_selected_chat_history():
    async def fake_stream():
        yield SimpleNamespace(text="answer")

    generate_stream = AsyncMock(return_value=fake_stream())
    gemini_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content_stream=generate_stream,
            count_tokens=AsyncMock(return_value=SimpleNamespace(total_tokens=1)),
        )
    )

    chunks = [
        chunk
        async for chunk in stream_chat_response(
            "Current question",
            gemini_client,
            "test-model",
            [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
            ],
        )
    ]

    assert chunks == ["answer"]
    call = generate_stream.await_args.kwargs
    contents = call["contents"]
    assert [item.role for item in contents] == ["user", "model", "user"]
    assert [item.parts[0].text for item in contents] == [
        "Previous question",
        "Previous answer",
        "Current question",
    ]


@pytest.mark.asyncio
async def test_stream_chat_response_keeps_most_recent_complete_turns(
    monkeypatch,
):
    monkeypatch.setenv("GEMINI_MAX_INPUT_TOKENS", "3")

    async def fake_stream():
        yield SimpleNamespace(text="answer")

    async def count_tokens(*, model, contents):
        assert model == "test-model"
        return SimpleNamespace(total_tokens=len(contents))

    generate_stream = AsyncMock(return_value=fake_stream())
    gemini_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content_stream=generate_stream,
            count_tokens=AsyncMock(side_effect=count_tokens),
        )
    )

    chunks = [
        chunk
        async for chunk in stream_chat_response(
            "Current question",
            gemini_client,
            "test-model",
            [
                {"role": "user", "content": "Old question"},
                {"role": "assistant", "content": "Old answer"},
                {"role": "user", "content": "Recent question"},
                {"role": "assistant", "content": "Recent answer"},
            ],
        )
    ]

    assert chunks == ["answer"]
    contents = generate_stream.await_args.kwargs["contents"]
    assert [item.parts[0].text for item in contents] == [
        "Recent question",
        "Recent answer",
        "Current question",
    ]
    assert len(contents) <= 3


@pytest.mark.asyncio
async def test_stream_chat_response_rejects_current_message_over_token_limit(
    monkeypatch,
):
    monkeypatch.setenv("GEMINI_MAX_INPUT_TOKENS", "2")
    gemini_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content_stream=AsyncMock(),
            count_tokens=AsyncMock(
                return_value=SimpleNamespace(total_tokens=3)
            ),
        )
    )

    with pytest.raises(ValueError, match="current message exceeds"):
        async for _ in stream_chat_response(
            "Too long", gemini_client, "test-model",
        ):
            pass

    gemini_client.models.generate_content_stream.assert_not_awaited()
