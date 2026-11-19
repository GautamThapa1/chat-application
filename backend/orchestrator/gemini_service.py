
"""Gemini response generation."""

from collections.abc import AsyncIterator


async def stream_chat_response(
    message: str,
    gemini_client,
    model: str,
) -> AsyncIterator[str]:
    """Generate a response and yield text chunks as Gemini produces them."""

    stream = await gemini_client.models.generate_content_stream(
        model=model,
        contents=message,
    )

    async for chunk in stream:
        if chunk.text:
            yield chunk.text
