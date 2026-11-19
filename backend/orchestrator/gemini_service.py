
"""Gemini response generation."""

from collections.abc import AsyncIterator

from google.genai import types


async def stream_chat_response(
    message: str,
    gemini_client,
    model: str,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Generate a response and yield text chunks as Gemini produces them."""

    contents = message
    if history:
        contents = [
            types.Content(
                role="user" if item["role"] == "user" else "model",
                parts=[types.Part.from_text(text=item["content"])],
            )
            for item in history
        ]
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=message)])
        )

    stream = await gemini_client.models.generate_content_stream(
        model=model,
        contents=contents,
    )

    async for chunk in stream:
        if chunk.text:
            yield chunk.text
