
"""Gemini response generation."""

from collections.abc import AsyncIterator
import os

from google.genai import types

DEFAULT_MAX_INPUT_TOKENS = 24_000
MAX_INPUT_TOKENS_ENV = "GEMINI_MAX_INPUT_TOKENS"


def _max_input_tokens() -> int:
    # Validates the number of input token
    value = os.getenv(MAX_INPUT_TOKENS_ENV, str(DEFAULT_MAX_INPUT_TOKENS))
    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError(f"{MAX_INPUT_TOKENS_ENV} must be a positive integer.") from exc
    if limit < 1:
        raise ValueError(f"{MAX_INPUT_TOKENS_ENV} must be a positive integer.")
    return limit


def _contents_for_history(
    # Convert conversation history and the current message into Gemini content format.
    history: list[dict[str, str]],
    message: str,
) -> list[types.Content]:
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
    return contents


async def _count_tokens(gemini_client, model: str, contents: list[types.Content]) -> int:
    result = await gemini_client.models.count_tokens(model=model, contents=contents)
    if result.total_tokens is None:
        raise ValueError("Gemini did not return an input token count.")
    return result.total_tokens


async def bounded_contents(
    message: str,
    gemini_client,
    model: str,
    history: list[dict[str, str]],
) -> list[types.Content]:
    token_limit = _max_input_tokens()
    contents = _contents_for_history(history, message)
    # for previous contents
    if await _count_tokens(gemini_client, model, contents) <= token_limit:
        return contents

    # for current message
    current_message = _contents_for_history([], message)
    if await _count_tokens(gemini_client, model, current_message) > token_limit:
        raise ValueError("The current message exceeds the Gemini input token limit.")

    starts = [0]
    starts.extend(
        index
        for index, item in enumerate(history)
        if index > 0 and item["role"] == "user"
    )
    starts.append(len(history))

    too_many_history = 0
    fits = len(starts) - 1
    while too_many_history + 1 < fits:
        midpoint = (too_many_history + fits) // 2
        candidate = _contents_for_history(history[starts[midpoint]:], message)
        if await _count_tokens(gemini_client, model, candidate) <= token_limit:
            fits = midpoint
        else:
            too_many_history = midpoint

    return _contents_for_history(history[starts[fits]:], message)


async def stream_chat_response(
    message: str,
    gemini_client,
    model: str,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Generate a response and yield text chunks as Gemini produces them."""

    contents = await bounded_contents(message, gemini_client, model, history or [])

    stream = await gemini_client.models.generate_content_stream(
        model=model,
        contents=contents,
    )

    async for chunk in stream:
        if chunk.text:
            yield chunk.text
