"""Translate between Google GenAI function calls and LangChain messages."""

from __future__ import annotations

import json
from typing import Any

from google.genai import types
from langchain_core.messages import AIMessage, ToolMessage

from backend.service.gemini_chat import _tool_idempotency_key


def _function_args(call: Any) -> dict[str, Any]:
    """Extracts function arguments from Gemini's tool call and converts them into a Python dictionary"""
    args = getattr(call, "args", None) or {}
    if isinstance(args, dict):
        return args
    if hasattr(args, "items"):
        return dict(args.items())
    raise ValueError("Gemini returned function arguments that are not an object.")


def gemini_content_to_ai_message(
    content: types.Content,
    function_calls: list[Any],
    *,
    request_id: str,
    round_index: int,
) -> AIMessage:
    """Converts Gemini's response and tool calls into a LangChain AIMessage so LangGraph can handle the tool calls."""
    tool_calls: list[dict[str, Any]] = []
    for call_index, call in enumerate(function_calls):
        name = getattr(call, "name", None)
        if not name:
            raise ValueError("Gemini returned a tool call without a name.")
        tool_calls.append(
            {
                "name": name,
                "args": _function_args(call),
                # The call ID is also the idempotency key used by the existing
                # dispatcher. That keeps retries safe without adding hidden
                # arguments to Gemini's public tool schema.
                "id": _tool_idempotency_key(
                    request_id, round_index, call_index, call
                ),
                "type": "tool_call",
            }
        )

    text = "".join(
        part.text or ""
        for part in (content.parts or [])
        if getattr(part, "text", None)
    )
    return AIMessage(content=text, tool_calls=tool_calls)


def tool_messages_to_gemini_content(
    tool_messages: list[ToolMessage],
) -> tuple[types.Content, str | None]:
    """Converts LangGraph's ToolMessage results into Gemini's expected function-response format."""
    parts: list[types.Part] = []
    confirmation_prompt: str | None = None

    for tool_message in tool_messages:
        if not tool_message.name:
            raise ValueError("LangGraph returned a tool result without a name.")

        raw_content = tool_message.content
        if isinstance(raw_content, str):
            try:
                response = json.loads(raw_content)
            except json.JSONDecodeError:
                response = {"result": raw_content}
        elif isinstance(raw_content, dict):
            response = raw_content
        else:
            response = {"result": raw_content}

        if not isinstance(response, dict):
            response = {"result": response}

        if response.get("confirmation_required"):
            prompt = response.get("message")
            if isinstance(prompt, str) and prompt:
                confirmation_prompt = prompt

        parts.append(
            types.Part.from_function_response(
                name=tool_message.name,
                response=response,
            )
        )

    return types.Content(role="user", parts=parts), confirmation_prompt


__all__ = [
    "gemini_content_to_ai_message",
    "tool_messages_to_gemini_content",
]
