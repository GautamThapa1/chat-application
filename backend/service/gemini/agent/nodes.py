"""Gemini node: invoke Google GenAI and adapt function calls for ToolNode."""

from __future__ import annotations

from contextlib import closing
from typing import Any

from google.genai import types
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_stream_writer
from langchain_core.runnables import RunnableConfig

from backend.service import gemini_chat as existing_tools
from backend.service.gemini.adapter import (
    gemini_content_to_ai_message,
    tool_messages_to_gemini_content,
)
from backend.service.gemini.agent.state import AgentState
from backend.service.gemini.config import (
    FINAL_RESPONSE_CONFIG,
    MAX_TOOL_CALLS_PER_ROUND,
    MAX_TOOL_ROUNDS,
    TOOL_CONFIG,
)


def _pending_tool_messages(messages: list[Any]) -> list[ToolMessage]:
    """Return ToolMessages produced after the most recent tool-call AIMessage.

    [
    ToolMessage(
        content="Sunny",
        tool_call_id="1",
        name="get_weather"
    ),

    ToolMessage(
        content="25°C",
        tool_call_id="2",
        name="get_temperature"
    )
    ]

    """
    last_tool_request_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            last_tool_request_index = index

    if last_tool_request_index < 0:
        return []

    return [
        message
        for message in messages[last_tool_request_index + 1 :]
        if isinstance(message, ToolMessage)
    ]


def _incomplete_note(finish_reason: Any) -> str:
    """A visible note for a reply the model did not finish, else ""."""
    if finish_reason in (
        None,
        types.FinishReason.STOP,
        types.FinishReason.FINISH_REASON_UNSPECIFIED,
    ):
        return ""
    if finish_reason == types.FinishReason.MAX_TOKENS:
        return "[This reply was cut off at the length limit.]"
    return "[This reply was stopped early and may be incomplete.]"


def _finish_reason(response: Any) -> Any:
    candidates = response.candidates or []
    return getattr(candidates[0], "finish_reason", None) if candidates else None


def _with_time(config: types.GenerateContentConfig, time_context: str):
    if not time_context:
        return config
    instruction = f"{config.system_instruction}\n{time_context}"
    return config.model_copy(update={"system_instruction": instruction})


def _content_from_stream(
    stream: Any, writer
) -> tuple[types.Content, list[Any], str, Any]:
    """Returns everything Gemini sends in streamed response's content part"""
    parts: list[Any] = []
    function_calls: list[Any] = []
    text_chunks: list[str] = []
    finish_reason = None

    with closing(stream):
        for chunk in stream:
            finish_reason = _finish_reason(chunk) or finish_reason
            candidates = chunk.candidates or []
            if not candidates or candidates[0].content is None:
                continue
            for part in candidates[0].content.parts or []:
                parts.append(part)
                if part.function_call is not None:
                    function_calls.append(part.function_call)
                elif part.text:
                    text_chunks.append(part.text)
                    writer({"kind": "chunk", "text": part.text})

    content = types.Content(role="model", parts=parts)
    return content, function_calls, "".join(text_chunks), finish_reason


def gemini_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    
    The `gemini_node()` extracts Gemini's response and checks whether it contains any tool calls.

    - If tools are requested, it converts the function calls into LangChain messages and sends them to LangGraph for tool execution.
    - If no tools are requested, it extracts Gemini's text response and returns it as the final answer.
    
    """
    # the above is to get just gemini context and gemini client id etc.
    writer = get_stream_writer()
    configurable = config.get("configurable", {})
    app_context = configurable.get("app_context")
    if not isinstance(app_context, dict):
        raise RuntimeError("Missing application context for the Gemini graph.")
    control = app_context["control"]
    control.checkpoint()  # no new model call after Stop

    gemini_client = app_context["gemini_client"]
    model = app_context["model"]
    request_id = app_context["request_id"]
    messages = list(state.get("messages", []))
    contents = list(state.get("contents", []))
    round_index = state.get("round_index", 0)
    emitted_text = state.get("emitted_text", False)


    pending_tools = _pending_tool_messages(messages)
    if not messages: # check to see if it is first Gemini call
        
        # First model response streams text directly to ChatService, preserving
        # the existing TextChunk behavior from gemini_chat.run_turn().
        stream = gemini_client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=_with_time(TOOL_CONFIG, control.time_context),
        )
        content, calls, streamed_text, finish_reason = _content_from_stream(
            stream, writer
        )
        emitted_text = bool(streamed_text)
        if not calls: # if it doesn't request a tool
            note = _incomplete_note(finish_reason)
            if note:
                note = f"\n\n{note}" if streamed_text else note
                writer({"kind": "chunk", "text": note})
                streamed_text += note
            return {
                "messages": [AIMessage(content=streamed_text)],
                "contents": contents,
                "round_index": round_index,
            }
    else: # if not first Gemini tools
        if not pending_tools:
            raise ValueError("Gemini graph expected tool results before the next model call.")

        function_response_content, confirmation_prompt = (
            tool_messages_to_gemini_content(pending_tools)
        )
        contents.append(function_response_content)

        # A confirmation request is returned without tools enabled, matching
        # the existing clear_notes/clear_tasks confirmation flow.
        generation_config = (
            FINAL_RESPONSE_CONFIG if confirmation_prompt else TOOL_CONFIG
        )
        response = gemini_client.models.generate_content(
            model=model,
            contents=contents,
            config=_with_time(generation_config, control.time_context),
        )
        content, calls = existing_tools._response_content_and_calls(response)

        if not calls:
            final_text = existing_tools._final_text(response, confirmation_prompt)
            note = _incomplete_note(_finish_reason(response))
            if note:
                final_text = f"{final_text}\n\n{note}"
            if emitted_text:
                final_text = f"\n\n{final_text}"
            writer({"kind": "final", "text": final_text})
            return {
                "messages": [AIMessage(content=final_text)],
                "contents": contents,
                "round_index": round_index,
            }

        if confirmation_prompt:
            raise ValueError("Gemini attempted another tool call during a confirmation response.")
        if content is None:
            raise ValueError("Gemini returned tool calls without response content.")
        # Text sent alongside a later tool call is part of the reply too.
        text = "".join(
            part.text for part in content.parts or []
            if part.text and not part.thought
        )
        if text:
            writer({"kind": "chunk", "text": f"\n\n{text}" if emitted_text else text})
            emitted_text = True

    if len(calls) > MAX_TOOL_CALLS_PER_ROUND:
        raise ValueError("Gemini requested too many tools at once.")
    if round_index >= MAX_TOOL_ROUNDS:
        raise ValueError("Gemini exceeded the maximum tool-call rounds.")

    contents.append(content)
    ai_message = gemini_content_to_ai_message(
        content,
        calls,
        request_id=request_id,
        round_index=round_index,
    )
    return {
        "messages": [ai_message],
        "contents": contents,
        "round_index": round_index + 1,
        "emitted_text": emitted_text,
    }


__all__ = ["gemini_node"] # from nodes import *, will only import this
