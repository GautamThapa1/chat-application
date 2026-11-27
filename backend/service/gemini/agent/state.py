"""LangGraph state used for one synchronous Gemini chat turn."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from google.genai import types
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# used for AgentGraph, to tell what it remembers
class AgentState(TypedDict, total=False):
    # Google GenAI Content remains the canonical model conversation format.
    contents: list[types.Content]

    # LangChain messages exist only to give ToolNode/tool routing a standard
    # representation of Gemini's function calls and tool results.
    messages: Annotated[list[AnyMessage], add_messages] # add_messages is a reducer to append in messages

    # Number of Gemini responses that have requested tool execution.
    round_index: int

    # True once reply text has been sent, so later text starts a new paragraph.
    emitted_text: bool

    # Used for diagnostic/stream correlation; never model-controlled.
    request_id: str


__all__ = ["AgentState"]
