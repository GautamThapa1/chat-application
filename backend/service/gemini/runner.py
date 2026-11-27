"""Compatibility-shaped synchronous runner around the LangGraph agent."""

from __future__ import annotations

import secrets
from collections.abc import Iterator

from backend.service.gemini.agent.graph import AGENT_GRAPH
from backend.service.gemini.client import build_initial_contents
from backend.service.gemini.config import MAX_TOOL_ROUNDS
from backend.service.turn_control import TurnControl

TurnEvent = (
    tuple[str, str]
    | tuple[str, str, dict]
)


def run_turn(
    message: str,
    gemini_client,
    model: str,
    history: list[dict[str, str]],
    user_id: int,
    request_id: str,
    *,
    control: TurnControl | None = None,
) -> Iterator[TurnEvent]:
    """Run one turn and yield the same events ChatService already consumes."""
    # Prepares user's message and conversation history.
    initial_contents = build_initial_contents(
        message,
        gemini_client,
        model,
        history,
    )
    # application context
    app_context = {
        "user_id": user_id,
        "user_message": message,
        "history": history,
        "confirmation_token": secrets.token_urlsafe(8),
        "gemini_client": gemini_client,
        "model": model,
        "request_id": request_id,
        "control": control or TurnControl(),
    }
    graph_config = {
        "configurable": {
            # No checkpointer is configured yet; thread_id is included so a
            # checkpointer can be added later without changing tool context.
            "thread_id": request_id or secrets.token_urlsafe(12),
            "app_context": app_context,
        },
        "recursion_limit": (MAX_TOOL_ROUNDS * 2) + 3,
    }

    stream = AGENT_GRAPH.stream(
        {
            "contents": initial_contents,
            "messages": [],
            "round_index": 0,
            "request_id": request_id,
        },
        config=graph_config,
        stream_mode="custom",
    )
    try:
        # these event are streamed, keeps on comming in stream mode
        for event in stream:
            if not isinstance(event, dict):
                continue
            kind = event.get("kind")
            if kind == "chunk":
                yield ("chunk", event["text"])
            elif kind == "tool_result":
                yield ("tool_result", event["tool"], event["result"])
            elif kind == "final":
                yield ("final", event["text"])
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


__all__ = ["run_turn", "TurnEvent"]
