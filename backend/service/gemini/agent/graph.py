"""Build the synchronous LangGraph agent used by ChatServicer."""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from backend.service.gemini.agent.nodes import gemini_node
from backend.service.gemini.agent.routing import route_after_gemini
from backend.service.gemini.agent.state import AgentState
from backend.service.gemini.config import MAX_TOOL_ROUNDS
from backend.service.gemini.tools import build_tools


def _run_tools_sequentially(
    state: AgentState,
    config: RunnableConfig,
    tool_node: ToolNode,
) -> dict[str, Any]:
    """Call the tools sequentially, extract them and run them sequentially. A, then B, then C"""
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], AIMessage):
        raise ValueError("Gemini graph expected a tool-call message.")

    tool_request = messages[-1]
    results = []
    for call in tool_request.tool_calls:
        single_call = AIMessage(
            content=tool_request.content,
            tool_calls=[call],
        )
        result = tool_node.invoke({"messages": [single_call]}, config)
        results.extend(result["messages"])
    return {"messages": results}


def build_graph():
    """Create a Gemini -> sequential tools -> Gemini graph."""
    builder = StateGraph(AgentState)
    tool_node = ToolNode(build_tools(), handle_tool_errors=False)

    # adding nodes
    builder.add_node("gemini", gemini_node)
    builder.add_node(
        "tools",
        lambda state, config: _run_tools_sequentially(state, config, tool_node),
    )

    # building the flow
    builder.add_edge(START, "gemini")
    builder.add_conditional_edges(
        "gemini",
        route_after_gemini, # routes here if no tools used
        {"tools": "tools", END: END}, # if it returns tools execute tools, if nothing then nothing
    )
    builder.add_edge("tools", "gemini")
    return builder.compile()


AGENT_GRAPH = build_graph()

__all__ = ["AGENT_GRAPH", "build_graph"]
