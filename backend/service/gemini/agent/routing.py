"""Conditional edges for the Gemini -> tools -> Gemini loop."""

from langgraph.prebuilt import tools_condition


def route_after_gemini(state):
    """Route to ToolNode when the latest AIMessage contains tool calls."""
    return tools_condition(state) # prebuilt function of langgraph to see if it returns tools or nah


__all__ = ["route_after_gemini"]
