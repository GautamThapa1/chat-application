"""Central registry for every model-callable tool in this project."""

from __future__ import annotations

from backend.service.gemini.schemas import DECLARATIONS_BY_NAME
from backend.service.gemini.tools.memories import MEMORY_TOOL_NAMES
from backend.service.gemini.tools.notes import NOTE_TOOL_NAMES
from backend.service.gemini.tools.tasks import TASK_TOOL_NAMES
from backend.service.gemini.tools.wrappers import make_dispatch_tool

ALL_TOOL_NAMES = NOTE_TOOL_NAMES + MEMORY_TOOL_NAMES + TASK_TOOL_NAMES


def build_tools():
    """Build the LangGraph tool list from the canonical Gemini declarations."""
    if len(ALL_TOOL_NAMES) != len(set(ALL_TOOL_NAMES)):
        raise RuntimeError("Duplicate tool name in the Gemini tool registry.")

    declared = set(DECLARATIONS_BY_NAME)
    registered = set(ALL_TOOL_NAMES)
    if declared != registered:
        missing = sorted(declared - registered)
        unknown = sorted(registered - declared)
        raise RuntimeError(
            f"Gemini tool registry mismatch; missing={missing}, unknown={unknown}."
        )

    return [make_dispatch_tool(name) for name in ALL_TOOL_NAMES]


__all__ = ["ALL_TOOL_NAMES", "build_tools"]
