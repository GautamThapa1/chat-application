"""Create LangGraph tools that delegate to the existing dispatcher."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer
from langgraph.prebuilt import ToolRuntime

from backend.service import gemini_chat as existing_tools
from backend.service.gemini.schemas import tool_args_schema


def make_dispatch_tool(tool_name: str) -> StructuredTool:
    """Wrap one existing dispatcher operation as a LangGraph tool.

    The model-visible argument schema comes from the original Google GenAI
    FunctionDeclaration. Application context and tool-call ID are injected by
    LangGraph and are never exposed as model-supplied arguments.
    """
    declaration = next(
        (
            item
            for item in existing_tools.FUNCTION_DECLARATIONS
            if item.name == tool_name
        ),
        None,
    )
    if declaration is None:
        raise KeyError(f"No existing Gemini tool named {tool_name!r}.")

    def execute(*, runtime: ToolRuntime, **arguments: Any) -> str:
        configurable = runtime.config.get("configurable", {})
        app_context = configurable.get("app_context")
        if not isinstance(app_context, dict):
            raise RuntimeError("Missing authenticated application context for tool execution.")

        required_context = {
            "user_id",
            "user_message",
            "history",
            "confirmation_token",
            "gemini_client",
        }
        missing = required_context - app_context.keys()
        if missing:
            raise RuntimeError(
                "Incomplete application context for tool execution: "
                + ", ".join(sorted(missing))
            )

        result = existing_tools.dispatch_tool_call(
            tool_name,
            arguments,
            app_context["user_id"],
            user_message=app_context["user_message"],
            history=app_context["history"],
            confirmation_token=app_context["confirmation_token"],
            gemini_client=app_context["gemini_client"],
            idempotency_key=runtime.tool_call_id,
        )

        # Stream the same ToolResult event the current gRPC service expects.
        get_stream_writer()(
            {
                "kind": "tool_result",
                "tool": tool_name,
                "result": result.response,
            }
        )
        return json.dumps(result.response, ensure_ascii=False, default=str)

    execute.__name__ = f"langgraph_{tool_name}"
    execute.__doc__ = declaration.description or f"Execute {tool_name}."

    return StructuredTool.from_function(
        func=execute,
        name=tool_name,
        description=declaration.description or f"Execute {tool_name}.",
        args_schema=tool_args_schema(tool_name),
        infer_schema=False,
    )
