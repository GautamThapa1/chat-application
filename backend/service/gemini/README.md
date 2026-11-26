# Gemini + LangGraph integration

This package is the agent orchestration layer used by `ChatServicer`.

## Runtime flow

1. `runner.run_turn()` reuses the existing token-bounded Google GenAI history builder.
2. `agent.nodes.gemini_node()` calls the existing synchronous `google.genai` client.
3. `adapter.gemini_content_to_ai_message()` converts Google's function calls into an `AIMessage` with `tool_calls`.
4. `tools_condition` routes that message to LangGraph's `ToolNode`.
5. The registered tools in `tools/` call the existing `gemini_chat.dispatch_tool_call()`; they do not duplicate database logic.
6. `adapter.tool_messages_to_gemini_content()` converts ToolNode results into Google's `Part.from_function_response()` format.
7. The graph calls Gemini again until it returns text without function calls.
8. Custom graph stream events are translated into the existing `chunk`, `tool_result`, and `final` events consumed by `ChatServicer`.

## Existing implementation retained

`backend/service/gemini_chat.py` remains the canonical home of the Google GenAI `FunctionDeclaration` definitions, argument validation, confirmation-token checks, idempotency helper, and `dispatch_tool_call()` implementation. This is deliberate: the LangGraph integration wraps the existing tool functions rather than rewriting them. Its old `run_turn()` loop is no longer called by `ChatServicer`; it is left in place as a compatibility path for existing callers and tests.

`backend/service/main.py` still creates one `google.genai.Client` and injects it into `ChatServicer`. The frontend, protobuf contract, gRPC service and WebSocket orchestration are unchanged.

## Dependency update

`pyproject.toml` adds `langgraph>=1.0,<2.0`. Run `uv lock` and `uv sync` in an environment that can reach the configured package index so `uv.lock` is refreshed.

## Tool-call execution policy

The graph executes multiple calls returned in one Gemini response sequentially, matching the old dispatcher loop. It does this by invoking `ToolNode` once per call rather than limiting the graph's global concurrency, which would deadlock nested synchronous graph work. Tool execution still gets its authenticated `user_id` from the gRPC application context, never from model arguments. The generated LangChain tool-call ID is also used as the existing idempotency key.
