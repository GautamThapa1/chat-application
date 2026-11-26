import json
from types import SimpleNamespace

from google.genai import types
from langchain_core.messages import ToolMessage

from backend.service.gemini.adapter import (
    gemini_content_to_ai_message,
    tool_messages_to_gemini_content,
)
from backend.service.gemini.schemas import DECLARATIONS_BY_NAME
from backend.service.gemini_chat import _tool_idempotency_key
from backend.service.gemini.tools.registry import ALL_TOOL_NAMES


def test_langgraph_registry_matches_all_existing_gemini_declarations():
    assert set(ALL_TOOL_NAMES) == set(DECLARATIONS_BY_NAME)
    assert len(ALL_TOOL_NAMES) == 16


def test_google_function_calls_become_langchain_tool_calls():
    content = types.Content(
        role="model",
        parts=[types.Part.from_text(text="I will save that.")],
    )
    call = SimpleNamespace(
        name="save_note",
        args={"title": "Project", "content": "Use LangGraph"},
        id="google-call-id",
    )

    message = gemini_content_to_ai_message(
        content,
        [call],
        request_id="request-123",
        round_index=0,
    )

    assert message.content == "I will save that."
    assert message.tool_calls[0]["name"] == "save_note"
    assert message.tool_calls[0]["args"] == {
        "title": "Project",
        "content": "Use LangGraph",
    }
    assert message.tool_calls[0]["id"] == _tool_idempotency_key(
        "request-123", 0, 0, call
    )


def test_tool_messages_become_google_function_responses():
    message = ToolMessage(
        name="clear_tasks",
        tool_call_id="call-1",
        content=json.dumps(
            {
                "confirmation_required": True,
                "message": "To confirm, reply with the token.",
            }
        ),
    )

    content, prompt = tool_messages_to_gemini_content([message])

    assert content.role == "user"
    assert content.parts[0].function_response.name == "clear_tasks"
    assert content.parts[0].function_response.response["confirmation_required"] is True
    assert prompt == "To confirm, reply with the token."
