from types import SimpleNamespace
from unittest.mock import AsyncMock

import grpc
import pytest
from google.genai import types

from backend.orchestrator.intent_handler import (
    TOOL_CONFIG,
    dispatch_tool_call,
    handle_function_calls,
    save_conversation,
)


def function_response(*calls, text=None):
    parts = [
        types.Part(
            function_call=types.FunctionCall(name=name, args=arguments)
        )
        for name, arguments in calls
    ]
    if text is not None:
        parts.append(types.Part(text=text))
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=types.Content(role="model", parts=parts))
        ],
        text=text,
    )


def gemini_client(*responses):
    return SimpleNamespace(
        models=SimpleNamespace(
            count_tokens=AsyncMock(
                return_value=SimpleNamespace(total_tokens=1)
            ),
            generate_content=AsyncMock(side_effect=responses),
        )
    )


@pytest.mark.asyncio
async def test_normal_chat_returns_to_streaming_path():
    client = gemini_client(function_response(text="A non-tool response."))

    result = await handle_function_calls(
        "Hello", client, "test-model", [], SimpleNamespace()
    )

    assert result is None
    client.models.generate_content.assert_awaited_once()


def test_native_tools_declare_all_supported_operations():
    assert {
        declaration.name
        for declaration in TOOL_CONFIG.tools[0].function_declarations
    } == {
        "save_note",
        "search_notes",
        "list_notes",
        "clear_notes",
        "create_task",
        "list_tasks",
        "complete_task",
    }


@pytest.mark.asyncio
async def test_save_note_dispatch_validates_and_maps_arguments():
    stub = SimpleNamespace(
        SaveNote=AsyncMock(
            return_value=SimpleNamespace(
                note=SimpleNamespace(id=4, title="Meeting")
            )
        )
    )

    result = await dispatch_tool_call(
        "save_note",
        {"title": " Meeting ", "content": " Discuss launch "},
        stub,
        user_message="Save a note",
        history=[],
        confirmation_token="unused",
    )

    assert result.response == {"note_id": 4, "title": "Meeting"}
    request = stub.SaveNote.await_args.args[0]
    assert (request.title, request.content) == ("Meeting", "Discuss launch")


@pytest.mark.asyncio
async def test_search_and_list_notes_map_to_search_notes_rpc():
    note = SimpleNamespace(id=1, title="Plan", content="Launch plan", created_at="")
    stub = SimpleNamespace(
        SearchNotes=AsyncMock(return_value=SimpleNamespace(notes=[note]))
    )

    searched = await dispatch_tool_call(
        "search_notes",
        {"keyword": " launch "},
        stub,
        user_message="Find launch",
        history=[],
        confirmation_token="unused",
    )
    listed = await dispatch_tool_call(
        "list_notes",
        {},
        stub,
        user_message="List notes",
        history=[],
        confirmation_token="unused",
    )

    assert searched.response["notes"][0]["title"] == "Plan"
    assert listed.response == searched.response
    assert [
        call.args[0].keyword for call in stub.SearchNotes.await_args_list
    ] == ["launch", ""]


@pytest.mark.asyncio
async def test_task_tools_dispatch_to_the_existing_grpc_methods():
    stub = SimpleNamespace(
        CreateTask=AsyncMock(
            return_value=SimpleNamespace(
                task=SimpleNamespace(id=9, title="Review", due_date="tomorrow")
            )
        ),
        ListTasks=AsyncMock(
            return_value=SimpleNamespace(
                tasks=[
                    SimpleNamespace(
                        id=9, title="Review", due_date="tomorrow", completed=False
                    )
                ]
            )
        ),
        CompleteTask=AsyncMock(
            return_value=SimpleNamespace(success=True)
        ),
    )

    created = await dispatch_tool_call(
        "create_task",
        {"title": " Review ", "due_date": " tomorrow "},
        stub,
        user_message="Add a task",
        history=[],
        confirmation_token="unused",
    )
    listed = await dispatch_tool_call(
        "list_tasks",
        {},
        stub,
        user_message="List tasks",
        history=[],
        confirmation_token="unused",
    )
    completed = await dispatch_tool_call(
        "complete_task",
        {"task_id": 9},
        stub,
        user_message="Complete task 9",
        history=[],
        confirmation_token="unused",
    )

    assert created.response == {
        "task_id": 9,
        "title": "Review",
        "due_date": "tomorrow",
    }
    assert listed.response["tasks"][0]["id"] == 9
    assert completed.response == {"task_id": 9, "success": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("save_note", {"title": "Only title"}),
        ("search_notes", {"keyword": "  "}),
        ("complete_task", {"task_id": True}),
        ("list_tasks", {"unexpected": True}),
    ],
)
async def test_invalid_tool_arguments_are_rejected_without_grpc_calls(
    name, arguments,
):
    stub = SimpleNamespace(
        SaveNote=AsyncMock(),
        SearchNotes=AsyncMock(),
        CompleteTask=AsyncMock(),
        ListTasks=AsyncMock(),
    )

    result = await dispatch_tool_call(
        name,
        arguments,
        stub,
        user_message="Do it",
        history=[],
        confirmation_token="unused",
    )

    assert result.response.get("error")
    for rpc in vars(stub).values():
        rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError, match="Unknown Gemini tool"):
        await dispatch_tool_call(
            "delete_account",
            {},
            SimpleNamespace(),
            user_message="Delete my account",
            history=[],
            confirmation_token="unused",
        )


@pytest.mark.asyncio
async def test_multiple_tool_calls_are_executed_and_results_returned_to_gemini():
    client = gemini_client(
        function_response(
            ("save_note", {"title": "Plan", "content": "Launch"}),
            ("list_tasks", {}),
        ),
        function_response(text="I saved the note and listed your tasks."),
    )
    stub = SimpleNamespace(
        SaveNote=AsyncMock(
            return_value=SimpleNamespace(
                note=SimpleNamespace(id=3, title="Plan")
            )
        ),
        ListTasks=AsyncMock(return_value=SimpleNamespace(tasks=[])),
    )

    result = await handle_function_calls(
        "Save a note and show my tasks",
        client,
        "test-model",
        [],
        stub,
    )

    assert result == "I saved the note and listed your tasks."
    assert stub.SaveNote.await_count == 1
    assert stub.ListTasks.await_count == 1
    second_request = client.models.generate_content.await_args_list[1].kwargs
    function_responses = second_request["contents"][-1]
    assert function_responses.role == "tool"
    assert [part.function_response.name for part in function_responses.parts] == [
        "save_note",
        "list_tasks",
    ]
    assert function_responses.parts[0].function_response.response == {
        "note_id": 3,
        "title": "Plan",
    }


@pytest.mark.asyncio
async def test_function_call_loop_supports_additional_tool_rounds():
    client = gemini_client(
        function_response(("search_notes", {"keyword": "launch"})),
        function_response(("list_tasks", {})),
        function_response(text="I found notes and listed your tasks."),
    )
    stub = SimpleNamespace(
        SearchNotes=AsyncMock(return_value=SimpleNamespace(notes=[])),
        ListTasks=AsyncMock(return_value=SimpleNamespace(tasks=[])),
    )

    result = await handle_function_calls(
        "Search launch notes and show my tasks",
        client,
        "test-model",
        [],
        stub,
    )

    assert result == "I found notes and listed your tasks."
    assert client.models.generate_content.await_count == 3
    assert stub.SearchNotes.await_count == 1
    assert stub.ListTasks.await_count == 1


@pytest.mark.asyncio
async def test_grpc_tool_failure_is_propagated():
    client = gemini_client(
        function_response(("list_tasks", {}))
    )
    stub = SimpleNamespace(
        ListTasks=AsyncMock(
            side_effect=grpc.aio.AioRpcError(
                grpc.StatusCode.UNAVAILABLE,
                grpc.aio.Metadata(),
                grpc.aio.Metadata(),
            )
        )
    )

    with pytest.raises(grpc.aio.AioRpcError):
        await handle_function_calls(
            "List my tasks", client, "test-model", [], stub
        )

    assert client.models.generate_content.await_count == 1


@pytest.mark.asyncio
async def test_clear_notes_requires_the_prompted_explicit_confirmation():
    client = gemini_client(
        function_response(("clear_notes", {})),
        function_response(text="I need confirmation first."),
    )
    stub = SimpleNamespace(
        ClearNotes=AsyncMock(return_value=SimpleNamespace(deleted_count=2))
    )

    reply = await handle_function_calls(
        "Clear my notes", client, "test-model", [], stub
    )

    stub.ClearNotes.assert_not_awaited()
    prompt = reply.splitlines()[-1]
    assert "confirm clear notes " in prompt
    token = prompt.split("confirm clear notes ", 1)[1].split("`", 1)[0]

    confirmed = await dispatch_tool_call(
        "clear_notes",
        {},
        stub,
        user_message=f"confirm clear notes {token}",
        history=[{"role": "assistant", "content": reply}],
        confirmation_token="unused",
    )

    assert confirmed.response == {"deleted_count": 2}
    stub.ClearNotes.assert_awaited_once()


@pytest.mark.asyncio
async def test_clear_notes_cannot_be_confirmed_without_a_pending_prompt():
    stub = SimpleNamespace(
        ClearNotes=AsyncMock(return_value=SimpleNamespace(deleted_count=1))
    )

    result = await dispatch_tool_call(
        "clear_notes",
        {},
        stub,
        user_message="confirm clear notes arbitrary",
        history=[],
        confirmation_token="one-time-token",
    )

    assert result.response["confirmation_required"] is True
    stub.ClearNotes.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_conversation():
    stub = SimpleNamespace(SaveConversation=AsyncMock())

    await save_conversation("Hello", "Hi there!", stub, 42)

    request = stub.SaveConversation.await_args.args[0]
    assert request.user_content == "Hello"
    assert request.assistant_content == "Hi there!"
    assert request.conversation_id == 42
