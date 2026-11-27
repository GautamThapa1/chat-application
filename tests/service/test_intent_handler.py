from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import types

from backend.service import db, gemini_chat, memory_store
from backend.service.turn_control import TurnControl


def _response(*calls, text=None):
    parts = [
        types.Part(function_call=types.FunctionCall(name=name, args=arguments))
        for name, arguments in calls
    ]
    if text is not None:
        parts.append(types.Part(text=text))
    content = types.Content(role="model", parts=parts)
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=content)],
        text=text,
    )


def _response_stream(response):
    yield response


def _client(*responses):
    return SimpleNamespace(
        models=SimpleNamespace(
            count_tokens=Mock(return_value=SimpleNamespace(total_tokens=1)),
            generate_content_stream=Mock(
                return_value=_response_stream(responses[0])
            ),
            generate_content=Mock(side_effect=responses[1:]),
        )
    )


def test_native_tools_declare_all_supported_operations():
    assert {
        declaration.name
        for declaration in gemini_chat.FUNCTION_DECLARATIONS
    } == {
        "save_note",
        "save_memory",
        "update_note",
        "update_memory",
        "delete_memory",
        "search_notes",
        "search_memories",
        "list_notes",
        "list_memories",
        "clear_notes",
        "create_task",
        "update_task",
        "list_tasks",
        "complete_task",
        "delete_task",
        "clear_tasks",
    }


def test_save_and_update_tools_use_in_process_storage(monkeypatch):
    note_calls = []
    memory_calls = []

    def save_note(user_id, title, content, idempotency_key):
        note_calls.append((user_id, title, content, idempotency_key))
        return {"id": 4, "title": title, "content": content, "created_at": ""}

    def update_note(user_id, note_id, title, content, idempotency_key):
        note_calls.append((user_id, note_id, title, content, idempotency_key))
        return {"found": True, "id": note_id, "title": title}

    def update_memory(user_id, memory_id, title, content, idempotency_key):
        memory_calls.append(
            (user_id, memory_id, title, content, idempotency_key)
        )
        return {"found": True, "id": memory_id, "title": title}

    monkeypatch.setattr(db, "save_note_idempotent", save_note)
    monkeypatch.setattr(db, "update_note_idempotent", update_note)
    monkeypatch.setattr(memory_store, "update_memory_idempotent", update_memory)

    saved = gemini_chat.dispatch_tool_call(
        "save_note",
        {"title": " Plan ", "content": " Launch "},
        user_id=7,
        user_message="Save a plan",
        history=[],
        confirmation_token="unused",
        idempotency_key="save-key",
    )
    updated_note = gemini_chat.dispatch_tool_call(
        "update_note",
        {"note_id": 4, "title": " Updated ", "content": " Revised "},
        user_id=7,
        user_message="Update the note",
        history=[],
        confirmation_token="unused",
        idempotency_key="update-note-key",
    )
    updated_memory = gemini_chat.dispatch_tool_call(
        "update_memory",
        {"memory_id": 9, "title": " Style ", "content": " Concise "},
        user_id=7,
        user_message="Update the memory",
        history=[],
        confirmation_token="unused",
        idempotency_key="update-memory-key",
    )

    assert saved.response == {
        "note_id": 4,
        "title": "Plan",
        "status": "📝 Note added",
    }
    assert updated_note.response["status"] == "📝 Note updated"
    assert updated_memory.response["status"] == "🧠 Memory updated"
    assert note_calls == [
        (7, "Plan", "Launch", "save-key"),
        (7, 4, "Updated", "Revised", "update-note-key"),
    ]
    assert memory_calls == [
        (7, 9, "Style", "Concise", "update-memory-key")
    ]


def test_memory_save_and_delete_use_the_memory_store(monkeypatch):
    calls = []

    def save_memory(user_id, title, content, idempotency_key):
        calls.append(("save", user_id, title, content, idempotency_key))
        return {"id": 12, "title": title, "content": content}

    def delete_memory(user_id, memory_id, idempotency_key):
        calls.append(("delete", user_id, memory_id, idempotency_key))
        return {"success": True}

    monkeypatch.setattr(memory_store, "save_memory_idempotent", save_memory)
    monkeypatch.setattr(memory_store, "delete_memory_idempotent", delete_memory)

    saved = gemini_chat.dispatch_tool_call(
        "save_memory",
        {"title": " Style ", "content": " Concise replies "},
        user_id=5,
        user_message="I prefer concise replies",
        history=[],
        confirmation_token="unused",
        idempotency_key="memory-key",
    )
    deleted = gemini_chat.dispatch_tool_call(
        "delete_memory",
        {"memory_id": 12},
        user_id=5,
        user_message="Forget that memory",
        history=[],
        confirmation_token="unused",
        idempotency_key="forget-key",
    )

    assert saved.response == {
        "memory_id": 12,
        "title": "Style",
        "content": "Concise replies",
        "status": "🧠 Memory saved",
    }
    assert deleted.response["success"] is True
    assert calls == [
        ("save", 5, "Style", "Concise replies", "memory-key"),
        ("delete", 5, 12, "forget-key"),
    ]


def test_search_notes_falls_back_to_keyword_search_when_embedding_fails(
    monkeypatch,
):
    notes = [{"id": 2, "title": "Garden", "content": "Tomatoes", "created_at": ""}]
    search = Mock(return_value=notes)
    monkeypatch.setattr(db, "search_notes", search)
    monkeypatch.setattr(
        gemini_chat, "embed_text", Mock(side_effect=RuntimeError("offline"))
    )
    monkeypatch.setattr(gemini_chat, "embedding_model", lambda: "embed-model")

    result = gemini_chat.dispatch_tool_call(
        "search_notes",
        {"keyword": "garden"},
        user_id=6,
        user_message="Find garden notes",
        history=[],
        confirmation_token="unused",
        gemini_client=object(),
    )

    assert result.response["notes"][0]["id"] == 2
    search.assert_called_once_with(6, "garden")


def test_list_notes_uses_storage_search_with_an_empty_keyword(monkeypatch):
    search = Mock(return_value=[])
    monkeypatch.setattr(db, "search_notes", search)

    result = gemini_chat.dispatch_tool_call(
        "list_notes",
        {},
        user_id=6,
        user_message="List notes",
        history=[],
        confirmation_token="unused",
    )

    assert result.response == {"notes": []}
    search.assert_called_once_with(6, "")


def test_task_tools_use_in_process_storage(monkeypatch):
    calls = []
    task = {
        "id": 3,
        "title": "Review",
        "due_at_utc": "",
        "legacy_due_date": "",
        "timezone": "UTC",
        "completed": False,
    }

    def update_deadline(user_id, task_id, due_at_local, idempotency_key):
        calls.append(("update", user_id, task_id, due_at_local, idempotency_key))
        return {
            "found": True,
            "id": task_id,
            "title": "Review",
            "due_at_utc": "2099-12-01T09:00:00Z",
            "timezone": "UTC",
        }

    def complete_task(user_id, task_id, idempotency_key):
        calls.append(("complete", user_id, task_id, idempotency_key))
        return {"success": True}

    def delete_task(user_id, task_id, idempotency_key):
        calls.append(("delete", user_id, task_id, idempotency_key))
        return {"success": True}

    monkeypatch.setattr(
        db, "update_task_deadline_idempotent", update_deadline
    )
    monkeypatch.setattr(db, "get_tasks", lambda user_id: [task])
    monkeypatch.setattr(db, "complete_task_idempotent", complete_task)
    monkeypatch.setattr(db, "delete_task_idempotent", delete_task)

    updated = gemini_chat.dispatch_tool_call(
        "update_task",
        {"task_id": 3, "due_at_local": " 2099-12-01 "},
        user_id=2,
        user_message="Move the task",
        history=[],
        confirmation_token="unused",
        idempotency_key="update-key",
    )
    listed = gemini_chat.dispatch_tool_call(
        "list_tasks",
        {},
        user_id=2,
        user_message="List tasks",
        history=[],
        confirmation_token="unused",
    )
    completed = gemini_chat.dispatch_tool_call(
        "complete_task",
        {"task_id": 3},
        user_id=2,
        user_message="Complete task",
        history=[],
        confirmation_token="unused",
        idempotency_key="complete-key",
    )
    deleted = gemini_chat.dispatch_tool_call(
        "delete_task",
        {"task_id": 3},
        user_id=2,
        user_message="Delete task",
        history=[],
        confirmation_token="unused",
        idempotency_key="delete-key",
    )

    assert updated.response["due_at_utc"] == "2099-12-01T09:00:00Z"
    assert listed.response["tasks"][0]["status"] == "⏳ Pending"
    assert completed.response["success"] is True
    assert deleted.response["success"] is True
    assert calls == [
        ("update", 2, 3, "2099-12-01", "update-key"),
        ("complete", 2, 3, "complete-key"),
        ("delete", 2, 3, "delete-key"),
    ]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("save_note", {"title": "Only title"}),
        ("update_note", {"note_id": True, "title": "x", "content": "y"}),
        ("update_memory", {"memory_id": True, "title": "x", "content": "y"}),
        ("delete_memory", {"memory_id": False}),
        ("search_notes", {"keyword": "  "}),
        ("search_memories", {"keyword": "  "}),
        ("complete_task", {"task_id": True}),
        ("delete_task", {"task_id": 0}),
        ("clear_tasks", {"unexpected": True}),
        ("list_tasks", {"unexpected": True}),
    ],
)
def test_invalid_tool_arguments_return_errors(name, arguments):
    result = gemini_chat.dispatch_tool_call(
        name,
        arguments,
        user_id=1,
        user_message="Do it",
        history=[],
        confirmation_token="unused",
    )

    assert result.response["error"]


def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError, match="Unknown Gemini tool"):
        gemini_chat.dispatch_tool_call(
            "delete_account",
            {},
            user_id=1,
            user_message="Delete my account",
            history=[],
            confirmation_token="unused",
        )


def test_run_turn_executes_multiple_tools_and_returns_results_to_gemini(
    monkeypatch,
):
    monkeypatch.setattr(
        db,
        "save_note_idempotent",
        lambda user_id, title, content, key: {
            "id": 3,
            "title": title,
            "content": content,
            "created_at": "",
        },
    )
    monkeypatch.setattr(db, "get_tasks", lambda user_id: [])
    client = _client(
        _response(
            ("save_note", {"title": "Plan", "content": "Launch"}),
            ("list_tasks", {}),
        ),
        _response(text="I saved the note and listed the tasks."),
    )

    events = list(gemini_chat.run_turn(
        "Save a note and show my tasks",
        client,
        "test-model",
        [],
        user_id=8,
        request_id="request",
    ))

    assert [event[0] for event in events] == [
        "tool_result", "tool_result", "final",
    ]
    assert events[0][2]["note_id"] == 3
    assert events[1][2] == {"tasks": []}
    function_responses = (
        client.models.generate_content.call_args.kwargs["contents"][-1]
    )
    assert function_responses.role == "user"
    assert [
        part.function_response.name for part in function_responses.parts
    ] == ["save_note", "list_tasks"]
    assert (
        function_responses.parts[0].function_response.response["note_id"] == 3
    )


def test_run_turn_supports_multiple_tool_rounds(monkeypatch):
    monkeypatch.setattr(db, "search_notes", lambda user_id, keyword: [])
    monkeypatch.setattr(db, "get_tasks", lambda user_id: [])
    client = _client(
        _response(("search_notes", {"keyword": "launch"})),
        _response(("list_tasks", {})),
        _response(text="I found notes and listed tasks."),
    )

    events = list(gemini_chat.run_turn(
        "Search launch notes and show tasks",
        client,
        "test-model",
        [],
        user_id=8,
        request_id="request",
    ))

    assert events == [
        ("tool_result", "search_notes", {"notes": []}),
        ("tool_result", "list_tasks", {"tasks": []}),
        ("final", "I found notes and listed tasks."),
    ]
    assert client.models.generate_content.call_count == 2


def test_run_turn_propagates_storage_failures(monkeypatch):
    monkeypatch.setattr(
        db, "get_tasks", Mock(side_effect=RuntimeError("storage unavailable"))
    )
    client = _client(_response(("list_tasks", {})), _response(text="unreachable"))

    with pytest.raises(RuntimeError, match="storage unavailable"):
        list(gemini_chat.run_turn(
            "List my tasks",
            client,
            "test-model",
            [],
            user_id=8,
            request_id="request",
        ))

    client.models.generate_content.assert_not_called()


def test_notes_confirmation_cannot_confirm_clearing_tasks():
    prompt = (
        "To confirm deleting all notes, reply exactly "
        "`confirm clear notes abc123`."
    )

    result = gemini_chat.dispatch_tool_call(
        "clear_tasks",
        {},
        user_id=3,
        user_message="confirm clear notes abc123",
        history=[{"role": "assistant", "content": prompt}],
        confirmation_token="fresh-token",
    )

    assert result.response["confirmation_required"] is True
    assert "confirm clear tasks fresh-token" in result.response["message"]


def test_listed_tool_without_a_handler_completes_nothing(monkeypatch):
    complete = Mock()
    monkeypatch.setattr(db, "complete_task_idempotent", complete)
    monkeypatch.setitem(
        gemini_chat._TOOL_FIELDS, "snooze_task", ({"task_id"}, {"task_id"})
    )
    with pytest.raises(ValueError, match="No handler for Gemini tool"):
        gemini_chat.dispatch_tool_call(
            "snooze_task", {"task_id": 1}, user_id=1,
            user_message="Snooze", history=[], confirmation_token="unused",
        )
    complete.assert_not_called()


READ_TOOLS = {
    "search_notes", "search_memories", "list_notes", "list_memories", "list_tasks",
}


def test_every_tool_that_changes_data_is_blocked_without_write_permission():
    declared = {declaration.name for declaration in gemini_chat.FUNCTION_DECLARATIONS}
    assert gemini_chat.WRITE_TOOLS == declared - READ_TOOLS
    arguments = {
        "title": "t", "content": "c", "keyword": "k", "due_at_local": "",
        "note_id": 1, "memory_id": 1, "task_id": 1,
    }
    for name in sorted(gemini_chat.WRITE_TOOLS):
        allowed = gemini_chat._TOOL_FIELDS[name][1]
        result = gemini_chat.dispatch_tool_call(
            name, {key: arguments[key] for key in allowed}, user_id=1,
            user_message="Do it", history=[], confirmation_token="unused",
            control=TurnControl(allow_writes=False),
        )
        assert result.response["error"].startswith("Changes are blocked")


def test_memory_saving_requires_an_explicit_request():
    declaration = next(
        d for d in gemini_chat.FUNCTION_DECLARATIONS if d.name == "save_memory"
    )
    prompt = " ".join(gemini_chat.SYSTEM_INSTRUCTION.split())
    assert "only when the user explicitly asks" in prompt
    assert "without waiting for an explicit request" not in prompt
    assert "only when the user explicitly asks" in declaration.description
