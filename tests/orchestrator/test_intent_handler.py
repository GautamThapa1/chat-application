import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.orchestrator.intent_handler import (
    classify_message,      # <-- was missing; tests would NameError without it
    execute_action,
    save_conversation,
)


def create_gemini_client(response_text):
    """Create a fake Gemini client for testing."""

    generate_content = AsyncMock(
        return_value=SimpleNamespace(text=response_text)
    )

    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=generate_content
        )
    )

    return client, generate_content


@pytest.mark.asyncio
async def test_classify_message_success():
    # Arrange
    decision = {
        "intent": "CHAT",
        "title": "",
        "content": "",
        "keyword": "",
        "due_date": "",
    }

    client, mock_generate = create_gemini_client(
        json.dumps(decision)
    )

    # Act
    result = await classify_message(
        "Hello",
        client,
        "test-model",
    )

    # Assert
    assert result == decision

    mock_generate.assert_awaited_once()

    call_args = mock_generate.await_args.kwargs

    assert call_args["model"] == "test-model"
    assert call_args["contents"] == 'Classify this user message:\n"Hello"'


@pytest.mark.asyncio
async def test_classify_message_empty_response():
    # Arrange
    client, _ = create_gemini_client(None)

    # Act and Assert
    with pytest.raises(ValueError, match="empty intent classification"):
        await classify_message("Hello", client, "test-model")


@pytest.mark.asyncio
async def test_classify_message_invalid_json():
    # Arrange
    client, _ = create_gemini_client("This is not JSON")

    # Act and Assert
    with pytest.raises(ValueError, match="invalid JSON"):
        await classify_message("Hello", client, "test-model")


@pytest.mark.asyncio
async def test_classify_message_unsupported_intent():
    # Arrange
    decision = {
        "intent": "DELETE_EVERYTHING",
        "title": "",
        "content": "",
        "keyword": "",
        "due_date": "",
    }

    client, _ = create_gemini_client(json.dumps(decision))

    # Act and Assert
    with pytest.raises(ValueError, match="unsupported intent"):
        await classify_message("Hello", client, "test-model")


@pytest.mark.asyncio
async def test_classify_message_missing_required_field():
    # Arrange
    decision = {
        "intent": "CHAT",
        "content": "",
        "keyword": "",
        "due_date": "",
    }

    client, _ = create_gemini_client(json.dumps(decision))

    # Act and Assert
    with pytest.raises(ValueError, match="invalid title"):
        await classify_message("Hello", client, "test-model")


@pytest.mark.asyncio
async def test_classify_message_missing_task_id():
    # Arrange
    decision = {
        "intent": "COMPLETE_TASK",
        "title": "",
        "content": "",
        "keyword": "",
        "due_date": "",
    }

    client, _ = create_gemini_client(json.dumps(decision))

    # Act and Assert
    with pytest.raises(ValueError, match="invalid task ID"):
        await classify_message(
            "Complete task 5",
            client,
            "test-model",
        )


@pytest.mark.asyncio
async def test_execute_save_note():
    # Arrange
    stub = SimpleNamespace(
        SaveNote=AsyncMock(
            return_value=SimpleNamespace(
                note=SimpleNamespace(
                    id=1,
                    title="Meeting"
                )
            )
        )
    )

    decision = {
        "intent": "SAVE_NOTE",
        "title": "Meeting",
        "content": "Meeting tomorrow",
    }

    # Act
    result = await execute_action(decision, stub)

    # Assert
    assert result == "✅ Note saved (ID: 1): 'Meeting'"

    stub.SaveNote.assert_awaited_once()

    request = stub.SaveNote.await_args.args[0]

    assert request.title == "Meeting"
    assert request.content == "Meeting tomorrow"


@pytest.mark.asyncio
async def test_execute_save_note_without_title():
    # Arrange
    stub = SimpleNamespace(SaveNote=AsyncMock())

    decision = {
        "intent": "SAVE_NOTE",
        "title": "",
        "content": "Meeting tomorrow",
    }

    # Act
    result = await execute_action(decision, stub)

    # Assert
    assert result == "What title should I use for that note?"
    stub.SaveNote.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_search_notes():
    # Arrange
    stub = SimpleNamespace(
        SearchNotes=AsyncMock(
            return_value=SimpleNamespace(
                notes=[
                    SimpleNamespace(
                        id=1,
                        title="Meeting",
                        content="Meeting tomorrow",
                    )
                ]
            )
        )
    )

    decision = {
        "intent": "SEARCH_NOTES",
        "keyword": "Meeting",
    }

    # Act
    result = await execute_action(decision, stub)

    # Assert
    assert "Found 1 note(s)" in result
    assert "[1] Meeting: Meeting tomorrow" in result

    request = stub.SearchNotes.await_args.args[0]
    assert request.keyword == "Meeting"


@pytest.mark.asyncio
async def test_execute_search_notes_without_keyword():
    # Arrange
    stub = SimpleNamespace(SearchNotes=AsyncMock())

    decision = {
        "intent": "SEARCH_NOTES",
        "keyword": "",
    }

    # Act
    result = await execute_action(decision, stub)

    # Assert
    assert result == "What should I search for in your notes?"
    stub.SearchNotes.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_create_task():
    # Arrange
    stub = SimpleNamespace(
        CreateTask=AsyncMock(
            return_value=SimpleNamespace(
                task=SimpleNamespace(
                    id=5,
                    title="Study Python",
                )
            )
        )
    )

    decision = {
        "intent": "CREATE_TASK",
        "title": "Study Python",
        "due_date": "2026-11-20",
    }

    # Act
    result = await execute_action(decision, stub)

    # Assert
    assert result == "✅ Task created (ID: 5): 'Study Python' due 2026-11-20"

    request = stub.CreateTask.await_args.args[0]

    assert request.title == "Study Python"
    assert request.due_date == "2026-11-20"


@pytest.mark.asyncio
async def test_execute_list_tasks():
    # Arrange
    stub = SimpleNamespace(
        ListTasks=AsyncMock(
            return_value=SimpleNamespace(
                tasks=[
                    SimpleNamespace(
                        id=1,
                        title="Study Python",
                        completed=False,
                        due_date="2026-11-20",
                    ),
                    SimpleNamespace(
                        id=2,
                        title="Read a book",
                        completed=True,
                        due_date="",
                    ),
                ]
            )
        )
    )

    decision = {"intent": "LIST_TASKS"}

    # Act
    result = await execute_action(decision, stub)

    # Assert
    assert "⬜ [1] Study Python (due 2026-11-20)" in result
    assert "✅ [2] Read a book" in result


@pytest.mark.asyncio
async def test_execute_list_tasks_empty():
    # Arrange
    stub = SimpleNamespace(
        ListTasks=AsyncMock(
            return_value=SimpleNamespace(tasks=[])
        )
    )

    # Act
    result = await execute_action(
        {"intent": "LIST_TASKS"},
        stub,
    )

    # Assert
    assert result == "📋 You have no tasks."


@pytest.mark.asyncio
async def test_execute_complete_task_success():
    # Arrange
    stub = SimpleNamespace(
        CompleteTask=AsyncMock(
            return_value=SimpleNamespace(success=True)
        )
    )

    decision = {
        "intent": "COMPLETE_TASK",
        "task_id": 5,
    }

    # Act
    result = await execute_action(decision, stub)

    # Assert
    assert result == "✅ Task 5 marked as complete!"

    request = stub.CompleteTask.await_args.args[0]
    assert request.task_id == 5


@pytest.mark.asyncio
async def test_execute_complete_task_not_found():
    # Arrange
    stub = SimpleNamespace(
        CompleteTask=AsyncMock(
            return_value=SimpleNamespace(success=False)
        )
    )

    decision = {
        "intent": "COMPLETE_TASK",
        "task_id": 99,
    }

    # Act
    result = await execute_action(decision, stub)

    # Assert
    assert result == "❌ Task 99 not found."


@pytest.mark.asyncio
async def test_execute_complete_task_invalid_id():
    # Arrange
    stub = SimpleNamespace(CompleteTask=AsyncMock())

    decision = {
        "intent": "COMPLETE_TASK",
        "task_id": 0,
    }

    # Act
    result = await execute_action(decision, stub)

    # Assert
    assert result == "Which task ID should I mark as complete?"
    stub.CompleteTask.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_conversation():
    # Arrange
    stub = SimpleNamespace(
        SaveConversation=AsyncMock()
    )

    # Act
    await save_conversation(
        "Hello",
        "Hi there!",
        stub,
        42,
    )

    # Assert
    stub.SaveConversation.assert_awaited_once()

    request = stub.SaveConversation.await_args.args[0]

    assert request.user_content == "Hello"
    assert request.assistant_content == "Hi there!"
    assert request.conversation_id == 42