"""Intent detection and routing to gRPC methods."""

import re

from backend.generated import chat_pb2


def detect_intent(message: str) -> str:
    """Identify the user's intent using simple string matching."""
    msg = message.lower().strip()

    if msg.startswith("remember that ") or "save note" in msg:
        return "SAVE_NOTE"
    elif msg in {"show my notes", "show notes", "list my notes", "list notes"}:
        return "LIST_NOTES"
    elif "search note" in msg or "find note" in msg:
        return "SEARCH_NOTES"
    elif "create task" in msg or "create a task" in msg or "add task" in msg:
        return "CREATE_TASK"
    elif "list task" in msg or "show task" in msg or "show my tasks" in msg:
        return "LIST_TASKS"
    elif (
        "complete task" in msg
        or "finish task" in msg
        or re.search(r"\bmark task \d+ (?:as )?completed?\b", msg)
    ):
        return "COMPLETE_TASK"
    else:
        return "GENERAL_CHAT"


async def handle_message(message: str, stub) -> str:
    """Route the message to the correct gRPC call and return a string response."""
    intent = detect_intent(message)
    msg_lower = message.lower()

    try:
        if intent == "SAVE_NOTE":
            # Syntax: "save note: Title | Content"
            if ":" in message:
                note_text = message.split(":", 1)[1]
                parts = note_text.split("|", 1)
                title = parts[0].strip()
                content = parts[1].strip() if len(parts) > 1 else ""
            else:
                content = re.sub(r"^\s*remember that\s+", "", message, flags=re.IGNORECASE).strip()
                title = content

            if not title:
                return "Error: Missing note title. Use format 'save note: Title | Content'"
            
            res = await stub.SaveNote(chat_pb2.SaveNoteRequest(title=title, content=content))
            return f"✅ Note saved (ID: {res.note.id}): '{res.note.title}'"

        elif intent in {"SEARCH_NOTES", "LIST_NOTES"}:
            # Syntax: "search notes: keyword"
            keyword = message.split(":", 1)[1].strip() if ":" in message else ""
            if intent == "SEARCH_NOTES" and not keyword:
                return "Error: Please provide a keyword to search."

            res = await stub.SearchNotes(chat_pb2.SearchNotesRequest(keyword=keyword))
            if not res.notes:
                return f"🔍 No notes found matching '{keyword}'."
            
            results = [f"- [{n.id}] {n.title}: {n.content}" for n in res.notes]
            return f"🔍 Found {len(res.notes)} note(s):\n" + "\n".join(results)

        elif intent == "CREATE_TASK":
            # Syntax: "create task: Title | Due Date (optional)"
            if ":" in message:
                task_text = message.split(":", 1)[1]
            else:
                task_text = re.sub(
                    r"^\s*(?:please\s+)?create\s+(?:a\s+)?task\s+to\s+",
                    "",
                    message,
                    flags=re.IGNORECASE,
                )
            parts = task_text.split("|", 1)
            title = parts[0].strip()
            due_date = parts[1].strip() if len(parts) > 1 else ""
            
            if not title:
                return "Error: Missing task title. Use format 'create task: Title | Due Date'"
                
            res = await stub.CreateTask(chat_pb2.CreateTaskRequest(title=title, due_date=due_date))
            return f"✅ Task created (ID: {res.task.id}): '{res.task.title}'" + (f" due {due_date}" if due_date else "")

        elif intent == "LIST_TASKS":
            res = await stub.ListTasks(chat_pb2.ListTasksRequest())
            if not res.tasks:
                return "📋 You have no tasks."
                
            lines = []
            for t in res.tasks:
                status = "✅" if t.completed else "⬜"
                due = f" (due {t.due_date})" if t.due_date else ""
                lines.append(f"{status} [{t.id}] {t.title}{due}")
            return "📋 Your Tasks:\n" + "\n".join(lines)

        elif intent == "COMPLETE_TASK":
            # Syntax: "complete task: ID"
            match = re.search(r"\b\d+\b", message)
            if not match:
                return "Error: Please provide a valid numeric task ID. (e.g., 'complete task: 1')"
            task_id = int(match.group())

            res = await stub.CompleteTask(chat_pb2.CompleteTaskRequest(task_id=task_id))
            if res.success:
                return f"✅ Task {task_id} marked as complete!"
            else:
                return f"❌ Task {task_id} not found."

        else:
            # GENERAL_CHAT
            res = await stub.SendMessage(chat_pb2.ChatMessage(role="user", content=message))
            return res.content

    except Exception as e:
        return f"⚠️ gRPC Error: {str(e)}"