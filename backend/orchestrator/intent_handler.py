"""Intent detection and routing to gRPC methods."""

import logging
import re

from backend.generated import chat_pb2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parsing patterns (compiled once)
#
# Each pattern is anchored to the *command phrase* rather than to the first
# ":" in the message. This keeps free text such as "meeting at 5:30pm" from
# being mistaken for the "command: argument" separator.
# ---------------------------------------------------------------------------
_FLAGS = re.IGNORECASE | re.DOTALL

_REMEMBER_THAT_RE = re.compile(r"^\s*remember\s+that\s+(?P<body>.*)$", _FLAGS)
_SAVE_NOTE_RE = re.compile(r"\bsave\s+notes?\b\s*:?\s*(?P<body>.*)$", _FLAGS)
_SEARCH_NOTES_RE = re.compile(
    r"\b(?:search|find)\s+notes?\b\s*"
    r"(?:(?:for|about|containing|matching)\b)?\s*:?\s*(?P<keyword>.*)$",
    _FLAGS,
)
_CREATE_TASK_RE = re.compile(
    r"\b(?:create|add)\s+(?:a\s+)?tasks?\b\s*(?::\s*|to(?:\s+|$))?(?P<body>.*)$",
    _FLAGS,
)
_MARK_COMPLETE_RE = re.compile(r"\bmark task #?\d+ (?:as )?completed?\b")
_TASK_ID_AFTER_CMD_RE = re.compile(
    r"\b(?:complete|finish|mark)\s+tasks?\b\s*[:#]?\s*#?(\d+)\b", re.IGNORECASE
)
_ANY_NUMBER_RE = re.compile(r"(?<![\w-])\d+\b")

_LIST_NOTES_PHRASES = {"show my notes", "show notes", "list my notes", "list notes"}


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------
def detect_intent(message: str) -> str:
    """Identify the user's intent using simple string matching."""
    if not isinstance(message, str):
        return "GENERAL_CHAT"

    # Lowercase and collapse runs of whitespace so "create   task" still matches.
    msg = " ".join(message.lower().split())

    if msg.startswith("remember that ") or "save note" in msg:
        return "SAVE_NOTE"
    elif msg.rstrip(".!?") in _LIST_NOTES_PHRASES:
        return "LIST_NOTES"
    elif "search note" in msg or "find note" in msg:
        return "SEARCH_NOTES"
    elif (
        "create task" in msg
        or "create a task" in msg
        or "add task" in msg
        or "add a task" in msg
    ):
        return "CREATE_TASK"
    elif "list task" in msg or "show task" in msg or "show my tasks" in msg:
        return "LIST_TASKS"
    elif (
        "complete task" in msg
        or "finish task" in msg
        or _MARK_COMPLETE_RE.search(msg)
    ):
        return "COMPLETE_TASK"
    else:
        return "GENERAL_CHAT"


# ---------------------------------------------------------------------------
# Per-intent handlers (private). Each takes (message, stub) and returns a str.
# ---------------------------------------------------------------------------
async def _handle_save_note(message: str, stub) -> str:
    # Syntax: "save note: Title | Content"  or  "remember that <text>"
    remember = _REMEMBER_THAT_RE.match(message)
    if remember:
        content = remember.group("body").strip()
        title = content
    else:
        match = _SAVE_NOTE_RE.search(message)
        body = match.group("body") if match else message
        title_part, _, content_part = body.partition("|")
        title = title_part.strip()
        content = content_part.strip()

    if not title:
        return "Error: Missing note title. Use format 'save note: Title | Content'"

    res = await stub.SaveNote(chat_pb2.SaveNoteRequest(title=title, content=content))
    return f"✅ Note saved (ID: {res.note.id}): '{res.note.title}'"


async def _query_notes(stub, keyword: str) -> str:
    res = await stub.SearchNotes(chat_pb2.SearchNotesRequest(keyword=keyword))
    if not res.notes:
        if keyword:
            return f"🔍 No notes found matching '{keyword}'."
        return "🔍 No notes found."

    results = [f"- [{n.id}] {n.title}: {n.content}" for n in res.notes]
    return f"🔍 Found {len(res.notes)} note(s):\n" + "\n".join(results)


async def _handle_search_notes(message: str, stub) -> str:
    # Syntax: "search notes: keyword"  (also "search notes for keyword")
    match = _SEARCH_NOTES_RE.search(message)
    keyword = (match.group("keyword") if match else message.partition(":")[2]).strip()
    if not keyword:
        return "Error: Please provide a keyword to search."
    return await _query_notes(stub, keyword)


async def _handle_list_notes(message: str, stub) -> str:
    # "show my notes" etc. -- an empty keyword lists everything.
    return await _query_notes(stub, "")


async def _handle_create_task(message: str, stub) -> str:
    # Syntax: "create task: Title | Due Date (optional)"
    #     or  "create a task to Title"  /  "add task Title"
    match = _CREATE_TASK_RE.search(message)
    task_text = match.group("body") if match else message.partition(":")[2]

    title_part, _, due_part = task_text.partition("|")
    title = title_part.strip()
    due_date = due_part.strip()

    if not title:
        return "Error: Missing task title. Use format 'create task: Title | Due Date'"

    res = await stub.CreateTask(chat_pb2.CreateTaskRequest(title=title, due_date=due_date))
    return f"✅ Task created (ID: {res.task.id}): '{res.task.title}'" + (
        f" due {due_date}" if due_date else ""
    )


async def _handle_list_tasks(message: str, stub) -> str:
    res = await stub.ListTasks(chat_pb2.ListTasksRequest())
    if not res.tasks:
        return "📋 You have no tasks."

    lines = []
    for t in res.tasks:
        status = "✅" if t.completed else "⬜"
        due = f" (due {t.due_date})" if t.due_date else ""
        lines.append(f"{status} [{t.id}] {t.title}{due}")
    return "📋 Your Tasks:\n" + "\n".join(lines)


async def _handle_complete_task(message: str, stub) -> str:
    # Syntax: "complete task: ID"  (also "mark task 3 as completed")
    invalid_id_msg = "Error: Please provide a valid numeric task ID. (e.g., 'complete task: 1')"

    # Prefer the number that directly follows the command; fall back to the
    # first standalone number (e.g. "complete task number 3").
    match = _TASK_ID_AFTER_CMD_RE.search(message)
    raw_id = match.group(1) if match else None
    if raw_id is None:
        fallback = _ANY_NUMBER_RE.search(message)
        raw_id = fallback.group() if fallback else None
    if raw_id is None:
        return invalid_id_msg

    task_id = int(raw_id)
    if task_id < 1:
        return invalid_id_msg

    try:
        request = chat_pb2.CompleteTaskRequest(task_id=task_id)
    except (ValueError, TypeError):
        # e.g. integer too large for the protobuf field type
        return invalid_id_msg

    res = await stub.CompleteTask(request)
    if res.success:
        return f"✅ Task {task_id} marked as complete!"
    return f"❌ Task {task_id} not found."


async def _handle_general_chat(message: str, stub) -> str:
    res = await stub.SendMessage(chat_pb2.ChatMessage(role="user", content=message))
    return res.content


_HANDLERS = {
    "SAVE_NOTE": _handle_save_note,
    "SEARCH_NOTES": _handle_search_notes,
    "LIST_NOTES": _handle_list_notes,
    "CREATE_TASK": _handle_create_task,
    "LIST_TASKS": _handle_list_tasks,
    "COMPLETE_TASK": _handle_complete_task,
    "GENERAL_CHAT": _handle_general_chat,
}


def _describe_error(exc: Exception) -> str:
    """Prefer a gRPC error's short details() over its verbose str()."""
    details = getattr(exc, "details", None)
    if callable(details):
        try:
            text = details()
        except Exception:
            text = None
        if text:
            return str(text)
    return str(exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
async def handle_message(message: str, stub) -> str:
    """Route the message to the correct gRPC call and return a string response."""
    if not isinstance(message, str):
        message = "" if message is None else str(message)
    if not message.strip():
        return "Error: Empty message."

    intent = detect_intent(message)
    handler = _HANDLERS.get(intent, _handle_general_chat)

    try:
        return await handler(message, stub)
    except Exception as exc:  # noqa: BLE001 - surface any failure as text
        logger.exception("Error while handling intent %s", intent)
        return f"⚠️ gRPC Error: {_describe_error(exc)}"