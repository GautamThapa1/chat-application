"""Regression tests for defects found while investigating the app."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import types

from backend.generated import chat_pb2
from backend.service import auth_interceptor, db, gemini_chat, memory_store, note_vector_index
from backend.service.chat_service import ChatServicer
from backend.service.gemini import runner
from backend.service.main import build_http_options
from backend.service.turn_control import TurnControl


@pytest.fixture
def user(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "fix.db"))
    monkeypatch.setenv("DATABASE_KEY", "defect-test-key")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("QDRANT_COLLECTION_PREFIX", f"fix_{tmp_path.name.replace('-', '_')}")
    db.init_db()
    return db.register_user("fix_user", "correct-horse")


def dispatch(user_id, name, arguments, **kwargs):
    return gemini_chat.dispatch_tool_call(
        name, arguments, user_id, user_message="go", history=[],
        confirmation_token="token", **kwargs,
    ).response


def part_text(value):
    return types.Part(text=value)


def part_call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def response(*parts, finish=None):
    return SimpleNamespace(
        candidates=[SimpleNamespace(
            content=types.Content(role="model", parts=list(parts)), finish_reason=finish)],
        text="".join(part.text or "" for part in parts),
    )


class Models:
    def __init__(self, first, replies=()):
        self.first, self.replies, self.configs = first, list(replies), []

    def count_tokens(self, **_):
        return SimpleNamespace(total_tokens=1)

    def generate_content_stream(self, *, model, contents, config):
        self.configs.append(config)
        yield from self.first

    def generate_content(self, *, model, contents, config):
        self.configs.append(config)
        return self.replies.pop(0)


def run(first, replies=(), user_id=1, **kwargs):
    models = Models(first, replies)
    events = list(runner.run_turn(
        "go", SimpleNamespace(models=models), "m", [], user_id, "req", **kwargs))
    saved = "".join(event[1] for event in events if event[0] in ("chunk", "final"))
    return events, saved, models


# ---- tool errors and task results ------------------------------------- #
@pytest.mark.parametrize("name", ["create_task", "update_task"])
def test_unparseable_deadline_is_returned_to_the_model(user, name):
    task = db.create_task(user, "existing")
    arguments = {"title": "t", "due_at_local": "next Friday"}
    if name == "update_task":
        arguments = {"task_id": task, "due_at_local": "next Friday"}

    result = dispatch(user, name, arguments)

    assert "ISO 8601" in result["error"]
    assert [t["title"] for t in db.get_tasks(user)] == ["existing"]


def test_every_task_result_has_a_local_deadline_and_flags_the_past(user):
    created = dispatch(user, "create_task", {"title": "t", "due_at_local": "2099-12-01T09:00"})
    updated = dispatch(user, "update_task",
                       {"task_id": created["task_id"], "due_at_local": "2020-01-01T09:00"})
    listed = dispatch(user, "list_tasks", {})

    assert created["due_at_local_display"] == "Dec 1, 2099 at 9:00 AM"
    assert "warning" not in created
    assert updated["due_at_local_display"] == "Jan 1, 2020 at 9:00 AM"
    assert "past" in updated["warning"]
    assert listed["tasks"][0]["due_at_local_display"] == "Jan 1, 2020 at 9:00 AM"


def test_delete_memory_is_idempotent_and_leaves_no_copy(user):
    saved = dispatch(user, "save_memory", {"title": "Pet", "content": "Dog Miso"},
                     idempotency_key="save")
    first = dispatch(user, "delete_memory", {"memory_id": saved["memory_id"]},
                     idempotency_key="delete")
    again = dispatch(user, "delete_memory", {"memory_id": saved["memory_id"]},
                     idempotency_key="delete")

    assert first == again and first["success"] is True
    conn = db.get_connection()
    stored = [row["response_json"] for row in conn.execute("SELECT response_json FROM api_idempotency")]
    conn.close()
    assert "Miso" not in json.dumps(stored)


@pytest.mark.parametrize("kind", ["note", "task"])
def test_cleared_items_leave_no_copy(user, kind):
    arguments = {"title": "SECRET", "content": "SECRET"} if kind == "note" else {"title": "SECRET"}
    dispatch(user, "save_note" if kind == "note" else "create_task", arguments, idempotency_key="k1")
    getattr(db, f"clear_{kind}s_idempotent")(user, "k2")

    conn = db.get_connection()
    stored = [row["response_json"] for row in conn.execute("SELECT response_json FROM api_idempotency")]
    conn.close()
    assert "SECRET" not in json.dumps(stored)


def test_long_lists_are_capped(user):
    for index in range(60):
        db.save_note(user, f"n{index}", "x" * 5000)

    result = dispatch(user, "list_notes", {})

    assert len(result["notes"]) == 50
    assert result["omitted"] == 10
    assert len(result["notes"][0]["content"]) < 2100


# ---- storage ----------------------------------------------------------- #
def test_expired_sessions_and_old_idempotency_rows_are_purged(user):
    expired = db.create_session(user, ttl_seconds=-1)
    db.save_note_idempotent(user, "n", "c", "old-key")
    conn = db.get_connection()
    with conn:
        conn.execute("UPDATE api_idempotency SET updated_at = 0")
    conn.close()

    live = db.create_session(user)

    conn = db.get_connection()
    assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM api_idempotency").fetchone()[0] == 0
    conn.close()
    assert db.get_session_user(live) == user and db.get_session_user(expired) is None


def test_vector_store_outage_does_not_block_search_or_delete(user, monkeypatch):
    note = db.save_note(user, "Garden", "tomatoes")
    memory = memory_store.save_memory(user, "Garden", "tomatoes")

    def down():
        raise ConnectionError("qdrant is down")

    monkeypatch.setattr(note_vector_index, "_client_for_config", down)

    assert [r["id"] for r in db.search_notes_hybrid(user, "Garden", [1.0], "m")] == [note]
    assert memory_store.delete_memory(user, memory) is True
    note_vector_index.delete_points("any", [1])
    assert db.delete_note(user, note) is True


def test_later_connections_skip_passphrase_derivation(user, monkeypatch):
    db.get_connection().close()
    used = []
    real = db._apply_database_key
    monkeypatch.setattr(db, "_apply_database_key",
                        lambda conn, key: used.append(key) or real(conn, key))

    db.get_connection().close()
    assert len(used) == 1 and used[0].startswith("x'")

    monkeypatch.setenv("DATABASE_KEY", "wrong-key")
    with pytest.raises(RuntimeError, match="DATABASE_KEY"):
        db.get_connection()


# ---- service handlers -------------------------------------------------- #
class Context:
    def abort(self, code, message):
        raise ValueError(f"{code.name}: {message}")


@pytest.mark.parametrize(("method", "request_message"), [
    ("SaveMemory", chat_pb2.SaveMemoryRequest(title=" ", content="c")),
    ("UpdateMemory", chat_pb2.UpdateMemoryRequest(memory_id=1, title="t", content="")),
    ("SaveNote", chat_pb2.SaveNoteRequest(title="t", content=" ")),
    ("UpdateNote", chat_pb2.UpdateNoteRequest(note_id=1)),
    ("CreateTask", chat_pb2.CreateTaskRequest(title="  ")),
    ("SaveConversation", chat_pb2.SaveConversationRequest(conversation_id=1, user_content="q")),
])
def test_service_rejects_blank_text(user, method, request_message):
    token = auth_interceptor._current_user.set(user)
    try:
        with pytest.raises(ValueError, match="INVALID_ARGUMENT"):
            getattr(ChatServicer(), method)(request_message, Context())
    finally:
        auth_interceptor._current_user.reset(token)


def test_conversation_search_embeds_the_query_in_the_service(user, monkeypatch):
    seen = []
    monkeypatch.setattr(db, "search_conversations_hybrid",
                        lambda uid, keyword, vector, model, limit: seen.append(vector) or [])
    client = SimpleNamespace(models=SimpleNamespace(embed_content=lambda **_: SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.5, 0.25])])))
    token = auth_interceptor._current_user.set(user)
    try:
        ChatServicer(gemini_client=client).SearchConversations(
            chat_pb2.SearchConversationsRequest(keyword="garden"), Context())
    finally:
        auth_interceptor._current_user.reset(token)

    assert seen == [[0.5, 0.25]]


# ---- agent replies ----------------------------------------------------- #
def test_text_around_tool_calls_is_kept_and_separated(user):
    events, saved, _ = run(
        [response(part_text("Looking."), part_call("list_notes"))],
        [response(part_text("Checking tasks."), part_call("list_tasks")),
         response(part_text("Done."))],
        user_id=user,
    )

    assert saved == "Looking.\n\nChecking tasks.\n\nDone."


@pytest.mark.parametrize(("finish", "note"), [
    (types.FinishReason.MAX_TOKENS, "cut off"),
    (types.FinishReason.SAFETY, "stopped early"),
    (types.FinishReason.STOP, None),
])
def test_incomplete_replies_are_marked(finish, note):
    _, saved, _ = run([response(part_text("Half"), finish=finish)])

    assert saved == "Half" if note is None else (saved.startswith("Half") and note in saved)


def test_incomplete_reply_after_a_tool_is_marked(user):
    _, saved, _ = run(
        [response(part_call("list_tasks"))],
        [response(part_text("Half"), finish=types.FinishReason.MAX_TOKENS)],
        user_id=user,
    )

    assert "cut off" in saved


def test_model_is_told_the_current_time(user):
    control = TurnControl(time_context="It is Friday in Asia/Kathmandu.")
    _, _, models = run([response(part_call("list_tasks"))], [response(part_text("ok"))],
                       user_id=user, control=control)

    assert all("Asia/Kathmandu" in config.system_instruction for config in models.configs)


def test_time_context_uses_the_users_timezone(user):
    db.set_user_timezone(user, "Asia/Kathmandu")
    from backend.service.chat_service import _time_context

    assert "Asia/Kathmandu" in _time_context(user)


# ---- configuration ----------------------------------------------------- #
def test_server_time_limit_covers_the_whole_turn():
    assert build_http_options().headers == {"X-Server-Timeout": "180"}


def test_database_path_from_env_file_is_applied(monkeypatch):
    from backend.service.main import apply_env_settings

    monkeypatch.setattr(db, "DB_PATH", "app.db")
    monkeypatch.setattr("backend.service.main.load_dotenv",
                        lambda *a, **k: monkeypatch.setenv("DB_PATH", "/tmp/from-env.db"))

    apply_env_settings()

    assert db.DB_PATH == "/tmp/from-env.db"


def test_web_app_reads_env_file_before_its_settings():
    code = (
        "import os, dotenv\n"
        "dotenv.load_dotenv = lambda *a, **k: os.environ.update("
        "ALLOWED_ORIGINS='https://from-env.example', COOKIE_SECURE='true')\n"
        "import backend.orchestrator.main\n"
        "from backend.orchestrator import auth\n"
        "print(sorted(auth.ALLOWED_ORIGINS), auth.COOKIE_SECURE)\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parents[2],
    ).stdout

    assert output.strip() == "['https://from-env.example'] True"


def test_expired_confirmations_are_purged(user):
    conversation = db.create_conversation(user)["id"]
    db.create_pending_confirmation(user, conversation, "clear_notes", "t", ttl_seconds=-1)

    db.create_session(user)

    conn = db.get_connection()
    assert conn.execute("SELECT count(*) FROM pending_confirmations").fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize("action", ["delete", "update"])
def test_failed_note_write_never_removes_its_vector(user, monkeypatch, action):
    note = db.save_note(user, "Old", "first")
    assert db.upsert_note_embedding(
        user, note, [1.0, 0.0], "m", db.note_content_hash("Old", "first"))
    removed = []
    monkeypatch.setattr(note_vector_index, "delete_points", lambda *a: removed.append(a))

    def fail(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(db, "_forget_stored_text" if action == "delete" else "_queue_note_embedding", fail)
    with pytest.raises(RuntimeError):
        if action == "delete":
            db.delete_note(user, note)
        else:
            db.update_note(user, note, "New", "second")

    assert removed == []


@pytest.mark.parametrize("name", ["Asia/Katmandu", "Asia/Calcutta"])
def test_older_timezone_names_reported_by_browsers_are_accepted(user, name):
    assert db.set_user_timezone(user, name) is True
    assert db.get_user_timezone(user) == name
