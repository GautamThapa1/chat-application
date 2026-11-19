"""Database tests for user-owned data and legacy-data migration."""

import sqlite3

import pytest

import backend.service.db as db


@pytest.fixture
def owners_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    user_a = db.register_user("alice", "correct-horse")
    user_b = db.register_user("bob", "another-horse")
    return user_a, user_b


def test_notes_are_isolated_by_owner(owners_db):
    user_a, user_b = owners_db
    note_id = db.save_note(user_a, "Grocery", "milk, eggs, bread")

    assert note_id > 0
    assert db.search_notes(user_a, "milk")[0]["title"] == "Grocery"
    assert db.search_notes(user_b, "milk") == []


def test_search_notes_is_case_insensitive_and_owner_scoped(owners_db):
    user_a, user_b = owners_db
    db.save_note(user_a, "Meeting", "discuss Q4 roadmap")
    db.save_note(user_a, "Shopping", "buy a new laptop")
    db.save_note(user_b, "Roadmap", "private plan")

    hits = db.search_notes(user_a, "roadmap")
    assert len(hits) == 1 and hits[0]["title"] == "Meeting"
    assert db.search_notes(user_a, "LAPTOP")[0]["title"] == "Shopping"
    assert db.search_notes(user_b, "roadmap")[0]["title"] == "Roadmap"


def test_tasks_are_isolated_and_updates_check_owner(owners_db):
    user_a, user_b = owners_db
    task_a = db.create_task(user_a, "Write report", due_date="2026-12-01")
    task_b = db.create_task(user_b, "Private task")

    assert task_a > 0 and task_b > 0
    assert [task["title"] for task in db.get_tasks(user_a)] == ["Write report"]
    assert [task["title"] for task in db.get_tasks(user_b)] == ["Private task"]
    assert db.complete_task(user_b, task_a) is False
    assert db.get_tasks(user_a)[0]["completed"] == 0
    assert db.complete_task(user_a, task_a) is True
    assert db.get_tasks(user_a)[0]["completed"] == 1


def test_conversations_messages_and_deletion_are_owner_scoped(owners_db):
    user_a, user_b = owners_db
    conversation_a = db.create_conversation(user_a)
    conversation_a2 = db.create_conversation(user_a)
    conversation_b = db.create_conversation(user_b)
    db.save_conversation(user_a, conversation_a["id"], "hello A", "reply A")
    db.save_conversation(user_a, conversation_a2["id"], "hello A2", "reply A2")
    db.save_conversation(user_b, conversation_b["id"], "hello B", "reply B")

    assert [row["id"] for row in db.list_conversations(user_a)] == [
        conversation_a2["id"],
        conversation_a["id"],
    ]
    assert db.list_conversations(user_b)[0]["id"] == conversation_b["id"]
    assert db.get_conversation(user_b, conversation_a["id"]) is None
    assert db.get_conversation_messages(user_b, conversation_a["id"]) == []
    assert db.delete_conversation(user_b, conversation_a["id"]) is False
    assert len(db.get_conversation_messages(user_a, conversation_a["id"])) == 2
    assert db.delete_conversation(user_a, conversation_a["id"]) is True
    assert db.get_conversation_messages(user_a, conversation_a["id"]) == []
    assert db.get_conversation(user_b, conversation_b["id"]) is not None


def test_unowned_legacy_records_are_preserved_without_assignment(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE notes (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            due_date TEXT,
            completed INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO messages (role, content) VALUES ('user', 'old message');
        INSERT INTO notes (title, content) VALUES ('old note', 'old content');
        INSERT INTO tasks (title) VALUES ('old task');
    """)
    conn.close()

    db.init_db()
    db.init_db()

    conn = sqlite3.connect(path)
    try:
        assert {
            row[1] for row in conn.execute("PRAGMA table_info(conversations)")
        } >= {"conversation_id", "user_id", "title", "created_at", "updated_at"}
        assert {
            row[1] for row in conn.execute("PRAGMA table_info(messages)")
        } >= {"message_id", "conversation_id", "user_id", "role", "content", "created_at"}
        assert "user_id" in {
            row[1] for row in conn.execute("PRAGMA table_info(notes)")
        }
        assert "user_id" in {
            row[1] for row in conn.execute("PRAGMA table_info(tasks)")
        }
        assert conn.execute("SELECT content FROM legacy_messages").fetchone()[0] == "old message"
        assert conn.execute("SELECT title FROM legacy_notes").fetchone()[0] == "old note"
        assert conn.execute("SELECT title FROM legacy_tasks").fetchone()[0] == "old task"
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    finally:
        conn.close()


def test_migration_refuses_to_overwrite_an_existing_legacy_table(tmp_path, monkeypatch):
    path = tmp_path / "conflict.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT);
        CREATE TABLE legacy_messages (id INTEGER PRIMARY KEY, content TEXT);
    """)
    conn.close()

    with pytest.raises(RuntimeError, match="legacy_messages already exists"):
        db.init_db()

    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='legacy_messages'"
        ).fetchone()
    finally:
        conn.close()
