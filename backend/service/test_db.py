"""Database tests for user-owned data and legacy-data migration."""

import sqlite3

import pytest

from backend.embeddings import note_content_hash
import backend.service.db as db


@pytest.fixture
def owners_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DATABASE_KEY", "test-database-key")
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


def test_missing_note_embeddings_are_backfilled_idempotently(owners_db):
    user_a, _ = owners_db
    note_id = db.save_note(user_a, "Research", "renewable energy")
    conn = db.get_connection()
    try:
        conn.execute(
            "DELETE FROM note_embedding_jobs WHERE note_id = ?",
            (note_id,),
        )
    finally:
        conn.close()

    db.init_db()
    db.init_db()

    jobs = db.get_pending_note_embeddings()
    assert [job["id"] for job in jobs] == [note_id]


def test_embedding_upsert_is_idempotent_and_requires_current_note_content(owners_db):
    user_a, _ = owners_db
    note_id = db.save_note(user_a, "Research", "renewable energy")
    content_hash = note_content_hash("Research", "renewable energy")

    assert db.upsert_note_embedding(
        user_a, note_id, [1.0, 0.0], "test-model", content_hash
    )
    assert db.upsert_note_embedding(
        user_a, note_id, [0.8, 0.2], "test-model", content_hash
    )

    conn = db.get_connection()
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM note_embeddings WHERE note_id = ?",
            (note_id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM note_embedding_jobs WHERE note_id = ?",
            (note_id,),
        ).fetchone()[0] == 0
    finally:
        conn.close()

    assert not db.upsert_note_embedding(
        user_a, note_id, [1.0, 0.0], "test-model", "stale-content"
    )


def test_note_updates_requeue_embeddings_and_clear_cascades_index(owners_db):
    user_a, _ = owners_db
    note_id = db.save_note(user_a, "Draft", "old content")
    assert db.upsert_note_embedding(
        user_a,
        note_id,
        [1.0, 0.0],
        "test-model",
        note_content_hash("Draft", "old content"),
    )

    assert db.update_note(user_a, note_id, "Final", "new content")
    assert not db.update_note(user_a + 1, note_id, "Private", "not owned")
    jobs = db.get_pending_note_embeddings()
    assert [(job["id"], job["title"], job["content"]) for job in jobs] == [
        (note_id, "Final", "new content")
    ]

    assert db.upsert_note_embedding(
        user_a,
        note_id,
        [0.0, 1.0],
        "test-model",
        note_content_hash("Final", "new content"),
    )
    assert db.clear_notes(user_a) == 1
    conn = db.get_connection()
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM note_embeddings WHERE note_id = ?",
            (note_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM note_embedding_jobs WHERE note_id = ?",
            (note_id,),
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_hybrid_search_ranks_semantically_and_isolates_users(owners_db):
    user_a, user_b = owners_db
    relevant_id = db.save_note(user_a, "Garden", "How to grow tomatoes")
    unrelated_id = db.save_note(user_a, "Travel", "Packing for a trip")
    private_id = db.save_note(user_b, "Private garden", "tomato plan")
    for user_id, note_id, title, content, vector in [
        (user_a, relevant_id, "Garden", "How to grow tomatoes", [1.0, 0.0]),
        (user_a, unrelated_id, "Travel", "Packing for a trip", [0.0, 1.0]),
        (user_b, private_id, "Private garden", "tomato plan", [1.0, 0.0]),
    ]:
        assert db.upsert_note_embedding(
            user_id,
            note_id,
            vector,
            "test-model",
            note_content_hash(title, content),
        )

    results = db.search_notes_hybrid(
        user_a, "", [1.0, 0.0], "test-model"
    )
    assert [note["id"] for note in results] == [relevant_id, unrelated_id]
    assert private_id not in {note["id"] for note in results}


def test_embedding_failures_are_scheduled_for_retry_without_losing_note(owners_db):
    user_a, _ = owners_db
    note_id = db.save_note(user_a, "Persistent", "saved before embedding")

    db.record_note_embedding_failure(note_id, "temporary API failure")

    conn = db.get_connection()
    try:
        job = conn.execute(
            "SELECT attempts, next_attempt_at, last_error "
            "FROM note_embedding_jobs WHERE note_id = ?",
            (note_id,),
        ).fetchone()
        assert job["attempts"] == 1
        assert job["next_attempt_at"] > 0
        assert job["last_error"] == "temporary API failure"
    finally:
        conn.close()
    assert db.search_notes(user_a, "saved before embedding")[0]["id"] == note_id
    assert db.get_pending_note_embeddings() == []
    conn = db.get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE note_embedding_jobs SET next_attempt_at = 0 WHERE note_id = ?",
                (note_id,),
            )
    finally:
        conn.close()
    assert [job["id"] for job in db.get_pending_note_embeddings()] == [note_id]


def test_search_notes_is_case_insensitive_and_owner_scoped(owners_db):
    user_a, user_b = owners_db
    db.save_note(user_a, "Meeting", "discuss Q4 roadmap")
    db.save_note(user_a, "Shopping", "buy a new laptop")
    db.save_note(user_b, "Roadmap", "private plan")

    hits = db.search_notes(user_a, "roadmap")
    assert len(hits) == 1 and hits[0]["title"] == "Meeting"
    assert db.search_notes(user_a, "LAPTOP")[0]["title"] == "Shopping"
    assert db.search_notes(user_b, "roadmap")[0]["title"] == "Roadmap"


def test_clear_notes_deletes_only_the_owners_notes(owners_db):
    user_a, user_b = owners_db
    db.save_note(user_a, "First", "first note")
    db.save_note(user_a, "Second", "second note")
    db.save_note(user_b, "Private", "another user's note")

    assert db.clear_notes(user_a) == 2
    assert db.search_notes(user_a, "") == []
    assert [note["title"] for note in db.search_notes(user_b, "")] == ["Private"]
    assert db.clear_notes(user_a) == 0


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
    conversation_a3 = db.create_conversation(user_a)
    conversation_b = db.create_conversation(user_b)

    db.save_conversation(user_a, conversation_a["id"], "hello A", "reply A")
    db.save_conversation(user_a, conversation_a2["id"], "hello A2", "reply A2")
    db.save_conversation(user_a, conversation_a3["id"], "hello A3", "reply A3")
    db.save_conversation(user_b, conversation_b["id"], "hello B", "reply B")

    # First page: newest 2 conversations
    page_1, total_a = db.list_conversations(
        user_a, limit=2, offset=0
    )

    assert [row["id"] for row in page_1] == [
        conversation_a3["id"],
        conversation_a2["id"],
    ]
    assert total_a == 3

    # Second page: remaining conversation
    page_2, total_a = db.list_conversations(
        user_a, limit=2, offset=2
    )

    assert [row["id"] for row in page_2] == [
        conversation_a["id"]
    ]
    assert total_a == 3

    # User B must only see their own conversations
    page_b, total_b = db.list_conversations(
        user_b, limit=20, offset=0
    )

    assert [row["id"] for row in page_b] == [
        conversation_b["id"]
    ]
    assert total_b == 1

    # Offset beyond available records returns an empty page
    empty_page, total = db.list_conversations(
        user_a, limit=2, offset=10
    )

    assert empty_page == []
    assert total == 3

    # Existing ownership checks
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
    monkeypatch.setenv("DATABASE_KEY", "test-database-key")
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

    conn = db.get_connection()
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
    monkeypatch.setenv("DATABASE_KEY", "test-database-key")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT);
        CREATE TABLE legacy_messages (id INTEGER PRIMARY KEY, content TEXT);
    """)
    conn.close()

    with pytest.raises(RuntimeError, match="legacy_messages already exists"):
        db.init_db()

    conn = db.get_connection()
    try:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='legacy_messages'"
        ).fetchone()
    finally:
        conn.close()


def test_plaintext_database_is_encrypted_and_data_is_preserved(
    tmp_path, monkeypatch,
):
    path = tmp_path / "plaintext.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    monkeypatch.setenv("DATABASE_KEY", "test's-database-key")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE private_data (content TEXT)")
    conn.execute("INSERT INTO private_data VALUES ('preserved')")
    conn.commit()
    conn.close()

    conn = db.get_connection()
    try:
        assert conn.execute(
            "SELECT content FROM private_data"
        ).fetchone()["content"] == "preserved"
    finally:
        conn.close()

    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(path).execute("SELECT * FROM sqlite_master").fetchall()


def test_encrypted_database_rejects_a_different_key(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "encrypted.db"))
    monkeypatch.setenv("DATABASE_KEY", "original-database-key")
    db.init_db()
    monkeypatch.setenv("DATABASE_KEY", "different-database-key")

    with pytest.raises(RuntimeError, match="Could not open the database"):
        db.get_connection()


def test_database_requires_a_key(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "missing-key.db"))
    monkeypatch.setattr(db, "ENV_PATH", tmp_path / "missing.env")
    monkeypatch.delenv("DATABASE_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Set DATABASE_KEY"):
        db.get_connection()


def test_list_conversations_rejects_invalid_limit(owners_db):
    user_a, _ = owners_db

    with pytest.raises(ValueError):
        db.list_conversations(user_a, limit=0)

    with pytest.raises(ValueError):
        db.list_conversations(user_a, limit=101)


def test_list_conversations_rejects_negative_offset(owners_db):
    user_a, _ = owners_db

    with pytest.raises(ValueError):
        db.list_conversations(user_a, limit=20, offset=-1)
