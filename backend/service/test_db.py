"""Database tests for user-owned data and legacy-data migration."""

import sqlite3
import struct
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.embeddings import embedding_model, note_content_hash
from backend.conversation_chunks import chunk_conversation, conversation_snapshot_hash
import backend.service.db as db
from backend.service import embedding_worker, memory_store


@pytest.fixture
def owners_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DATABASE_KEY", "test-database-key")
    # Keep vector tests isolated and deterministic; no external Qdrant service
    # is needed for database-level search and migration tests.
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv(
        "QDRANT_COLLECTION_PREFIX",
        f"test_{tmp_path.name.replace('-', '_')}",
    )
    db.init_db()
    user_a = db.register_user("alice", "correct-horse")
    user_b = db.register_user("bob", "another-horse")
    return user_a, user_b


def test_edit_message_updates_and_deletes_later_messages(owners_db):
    user_a, user_b = owners_db
    conversation = db.create_conversation(user_a, "Edit test")
    db.save_conversation(user_a, conversation["id"], "First", "First answer")
    db.save_conversation(user_a, conversation["id"], "Second", "Second answer")

    messages = db.get_conversation_messages(user_a, conversation["id"])
    first_user = next(row for row in messages if row["role"] == "user")

    success, content = db.edit_conversation_message(
        user_a,
        conversation["id"],
        first_user["id"],
        "Edited first",
    )
    assert success
    assert content == "Edited first"
    remaining = db.get_conversation_messages(user_a, conversation["id"])
    assert [row["content"] for row in remaining] == ["Edited first"]

    assert not db.edit_conversation_message(
        user_b,
        conversation["id"],
        first_user["id"],
        "Private edit",
    )[0]


def test_saving_after_an_edit_does_not_duplicate_the_user_message(owners_db):
    user_a, _ = owners_db
    conversation = db.create_conversation(user_a, "Edit save")
    db.save_conversation(user_a, conversation["id"], "Hi", "Hello", "r1")
    first_user = db.get_conversation_messages(user_a, conversation["id"])[0]

    # The websocket does exactly this: edit first, then save the new answer.
    db.edit_conversation_message(user_a, conversation["id"], first_user["id"], "Hi there")
    db.save_conversation(user_a, conversation["id"], "Hi there", "New answer", "r2")

    rows = db.get_conversation_messages(user_a, conversation["id"])
    assert [(row["role"], row["content"]) for row in rows] == [
        ("user", "Hi there"),
        ("assistant", "New answer"),
    ]


def test_saving_the_same_text_after_an_answered_turn_still_adds_a_pair(owners_db):
    user_a, _ = owners_db
    conversation = db.create_conversation(user_a, "Repeat")
    db.save_conversation(user_a, conversation["id"], "Hi", "Hello", "r1")
    db.save_conversation(user_a, conversation["id"], "Hi", "Hello again", "r2")

    rows = db.get_conversation_messages(user_a, conversation["id"])
    assert [row["role"] for row in rows] == [
        "user", "assistant", "user", "assistant",
    ]


def test_regenerate_context_and_assistant_replacement(owners_db):
    user_a, _ = owners_db
    conversation = db.create_conversation(user_a, "Regenerate test")
    db.save_conversation(user_a, conversation["id"], "Question", "Old answer")
    db.save_conversation(user_a, conversation["id"], "Later", "Later answer")

    messages = db.get_conversation_messages(user_a, conversation["id"])
    first_assistant = next(row for row in messages if row["role"] == "assistant")

    success, user_content, history = db.get_regeneration_context(
        user_a,
        conversation["id"],
        first_assistant["id"],
    )
    assert success
    assert user_content == "Question"
    assert history == []

    assert db.replace_assistant_response(
        user_a,
        conversation["id"],
        first_assistant["id"],
        "New answer",
        "regenerate-request",
    )
    remaining = db.get_conversation_messages(user_a, conversation["id"])
    assert [row["content"] for row in remaining] == ["Question", "New answer"]

    # The same request ID does not create or replace the response twice.
    assert db.replace_assistant_response(
        user_a,
        conversation["id"],
        first_assistant["id"],
        "Should not replace",
        "regenerate-request",
    )
    remaining = db.get_conversation_messages(user_a, conversation["id"])
    assert remaining[-1]["content"] == "New answer"


def test_first_user_message_sets_a_short_relevant_conversation_title(owners_db):
    user_a, _ = owners_db
    conversation = db.create_conversation(user_a)

    db.save_conversation(
        user_a,
        conversation["id"],
        "Could you help me plan a weekend trip to Pokhara? Include hiking.",
        "Sure, let's plan it.",
    )
    db.save_conversation(
        user_a,
        conversation["id"],
        "Also suggest restaurants.",
        "Here are some options.",
    )

    rows, _ = db.list_conversations(user_a)
    assert rows[0]["title"] == "Plan a weekend trip to Pokhara"


def test_notes_are_isolated_by_owner(owners_db):
    user_a, user_b = owners_db
    note_id = db.save_note(user_a, "Grocery", "milk, eggs, bread")

    assert note_id > 0
    assert db.search_notes(user_a, "milk")[0]["title"] == "Grocery"
    assert db.search_notes(user_b, "milk") == []


def test_memories_use_separate_owner_scoped_storage(owners_db):
    user_a, user_b = owners_db
    note_id = db.save_note(user_a, "Preference", "Prefers concise answers")
    memory_id = memory_store.save_memory(
        user_a, "Answer style", "Prefers concise answers"
    )

    note_rows = db.search_notes(user_a, "concise")
    assert [(row["id"], row["title"]) for row in note_rows] == [
        (note_id, "Preference")
    ]
    assert [row["id"] for row in memory_store.search_memories(user_a, "concise")] == [
        memory_id
    ]
    assert memory_store.search_memories(user_b, "concise") == []
    memories, total_count = memory_store.list_memories(user_a)
    assert total_count == 1
    assert [row["id"] for row in memories] == [memory_id]


def test_memory_embeddings_are_separate_and_deleted_with_the_memory(
    owners_db, monkeypatch,
):
    user_a, user_b = owners_db
    memory_id = memory_store.save_memory(
        user_a, "Project", "A stable project context"
    )
    content_hash = note_content_hash("Project", "A stable project context")
    assert memory_store.upsert_memory_embedding(
        user_a, memory_id, [1.0, 0.0], "test-model", content_hash
    )

    deleted_points = []
    delete_points = db.note_vector_index.delete_points

    def record_delete(collection, ids):
        deleted_points.append((collection, list(ids)))
        return delete_points(collection, ids)

    monkeypatch.setattr(db.note_vector_index, "delete_points", record_delete)
    assert memory_store.update_memory(
        user_a, memory_id, "Project", "An updated project context"
    )
    assert not memory_store.upsert_memory_embedding(
        user_a, memory_id, [1.0, 0.0], "test-model", content_hash
    )
    updated_hash = note_content_hash("Project", "An updated project context")
    assert memory_store.upsert_memory_embedding(
        user_a, memory_id, [0.0, 1.0], "test-model", updated_hash
    )
    assert not memory_store.delete_memory(user_b, memory_id)
    assert memory_store.delete_memory(user_a, memory_id)
    assert len(deleted_points) == 2
    assert all(point_ids == [memory_id] for _, point_ids in deleted_points)
    assert not memory_store.upsert_memory_embedding(
        user_a, memory_id, [0.0, 1.0], "test-model", updated_hash
    )
    assert memory_store.search_memories(user_a, "stable project") == []

    assert db.note_vector_index.collection_name(
        "test-model", 2, "notes"
    ) != db.note_vector_index.collection_name("test-model", 2, "memories")


def test_legacy_memory_notes_move_to_memory_storage(owners_db, monkeypatch):
    user_id, _ = owners_db
    legacy_id = db.save_note(
        user_id, "Memory: Answer style", "Prefers concise answers"
    )
    content_hash = note_content_hash(
        "Memory: Answer style", "Prefers concise answers"
    )
    assert db.upsert_note_embedding(
        user_id, legacy_id, [1.0, 0.0], "test-model", content_hash
    )

    deleted_points = []
    delete_points = db.note_vector_index.delete_points

    def record_delete(collection, ids):
        deleted_points.append((collection, list(ids)))
        return delete_points(collection, ids)

    monkeypatch.setattr(db.note_vector_index, "delete_points", record_delete)
    db.init_db()

    assert db.search_notes(user_id, "concise") == []
    migrated, count = memory_store.list_memories(user_id)
    assert count == 1
    assert migrated[0]["title"] == "Answer style"
    assert migrated[0]["content"] == "Prefers concise answers"
    assert len(deleted_points) == 1
    assert deleted_points[0][1] == [legacy_id]


def test_mutation_idempotency_retries_and_payload_conflicts(owners_db):
    user_a, user_b = owners_db

    first = db.save_note_idempotent(
        user_a, "Retry note", "same payload", "save-key"
    )
    repeated = db.save_note_idempotent(
        user_a, "Retry note", "same payload", "save-key"
    )
    assert repeated == first
    assert db.search_notes(user_a, "Retry note")[0]["id"] == first["id"]
    assert len(db.search_notes(user_a, "Retry note")) == 1

    with pytest.raises(db.IdempotencyConflict):
        db.save_note_idempotent(
            user_a, "Different", "payload", "save-key"
        )
    # Key scope includes user_id, so the same key is independent for another owner.
    other_owner = db.save_note_idempotent(
        user_b, "Retry note", "same payload", "save-key"
    )
    assert other_owner["id"] != first["id"]

    clear_first = db.clear_notes_idempotent(user_a, "clear-key")
    clear_repeated = db.clear_notes_idempotent(user_a, "clear-key")
    assert clear_repeated == clear_first
    assert clear_repeated["deleted_count"] == 1


def test_mutation_idempotency_handles_concurrent_creates(owners_db):
    user_a, _ = owners_db

    # BEGIN IMMEDIATE serializes the key claim, so exactly one worker executes
    # the insert and both callers receive the same original note response.
    def create():
        return db.save_note_idempotent(
            user_a, "Concurrent", "one note", "concurrent-key"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: create(), range(2)))

    assert results[0] == results[1]
    assert len(db.search_notes(user_a, "Concurrent")) == 1


def test_stale_embedding_upsert_cannot_replace_updated_note(owners_db):
    user_a, _ = owners_db
    note_id = db.save_note(user_a, "Old", "first content")
    old_hash = note_content_hash("Old", "first content")

    assert db.update_note(user_a, note_id, "New", "second content")
    assert not db.upsert_note_embedding(
        user_a,
        note_id,
        [1.0, 0.0],
        "test-model",
        old_hash,
    )

    new_hash = note_content_hash("New", "second content")
    assert db.upsert_note_embedding(
        user_a,
        note_id,
        [0.0, 1.0],
        "test-model",
        new_hash,
    )


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
            "SELECT COUNT(*) FROM note_embedding_index WHERE note_id = ?",
            (note_id,),
        ).fetchone()[0] == 1
        assert "embedding" not in {
            row[1] for row in conn.execute("PRAGMA table_info(note_embedding_index)")
        }
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

    assert not db.update_note(user_a + 1, note_id, "Private", "not owned")
    assert [
        row["id"]
        for row in db.search_notes_hybrid(user_a, "", [1.0, 0.0], "test-model")
    ] == [note_id]

    assert db.update_note(user_a, note_id, "Final", "new content")
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
            "SELECT COUNT(*) FROM note_embedding_index WHERE note_id = ?",
            (note_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM note_embedding_jobs WHERE note_id = ?",
            (note_id,),
        ).fetchone()[0] == 0
    finally:
        conn.close()
    assert db.search_notes_hybrid(user_a, "", [0.0, 1.0], "test-model") == []


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


def test_delete_note_is_owner_scoped_and_cleans_up_its_vector(owners_db, monkeypatch):
    user_a, user_b = owners_db
    note_id = db.save_note(user_a, "Project", "A stable project context")
    other_note_id = db.save_note(user_b, "Private", "Another user's memory")
    content_hash = note_content_hash("Project", "A stable project context")
    assert db.upsert_note_embedding(
        user_a, note_id, [1.0, 0.0], "test-model", content_hash
    )

    deleted_points = []
    delete_points = db.note_vector_index.delete_points

    def record_delete(collection, note_ids):
        deleted_points.append((collection, list(note_ids)))
        return delete_points(collection, note_ids)

    monkeypatch.setattr(db.note_vector_index, "delete_points", record_delete)

    assert not db.delete_note(user_b, note_id)
    assert db.delete_note(user_a, note_id)
    assert not db.delete_note(user_a, note_id)
    assert len(deleted_points) == 1
    assert deleted_points[0][1] == [note_id]
    assert db.search_notes(user_a, "stable project") == []
    assert [note["id"] for note in db.search_notes(user_b, "memory")] == [
        other_note_id
    ]
    assert not db.upsert_note_embedding(
        user_a, note_id, [1.0, 0.0], "test-model", content_hash
    )

    conn = db.get_connection()
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM note_embedding_index WHERE note_id = ?",
            (note_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM note_embedding_jobs WHERE note_id = ?",
            (note_id,),
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_legacy_sql_vectors_are_moved_to_qdrant_without_reembedding(
    owners_db, monkeypatch,
):
    user_a, _ = owners_db
    title, content = "Research", "renewable energy"
    note_id = db.save_note(user_a, title, content)
    conn = db.get_connection()
    try:
        conn.execute(
            """CREATE TABLE note_embeddings (
                   note_id INTEGER PRIMARY KEY,
                   user_id INTEGER NOT NULL,
                   model TEXT NOT NULL,
                   dimensions INTEGER NOT NULL,
                   embedding BLOB NOT NULL,
                   content_hash TEXT NOT NULL,
                   updated_at INTEGER NOT NULL
               )"""
        )
        conn.execute(
            """INSERT INTO note_embeddings
                   (note_id, user_id, model, dimensions, embedding,
                    content_hash, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (
                note_id,
                user_a,
                embedding_model(),
                2,
                struct.pack("<2f", 1.0, 0.0),
                note_content_hash(title, content),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    db.init_db()
    worker = embedding_worker.NoteEmbeddingWorker()
    original_upsert = db.upsert_note_embedding

    def upsert_and_stop(*args):
        indexed = original_upsert(*args)
        worker._stop_event.set()
        return indexed

    monkeypatch.setattr(db, "upsert_note_embedding", upsert_and_stop)
    monkeypatch.setattr(
        embedding_worker,
        "embed_text",
        lambda *args, **kwargs: pytest.fail("legacy vector should be reused"),
    )
    worker._process_jobs(object())

    results = db.search_notes_hybrid(
        user_a, "", [1.0, 0.0], embedding_model()
    )
    assert [row["id"] for row in results] == [note_id]
    conn = db.get_connection()
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM note_embeddings_legacy"
        ).fetchone()[0] == 0
    finally:
        conn.close()


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



def test_delete_task_and_clear_tasks_are_owner_scoped(owners_db):
    user_a, user_b = owners_db
    first = db.create_task(user_a, "First")
    db.create_task(user_a, "Second")
    private = db.create_task(user_b, "Private task")

    assert db.delete_task(user_b, first) is False  # not B's task
    assert len(db.get_tasks(user_a)) == 2
    assert db.delete_task(user_a, first) is True
    assert db.delete_task(user_a, first) is False  # already gone
    assert [task["title"] for task in db.get_tasks(user_a)] == ["Second"]

    assert db.clear_tasks(user_a) == 1
    assert db.get_tasks(user_a) == []
    assert db.clear_tasks(user_a) == 0
    assert [task["id"] for task in db.get_tasks(user_b)] == [private]


def test_delete_and_clear_tasks_are_idempotent_with_a_key(owners_db):
    user_a, _ = owners_db
    task = db.create_task(user_a, "Once")
    db.create_task(user_a, "Twice")

    first = db.delete_task_idempotent(user_a, task, "delete-key")
    repeat = db.delete_task_idempotent(user_a, task, "delete-key")
    assert first == repeat == {"success": True}  # a retry repeats the original answer

    cleared = db.clear_tasks_idempotent(user_a, "clear-tasks-key")
    retried = db.clear_tasks_idempotent(user_a, "clear-tasks-key")
    assert cleared == retried == {"deleted_count": 1}


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


def _index_conversation(user_id, conversation_id):
    messages = db.get_conversation_messages(user_id, conversation_id)
    chunks = chunk_conversation(conversation_id, messages)
    vectors = {chunk["chunk_id"]: [1.0, 0.0] for chunk in chunks}
    indexed = db.upsert_conversation_embeddings(
        user_id,
        conversation_id,
        chunks,
        vectors,
        "gemini-embedding-001",
        conversation_snapshot_hash(messages),
    )
    assert indexed
    return chunks


def test_conversation_semantic_search_is_owner_scoped_and_uses_its_own_collection(owners_db):
    user_a, user_b = owners_db
    conversation_a = db.create_conversation(user_a, "Hiking plans")
    conversation_b = db.create_conversation(user_b, "Private hiking")
    db.save_conversation(
        user_a, conversation_a["id"], "How should I prepare for a long trek?",
        "Build endurance gradually and pack layered clothing.",
    )
    db.save_conversation(
        user_b, conversation_b["id"], "My private route",
        "This belongs to another account.",
    )
    chunks_a = _index_conversation(user_a, conversation_a["id"])
    _index_conversation(user_b, conversation_b["id"])

    results = db.search_conversations_hybrid(
        user_a, "hiking", [1.0, 0.0], "gemini-embedding-001"
    )

    assert [row["conversation_id"] for row in results] == [conversation_a["id"]]
    assert "long trek" in results[0]["excerpt"]
    assert db.search_conversations_hybrid(
        user_b, "", [1.0, 0.0], "gemini-embedding-001"
    )[0]["conversation_id"] == conversation_b["id"]
    assert db.search_conversations_keyword(user_a, "another account") == []
    assert db.search_conversations_keyword(user_b, "another account")[0]["conversation_id"] == conversation_b["id"]
    assert db.note_vector_index.collection_name(
        "gemini-embedding-001", 2, "conversations"
    ) != db.note_vector_index.collection_name("gemini-embedding-001", 2, "notes")
    assert len(chunks_a) == 1


def test_hybrid_conversation_search_includes_unindexed_keyword_matches(owners_db):
    user_a, _ = owners_db
    conversation = db.create_conversation(user_a, "Recent deployment notes")
    db.save_conversation(
        user_a,
        conversation["id"],
        "How do I deploy the worker?",
        "Use the service container and configure Qdrant.",
    )

    # Search is available immediately, before the asynchronous index worker
    # has embedded this new conversation.
    results = db.search_conversations_hybrid(
        user_a, "deploy the worker", [1.0, 0.0], "gemini-embedding-001"
    )

    assert [row["conversation_id"] for row in results] == [conversation["id"]]
    assert results[0]["score"] == 0.2
    assert "deploy the worker" in results[0]["excerpt"]


def test_conversation_edit_invalidates_old_chunks_and_delete_removes_index(owners_db):
    user_a, _ = owners_db
    conversation = db.create_conversation(user_a, "Edit indexed chat")
    db.save_conversation(
        user_a, conversation["id"], "Original topic", "Original answer"
    )
    _index_conversation(user_a, conversation["id"])

    user_message = db.get_conversation_messages(user_a, conversation["id"])[0]
    assert db.edit_conversation_message(
        user_a, conversation["id"], user_message["id"], "Revised topic"
    )[0]
    assert db.search_conversations_hybrid(
        user_a, "original", [1.0, 0.0], "gemini-embedding-001"
    ) == []
    assert db.get_conversation_embedding_index(user_a, conversation["id"]) == []

    # Reindex the edited snapshot, then verify that deleting the conversation
    # removes both its SQL index rows and Qdrant points.
    db.save_conversation(
        user_a, conversation["id"], "Revised topic", "Revised answer"
    )
    _index_conversation(user_a, conversation["id"])
    assert db.delete_conversation(user_a, conversation["id"])
    assert db.get_conversation_embedding_index(user_a, conversation["id"]) == []
    assert db.search_conversations_hybrid(
        user_a, "revised", [1.0, 0.0], "gemini-embedding-001"
    ) == []


def test_stale_conversation_embedding_snapshot_is_rejected(owners_db):
    user_a, _ = owners_db
    conversation = db.create_conversation(user_a, "Stale snapshot")
    db.save_conversation(user_a, conversation["id"], "First prompt", "First reply")
    old_messages = db.get_conversation_messages(user_a, conversation["id"])
    old_chunks = chunk_conversation(conversation["id"], old_messages)
    old_snapshot = conversation_snapshot_hash(old_messages)

    db.save_conversation(user_a, conversation["id"], "Second prompt", "Second reply")
    result = db.upsert_conversation_embeddings(
        user_a,
        conversation["id"],
        old_chunks,
        {chunk["chunk_id"]: [1.0, 0.0] for chunk in old_chunks},
        "gemini-embedding-001",
        old_snapshot,
    )

    assert result is False
    assert db.get_conversation_embedding_index(user_a, conversation["id"]) == []
