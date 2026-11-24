import backend.service.db as db
from backend.service import embedding_worker, memory_store


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DATABASE_KEY", "test-database-key")
    # Worker tests use an isolated in-memory Qdrant instance.
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv(
        "QDRANT_COLLECTION_PREFIX",
        f"test_{tmp_path.name.replace('-', '_')}",
    )
    db.init_db()
    return db.register_user("alice", "correct-horse")


def test_worker_indexes_queued_note_with_existing_note_id(tmp_path, monkeypatch):
    user_id = _setup_database(tmp_path, monkeypatch)
    note_id = db.save_note(user_id, "Plan", "launch checklist")
    worker = embedding_worker.NoteEmbeddingWorker()
    original_upsert = db.upsert_note_embedding

    def upsert_and_stop(*args):
        indexed = original_upsert(*args)
        worker._stop_event.set()
        return indexed

    monkeypatch.setattr(
        embedding_worker,
        "embed_text",
        lambda *args, **kwargs: [1.0, 0.0],
    )
    monkeypatch.setattr(db, "upsert_note_embedding", upsert_and_stop)

    worker._process_jobs(object())

    results = db.search_notes_hybrid(user_id, "", [1.0, 0.0], "gemini-embedding-001")
    assert [note["id"] for note in results] == [note_id]
    assert results[0]["title"] == "Plan"


def test_worker_indexes_queued_memory_in_its_own_collection(tmp_path, monkeypatch):
    user_id = _setup_database(tmp_path, monkeypatch)
    memory_id = memory_store.save_memory(
        user_id, "Answer style", "Prefers concise answers"
    )
    worker = embedding_worker.EmbeddingWorker()
    original_upsert = memory_store.upsert_memory_embedding

    def upsert_and_stop(*args):
        indexed = original_upsert(*args)
        worker._stop_event.set()
        return indexed

    monkeypatch.setattr(
        embedding_worker,
        "embed_text",
        lambda *args, **kwargs: [1.0, 0.0],
    )
    monkeypatch.setattr(memory_store, "upsert_memory_embedding", upsert_and_stop)

    worker._process_jobs(object())

    results = memory_store.search_memories_hybrid(
        user_id,
        "",
        [1.0, 0.0],
        "gemini-embedding-001",
    )
    assert [memory["id"] for memory in results] == [memory_id]


def test_worker_retries_a_failed_memory_embedding(tmp_path, monkeypatch):
    user_id = _setup_database(tmp_path, monkeypatch)
    memory_id = memory_store.save_memory(
        user_id, "Answer style", "Prefers concise answers"
    )
    worker = embedding_worker.EmbeddingWorker()
    original_record_failure = memory_store.record_memory_embedding_failure

    def record_failure_and_stop(failed_id, error):
        original_record_failure(failed_id, error)
        worker._stop_event.set()

    def fail_embedding(*args, **kwargs):
        raise RuntimeError("temporary embedding failure")

    monkeypatch.setattr(embedding_worker, "embed_text", fail_embedding)
    monkeypatch.setattr(
        memory_store,
        "record_memory_embedding_failure",
        record_failure_and_stop,
    )

    worker._process_jobs(object())

    assert memory_store.search_memories(user_id, "concise")[0]["id"] == memory_id
    conn = db.get_connection()
    try:
        job = conn.execute(
            """SELECT attempts, last_error FROM memory_embedding_jobs
                 WHERE memory_id = ?""",
            (memory_id,),
        ).fetchone()
        assert job["attempts"] == 1
        assert job["last_error"] == "temporary embedding failure"
    finally:
        conn.close()


def test_worker_records_embedding_failure_and_keeps_note(tmp_path, monkeypatch):
    user_id = _setup_database(tmp_path, monkeypatch)
    note_id = db.save_note(user_id, "Plan", "launch checklist")
    worker = embedding_worker.NoteEmbeddingWorker()
    original_record_failure = db.record_note_embedding_failure

    def record_failure_and_stop(failed_note_id, error):
        original_record_failure(failed_note_id, error)
        worker._stop_event.set()

    def fail_embedding(*args, **kwargs):
        raise RuntimeError("temporary embedding failure")

    monkeypatch.setattr(embedding_worker, "embed_text", fail_embedding)
    monkeypatch.setattr(
        db, "record_note_embedding_failure", record_failure_and_stop
    )

    worker._process_jobs(object())

    assert db.search_notes(user_id, "launch checklist")[0]["id"] == note_id
    assert db.get_pending_note_embeddings() == []
    conn = db.get_connection()
    try:
        job = conn.execute(
            "SELECT attempts, last_error FROM note_embedding_jobs WHERE note_id = ?",
            (note_id,),
        ).fetchone()
        assert job["attempts"] == 1
        assert job["last_error"] == "temporary embedding failure"
    finally:
        conn.close()


def test_worker_restart_reclaims_expired_embedding_job(tmp_path, monkeypatch):
    user_id = _setup_database(tmp_path, monkeypatch)
    db.save_note(user_id, "Restart", "worker lease")
    first = embedding_worker.NoteEmbeddingWorker()
    second = embedding_worker.NoteEmbeddingWorker()

    first_jobs = db.get_pending_note_embeddings(
        20, lease_owner=first._lease_owner
    )
    second_jobs = db.get_pending_note_embeddings(
        20, lease_owner=second._lease_owner
    )
    assert len(first_jobs) == 1
    assert second_jobs == []

    conn = db.get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE note_embedding_jobs SET lease_until = 0"
            )
    finally:
        conn.close()

    reclaimed = db.get_pending_note_embeddings(
        20, lease_owner=second._lease_owner
    )
    assert [job["id"] for job in reclaimed] == [first_jobs[0]["id"]]


def test_worker_indexes_conversation_turns_and_reuses_unchanged_chunks(
    tmp_path, monkeypatch,
):
    user_id = _setup_database(tmp_path, monkeypatch)
    conversation = db.create_conversation(user_id, "Semantic history")
    db.save_conversation(
        user_id, conversation["id"], "How do I deploy with Docker?",
        "Build an image, then start the Compose stack.",
    )
    worker = embedding_worker.EmbeddingWorker()
    embedded_texts = []
    original_upsert = db.upsert_conversation_embeddings

    def fake_embed(_client, text, **kwargs):
        embedded_texts.append(text)
        return [1.0, 0.0]

    def upsert_and_stop(*args, **kwargs):
        indexed = original_upsert(*args, **kwargs)
        worker._stop_event.set()
        return indexed

    monkeypatch.setattr(embedding_worker, "embed_text", fake_embed)
    monkeypatch.setattr(db, "upsert_conversation_embeddings", upsert_and_stop)
    worker._process_jobs(object())

    assert len(embedded_texts) == 1
    first_results = db.search_conversations_hybrid(
        user_id, "", [1.0, 0.0], "gemini-embedding-001"
    )
    assert first_results[0]["conversation_id"] == conversation["id"]

    db.save_conversation(
        user_id, conversation["id"], "How can I scale the service?",
        "Run one app worker and scale the backing services independently.",
    )
    second_worker = embedding_worker.EmbeddingWorker()
    original_stop = second_worker._stop_event.set

    def stop_after_second_upsert(*args, **kwargs):
        indexed = original_upsert(*args, **kwargs)
        original_stop()
        return indexed

    monkeypatch.setattr(
        db, "upsert_conversation_embeddings", stop_after_second_upsert
    )
    second_worker._process_jobs(object())

    # Only the newly appended turn needs a document embedding.
    assert len(embedded_texts) == 2
    assert len(db.get_conversation_embedding_index(user_id, conversation["id"])) == 2


def test_worker_retries_failed_conversation_embedding(tmp_path, monkeypatch):
    user_id = _setup_database(tmp_path, monkeypatch)
    conversation = db.create_conversation(user_id, "Retry history")
    db.save_conversation(user_id, conversation["id"], "Prompt", "Response")
    worker = embedding_worker.EmbeddingWorker()
    original_record_failure = db.record_conversation_embedding_failure

    def record_failure_and_stop(conversation_id, error):
        original_record_failure(conversation_id, error)
        worker._stop_event.set()

    def fail_embedding(*args, **kwargs):
        raise RuntimeError("temporary conversation embedding failure")

    monkeypatch.setattr(embedding_worker, "embed_text", fail_embedding)
    monkeypatch.setattr(
        db, "record_conversation_embedding_failure", record_failure_and_stop
    )
    worker._process_jobs(object())

    conn = db.get_connection()
    try:
        job = conn.execute(
            """SELECT attempts, last_error FROM conversation_embedding_jobs
                 WHERE conversation_id = ?""",
            (conversation["id"],),
        ).fetchone()
        assert job["attempts"] == 1
        assert job["last_error"] == "temporary conversation embedding failure"
    finally:
        conn.close()
