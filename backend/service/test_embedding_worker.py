import backend.service.db as db
from backend.service import embedding_worker


def _setup_database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DATABASE_KEY", "test-database-key")
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
