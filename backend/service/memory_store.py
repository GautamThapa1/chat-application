"""Persistence and embedding lifecycle for user memories."""

import time

from backend.embeddings import (
    embedding_model,
    note_content_hash,
    validate_embedding_vector,
)
from backend.service import db, note_vector_index


def _queue_memory_embedding(
    conn,
    memory_id: int,
    user_id: int,
    title: str,
    content: str,
    *,
    reset_attempts: bool = True,
) -> None:
    model = embedding_model()
    content_hash = note_content_hash(title, content)
    conn.execute(
        """INSERT OR IGNORE INTO memory_embedding_dedupe
               (memory_id, content_hash, model)
           VALUES (?, ?, ?)""",
        (memory_id, content_hash, model),
    )
    if reset_attempts:
        conn.execute(
            """INSERT INTO memory_embedding_jobs (memory_id, user_id)
               VALUES (?, ?)
               ON CONFLICT(memory_id) DO UPDATE SET
                   user_id = excluded.user_id,
                   attempts = 0,
                   next_attempt_at = 0,
                   last_error = NULL""",
            (memory_id, user_id),
        )
    else:
        conn.execute(
            """INSERT OR IGNORE INTO memory_embedding_jobs (memory_id, user_id)
               VALUES (?, ?)""",
            (memory_id, user_id),
        )


def _migrate_legacy_memory_notes(conn) -> None:
    rows = conn.execute(
        """SELECT id, user_id, title, content, created_at
             FROM notes
            WHERE lower(trim(title)) LIKE 'memory:%'"""
    ).fetchall()
    for row in rows:
        title = row["title"].strip()[7:].strip() or "Memory"
        cursor = conn.execute(
            """INSERT INTO memories (user_id, title, content, created_at)
               VALUES (?, ?, ?, ?)""",
            (row["user_id"], title, row["content"], row["created_at"]),
        )
        _queue_memory_embedding(
            conn,
            cursor.lastrowid,
            row["user_id"],
            title,
            row["content"],
        )
        indexed = conn.execute(
            """SELECT collection FROM note_embedding_index
                 WHERE note_id = ? AND user_id = ?""",
            (row["id"], row["user_id"]),
        ).fetchone()
        if indexed is not None:
            note_vector_index.delete_points(indexed["collection"], [row["id"]])
        if db._table_exists(conn, "note_embeddings_legacy"):
            conn.execute(
                "DELETE FROM note_embeddings_legacy WHERE note_id = ?",
                (row["id"],),
            )
        conn.execute("DELETE FROM notes WHERE id = ?", (row["id"],))


def _reconcile_memory_embedding_jobs(conn) -> None:
    rows = conn.execute(
        """SELECT m.id, m.user_id, m.title, m.content,
                  i.model, i.content_hash
             FROM memories AS m
             LEFT JOIN memory_embedding_index AS i ON i.memory_id = m.id"""
    ).fetchall()
    current_model = embedding_model()
    for row in rows:
        if (
            row["model"] == current_model
            and row["content_hash"] == note_content_hash(row["title"], row["content"])
        ):
            continue
        _queue_memory_embedding(
            conn,
            row["id"],
            row["user_id"],
            row["title"],
            row["content"],
            reset_attempts=False,
        )


def init_memory_storage(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title      TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_embedding_index (
            memory_id     INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            model         TEXT NOT NULL,
            dimensions    INTEGER NOT NULL CHECK (dimensions > 0),
            collection    TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            updated_at    INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_embedding_jobs (
            memory_id       INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            attempts        INTEGER NOT NULL DEFAULT 0,
            next_attempt_at INTEGER NOT NULL DEFAULT 0,
            last_error      TEXT,
            lease_owner     TEXT,
            lease_until     INTEGER NOT NULL DEFAULT 0
        )
    """)
    job_columns = {
        row["name"] for row in conn.execute(
            'PRAGMA table_info("memory_embedding_jobs")'
        )
    }
    if "lease_owner" not in job_columns:
        conn.execute(
            "ALTER TABLE memory_embedding_jobs ADD COLUMN lease_owner TEXT"
        )
    if "lease_until" not in job_columns:
        conn.execute(
            "ALTER TABLE memory_embedding_jobs "
            "ADD COLUMN lease_until INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_embedding_dedupe (
            memory_id     INTEGER NOT NULL,
            content_hash  TEXT NOT NULL,
            model         TEXT NOT NULL,
            PRIMARY KEY (memory_id, content_hash, model),
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_user "
        "ON memories(user_id, id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_embedding_index_user "
        "ON memory_embedding_index(user_id, memory_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_embedding_jobs_due "
        "ON memory_embedding_jobs(next_attempt_at, memory_id)"
    )
    _migrate_legacy_memory_notes(conn)
    _reconcile_memory_embedding_jobs(conn)


def save_memory_idempotent_in_transaction(
    conn,
    user_id: int,
    title: str,
    content: str,
) -> dict:
    if conn is None:
        memory_id = save_memory(user_id, title, content)
        return {
            "id": memory_id,
            "title": title,
            "content": content,
            "created_at": "",
        }
    cursor = conn.execute(
        "INSERT INTO memories (user_id, title, content) VALUES (?, ?, ?)",
        (user_id, title, content),
    )
    memory_id = cursor.lastrowid
    _queue_memory_embedding(conn, memory_id, user_id, title, content)
    return {
        "id": memory_id,
        "title": title,
        "content": content,
        "created_at": "",
    }


def save_memory(user_id: int, title: str, content: str) -> int:
    conn = db.get_connection()
    try:
        with conn:
            return save_memory_idempotent_in_transaction(
                conn, user_id, title, content
            )["id"]
    finally:
        conn.close()


def save_memory_idempotent(
    user_id: int,
    title: str,
    content: str,
    idempotency_key: str,
) -> dict:
    return db.run_idempotent_request(
        user_id,
        idempotency_key,
        "save_memory",
        {"title": title, "content": content},
        lambda conn: save_memory_idempotent_in_transaction(
            conn, user_id, title, content
        ),
    )


def list_memories(
    user_id: int,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict], int]:
    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100.")
    if offset < 0:
        raise ValueError("Offset cannot be negative.")
    conn = db.get_connection()
    try:
        rows = conn.execute(
            """SELECT id, title, content, created_at
                 FROM memories
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?""",
            (user_id, limit, offset),
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
        return [dict(row) for row in rows], total
    finally:
        conn.close()


def search_memories(user_id: int, keyword: str) -> list[dict]:
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("Keyword cannot be blank.")
    conn = db.get_connection()
    try:
        like = f"%{keyword}%"
        rows = conn.execute(
            """SELECT id, title, content, created_at
                 FROM memories
                WHERE user_id = ? AND (title LIKE ? OR content LIKE ?)
                ORDER BY id""",
            (user_id, like, like),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _update_memory_in_transaction(
    conn,
    user_id: int,
    memory_id: int,
    title: str,
    content: str,
) -> dict:
    indexed = conn.execute(
        """SELECT collection FROM memory_embedding_index
             WHERE memory_id = ? AND user_id = ?""",
        (memory_id, user_id),
    ).fetchone()
    memory = conn.execute(
        "SELECT 1 FROM memories WHERE id = ? AND user_id = ?",
        (memory_id, user_id),
    ).fetchone()
    if memory is None:
        return {"found": False}
    if indexed is not None:
        note_vector_index.delete_points(indexed["collection"], [memory_id])
    updated = conn.execute(
        "UPDATE memories SET title = ?, content = ? WHERE id = ? AND user_id = ?",
        (title, content, memory_id, user_id),
    )
    if not updated.rowcount:
        return {"found": False}
    conn.execute(
        "DELETE FROM memory_embedding_index WHERE memory_id = ?",
        (memory_id,),
    )
    _queue_memory_embedding(conn, memory_id, user_id, title, content)
    return {
        "found": True,
        "id": memory_id,
        "title": title,
        "content": content,
        "created_at": "",
    }


def update_memory(user_id: int, memory_id: int, title: str, content: str) -> bool:
    conn = db.get_connection()
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            return _update_memory_in_transaction(
                conn, user_id, memory_id, title, content
            )["found"]
    finally:
        conn.close()


def update_memory_idempotent(
    user_id: int,
    memory_id: int,
    title: str,
    content: str,
    idempotency_key: str,
) -> dict:
    def execute(conn) -> dict:
        if conn is None:
            if not update_memory(user_id, memory_id, title, content):
                return {"found": False}
            return {
                "found": True,
                "id": memory_id,
                "title": title,
                "content": content,
                "created_at": "",
            }
        return _update_memory_in_transaction(
            conn, user_id, memory_id, title, content
        )

    return db.run_idempotent_request(
        user_id,
        idempotency_key,
        "update_memory",
        {"memory_id": memory_id, "title": title, "content": content},
        execute,
    )


def delete_memory(user_id: int, memory_id: int) -> bool:
    conn = db.get_connection()
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            memory = conn.execute(
                "SELECT 1 FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            ).fetchone()
            if memory is None:
                return False
            indexed = conn.execute(
                """SELECT collection FROM memory_embedding_index
                     WHERE memory_id = ? AND user_id = ?""",
                (memory_id, user_id),
            ).fetchone()
            if indexed is not None:
                note_vector_index.delete_points(indexed["collection"], [memory_id])
            deleted = conn.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
            return deleted.rowcount > 0
    finally:
        conn.close()


def get_pending_memory_embeddings(
    limit: int = 20,
    lease_owner: str | None = None,
) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ValueError("Embedding batch size must be between 1 and 100.")
    conn = db.get_connection()
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            now = int(time.time())
            if lease_owner:
                conn.execute(
                    """UPDATE memory_embedding_jobs
                          SET lease_owner = NULL, lease_until = 0
                        WHERE lease_until > 0 AND lease_until <= ?""",
                    (now,),
                )
            rows = conn.execute(
                """SELECT m.id, m.user_id, m.title, m.content
                     FROM memory_embedding_jobs AS j
                     JOIN memories AS m
                       ON m.id = j.memory_id AND m.user_id = j.user_id
                    WHERE j.next_attempt_at <= ?
                      AND (
                           j.lease_owner IS NULL
                           OR j.lease_until <= ?
                           OR j.lease_owner = ?
                          )
                    ORDER BY j.next_attempt_at, j.memory_id
                    LIMIT ?""",
                (now, now, lease_owner or "", limit),
            ).fetchall()
            if lease_owner:
                lease_until = now + 300
                conn.executemany(
                    """UPDATE memory_embedding_jobs
                          SET lease_owner = ?, lease_until = ?
                        WHERE memory_id = ?""",
                    [(lease_owner, lease_until, row["id"]) for row in rows],
                )
            return [dict(row) for row in rows]
    finally:
        conn.close()


def record_memory_embedding_failure(memory_id: int, error: str) -> None:
    conn = db.get_connection()
    try:
        with conn:
            row = conn.execute(
                "SELECT attempts FROM memory_embedding_jobs WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return
            attempts = row["attempts"] + 1
            retry_delay = min(3600, 2 ** min(attempts, 12))
            conn.execute(
                """UPDATE memory_embedding_jobs
                      SET attempts = ?, next_attempt_at = ?, last_error = ?,
                          lease_owner = NULL, lease_until = 0
                    WHERE memory_id = ?""",
                (
                    attempts,
                    int(time.time()) + retry_delay,
                    error[:500],
                    memory_id,
                ),
            )
    finally:
        conn.close()


def upsert_memory_embedding(
    user_id: int,
    memory_id: int,
    embedding: list[float],
    model: str,
    content_hash: str,
) -> bool:
    vector = validate_embedding_vector(embedding)
    if not model or not content_hash:
        raise ValueError("Embedding model and content hash are required.")

    conn = db.get_connection()
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            memory = conn.execute(
                "SELECT title, content FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            ).fetchone()
            if memory is None or note_content_hash(
                memory["title"], memory["content"]
            ) != content_hash:
                return False
            collection = note_vector_index.upsert(
                record_id=memory_id,
                user_id=user_id,
                vector=vector,
                model=model,
                content_hash=content_hash,
                entity="memories",
            )
            previous = conn.execute(
                "SELECT collection FROM memory_embedding_index WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if previous is not None and previous["collection"] != collection:
                note_vector_index.delete_points(previous["collection"], [memory_id])
            conn.execute(
                """INSERT INTO memory_embedding_index
                       (memory_id, user_id, model, dimensions, collection,
                        content_hash, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(memory_id) DO UPDATE SET
                       user_id = excluded.user_id,
                       model = excluded.model,
                       dimensions = excluded.dimensions,
                       collection = excluded.collection,
                       content_hash = excluded.content_hash,
                       updated_at = excluded.updated_at""",
                (
                    memory_id,
                    user_id,
                    model,
                    len(vector),
                    collection,
                    content_hash,
                    int(time.time()),
                ),
            )
            conn.execute(
                "DELETE FROM memory_embedding_jobs WHERE memory_id = ?",
                (memory_id,),
            )
            return True
    finally:
        conn.close()


def search_memories_hybrid(
    user_id: int,
    keyword: str,
    query_embedding: list[float],
    model: str,
    limit: int = 20,
) -> list[dict]:
    query_vector = validate_embedding_vector(query_embedding)
    if not model:
        raise ValueError("Embedding model is required.")
    if not 1 <= limit <= 100:
        raise ValueError("Search limit must be between 1 and 100.")

    conn = db.get_connection()
    try:
        vector_matches = dict(
            note_vector_index.search(
                user_id=user_id,
                vector=query_vector,
                model=model,
                limit=limit,
                entity="memories",
            )
        )
        keyword_rows = (
            conn.execute(
                """SELECT id FROM memories
                     WHERE user_id = ? AND (title LIKE ? OR content LIKE ?)""",
                (user_id, f"%{keyword}%", f"%{keyword}%"),
            ).fetchall()
            if keyword
            else []
        )
        keyword_ids = {row["id"] for row in keyword_rows}
        candidate_ids = set(vector_matches) | keyword_ids
        if not candidate_ids:
            return []
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = conn.execute(
            f"""SELECT id, title, content, created_at
                  FROM memories
                 WHERE user_id = ? AND id IN ({placeholders})""",
            [user_id, *candidate_ids],
        ).fetchall()
        indexed_rows = conn.execute(
            f"""SELECT memory_id, model, dimensions, content_hash
                  FROM memory_embedding_index
                 WHERE user_id = ? AND memory_id IN ({placeholders})""",
            [user_id, *candidate_ids],
        ).fetchall()
        indexed_by_id = {row["memory_id"]: row for row in indexed_rows}
        ranked = []
        for row in rows:
            index_entry = indexed_by_id.get(row["id"])
            similarity = vector_matches.get(row["id"])
            if (
                index_entry is None
                or index_entry["model"] != model
                or index_entry["dimensions"] != len(query_vector)
                or index_entry["content_hash"]
                != note_content_hash(row["title"], row["content"])
            ):
                similarity = None
            is_keyword_match = row["id"] in keyword_ids
            if similarity is None and not is_keyword_match:
                continue
            semantic_score = (similarity + 1) / 2 if similarity is not None else 0
            score = 0.7 * semantic_score + 0.3 * is_keyword_match
            ranked.append((score, row))
        ranked.sort(key=lambda result: (-result[0], result[1]["id"]))
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for _, row in ranked[:limit]
        ]
    finally:
        conn.close()
