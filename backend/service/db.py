"""SQLite persistence for accounts and user-owned application data."""

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import struct
import tempfile
import time
from pathlib import Path

import sqlcipher3
from dotenv import load_dotenv

from backend.embeddings import embedding_model, note_content_hash
from backend.service import note_vector_index


# DB_PATH lets Docker keep the database on a mounted volume (e.g. /data/app.db).
DB_PATH = os.getenv("DB_PATH", "app.db")
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _database_key() -> str:
    load_dotenv(ENV_PATH)
    key = os.getenv("DATABASE_KEY")
    if not key:
        raise RuntimeError("Set DATABASE_KEY before opening the database.")
    return key


def _apply_database_key(conn, key: str) -> None:
    # Apply the SQLCipher encryption key and verify 
    # it by attempting to read the database schema.
    escaped_key = key.replace("'", "''")
    conn.execute(f"PRAGMA key = '{escaped_key}'")
    conn.execute("SELECT count(*) FROM sqlite_master").fetchone()


def _is_plaintext_database(path: str) -> bool:
    # Check whether the database is a plaintext SQLite file.
    # so we know if we are migrating encrypted or unencrypted database
    try:
        with open(path, "rb") as database_file:
            return database_file.read(16) == b"SQLite format 3\x00"
    except FileNotFoundError:
        return False


def _migrate_plaintext_database(path: str, key: str) -> None:
    # use a temporary database to migrate our old database to newer one
    path_obj = Path(path)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path_obj.name}.",
        suffix=".encrypted",
        dir=path_obj.parent,
    )
    os.close(descriptor)
    try:
        conn = sqlcipher3.connect(temporary_path)
        try:
            _apply_database_key(conn, key)
            conn.execute(
                "ATTACH DATABASE ? AS plaintext KEY ''",
                (str(path_obj.resolve()),),
            )
            conn.execute("SELECT sqlcipher_export('main', 'plaintext')").fetchone()
            conn.execute("DETACH DATABASE plaintext")
        finally:
            conn.close()

        conn = sqlcipher3.connect(temporary_path)
        try:
            _apply_database_key(conn, key)
        finally:
            conn.close()

        source = sqlite3.connect(path)
        try:
            checkpoint = source.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if checkpoint and checkpoint[0] != 0:
                raise RuntimeError("Could not safely checkpoint the database.")
        finally:
            source.close()

        for suffix in ("-wal", "-shm"):
            try:
                os.unlink(str(path_obj) + suffix)
            except FileNotFoundError:
                pass
        os.replace(temporary_path, path)
    finally:
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                os.unlink(temporary_path + suffix)
            except FileNotFoundError:
                pass


def get_connection():
    key = _database_key()
    if _is_plaintext_database(DB_PATH):
        _migrate_plaintext_database(DB_PATH, key)

    conn = sqlcipher3.connect(DB_PATH)
    try:
        _apply_database_key(conn, key)
    except sqlcipher3.DatabaseError as exc:
        conn.close()
        raise RuntimeError(
            "Could not open the database with DATABASE_KEY; check the key "
            "or database integrity."
        ) from exc

    conn.row_factory = sqlcipher3.Row
    conn.execute("PRAGMA foreign_keys = ON") # don't allow invalid user and CASCADE on delete functionality
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _table_exists(conn, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone() is not None


def _table_columns(conn, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _preserve_unowned_table(conn, table: str, ownership_columns: set[str]) -> None:
    if not _table_exists(conn, table):
        return
    if ownership_columns.issubset(_table_columns(conn, table)):
        return

    legacy_table = f"legacy_{table}"
    if _table_exists(conn, legacy_table):
        raise RuntimeError(
            f"Cannot migrate {table}: {legacy_table} already exists. "
            "Resolve the existing legacy table before retrying."
        )
    conn.execute(f'ALTER TABLE "{table}" RENAME TO "{legacy_table}"')


def _queue_note_embedding(
    conn,
    note_id: int,
    user_id: int,
    title: str,
    content: str,
    *,
    reset_attempts: bool = True,
) -> None:
    model = embedding_model()
    content_hash = note_content_hash(title, content)
    # A persistent dedupe key makes worker restarts and duplicate queue calls
    # converge on the same note/model/content combination.
    conn.execute(
        """INSERT OR IGNORE INTO note_embedding_dedupe
               (note_id, content_hash, model)
           VALUES (?, ?, ?)""",
        (note_id, content_hash, model),
    )
    if reset_attempts:
        conn.execute(
            """INSERT INTO note_embedding_jobs (note_id, user_id)
               VALUES (?, ?)
               ON CONFLICT(note_id) DO UPDATE SET
                   user_id = excluded.user_id,
                   attempts = 0,
                   next_attempt_at = 0,
                   last_error = NULL""",
            (note_id, user_id),
        )
    else:
        conn.execute(
            """INSERT OR IGNORE INTO note_embedding_jobs (note_id, user_id)
               VALUES (?, ?)""",
            (note_id, user_id),
        )


def _reconcile_note_embedding_jobs(conn) -> None:
    # Startup reconciliation queues notes whose model or content changed.
    # This also covers notes that existed before the embedding job table.
    rows = conn.execute(
        """SELECT n.id, n.user_id, n.title, n.content,
                  e.model, e.content_hash
             FROM notes AS n
             LEFT JOIN note_embedding_index AS e ON e.note_id = n.id"""
    ).fetchall()
    current_model = embedding_model()
    for row in rows:
        expected_hash = note_content_hash(row["title"], row["content"])
        if (
            row["model"] == current_model
            and row["content_hash"] == expected_hash
        ):
            continue
        _queue_note_embedding(
            conn,
            row["id"],
            row["user_id"],
            row["title"],
            row["content"],
            reset_attempts=False,
        )
    if _table_exists(conn, "note_embeddings_legacy"):
        legacy_rows = conn.execute(
            """SELECT e.note_id, e.user_id, n.title, n.content
                 FROM note_embeddings_legacy AS e
                 JOIN notes AS n ON n.id = e.note_id AND n.user_id = e.user_id"""
        ).fetchall()
        for row in legacy_rows:
            _queue_note_embedding(
                conn,
                row["note_id"],
                row["user_id"],
                row["title"],
                row["content"],
                reset_attempts=False,
            )


def init_db():
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            # Idempotency records survive crashes and provide the original
            # response for retries. The processing lease lets a crashed owner
            # be recovered without allowing concurrent owners to mutate twice.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_idempotency (
                    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    idempotency_key TEXT NOT NULL,
                    operation     TEXT NOT NULL,
                    payload_hash  TEXT NOT NULL,
                    status        TEXT NOT NULL CHECK (
                        status IN ('processing', 'completed')
                    ),
                    response_json TEXT,
                    lease_until   INTEGER NOT NULL DEFAULT 0,
                    created_at    INTEGER NOT NULL,
                    updated_at    INTEGER NOT NULL,
                    PRIMARY KEY (user_id, idempotency_key)
                )
            """)
            # Conversation responses use this small marker table to make
            # WebSocket retries idempotent without adding hidden chat messages.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_response_requests (
                    conversation_id INTEGER NOT NULL,
                    user_id         INTEGER NOT NULL,
                    request_id      TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, user_id, request_id),
                    FOREIGN KEY (conversation_id, user_id)
                        REFERENCES conversations(conversation_id, user_id)
                        ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                )
            """)

            _preserve_unowned_table(conn, "messages", {"user_id", "conversation_id"})
            _preserve_unowned_table(conn, "notes", {"user_id"})
            _preserve_unowned_table(conn, "tasks", {"user_id"})

            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title           TEXT NOT NULL,
                    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (conversation_id, user_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content         TEXT NOT NULL,
                    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id, user_id)
                        REFERENCES conversations(conversation_id, user_id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title      TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Preserve the old SQLite vector table as a migration source. The
            # worker copies its vectors into Qdrant, then removes them from here.
            if _table_exists(conn, "note_embeddings"):
                if _table_exists(conn, "note_embeddings_legacy"):
                    raise RuntimeError(
                        "Cannot migrate note_embeddings: "
                        "note_embeddings_legacy already exists."
                    )
                conn.execute(
                    "ALTER TABLE note_embeddings RENAME TO note_embeddings_legacy"
                )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS note_embedding_index (
                    note_id       INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
                    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    model         TEXT NOT NULL,
                    dimensions    INTEGER NOT NULL CHECK (dimensions > 0),
                    collection    TEXT NOT NULL,
                    content_hash  TEXT NOT NULL,
                    updated_at    INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS note_embedding_jobs (
                    note_id         INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
                    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    attempts        INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at INTEGER NOT NULL DEFAULT 0,
                    last_error      TEXT,
                    lease_owner     TEXT,
                    lease_until     INTEGER NOT NULL DEFAULT 0
                )
            """)
            job_columns = _table_columns(conn, "note_embedding_jobs")
            if "lease_owner" not in job_columns:
                conn.execute(
                    "ALTER TABLE note_embedding_jobs ADD COLUMN lease_owner TEXT"
                )
            if "lease_until" not in job_columns:
                conn.execute(
                    "ALTER TABLE note_embedding_jobs "
                    "ADD COLUMN lease_until INTEGER NOT NULL DEFAULT 0"
                )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS note_embedding_dedupe (
                    note_id       INTEGER NOT NULL,
                    content_hash  TEXT NOT NULL,
                    model         TEXT NOT NULL,
                    PRIMARY KEY (note_id, content_hash, model),
                    FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title      TEXT NOT NULL,
                    due_date   TEXT,
                    completed  INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_user_updated "
                "ON conversations(user_id, updated_at DESC, conversation_id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation "
                "ON messages(user_id, conversation_id, message_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_user "
                "ON notes(user_id, id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_note_embedding_index_user "
                "ON note_embedding_index(user_id, note_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_note_embedding_jobs_due "
                "ON note_embedding_jobs(next_attempt_at, note_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_user_status "
                "ON tasks(user_id, completed, id)"
            )
            _reconcile_note_embedding_jobs(conn)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Conversations and messages
# --------------------------------------------------------------------------- #
class ConversationNotFound(LookupError):
    """Raised when a conversation is missing or belongs to another user."""


def create_conversation(user_id: int, title: str = "New chat") -> dict:
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
                (user_id, title.strip()[:120] or "New chat"),
            )
            row = conn.execute(
                "SELECT conversation_id AS id, user_id, title, created_at, updated_at "
                "FROM conversations WHERE conversation_id = ? AND user_id = ?",
                (cur.lastrowid, user_id),
            ).fetchone()
            return dict(row)
    finally:
        conn.close()


def list_conversations(
    user_id: int,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict], int]:

    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100.")

    if offset < 0:
        raise ValueError("Offset cannot be negative.")

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT conversation_id AS id, user_id, title,
                   created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC, conversation_id DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset),
        ).fetchall()

        total_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM conversations
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()[0]

        return [dict(row) for row in rows], total_count

    finally:
        conn.close()


def get_conversation(user_id: int, conversation_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT conversation_id AS id, user_id, title, created_at, updated_at "
            "FROM conversations WHERE conversation_id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_conversation_messages(user_id: int, conversation_id: int) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT message_id AS id, conversation_id, user_id, role, content, created_at
                 FROM messages
                WHERE conversation_id = ? AND user_id = ?
                ORDER BY id""",
            (conversation_id, user_id),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def edit_conversation_message(
    user_id: int,
    conversation_id: int,
    message_id: int,
    content: str,
) -> tuple[bool, str]:
    """Edit a user message and remove every later message atomically."""
    content = content.strip()
    if not content:
        raise ValueError("Message content must not be empty.")

    conn = get_connection()
    try:
        with conn:
            message = conn.execute(
                """SELECT message_id, role
                     FROM messages
                    WHERE message_id = ? AND conversation_id = ?
                      AND user_id = ?""",
                (message_id, conversation_id, user_id),
            ).fetchone()
            if message is None or message["role"] != "user":
                return False, ""

            # Deleting by message ID uses SQLite's creation order: everything
            # after this user message is part of the abandoned branch.
            conn.execute(
                """DELETE FROM messages
                  WHERE conversation_id = ? AND user_id = ?
                    AND message_id > ?""",
                (conversation_id, user_id, message_id),
            )
            conn.execute(
                """UPDATE messages SET content = ?
                  WHERE message_id = ? AND conversation_id = ? AND user_id = ?""",
                (content, message_id, conversation_id, user_id),
            )
            conn.execute(
                """UPDATE conversations SET updated_at = CURRENT_TIMESTAMP
                  WHERE conversation_id = ? AND user_id = ?""",
                (conversation_id, user_id),
            )
            return True, content
    finally:
        conn.close()


def get_regeneration_context(
    user_id: int,
    conversation_id: int,
    assistant_message_id: int,
) -> tuple[bool, str, list[dict]]:
    """Find the assistant's preceding user message and history before it."""
    conn = get_connection()
    try:
        assistant = conn.execute(
            """SELECT message_id
                 FROM messages
                WHERE message_id = ? AND conversation_id = ?
                  AND user_id = ? AND role = 'assistant'""",
            (assistant_message_id, conversation_id, user_id),
        ).fetchone()
        if assistant is None:
            return False, "", []

        user_message = conn.execute(
            """SELECT message_id, content
                 FROM messages
                WHERE conversation_id = ? AND user_id = ?
                  AND role = 'user' AND message_id < ?
                ORDER BY message_id DESC
                LIMIT 1""",
            (conversation_id, user_id, assistant_message_id),
        ).fetchone()
        if user_message is None:
            return False, "", []

        history = conn.execute(
            """SELECT role, content
                 FROM messages
                WHERE conversation_id = ? AND user_id = ?
                  AND message_id < ? AND role IN ('user', 'assistant')
                ORDER BY message_id""",
            (conversation_id, user_id, user_message["message_id"]),
        ).fetchall()
        return True, user_message["content"], [dict(row) for row in history]
    finally:
        conn.close()


def replace_assistant_response(
    user_id: int,
    conversation_id: int,
    assistant_message_id: int,
    response: str,
    request_id: str = "",
) -> bool:
    """Replace one assistant response atomically and prevent duplicate retries."""
    conn = get_connection()
    try:
        with conn:
            if request_id:
                existing = conn.execute(
                    """SELECT 1 FROM conversation_response_requests
                        WHERE conversation_id = ? AND user_id = ? AND request_id = ?""",
                    (conversation_id, user_id, request_id),
                ).fetchone()
                if existing is not None:
                    return True

            assistant = conn.execute(
                """SELECT message_id
                     FROM messages
                    WHERE message_id = ? AND conversation_id = ?
                      AND user_id = ? AND role = 'assistant'""",
                (assistant_message_id, conversation_id, user_id),
            ).fetchone()
            if assistant is None:
                return False

            # Regeneration branches from this response: discard any later turns
            # so stale answers cannot remain after the new branch.
            conn.execute(
                """DELETE FROM messages
                  WHERE conversation_id = ? AND user_id = ?
                    AND message_id > ?""",
                (conversation_id, user_id, assistant_message_id),
            )

            conn.execute(
                """UPDATE messages SET content = ?, created_at = CURRENT_TIMESTAMP
                  WHERE message_id = ? AND conversation_id = ? AND user_id = ?""",
                (response, assistant_message_id, conversation_id, user_id),
            )
            if request_id:
                conn.execute(
                    """INSERT INTO conversation_response_requests
                       (conversation_id, user_id, request_id)
                       VALUES (?, ?, ?)""",
                    (conversation_id, user_id, request_id),
                )
            conn.execute(
                """UPDATE conversations SET updated_at = CURRENT_TIMESTAMP
                  WHERE conversation_id = ? AND user_id = ?""",
                (conversation_id, user_id),
            )
            return True
    finally:
        conn.close()


def delete_conversation(user_id: int, conversation_id: int) -> bool:
    conn = get_connection()
    try:
        with conn:
            deleted = conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            return deleted.rowcount == 1
    finally:
        conn.close()


def conversation_title_from_message(message: str) -> str:
    """Make a short sidebar title from the first user message."""
    title = re.split(r"(?<=[.!?])\s+|\r?\n", message.strip(), maxsplit=1)[0]
    title = title.strip(" \t\r\n.,!?;:-")
    title = re.sub(
        r"^(?:please\s+)?(?:can you help me|could you help me|can you|could you|"
        r"help me|remember that|i want to|i need to)\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )
    if not title:
        return "New chat"

    # Keep a whole-word title where possible so sidebar labels stay readable.
    if len(title) > 48:
        shortened = title[:48]
        if " " in shortened:
            shortened = shortened.rsplit(" ", 1)[0]
        title = shortened.rstrip(" ,.:;!?")
    return title[0].upper() + title[1:]


def save_conversation(
    user_id: int,
    conversation_id: int,
    user_content: str,
    assistant_content: str,
    request_id: str = "",
) -> None:
    conn = get_connection()
    try:
        with conn:
            conversation = conn.execute(
                "SELECT title FROM conversations WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if conversation is None:
                raise ConversationNotFound

            # A request ID prevents duplicate response pairs when a WebSocket
            # completion/retry is delivered more than once.
            if request_id:
                existing = conn.execute(
                    """SELECT 1 FROM conversation_response_requests
                        WHERE conversation_id = ? AND user_id = ? AND request_id = ?""",
                    (conversation_id, user_id, request_id),
                ).fetchone()
                if existing is not None:
                    return
                conn.execute(
                    """INSERT INTO conversation_response_requests
                       (conversation_id, user_id, request_id)
                       VALUES (?, ?, ?)""",
                    (conversation_id, user_id, request_id),
                )

            # After an edit the (edited) user message is already stored and has
            # no answer yet. Saving it again would show the prompt twice, so in
            # that case only the assistant reply is appended.
            last = conn.execute(
                """SELECT role, content FROM messages
                    WHERE conversation_id = ? AND user_id = ?
                    ORDER BY message_id DESC LIMIT 1""",
                (conversation_id, user_id),
            ).fetchone()
            user_already_saved = (
                last is not None
                and last["role"] == "user"
                and last["content"] == user_content
            )
            new_rows = [(conversation_id, user_id, "assistant", assistant_content)]
            if not user_already_saved:
                new_rows.insert(0, (conversation_id, user_id, "user", user_content))
            conn.executemany(
                """INSERT INTO messages
                   (conversation_id, user_id, role, content)
                   VALUES (?, ?, ?, ?)""",
                new_rows,
            )
            title = conversation["title"]
            if title == "New chat":
                # Set the title once, from the first user message only.
                title = conversation_title_from_message(user_content)
            conn.execute(
                """UPDATE conversations
                      SET title = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE conversation_id = ? AND user_id = ?""",
                (title, conversation_id, user_id),
            )
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #
class IdempotencyConflict(ValueError):
    """Raised when an idempotency key is reused for a different request."""


class IdempotencyInProgress(TimeoutError):
    """Raised when a concurrent request owns the processing lease."""


def _payload_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_idempotent_request(
    user_id: int,
    idempotency_key: str,
    operation: str,
    payload: dict,
    execute,
) -> dict:
    """Run a mutation once per user/key pair and persist its original result.

    A processing row is claimed atomically. A crash leaves it recoverable after
    the short lease expires; concurrent callers wait for completion instead of
    running the mutation twice.
    """
    if not idempotency_key:
        return execute(None)

    key = idempotency_key.strip()
    if not key or len(key) > 128:
        raise ValueError("Idempotency key must contain 1 to 128 characters.")

    payload_hash = _payload_hash(payload)
    now = int(time.time())
    lease_until = now + 30

    for _ in range(100):
        conn = get_connection()
        try:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """SELECT operation, payload_hash, status,
                              response_json, lease_until
                         FROM api_idempotency
                        WHERE user_id = ? AND idempotency_key = ?""",
                    (user_id, key),
                ).fetchone()

                if row is not None:
                    if row["operation"] != operation or row["payload_hash"] != payload_hash:
                        raise IdempotencyConflict(
                            "Idempotency key is already used for a different operation or payload."
                        )
                    if row["status"] == "completed":
                        return json.loads(row["response_json"])

                    if row["lease_until"] > now:
                        conn.rollback()
                        should_wait = True
                        can_execute = False
                    else:
                        conn.execute(
                            """UPDATE api_idempotency
                                  SET lease_until = ?, updated_at = ?
                                WHERE user_id = ? AND idempotency_key = ?""",
                            (lease_until, now, user_id, key),
                        )
                        should_wait = False
                        can_execute = True
                else:
                    conn.execute(
                        """INSERT INTO api_idempotency
                               (user_id, idempotency_key, operation, payload_hash,
                                status, response_json, lease_until, created_at, updated_at)
                           VALUES (?, ?, ?, ?, 'processing', NULL, ?, ?, ?)""",
                        (user_id, key, operation, payload_hash,
                         lease_until, now, now),
                    )
                    should_wait = False
                    can_execute = True

                if not can_execute:
                    pass
                else:
                    try:
                        response = execute(conn)
                        if not isinstance(response, dict):
                            raise TypeError(
                                "Idempotent mutation responses must be dictionaries."
                            )
                        conn.execute(
                            """UPDATE api_idempotency
                                  SET status = 'completed',
                                      response_json = ?,
                                      lease_until = 0,
                                      updated_at = ?
                                WHERE user_id = ? AND idempotency_key = ?""",
                            (json.dumps(response, sort_keys=True),
                             int(time.time()), user_id, key),
                        )
                        return response
                    except Exception:
                        conn.execute(
                            """UPDATE api_idempotency
                                  SET lease_until = 0, updated_at = ?
                                WHERE user_id = ? AND idempotency_key = ?""",
                            (int(time.time()), user_id, key),
                        )
                        conn.commit()
                        raise
        finally:
            conn.close()

        if should_wait:
            time.sleep(0.05)
            now = int(time.time())
            continue

    raise IdempotencyInProgress(
        "The same idempotent request is still being processed."
    )


def save_note(user_id: int, title: str, content: str) -> int:
    """Insert a note and return its id."""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO notes (user_id, title, content) VALUES (?, ?, ?)",
                (user_id, title, content),
            )
            note_id = cur.lastrowid
            _queue_note_embedding(conn, note_id, user_id, title, content)
            return note_id
    finally:
        conn.close()


def save_note_idempotent(
    user_id: int,
    title: str,
    content: str,
    idempotency_key: str,
) -> dict:
    # The payload hash is part of the key claim, preventing accidental reuse
    # for a different note while preserving the exact first response.
    return run_idempotent_request(
        user_id,
        idempotency_key,
        "save_note",
        {"title": title, "content": content},
        lambda conn: save_note_idempotent_in_transaction(
            conn, user_id, title, content
        ),
    )


def save_note_idempotent_in_transaction(conn, user_id, title, content) -> dict:
    if conn is None:
        note_id = save_note(user_id, title, content)
        return {
            "id": note_id,
            "title": title,
            "content": content,
            "created_at": "",
        }
    cur = conn.execute(
        "INSERT INTO notes (user_id, title, content) VALUES (?, ?, ?)",
        (user_id, title, content),
    )
    note_id = cur.lastrowid
    _queue_note_embedding(conn, note_id, user_id, title, content)
    return {
        "id": note_id,
        "title": title,
        "content": content,
        "created_at": "",
    }


def update_note(user_id: int, note_id: int, title: str, content: str) -> bool:
    """Update an owned note and queue its replacement vector."""
    conn = get_connection()
    try:
        with conn:
            # A content update invalidates the old point immediately and leaves
            # re-indexing to the existing retry queue.
            indexed = conn.execute(
                """SELECT i.collection
                     FROM note_embedding_index AS i
                     JOIN notes AS n ON n.id = i.note_id
                    WHERE i.note_id = ? AND n.user_id = ?""",
                (note_id, user_id),
            ).fetchone()
            note = conn.execute(
                "SELECT 1 FROM notes WHERE id = ? AND user_id = ?",
                (note_id, user_id),
            ).fetchone()
            if note is None:
                return False
            if indexed is not None:
                note_vector_index.delete_points(indexed["collection"], [note_id])
            updated = conn.execute(
                "UPDATE notes SET title = ?, content = ? WHERE id = ? AND user_id = ?",
                (title, content, note_id, user_id),
            )
            if not updated.rowcount:
                return False
            conn.execute(
                "DELETE FROM note_embedding_index WHERE note_id = ?", (note_id,)
            )
            _queue_note_embedding(conn, note_id, user_id, title, content)
            return True
    finally:
        conn.close()


def update_note_idempotent(
    user_id: int,
    note_id: int,
    title: str,
    content: str,
    idempotency_key: str,
) -> dict:
    def execute(conn) -> dict:
        if conn is None:
            if not update_note(user_id, note_id, title, content):
                return {"found": False}
            return {
                "found": True,
                "id": note_id,
                "title": title,
                "content": content,
                "created_at": "",
            }
        indexed = conn.execute(
            """SELECT i.collection
                 FROM note_embedding_index AS i
                 JOIN notes AS n ON n.id = i.note_id
                WHERE i.note_id = ? AND n.user_id = ?""",
            (note_id, user_id),
        ).fetchone()
        note = conn.execute(
            "SELECT 1 FROM notes WHERE id = ? AND user_id = ?",
            (note_id, user_id),
        ).fetchone()
        if note is None:
            return {"found": False}
        if indexed is not None:
            note_vector_index.delete_points(indexed["collection"], [note_id])
        updated = conn.execute(
            "UPDATE notes SET title = ?, content = ? WHERE id = ? AND user_id = ?",
            (title, content, note_id, user_id),
        )
        if not updated.rowcount:
            return {"found": False}
        conn.execute(
            "DELETE FROM note_embedding_index WHERE note_id = ?", (note_id,)
        )
        _queue_note_embedding(conn, note_id, user_id, title, content)
        return {
            "found": True,
            "id": note_id,
            "title": title,
            "content": content,
            "created_at": "",
        }

    return run_idempotent_request(
        user_id,
        idempotency_key,
        "update_note",
        {"note_id": note_id, "title": title, "content": content},
        execute,
    )


def search_notes(user_id: int, keyword: str) -> list[dict]:
    """Return notes whose title or content contains `keyword` (case-insensitive)."""
    conn = get_connection()
    try:
        like = f"%{keyword}%"
        rows = conn.execute(
            """SELECT id, title, content, created_at
                 FROM notes
                WHERE user_id = ? AND (title LIKE ? OR content LIKE ?)
                ORDER BY id""",
            (user_id, like, like),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_pending_note_embeddings(limit: int = 20, lease_owner: str | None = None) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ValueError("Embedding batch size must be between 1 and 100.")
    conn = get_connection()
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            now = int(time.time())
            if lease_owner:
                conn.execute(
                    """UPDATE note_embedding_jobs
                          SET lease_owner = NULL, lease_until = 0
                        WHERE lease_until > 0 AND lease_until <= ?""",
                    (now,),
                )
            rows = conn.execute(
                """SELECT n.id, n.user_id, n.title, n.content
                     FROM note_embedding_jobs AS j
                     JOIN notes AS n ON n.id = j.note_id AND n.user_id = j.user_id
                    WHERE j.next_attempt_at <= ?
                      AND (
                           j.lease_owner IS NULL
                           OR j.lease_until <= ?
                           OR j.lease_owner = ?
                          )
                    ORDER BY j.next_attempt_at, j.note_id
                    LIMIT ?""",
                (now, now, lease_owner or "", limit),
            ).fetchall()
            if lease_owner:
                lease_until = now + 300
                conn.executemany(
                    """UPDATE note_embedding_jobs
                          SET lease_owner = ?, lease_until = ?
                        WHERE note_id = ?""",
                    [(lease_owner, lease_until, row["id"]) for row in rows],
                )
            return [dict(row) for row in rows]
    finally:
        conn.close()


def record_note_embedding_failure(note_id: int, error: str) -> None:
    conn = get_connection()
    try:
        with conn:
            row = conn.execute(
                "SELECT attempts FROM note_embedding_jobs WHERE note_id = ?",
                (note_id,),
            ).fetchone()
            if row is None:
                return
            attempts = row["attempts"] + 1
            retry_delay = min(3600, 2 ** min(attempts, 12))
            conn.execute(
                """UPDATE note_embedding_jobs
                      SET attempts = ?, next_attempt_at = ?, last_error = ?,
                          lease_owner = NULL, lease_until = 0
                    WHERE note_id = ?""",
                (
                    attempts,
                    int(time.time()) + retry_delay,
                    error[:500],
                    note_id,
                ),
            )
    finally:
        conn.close()


def get_legacy_note_embedding(note_id: int) -> dict | None:
    # Read a legacy vector only for migration. It avoids re-calling Gemini
    # when the stored model and content hash still match the current note.
    conn = get_connection()
    try:
        if not _table_exists(conn, "note_embeddings_legacy"):
            return None
        row = conn.execute(
            """SELECT model, dimensions, embedding, content_hash
                 FROM note_embeddings_legacy WHERE note_id = ?""",
            (note_id,),
        ).fetchone()
        if row is None:
            return None
        dimensions = row["dimensions"]
        if dimensions <= 0 or len(row["embedding"]) != dimensions * 4:
            raise ValueError(f"Legacy embedding for note {note_id} is malformed.")
        return {
            "model": row["model"],
            "embedding": list(struct.unpack(f"<{dimensions}f", row["embedding"])),
            "content_hash": row["content_hash"],
        }
    finally:
        conn.close()


def _float_vector(values: list[float]) -> list[float]:
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding must contain numeric values.") from exc
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding must contain finite values.")
    return vector


def upsert_note_embedding(
    user_id: int,
    note_id: int,
    embedding: list[float],
    model: str,
    content_hash: str,
) -> bool:
    """Idempotently replace the searchable vector for an owned note."""
    vector = _float_vector(embedding)
    if not model or not content_hash:
        raise ValueError("Embedding model and content hash are required.")

    conn = get_connection()
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            note = conn.execute(
                "SELECT title, content FROM notes WHERE id = ? AND user_id = ?",
                (note_id, user_id),
            ).fetchone()
            # Validate both ownership and the note snapshot before touching
            # Qdrant. A stale worker can therefore never replace a newer vector.
            if note is None or note_content_hash(
                note["title"], note["content"]
            ) != content_hash:
                return False

            # Qdrant is updated first, while SQL remains the stale-result and
            # retry source of truth. The index reference is committed afterward.
            collection = note_vector_index.upsert(
                note_id=note_id,
                user_id=user_id,
                vector=vector,
                model=model,
                content_hash=content_hash,
            )
            previous = conn.execute(
                "SELECT collection FROM note_embedding_index WHERE note_id = ?",
                (note_id,),
            ).fetchone()
            if previous is not None and previous["collection"] != collection:
                note_vector_index.delete_points(previous["collection"], [note_id])
            conn.execute(
                """INSERT INTO note_embedding_index
                       (note_id, user_id, model, dimensions, collection,
                        content_hash, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(note_id) DO UPDATE SET
                       user_id = excluded.user_id,
                       model = excluded.model,
                       dimensions = excluded.dimensions,
                       collection = excluded.collection,
                       content_hash = excluded.content_hash,
                       updated_at = excluded.updated_at""",
                (
                    note_id,
                    user_id,
                    model,
                    len(vector),
                    collection,
                    content_hash,
                    int(time.time()),
                ),
            )
            if _table_exists(conn, "note_embeddings_legacy"):
                conn.execute(
                    "DELETE FROM note_embeddings_legacy WHERE note_id = ?",
                    (note_id,),
                )
            conn.execute(
                "DELETE FROM note_embedding_jobs WHERE note_id = ?",
                (note_id,),
            )
            return True
    finally:
        conn.close()


def search_notes_hybrid(
    user_id: int,
    keyword: str,
    query_embedding: list[float],
    model: str,
    limit: int = 20,
) -> list[dict]:
    """Combine Qdrant similarity with the existing case-insensitive search."""
    query_vector = _float_vector(query_embedding)
    if not model:
        raise ValueError("Embedding model is required.")
    if not 1 <= limit <= 100:
        raise ValueError("Search limit must be between 1 and 100.")

    conn = get_connection()
    try:
        # Qdrant provides semantic candidates; SQLite adds exact keyword
        # matches and then validates ownership, model, dimensions, and hash.
        vector_matches = dict(
            note_vector_index.search(
                user_id=user_id,
                vector=query_vector,
                model=model,
                limit=limit,
            )
        )
        keyword_rows = (
            conn.execute(
                """SELECT id FROM notes
                     WHERE user_id = ? AND (title LIKE ? OR content LIKE ?)""",
                (user_id, f"%{keyword}%", f"%{keyword}%"),
            ).fetchall()
            if keyword
            else []
        )
        keyword_ids = {
            row["id"] for row in keyword_rows
        } if keyword else set()
        candidate_ids = set(vector_matches) | keyword_ids
        if not candidate_ids:
            return []
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = conn.execute(
            f"""SELECT id, title, content, created_at
                  FROM notes
                 WHERE user_id = ? AND id IN ({placeholders})""",
            [user_id, *candidate_ids],
        ).fetchall()
        indexed_rows = conn.execute(
            f"""SELECT note_id, model, dimensions, content_hash
                  FROM note_embedding_index
                 WHERE user_id = ? AND note_id IN ({placeholders})""",
            [user_id, *candidate_ids],
        ).fetchall()
        indexed_by_id = {row["note_id"]: row for row in indexed_rows}
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


def clear_notes(user_id: int) -> int:
    """Delete all notes owned by a user and return the number deleted."""
    conn = get_connection()
    try:
        with conn:
            # Remove Qdrant points before deleting the SQL note rows so each
            # collection is cleaned only for the signed-in user's points.
            indexed_rows = conn.execute(
                """SELECT note_id, collection FROM note_embedding_index
                     WHERE user_id = ?""",
                (user_id,),
            ).fetchall()
            by_collection: dict[str, list[int]] = {}
            for row in indexed_rows:
                by_collection.setdefault(row["collection"], []).append(row["note_id"])
            for collection, note_ids in by_collection.items():
                note_vector_index.delete_points(collection, note_ids)
            cur = conn.execute("DELETE FROM notes WHERE user_id = ?", (user_id,))
            return cur.rowcount
    finally:
        conn.close()


def delete_note(user_id: int, note_id: int) -> bool:
    """Delete one owned note and its searchable vector."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            note = conn.execute(
                "SELECT id FROM notes WHERE id = ? AND user_id = ?",
                (note_id, user_id),
            ).fetchone()
            if note is None:
                return False

            indexed = conn.execute(
                """SELECT collection FROM note_embedding_index
                     WHERE note_id = ? AND user_id = ?""",
                (note_id, user_id),
            ).fetchone()
            if indexed is not None:
                note_vector_index.delete_points(indexed["collection"], [note_id])
            if _table_exists(conn, "note_embeddings_legacy"):
                conn.execute(
                    "DELETE FROM note_embeddings_legacy WHERE note_id = ?",
                    (note_id,),
                )
            deleted = conn.execute(
                "DELETE FROM notes WHERE id = ? AND user_id = ?",
                (note_id, user_id),
            )
            return deleted.rowcount > 0
    finally:
        conn.close()


def clear_notes_in_transaction(conn, user_id: int) -> dict:
    if conn is None:
        return {"deleted_count": clear_notes(user_id)}
    indexed_rows = conn.execute(
        """SELECT note_id, collection FROM note_embedding_index
             WHERE user_id = ?""",
        (user_id,),
    ).fetchall()
    by_collection: dict[str, list[int]] = {}
    for row in indexed_rows:
        by_collection.setdefault(row["collection"], []).append(row["note_id"])
    for collection, note_ids in by_collection.items():
        note_vector_index.delete_points(collection, note_ids)
    cur = conn.execute("DELETE FROM notes WHERE user_id = ?", (user_id,))
    return {"deleted_count": cur.rowcount}


def clear_notes_idempotent(
    user_id: int,
    idempotency_key: str,
) -> dict:
    # Clearing has no payload beyond the owner scope. The stored count prevents
    # a timeout retry from reporting a second deletion as 0 instead of 1.
    return run_idempotent_request(
        user_id,
        idempotency_key,
        "clear_notes",
        {},
        lambda conn: clear_notes_in_transaction(conn, user_id),
    )


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def create_task(user_id: int, title: str, due_date: str | None = None) -> int:
    """Insert a task and return its id. `due_date` is a free-form string (e.g. '2026-12-01')."""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO tasks (user_id, title, due_date) VALUES (?, ?, ?)",
                (user_id, title, due_date),
            )
            return cur.lastrowid
    finally:
        conn.close()


def create_task_idempotent(
    user_id: int,
    title: str,
    due_date: str | None,
    idempotency_key: str,
) -> dict:
    return run_idempotent_request(
        user_id,
        idempotency_key,
        "create_task",
        {"title": title, "due_date": due_date or ""},
        lambda conn: create_task_in_transaction(
            conn, user_id, title, due_date
        ),
    )


def create_task_in_transaction(
    conn,
    user_id: int,
    title: str,
    due_date: str | None,
) -> dict:
    if conn is None:
        task_id = create_task(user_id, title, due_date)
        return {
            "id": task_id,
            "title": title,
            "due_date": due_date or "",
            "completed": False,
            "created_at": "",
        }
    cur = conn.execute(
        "INSERT INTO tasks (user_id, title, due_date) VALUES (?, ?, ?)",
        (user_id, title, due_date),
    )
    return {
        "id": cur.lastrowid,
        "title": title,
        "due_date": due_date or "",
        "completed": False,
        "created_at": "",
    }


def get_tasks(user_id: int) -> list[dict]:
    """Return all tasks, incomplete first, then by id."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, title, due_date, completed, created_at
                 FROM tasks
                WHERE user_id = ?
                ORDER BY completed ASC, id ASC""",
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def complete_task(user_id: int, task_id: int) -> bool:
    """Mark a task complete. Returns True if a row was updated, False if not found."""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "UPDATE tasks SET completed = 1 WHERE id = ? AND user_id = ?",
                (task_id, user_id),
            )
            return cur.rowcount > 0
    finally:
        conn.close()


def complete_task_idempotent(
    user_id: int,
    task_id: int,
    idempotency_key: str,
) -> dict:
    def execute(conn) -> dict:
        if conn is None:
            return {"success": complete_task(user_id, task_id)}
        return {
            "success": conn.execute(
                "UPDATE tasks SET completed = 1 WHERE id = ? AND user_id = ?",
                (task_id, user_id),
            ).rowcount
            > 0
        }

    return run_idempotent_request(
        user_id,
        idempotency_key,
        "complete_task",
        {"task_id": task_id},
        execute,
    )

def delete_task(user_id: int, task_id: int) -> bool:
    """Delete one task. Returns False if it does not exist or is not the user's."""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM tasks WHERE id = ? AND user_id = ?",
                (task_id, user_id),
            )
            return cur.rowcount > 0
    finally:
        conn.close()


def delete_task_idempotent(
    user_id: int,
    task_id: int,
    idempotency_key: str,
) -> dict:
    def execute(conn) -> dict:
        if conn is None:
            return {"success": delete_task(user_id, task_id)}
        return {
            "success": conn.execute(
                "DELETE FROM tasks WHERE id = ? AND user_id = ?",
                (task_id, user_id),
            ).rowcount
            > 0
        }

    return run_idempotent_request(
        user_id,
        idempotency_key,
        "delete_task",
        {"task_id": task_id},
        execute,
    )


def clear_tasks(user_id: int) -> int:
    """Delete every task owned by the user and return how many were deleted."""
    conn = get_connection()
    try:
        with conn:
            return conn.execute(
                "DELETE FROM tasks WHERE user_id = ?", (user_id,)
            ).rowcount
    finally:
        conn.close()


def clear_tasks_idempotent(user_id: int, idempotency_key: str) -> dict:
    # The stored count stops a timeout retry from reporting 0 on the second call.
    def execute(conn) -> dict:
        if conn is None:
            return {"deleted_count": clear_tasks(user_id)}
        return {
            "deleted_count": conn.execute(
                "DELETE FROM tasks WHERE user_id = ?", (user_id,)
            ).rowcount
        }

    return run_idempotent_request(
        user_id, idempotency_key, "clear_tasks", {}, execute,
    )


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #
# scrypt cost parameters. n=2**14, r=8 uses ~16 MB of RAM per hash.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


def hash_password(password: str) -> str:
    """Salted, deliberately slow hash. Format: scrypt$n$r$p$salt$hash (hex)."""
    salt = os.urandom(16)  # unique per hash: same password -> different result
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt_hex, hash_hex = stored.split("$")
        digest = hashlib.scrypt(
            password.encode(),
            salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, bytes.fromhex(hash_hex))  # constant-time

# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
class UsernameTaken(ValueError):
    pass


_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")
_MIN_PASSWORD, _MAX_PASSWORD = 8, 128  # the max stops huge-password CPU abuse

# Burns the same CPU time when the username doesn't exist, so response timing
# can't be used to discover valid usernames.
_DUMMY_HASH = hash_password("dummy-password-for-timing")


def register_user(username: str, password: str) -> int:
    """Create a user and return its id. Raises ValueError / UsernameTaken."""
    if not _USERNAME_RE.match(username):
        raise ValueError("Username must be 3-32 chars: letters, digits, underscore.")
    if not _MIN_PASSWORD <= len(password) <= _MAX_PASSWORD:
        raise ValueError(
            f"Password must be {_MIN_PASSWORD}-{_MAX_PASSWORD} characters."
        )
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, hash_password(password)),
            )
        return cur.lastrowid
    except sqlite3.IntegrityError as exc: # two simultaneous sign-ups both pass the check
        raise UsernameTaken("Username already taken.") from exc
    finally:
        conn.close()


def authenticate(username: str, password: str) -> int | None:
    """Return the user id if the credentials are right, else None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()
    if row is None: # kill time to make somone trying to guess our username
        verify_password(password, _DUMMY_HASH)  # equalize timing
        return None
    return row["id"] if verify_password(password, row["password_hash"]) else None

# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days


def _hash_token(token: str) -> str:
    """Store only a hash, so a leaked DB file does not leak usable sessions."""
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: int, ttl_seconds: int = SESSION_TTL_SECONDS) -> str:
    """Create a session for a user. The raw token is returned ONCE."""
    token = secrets.token_urlsafe(32)  # 256 bits of randomness
    now = int(time.time())
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (_hash_token(token), user_id, now, now + ttl_seconds),
            )
    finally:
        conn.close()
    return token


def get_session_user(token: str) -> int | None:
    """Return the user_id for a valid, unexpired token, else None."""
    if not token:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT user_id FROM sessions WHERE token_hash = ? AND expires_at > ?",
            (_hash_token(token), int(time.time())),
        ).fetchone()
    finally:
        conn.close()
    return row["user_id"] if row else None


def get_session_username(token: str) -> str | None:
    """Return the username for a valid session without exposing session tokens."""
    if not token:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT u.username
                 FROM sessions AS s
                 JOIN users AS u ON u.id = s.user_id
                WHERE s.token_hash = ? AND s.expires_at > ?""",
            (_hash_token(token), int(time.time())),
        ).fetchone()
    finally:
        conn.close()
    return row["username"] if row else None


def delete_session(token: str) -> None:
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),)
            )
    finally:
        conn.close()