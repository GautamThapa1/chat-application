"""SQLite persistence for accounts and user-owned application data."""

import hashlib
import hmac
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


DB_PATH = "app.db"
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
    *,
    reset_attempts: bool = True,
) -> None:
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
    rows = conn.execute(
        """SELECT n.id, n.user_id, n.title, n.content,
                  e.model, e.content_hash
             FROM notes AS n
             LEFT JOIN note_embeddings AS e ON e.note_id = n.id"""
    ).fetchall()
    current_model = embedding_model()
    for row in rows:
        expected_hash = note_content_hash(row["title"], row["content"])
        if (
            row["model"] == current_model
            and row["content_hash"] == expected_hash
        ):
            continue
        if row["model"] is not None:
            conn.execute(
                "DELETE FROM note_embeddings WHERE note_id = ?",
                (row["id"],),
            )
        _queue_note_embedding(conn, row["id"], row["user_id"], reset_attempts=False)


def init_db():
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        with conn:
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS note_embeddings (
                    note_id      INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
                    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    model        TEXT NOT NULL,
                    dimensions   INTEGER NOT NULL CHECK (dimensions > 0),
                    embedding    BLOB NOT NULL,
                    content_hash TEXT NOT NULL,
                    updated_at   INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS note_embedding_jobs (
                    note_id         INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
                    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    attempts        INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at INTEGER NOT NULL DEFAULT 0,
                    last_error      TEXT
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
                "CREATE INDEX IF NOT EXISTS idx_note_embeddings_user "
                "ON note_embeddings(user_id, note_id)"
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


def save_conversation(
    user_id: int,
    conversation_id: int,
    user_content: str,
    assistant_content: str,
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

            conn.executemany(
                """INSERT INTO messages
                   (conversation_id, user_id, role, content)
                   VALUES (?, ?, ?, ?)""",
                (
                    (conversation_id, user_id, "user", user_content),
                    (conversation_id, user_id, "assistant", assistant_content),
                ),
            )
            title = conversation["title"]
            if title == "New chat":
                first_line = user_content.strip().splitlines()
                if first_line:
                    title = first_line[0][:80]
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
            _queue_note_embedding(conn, note_id, user_id)
            return note_id
    finally:
        conn.close()


def update_note(user_id: int, note_id: int, title: str, content: str) -> bool:
    """Update an owned note and queue its replacement vector."""
    conn = get_connection()
    try:
        with conn:
            updated = conn.execute(
                "UPDATE notes SET title = ?, content = ? WHERE id = ? AND user_id = ?",
                (title, content, note_id, user_id),
            )
            if not updated.rowcount:
                return False
            conn.execute("DELETE FROM note_embeddings WHERE note_id = ?", (note_id,))
            _queue_note_embedding(conn, note_id, user_id)
            return True
    finally:
        conn.close()


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


def get_pending_note_embeddings(limit: int = 20) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ValueError("Embedding batch size must be between 1 and 100.")
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT n.id, n.user_id, n.title, n.content
                 FROM note_embedding_jobs AS j
                 JOIN notes AS n ON n.id = j.note_id AND n.user_id = j.user_id
                WHERE j.next_attempt_at <= ?
                ORDER BY j.next_attempt_at, j.note_id
                LIMIT ?""",
            (int(time.time()), limit),
        ).fetchall()
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
                      SET attempts = ?, next_attempt_at = ?, last_error = ?
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
    try:
        packed = struct.pack(f"<{len(vector)}f", *vector)
    except (OverflowError, struct.error) as exc:
        raise ValueError("Embedding values exceed supported precision.") from exc

    conn = get_connection()
    try:
        with conn:
            note = conn.execute(
                "SELECT title, content FROM notes WHERE id = ? AND user_id = ?",
                (note_id, user_id),
            ).fetchone()
            if note is None or note_content_hash(
                note["title"], note["content"]
            ) != content_hash:
                return False

            conn.execute(
                """INSERT INTO note_embeddings
                       (note_id, user_id, model, dimensions, embedding,
                        content_hash, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(note_id) DO UPDATE SET
                       user_id = excluded.user_id,
                       model = excluded.model,
                       dimensions = excluded.dimensions,
                       embedding = excluded.embedding,
                       content_hash = excluded.content_hash,
                       updated_at = excluded.updated_at""",
                (
                    note_id,
                    user_id,
                    model,
                    len(vector),
                    packed,
                    content_hash,
                    int(time.time()),
                ),
            )
            conn.execute(
                "DELETE FROM note_embedding_jobs WHERE note_id = ?",
                (note_id,),
            )
            return True
    finally:
        conn.close()


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def search_notes_hybrid(
    user_id: int,
    keyword: str,
    query_embedding: list[float],
    model: str,
    limit: int = 20,
) -> list[dict]:
    """Combine exact cosine similarity with the existing case-insensitive search."""
    query_vector = _float_vector(query_embedding)
    if not model:
        raise ValueError("Embedding model is required.")
    if not 1 <= limit <= 100:
        raise ValueError("Search limit must be between 1 and 100.")

    conn = get_connection()
    try:
        keyword_ids = {
            row["id"]
            for row in conn.execute(
                """SELECT id FROM notes
                     WHERE user_id = ? AND (title LIKE ? OR content LIKE ?)""",
                (user_id, f"%{keyword}%", f"%{keyword}%"),
            ).fetchall()
        } if keyword else set()
        rows = conn.execute(
            """SELECT n.id, n.title, n.content, n.created_at,
                      e.embedding, e.dimensions
                 FROM notes AS n
                 LEFT JOIN note_embeddings AS e
                   ON e.note_id = n.id
                  AND e.user_id = n.user_id
                  AND e.model = ?
                WHERE n.user_id = ?""",
            (model, user_id),
        ).fetchall()

        ranked = []
        for row in rows:
            similarity = None
            if row["embedding"] is not None:
                dimensions = row["dimensions"]
                if dimensions == len(query_vector) and len(row["embedding"]) == dimensions * 4:
                    stored_vector = list(
                        struct.unpack(f"<{dimensions}f", row["embedding"])
                    )
                    similarity = max(
                        -1.0,
                        min(
                            1.0,
                            _cosine_similarity(query_vector, stored_vector),
                        ),
                    )
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
            cur = conn.execute("DELETE FROM notes WHERE user_id = ?", (user_id,))
            return cur.rowcount
    finally:
        conn.close()


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


def delete_session(token: str) -> None:
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),)
            )
    finally:
        conn.close()