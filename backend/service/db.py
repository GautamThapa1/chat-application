"""SQLite persistence for accounts and user-owned application data."""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time

DB_PATH = "app.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
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
                "CREATE INDEX IF NOT EXISTS idx_tasks_user_status "
                "ON tasks(user_id, completed, id)"
            )
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


def list_conversations(user_id: int) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT conversation_id AS id, user_id, title, created_at, updated_at "
            "FROM conversations WHERE user_id = ? "
            "ORDER BY updated_at DESC, conversation_id DESC",
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]
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
            return cur.lastrowid
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