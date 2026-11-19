"""SQLite persistence: messages, notes, tasks."""

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

# create my tables: messages, notes, tasks
def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS notes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT NOT NULL,
            due_date   TEXT,
            completed  INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL
        );
    """)
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Messages (existing)
# --------------------------------------------------------------------------- #
def save_message(role: str, content: str) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO messages (role, content) VALUES (?, ?)",
        (role, content),
    )
    conn.commit()
    msg_id = cur.lastrowid
    conn.close()
    return msg_id


def save_conversation(user_content: str, assistant_content: str) -> None:
    conn = get_connection()
    try:
        with conn:
            conn.executemany(
                "INSERT INTO messages (role, content) VALUES (?, ?)",
                (
                    ("user", user_content),
                    ("assistant", assistant_content),
                ),
            )
    finally:
        conn.close()


def get_messages() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, role, content, created_at FROM messages ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #
def save_note(title: str, content: str) -> int:
    """Insert a note and return its id."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO notes (title, content) VALUES (?, ?)",
        (title, content),
    )
    conn.commit()
    note_id = cur.lastrowid
    conn.close()
    return note_id


def search_notes(keyword: str) -> list[dict]:
    """Return notes whose title or content contains `keyword` (case-insensitive)."""
    conn = get_connection()
    like = f"%{keyword}%"
    rows = conn.execute(
        """SELECT id, title, content, created_at
             FROM notes
            WHERE title LIKE ? OR content LIKE ?
            ORDER BY id""",
        (like, like),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def create_task(title: str, due_date: str | None = None) -> int:
    """Insert a task and return its id. `due_date` is a free-form string (e.g. '2026-12-01')."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO tasks (title, due_date) VALUES (?, ?)",
        (title, due_date),
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()
    return task_id


def get_tasks() -> list[dict]:
    """Return all tasks, incomplete first, then by id."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT id, title, due_date, completed, created_at
             FROM tasks
            ORDER BY completed ASC, id ASC"""
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def complete_task(task_id: int) -> bool:
    """Mark a task complete. Returns True if a row was updated, False if not found."""
    conn = get_connection()
    cur = conn.execute(
        "UPDATE tasks SET completed = 1 WHERE id = ?",
        (task_id,),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated

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