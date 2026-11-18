"""SQLite persistence: messages, notes, tasks."""

import sqlite3

DB_PATH = "app.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
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