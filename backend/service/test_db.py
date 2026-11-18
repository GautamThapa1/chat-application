"""Independent tests for notes + tasks. Uses a fresh temp DB per test."""
import os
import tempfile
from pathlib import Path

# Point DB_PATH at a temp file BEFORE importing db
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ.setdefault("_IGNORE", "1")
_tmp.close()

import backend.service.db as db
db.DB_PATH = _tmp.name
db.init_db()


def test_save_note():
    nid = db.save_note("Grocery", "milk, eggs, bread")
    assert isinstance(nid, int) and nid > 0
    print("save_note OK ->", nid)


def test_search_notes():
    db.save_note("Meeting", "discuss Q4 roadmap")
    db.save_note("Shopping", "buy a new laptop")
    hits = db.search_notes("roadmap")
    assert len(hits) == 1 and hits[0]["title"] == "Meeting"
    hits_ci = db.search_notes("LAPTOP")  # case-insensitive
    assert len(hits_ci) == 1
    print("search_notes OK ->", [h["title"] for h in hits])


def test_create_task():
    tid = db.create_task("Write report", due_date="2026-12-01")
    tid2 = db.create_task("No deadline task")
    assert tid > 0 and tid2 > 0
    print("create_task OK ->", tid, tid2)


def test_get_tasks():
    tasks = db.get_tasks()
    assert len(tasks) >= 2
    assert all("completed" in t for t in tasks)
    print("get_tasks OK ->", [(t["id"], t["title"], t["completed"]) for t in tasks])


def test_complete_task():
    tid = db.create_task("Temporary task")
    assert db.complete_task(tid) is True
    assert db.complete_task(999999) is False   # non-existent
    done = [t for t in db.get_tasks() if t["id"] == tid][0]
    assert done["completed"] == 1
    print("complete_task OK ->", tid)


if __name__ == "__main__":
    for fn in (test_save_note, test_search_notes,
               test_create_task, test_get_tasks, test_complete_task):
        fn()
        print(f"  ✓ {fn.__name__}")
    os.unlink(_tmp.name)
    print("\nAll tests passed.")

# PYTHONPATH=backend uv run python -m service.test_db