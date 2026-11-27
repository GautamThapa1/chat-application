"""Keyword matching and the relevance cutoff for memories and notes."""

import pytest

from backend.service import db, memory_store, search_text

MODEL = "test-embedding"


@pytest.fixture
def user(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "search.db"))
    monkeypatch.setenv("DATABASE_KEY", "search-test-key")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("QDRANT_COLLECTION_PREFIX", f"search_{tmp_path.name.replace('-', '_')}")
    db.init_db()
    return db.register_user("search_user", "correct-horse")


SEEDS = [
    ("Backend", "My backend uses FastAPI"),
    ("Database", "The project database is SQLite"),
    ("Garden", "I grow tomatoes on the balcony"),
]


def seed(kind, user):
    save = memory_store.save_memory if kind == "memory" else db.save_note
    return [save(user, title, content) for title, content in SEEDS]


def keyword(kind, user, text):
    search = memory_store.search_memories if kind == "memory" else db.search_notes
    return [row["title"] for row in search(user, text)]


@pytest.mark.parametrize("kind", ["memory", "note"])
@pytest.mark.parametrize(("question", "expected"), [
    ("backend framework", ["Backend"]),
    ("what is my database", ["Database"]),
    ("tomato plants", ["Garden"]),
    ("FastAPI", ["Backend"]),
    ("my backend uses", ["Backend"]),
    ("the project database backend", ["Database", "Backend"]),
    ("favourite holiday", []),
    ("what do I do with it", []),
])
def test_keyword_search_matches_words_and_ranks_by_how_many(user, kind, question, expected):
    seed(kind, user)

    assert keyword(kind, user, question) == expected


def test_listing_notes_still_returns_everything(user):
    seed("note", user)

    assert len(db.search_notes(user, "")) == 3


def test_short_words_do_not_match_inside_longer_ones(user):
    db.save_note(user, "Career", "Interested in category theory")

    assert db.search_notes(user, "car cat") == []


def index(kind, user, record_id, title, content, vector):
    upsert = memory_store.upsert_memory_embedding if kind == "memory" else db.upsert_note_embedding
    assert upsert(user, record_id, vector, MODEL, db.note_content_hash(title, content))


def hybrid(kind, user, text, vector):
    search = memory_store.search_memories_hybrid if kind == "memory" else db.search_notes_hybrid
    return [row["title"] for row in search(user, text, vector, MODEL)]


@pytest.mark.parametrize("kind", ["memory", "note"])
def test_cutoff_drops_unrelated_meaning_matches_only(user, monkeypatch, kind):
    ids = seed(kind, user)
    for record_id, (title, content), vector in zip(ids, SEEDS, ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0])):
        index(kind, user, record_id, title, content, vector)

    # Off by default: every indexed item still comes back.
    assert len(hybrid(kind, user, "zzz", [1.0, 0.0])) == 3

    monkeypatch.setattr(search_text, "MIN_SIMILARITY", 0.5)
    assert hybrid(kind, user, "zzz", [1.0, 0.0]) == ["Backend"]
    # A word match is kept however low its similarity.
    assert hybrid(kind, user, "tomatoes", [1.0, 0.0]) == ["Backend", "Garden"]
