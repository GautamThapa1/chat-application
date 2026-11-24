from backend.conversation_chunks import (
    CHUNK_OVERLAP_CHARS,
    MAX_CHUNK_CHARS,
    chunk_conversation,
    conversation_snapshot_hash,
)


def test_chunk_conversation_groups_user_and_assistant_turns():
    messages = [
        {"id": 10, "role": "user", "content": "Explain caching"},
        {"id": 11, "role": "assistant", "content": "Caching stores reused data."},
        {"id": 12, "role": "user", "content": "What about Redis?"},
        {"id": 13, "role": "assistant", "content": "Redis is an in-memory store."},
    ]

    chunks = chunk_conversation(7, messages)

    assert len(chunks) == 2
    assert chunks[0]["message_start_id"] == 10
    assert chunks[0]["message_end_id"] == 11
    assert "[USER]" in chunks[0]["content"]
    assert "[ASSISTANT]" in chunks[0]["content"]
    assert chunks[1]["message_start_id"] == 12
    assert chunks[1]["message_end_id"] == 13


def test_chunk_conversation_splits_long_turns_with_overlap():
    messages = [
        {"id": 20, "role": "user", "content": "x" * (MAX_CHUNK_CHARS * 2)},
    ]

    chunks = chunk_conversation(8, messages)

    assert len(chunks) == 3
    assert all(len(chunk["content"]) <= MAX_CHUNK_CHARS for chunk in chunks)
    assert chunks[0]["content"][-CHUNK_OVERLAP_CHARS:] in chunks[1]["content"]


def test_snapshot_hash_is_order_and_content_sensitive():
    first = [{"id": 1, "role": "user", "content": "hello"}]
    changed = [{"id": 1, "role": "user", "content": "goodbye"}]

    assert conversation_snapshot_hash(first) != conversation_snapshot_hash(changed)
