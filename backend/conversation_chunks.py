"""Deterministic, turn-aware chunks for semantic conversation search."""

import hashlib
import json
import uuid

CHUNK_NAMESPACE = uuid.UUID("4d5f05d8-7d9c-4ed8-b4ca-d5664bfc0e5d")
MAX_CHUNK_CHARS = 4_000
CHUNK_OVERLAP_CHARS = 320


def conversation_snapshot_hash(messages: list[dict]) -> str:
    """Hash the ordered message snapshot used to guard background indexing."""
    canonical = json.dumps(
        [(int(row["id"]), row["role"], row["content"]) for row in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _split_long_turn(text: str) -> list[str]:
    """Split oversized turns near paragraph/sentence boundaries with small overlap."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n", start + MAX_CHUNK_CHARS // 2, end),
                text.rfind(". ", start + MAX_CHUNK_CHARS // 2, end),
                text.rfind(" ", start + MAX_CHUNK_CHARS // 2, end),
            )
            if boundary > start:
                end = min(end, boundary + (1 if text[boundary] == "\n" else 2))
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP_CHARS)
    return [piece for piece in pieces if piece]


def chunk_conversation(conversation_id: int, messages: list[dict]) -> list[dict]:
    """Return one semantic chunk per user/assistant turn."""
    turns: list[list[dict]] = []
    current: list[dict] = []
    for message in messages:
        role = message["role"]
        if role == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)

    chunks = []
    for turn in turns:
        start_id = int(turn[0]["id"])
        end_id = int(turn[-1]["id"])
        content = "\n\n".join(
            f"[{message['role'].upper()}]\n{message['content'].strip()}"
            for message in turn
            if message["content"].strip()
        ).strip()
        if not content:
            continue

        for part_index, part in enumerate(_split_long_turn(content)):
            chunk_id = str(
                uuid.uuid5(
                    CHUNK_NAMESPACE,
                    f"{conversation_id}:{start_id}:{end_id}:{part_index}",
                )
            )
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "conversation_id": conversation_id,
                    "message_start_id": start_id,
                    "message_end_id": end_id,
                    "chunk_index": len(chunks),
                    "content": part,
                    "content_hash": hashlib.sha256(part.encode("utf-8")).hexdigest(),
                }
            )
    return chunks
