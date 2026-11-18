"""Simulated async response stream; replace this provider without changing the WebSocket protocol."""

import asyncio
from collections.abc import AsyncIterator

CHUNK_SIZE = 12
CHUNK_DELAY_SECONDS = 0.015


async def generate_response_chunks(response: str) -> AsyncIterator[str]:
    """Yield a completed response in small chunks to simulate token streaming."""
    for offset in range(0, len(response), CHUNK_SIZE):
        yield response[offset : offset + CHUNK_SIZE]
        await asyncio.sleep(CHUNK_DELAY_SECONDS)
