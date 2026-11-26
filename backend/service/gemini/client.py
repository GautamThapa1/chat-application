"""Helpers around the existing synchronous Google GenAI client.

The client is still created by ``backend.service.main`` and injected into
ChatServicer. This module deliberately does not create a second client.
"""

from __future__ import annotations

from typing import Any

from google.genai import types

from backend.service.gemini_chat import bounded_contents_sync


def build_initial_contents(
    message: str,
    gemini_client: Any,
    model: str,
    history: list[dict[str, str]],
) -> list[types.Content]:
    """Reuse the existing token-bounded history selection for a graph run."""
    return bounded_contents_sync(message, gemini_client, model, history)


__all__ = ["build_initial_contents"]
