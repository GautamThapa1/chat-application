"""Gemini generation settings shared by the LangGraph integration.

The existing Google GenAI declarations and prompt remain the source of truth
in ``backend.service.gemini_chat`` during this incremental migration.
"""

from backend.service.gemini_chat import (
    DEFAULT_MAX_INPUT_TOKENS,
    MAX_INPUT_TOKENS_ENV,
    MAX_TOOL_CALLS_PER_ROUND,
    MAX_TOOL_ROUNDS,
    SYSTEM_INSTRUCTION,
    TOOL_CONFIG,
    FINAL_RESPONSE_CONFIG,
)

__all__ = [
    "DEFAULT_MAX_INPUT_TOKENS",
    "MAX_INPUT_TOKENS_ENV",
    "MAX_TOOL_CALLS_PER_ROUND",
    "MAX_TOOL_ROUNDS",
    "SYSTEM_INSTRUCTION",
    "TOOL_CONFIG",
    "FINAL_RESPONSE_CONFIG",
]
