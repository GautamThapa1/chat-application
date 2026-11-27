"""Per-turn rules that travel from ChatService to the agent and its tools."""

import threading
import time
from dataclasses import dataclass, field


class TurnStopped(Exception):
    """The user pressed Stop, so no new work may start."""


class TurnTimedOut(Exception):
    """The turn ran past its time limit."""


@dataclass(frozen=True)
class TurnControl:
    allow_writes: bool = True
    can_confirm: bool = False
    conversation_id: int | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    deadline: float | None = None  # time.monotonic() value

    def checkpoint(self) -> None:
        """Raise TurnStopped if cancelled, else TurnTimedOut if past the deadline."""
        if self.cancelled.is_set():
            raise TurnStopped
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TurnTimedOut
