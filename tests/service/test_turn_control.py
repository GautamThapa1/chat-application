"""Per-turn rules: Stop and time limits, run through the real agent graph."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import types

from backend.service import db
from backend.service.gemini import runner
from backend.service.main import build_http_options
from backend.service.turn_control import TurnControl, TurnStopped, TurnTimedOut


def _response(*calls, text=None):
    parts = [
        types.Part(function_call=types.FunctionCall(name=name, args=arguments))
        for name, arguments in calls
    ]
    if text is not None:
        parts.append(types.Part(text=text))
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=types.Content(role="model", parts=parts))
        ],
        text=text,
    )


def _stream(response):
    yield response


def _client(*responses):
    return SimpleNamespace(
        models=SimpleNamespace(
            count_tokens=Mock(return_value=SimpleNamespace(total_tokens=1)),
            generate_content_stream=Mock(return_value=_stream(responses[0])),
            generate_content=Mock(side_effect=responses[1:]),
        )
    )


def test_stop_prevents_the_next_tool(monkeypatch):
    control = TurnControl()
    saved = []

    def save_note(user_id, title, content, idempotency_key):
        saved.append(title)
        control.cancelled.set()
        return {"id": 1, "title": title, "content": content, "created_at": ""}

    monkeypatch.setattr(db, "save_note_idempotent", save_note)
    client = _client(_response(
        ("save_note", {"title": "A", "content": "c"}),
        ("save_note", {"title": "B", "content": "c"}),
    ))

    with pytest.raises(TurnStopped):
        list(runner.run_turn("save", client, "m", [], 1, "r", control=control))
    assert saved == ["A"]


def test_gemini_requests_have_a_time_limit():
    options = build_http_options()
    assert options.timeout == 60_000  # milliseconds
    assert options.retry_options.attempts == 2
    assert options.retry_options.http_status_codes == [503]


def test_expired_turn_makes_no_model_call():
    client = _client(_response(text="unused"))
    control = TurnControl(deadline=time.monotonic() - 1)
    with pytest.raises(TurnTimedOut):
        list(runner.run_turn("hi", client, "m", [], 1, "r", control=control))
    client.models.generate_content_stream.assert_not_called()
