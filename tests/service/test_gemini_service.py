from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import types

from backend.service import gemini_chat


def _response(*parts):
    content = types.Content(role="model", parts=list(parts))
    text = "".join(part.text or "" for part in parts)
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=content)],
        text=text,
    )


def _response_stream(response):
    yield response


def _client(*responses, token_counts=None):
    if token_counts is None:
        token_counts = [1]
    return SimpleNamespace(
        models=SimpleNamespace(
            count_tokens=Mock(
                side_effect=[
                    SimpleNamespace(total_tokens=count)
                    for count in token_counts
                ]
            ),
            generate_content_stream=Mock(
                return_value=_response_stream(responses[0])
            ),
            generate_content=Mock(side_effect=responses[1:]),
        )
    )


def test_run_turn_yields_generated_text_chunks():
    client = _client(_response(
        types.Part(text="Hello"),
        types.Part(text=" there"),
        types.Part(text="!"),
    ))

    result = list(gemini_chat.run_turn(
        "Hi", client, "test-model", [], user_id=4, request_id="request",
    ))

    assert result == [
        ("chunk", "Hello"),
        ("chunk", " there"),
        ("chunk", "!"),
    ]
    client.models.generate_content_stream.assert_called_once()
    request = client.models.generate_content_stream.call_args.kwargs
    assert request["model"] == "test-model"
    assert request["contents"][0].role == "user"
    assert request["contents"][0].parts[0].text == "Hi"


def test_run_turn_skips_empty_and_missing_candidate_content():
    responses = [
        SimpleNamespace(candidates=[SimpleNamespace(content=None)]),
        _response(
            types.Part(text="Hello"),
            types.Part(text=""),
            types.Part(text=None),
            types.Part(text="!"),
        ),
    ]
    client = _client(responses[0])

    def stream():
        yield from responses

    client.models.generate_content_stream.return_value = stream()

    result = list(gemini_chat.run_turn(
        "Hi", client, "test-model", [], user_id=4, request_id="request",
    ))

    assert result == [("chunk", "Hello"), ("chunk", "!")]


def test_run_turn_includes_the_selected_chat_history():
    client = _client(_response(types.Part(text="answer")))
    history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]

    result = list(gemini_chat.run_turn(
        "Current question",
        client,
        "test-model",
        history,
        user_id=4,
        request_id="request",
    ))

    assert result == [("chunk", "answer")]
    contents = client.models.generate_content_stream.call_args.kwargs["contents"]
    assert [item.role for item in contents] == ["user", "model", "user"]
    assert [item.parts[0].text for item in contents] == [
        "Previous question",
        "Previous answer",
        "Current question",
    ]


def test_bounded_contents_rejects_an_oversized_current_message(monkeypatch):
    monkeypatch.setenv(gemini_chat.MAX_INPUT_TOKENS_ENV, "2")
    client = _client(
        _response(types.Part(text="unused")),
        token_counts=[3, 3],
    )

    with pytest.raises(ValueError, match="current message exceeds"):
        gemini_chat.bounded_contents_sync(
            "Too long",
            client,
            "test-model",
            [{"role": "user", "content": "Earlier"}],
        )

    client.models.generate_content_stream.assert_not_called()


def test_bounded_contents_requires_a_token_count():
    client = SimpleNamespace(
        models=SimpleNamespace(
            count_tokens=Mock(return_value=SimpleNamespace(total_tokens=None))
        )
    )

    with pytest.raises(ValueError, match="did not return an input token count"):
        gemini_chat.bounded_contents_sync(
            "Hello", client, "test-model", []
        )
