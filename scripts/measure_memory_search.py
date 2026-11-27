"""Measure memory search on made-up data, and suggest a relevance cutoff.

    uv run python scripts/measure_memory_search.py          # word matching only, no API calls
    uv run python scripts/measure_memory_search.py --live   # 50 Gemini embedding calls

--live needs GEMINI_API_KEY (read from .env). No stored user data is used.
Put the suggested value in MIN_SIMILARITY in backend/service/search_text.py.
Add questions that went wrong for you: the cutoff is only as good as this set.
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.embeddings import (  # noqa: E402
    DOCUMENT_TASK_TYPE,
    QUERY_TASK_TYPE,
    embed_text,
    embedding_model,
    note_embedding_text,
)
from backend.service.search_text import keyword_score  # noqa: E402

MEMORIES = [
    ("backend", "Backend", "My backend uses FastAPI"),
    ("database", "Database", "The project database is SQLite encrypted with SQLCipher"),
    ("frontend", "Frontend", "I build the user interface with React and Vite"),
    ("editor", "Editor", "I write code in Neovim"),
    ("language", "Language", "Python is my main programming language"),
    ("style", "Answer style", "I prefer short answers with code examples"),
    ("deploy", "Hosting", "The app is deployed on Render and Vercel"),
    ("goal", "Career goal", "I want to become an AI engineer"),
    ("learning", "Learning", "I am studying LangGraph and agent design"),
    ("timezone", "Location", "I live in Kathmandu"),
    ("schedule", "Work hours", "I usually work on side projects late in the evening"),
    ("food", "Food", "I do not eat meat"),
    ("drink", "Drink", "I like black coffee in the morning"),
    ("sport", "Sport", "I play football on Saturdays"),
    ("music", "Music", "I listen to lo-fi music while programming"),
    ("garden", "Garden", "I grow tomatoes on my balcony"),
    ("book", "Reading", "I am reading a book about distributed systems"),
    ("testing", "Testing", "I write failing tests before changing code"),
    ("pet", "Pet", "My dog is called Miso"),
    ("travel", "Travel", "I plan to trek to Annapurna base camp"),
]

QUESTIONS = [
    ("backend framework", {"backend"}),
    ("which web server library do I use", {"backend"}),
    ("where is the data stored", {"database"}),
    ("what database does my project use", {"database"}),
    ("UI technology", {"frontend"}),
    ("favourite text editor", {"editor"}),
    ("what programming language should the example be in", {"language", "style"}),
    ("how should you format your replies", {"style"}),
    ("where is the app hosted", {"deploy"}),
    ("what job am I aiming for", {"goal"}),
    ("what am I currently learning", {"learning", "book"}),
    ("which city am I in", {"timezone"}),
    ("when do I usually code", {"schedule"}),
    ("suggest a dinner recipe", {"food"}),
    ("what do I drink when I wake up", {"drink"}),
    ("plans for the weekend", {"sport"}),
    ("recommend something to listen to", {"music"}),
    ("tips for my plants", {"garden"}),
    ("what is my dog's name", {"pet"}),
    ("upcoming hiking trip", {"travel"}),
    ("how do I like to fix bugs", {"testing"}),
    ("tomato care", {"garden"}),
    # Nothing saved answers these.
    ("what is the capital of France", set()),
    ("convert 100 dollars to euros", set()),
    ("who is my manager", set()),
    ("what car do I drive", set()),
    ("explain how vaccines work", set()),
    ("my favourite movie", set()),
    ("what is 17 times 23", set()),
    ("do I have any siblings", set()),
]


def evaluate(similarities, word_scores, expected, cutoff):
    """What a search returns at `cutoff`, given each question's scores per memory."""
    found = wanted = unrelated = quiet = silent_questions = 0
    for sims, words, answers in zip(similarities, word_scores, expected):
        returned = {
            index
            for index, (similarity, word) in enumerate(zip(sims, words))
            if word > 0
            or (similarity is not None and (cutoff is None or similarity >= cutoff))
        }
        found += len(returned & answers)
        wanted += len(answers)
        unrelated += len(returned - answers)
        if not answers:
            silent_questions += 1
            quiet += not returned
    return {
        "recall": found / wanted if wanted else 1.0,
        "unrelated": unrelated / len(expected),
        "quiet": quiet / silent_questions if silent_questions else 1.0,
    }


def suggest_cutoff(similarities, word_scores, expected, margin=0.02):
    """The highest cutoff that loses no right answer, less a safety margin."""
    baseline = evaluate(similarities, word_scores, expected, None)["recall"]
    lossless = [
        step / 100
        for step in range(0, 101)
        if evaluate(similarities, word_scores, expected, step / 100)["recall"] >= baseline
    ]
    return round(max(lossless) - margin, 2) if lossless else None


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.hypot(*a) * math.hypot(*b))


def _report(label, result):
    print(
        f"{label:<26} right answers found {result['recall']:>5.0%} | "
        f"unrelated per question {result['unrelated']:>4.1f} | "
        f"silent when nothing fits {result['quiet']:>4.0%}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="call Gemini for embeddings")
    live = parser.parse_args().live

    keys = [key for key, _, _ in MEMORIES]
    expected = [{keys.index(key) for key in answers} for _, answers in QUESTIONS]
    phrase = [
        [float(q.casefold() in f"{title}\n{content}".casefold()) for _, title, content in MEMORIES]
        for q, _ in QUESTIONS
    ]
    words = [
        [keyword_score(q, title, content) for _, title, content in MEMORIES]
        for q, _ in QUESTIONS
    ]
    none = [[None] * len(MEMORIES)] * len(QUESTIONS)

    print(f"{len(MEMORIES)} memories, {len(QUESTIONS)} questions\n")
    _report("whole phrase (before)", evaluate(none, phrase, expected, None))
    _report("word matching (now)", evaluate(none, words, expected, None))
    if not live:
        print(f"\n--live would make {len(MEMORIES) + len(QUESTIONS)} embedding calls.")
        return

    from dotenv import load_dotenv
    from google import genai

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY to use --live.")
    model = embedding_model()
    with genai.Client(api_key=os.environ["GEMINI_API_KEY"]) as client:
        documents = [
            embed_text(client, note_embedding_text(title, content),
                       task_type=DOCUMENT_TASK_TYPE, model=model)
            for _, title, content in MEMORIES
        ]
        queries = [
            embed_text(client, q, task_type=QUERY_TASK_TYPE, model=model)
            for q, _ in QUESTIONS
        ]
    sims = [[_cosine(query, document) for document in documents] for query in queries]

    print(f"\nmeaning + words, model {model}")
    _report("no cutoff", evaluate(sims, words, expected, None))
    suggestion = suggest_cutoff(sims, words, expected)
    for cutoff in sorted({0.5, 0.55, 0.6, 0.65, 0.7, 0.75, suggestion}):
        _report(f"cutoff {cutoff:.2f}", evaluate(sims, words, expected, cutoff))
    print(f"\nSuggested MIN_SIMILARITY = {suggestion:.2f}")


if __name__ == "__main__":
    main()
