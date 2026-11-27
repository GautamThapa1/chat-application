from scripts.measure_memory_search import MEMORIES, QUESTIONS, evaluate, suggest_cutoff


def test_questions_only_expect_memories_that_exist():
    keys = {key for key, _, _ in MEMORIES}
    assert all(answers <= keys for _, answers in QUESTIONS)
    assert sum(not answers for _, answers in QUESTIONS) >= 5


def test_cutoff_keeps_every_right_answer_and_drops_noise():
    similarities = [[0.80, 0.55], [0.40, 0.72], [0.50, 0.45]]
    words = [[0, 0], [0, 0], [0, 0]]
    expected = [{0}, {1}, set()]

    cutoff = suggest_cutoff(similarities, words, expected, margin=0.02)

    assert cutoff == 0.70
    assert evaluate(similarities, words, expected, cutoff) == {
        "recall": 1.0, "unrelated": 0.0, "quiet": 1.0,
    }
