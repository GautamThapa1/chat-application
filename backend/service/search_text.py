"""Word matching for memory and note search, and the meaning-match cutoff."""

import re

# Lowest similarity a result needs when it shares no word with the question.
# None turns the cutoff off. Pick a value with scripts/measure_memory_search.py;
# a guessed one can hide correct memories.
MIN_SIMILARITY: float | None = None

_STOPWORDS = frozenset(
    "a about all am an and any are as at be been but by can could did do does for "
    "from had has have how i if in is it its me my no not of on or our should so "
    "tell that the their them there these they this those to was we were what when "
    "where which who why will with would you your".split()
)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _forms(word: str) -> set[str]:
    """A word and its likely singular, so "tomato" finds "tomatoes"."""
    forms = {word}
    if len(word) > 3 and word.endswith("s"):
        forms.add(word[:-1])
        if word.endswith("es"):
            forms.add(word[:-2])
    return forms


def keyword_score(query: str, title: str, content: str) -> float:
    """1.0 if the whole query appears, else the share of its words that do."""
    phrase = query.strip().casefold()
    if not phrase:
        return 0.0
    text = f"{title}\n{content}"
    if phrase in title.casefold() or phrase in content.casefold():
        return 1.0
    terms = [w for w in dict.fromkeys(_words(query)) if w not in _STOPWORDS]
    if not terms:
        return 0.0
    present = set().union(*(_forms(word) for word in _words(text)))
    return sum(bool(_forms(term) & present) for term in terms) / len(terms)


def keyword_scores(rows, query: str) -> dict[int, float]:
    """Row id to score, for the rows that match at all."""
    scores = {
        row["id"]: keyword_score(query, row["title"], row["content"]) for row in rows
    }
    return {row_id: score for row_id, score in scores.items() if score > 0}


def rank_by_keywords(rows, query: str) -> list[dict]:
    """Matching rows, best first; equal scores keep their id order."""
    scores = keyword_scores(rows, query)
    matches = [dict(row) for row in rows if row["id"] in scores]
    return sorted(matches, key=lambda row: (-scores[row["id"]], row["id"]))


def is_relevant(similarity: float | None, score: float) -> bool:
    """Whether a search candidate should be returned at all."""
    if score > 0:
        return True
    if similarity is None:
        return False
    return MIN_SIMILARITY is None or similarity >= MIN_SIMILARITY
