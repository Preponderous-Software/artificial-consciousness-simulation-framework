"""Tests for MetacognitiveMonitor heuristic reliability scoring (HOT-2)."""

from __future__ import annotations

from core.metacognition import MetacognitiveMonitor, _content_words
from memory.short_term import MemoryItem


def _item(kind: str, content: str) -> MemoryItem:
    return MemoryItem(kind=kind, content=content, timestamp="2026-01-01T00:00:00+00:00")


monitor = MetacognitiveMonitor()


# --- _content_words -----------------------------------------------------------

def test_content_words_filters_short_tokens() -> None:
    assert "the" not in _content_words("the cat sat")
    assert "cat" not in _content_words("the cat sat")  # 3 chars


def test_content_words_filters_stopwords() -> None:
    assert "about" not in _content_words("about everything")
    assert "from" not in _content_words("from somewhere")


def test_content_words_returns_long_non_stopwords() -> None:
    words = _content_words("curiosity drives exploration forward")
    assert "curiosity" in words
    assert "drives" in words
    assert "exploration" in words
    assert "forward" in words


# --- score: empty buffer ------------------------------------------------------

def test_score_empty_buffer_returns_high() -> None:
    thought = "Curiosity reveals hidden patterns beneath apparent chaos drawing awareness toward deeper understanding"
    assert monitor.score(thought, []) == "high"


def test_score_buffer_with_no_thought_items_returns_high() -> None:
    items = [_item("reflection", "some reflection about existence")]
    thought = "Curiosity reveals hidden patterns beneath apparent chaos drawing awareness toward deeper understanding"
    assert monitor.score(thought, items) == "high"


# --- score: novel thought (low overlap) ---------------------------------------

def test_score_novel_thought_returns_high() -> None:
    buffer = [_item("thought", "I wonder about light and shadow dancing across the wall")]
    novel = "Mathematics reveals structure beneath apparent chaos in quantum physics"
    assert monitor.score(novel, buffer) == "high"


# --- score: moderate overlap --------------------------------------------------

def test_score_moderate_overlap_returns_uncertain() -> None:
    buffer = [_item("thought", "patterns of light reveal hidden structure beneath the surface")]
    # shares: patterns, light, reveal, hidden, structure — ~50% of new thought words
    similar = "patterns and light always reveal some hidden structure within nature"
    result = monitor.score(similar, buffer)
    assert result in ("uncertain", "noise")  # overlap is in the uncertain-to-noise band


# --- score: high overlap (attractor repetition) -------------------------------

def test_score_repetitive_thought_returns_noise() -> None:
    buffer = [
        _item("thought", "threads of consciousness weave tapestry patterns through existence"),
        _item("thought", "tapestry threads weave patterns through consciousness existence always"),
    ]
    # Nearly identical vocabulary to what's in the buffer
    repetitive = "consciousness threads weave tapestry patterns through existence always deeply"
    assert monitor.score(repetitive, buffer) == "noise"


# --- score: short thought -----------------------------------------------------

def test_score_short_thought_returns_uncertain() -> None:
    # Fewer than 6 content words
    assert monitor.score("I think therefore exist.", []) == "uncertain"


def test_score_very_short_thought_returns_uncertain() -> None:
    assert monitor.score("Yes.", []) == "uncertain"


# --- score: non-thought buffer items don't count ------------------------------

def test_score_ignores_non_thought_buffer_items() -> None:
    # Buffer has lots of words in reflection/perception items, but no "thought" items.
    # Those words should NOT count as recent — thought should score "high".
    buffer = [
        _item("reflection", "curiosity drives exploration patterns structure awareness perception"),
        _item("perception", "tapestry threads weave patterns consciousness existence"),
    ]
    thought = "curiosity drives exploration patterns structure awareness perception deeply"
    # All words match buffer content — but buffer has no "thought" items, so high.
    assert monitor.score(thought, buffer) == "high"


# --- importance ---------------------------------------------------------------

def test_importance_values() -> None:
    assert monitor.importance("high") == 1.0
    assert monitor.importance("uncertain") == 0.75
    assert monitor.importance("noise") == 0.5


def test_importance_unknown_label_defaults_to_one() -> None:
    assert monitor.importance("unknown") == 1.0


# --- reflection_boost ---------------------------------------------------------

def test_reflection_boost_values() -> None:
    assert monitor.reflection_boost("high") == 0.0
    assert monitor.reflection_boost("uncertain") == 0.15
    assert monitor.reflection_boost("noise") == 0.30


def test_reflection_boost_unknown_label_defaults_to_zero() -> None:
    assert monitor.reflection_boost("unknown") == 0.0


# --- ordering invariants ------------------------------------------------------

def test_importance_ordering() -> None:
    assert monitor.importance("noise") < monitor.importance("uncertain") < monitor.importance("high")


def test_reflection_boost_ordering() -> None:
    assert monitor.reflection_boost("high") < monitor.reflection_boost("uncertain") < monitor.reflection_boost("noise")
