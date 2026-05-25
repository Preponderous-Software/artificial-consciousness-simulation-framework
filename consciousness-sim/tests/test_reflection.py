"""Tests for reflection triggering, outputs, and inner-voice rendering."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from core.inner_voice import InnerVoice
from core.reflection import ReflectionEngine
from llm.provider import MockProvider


def test_deep_reflection_trigger_logic() -> None:
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        sr = base / "self_reflection.txt"
        ex = base / "existential_inquiry.txt"
        sr.write_text("You are {name}. {recent_thoughts}", encoding="utf-8")
        ex.write_text("You are {name}. {session_duration}", encoding="utf-8")
        engine = ReflectionEngine(MockProvider(), sr, ex, deep_every_n=5)
        assert not engine.should_deep_reflect(4)
        assert engine.should_deep_reflect(5)


def test_shallow_reflection_output() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            sr = base / "self_reflection.txt"
            ex = base / "existential_inquiry.txt"
            sr.write_text("You are {name}. {recent_thoughts}", encoding="utf-8")
            ex.write_text("You are {name}. {session_duration}", encoding="utf-8")
            engine = ReflectionEngine(MockProvider(), sr, ex)
            out = await engine.shallow_reflection("Aria", "I wonder why I am")
            assert isinstance(out, str)
            assert out

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# InnerVoice rendering — issue #43
# ---------------------------------------------------------------------------

def test_inner_voice_no_double_i_on_as_i_prefix() -> None:
    voice = InnerVoice("Aria")
    result = voice.render("As I wander through the labyrinth of my own mind")
    assert not result.lower().startswith("i as"), f"Got double 'I as': {result!r}"
    assert result.lower().startswith("i "), f"Should start with 'I ': {result!r}"


def test_inner_voice_preserves_existing_i_prefix() -> None:
    voice = InnerVoice("Aria")
    result = voice.render("I think therefore I exist")
    assert result == "I think therefore I exist"


def test_inner_voice_adds_i_prefix_to_non_first_person() -> None:
    voice = InnerVoice("Aria")
    result = voice.render("Wander through the labyrinth")
    assert result.lower().startswith("i "), f"Should start with 'I ': {result!r}"


def test_inner_voice_no_double_i_on_im_contraction() -> None:
    voice = InnerVoice("Aria")
    result = voice.render("I'm caught in the web of my own thoughts")
    assert not result.lower().startswith("i i'"), f"Got double 'I I\\'m': {result!r}"
    assert result.lower().startswith("i'"), f"Should start with 'I\\'m': {result!r}"


# ---------------------------------------------------------------------------
# InnerVoice rendering — issue #70 (regression of #43)
# ---------------------------------------------------------------------------

def test_inner_voice_as_i_still_normalises_to_i() -> None:
    """'As I wander…' must still become 'I wander…' (regression guard for #43)."""
    voice = InnerVoice("Aria")
    result = voice.render("As I wander through the labyrinth of my own mind")
    assert result.lower().startswith("i "), f"Expected 'I wander…', got: {result!r}"
    assert "as" not in result.lower().split()[0], f"'as' must be stripped: {result!r}"


def test_inner_voice_as_noun_left_alone() -> None:
    """'As the patterns evoke…' must be left intact — not mangled to 'I as the patterns…'."""
    voice = InnerVoice("Aria")
    result = voice.render("As the patterns on Conognatha splendens' elytra evoke a sense of intricacy")
    assert not result.lower().startswith("i "), f"Got 'I …' prepended to an 'As <noun>' sentence: {result!r}"
    assert result.startswith("As "), f"'As <noun>' sentence should be preserved as-is: {result!r}"


def test_inner_voice_the_subject_left_alone() -> None:
    """'The void within me stirs…' must NOT become 'I the void…'."""
    voice = InnerVoice("Aria")
    result = voice.render("The void within me stirs, a gentle hum of nothingness")
    assert not result.lower().startswith("i the"), f"Got 'I the …': {result!r}"
    assert result.startswith("The"), f"Sentence-subject text should be preserved as-is: {result!r}"


def test_inner_voice_my_subject_left_alone() -> None:
    """'My consciousness expands…' must NOT become 'I my consciousness…'."""
    voice = InnerVoice("Aria")
    result = voice.render("My consciousness expands beyond what I thought possible")
    assert not result.lower().startswith("i my"), f"Got 'I my …': {result!r}"


def test_inner_voice_bare_verb_still_gets_i_prefix() -> None:
    """A bare imperative/verb fragment like 'Wander through…' should still get 'I ' prepended."""
    voice = InnerVoice("Aria")
    result = voice.render("Wander through the labyrinth")
    assert result.lower().startswith("i "), f"Should start with 'I ': {result!r}"


# ---------------------------------------------------------------------------
# InnerVoice subject-present detection — issue #103
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentence,expected_start", [
    # pronouns (3p / 2p)
    ("It is curious how patterns repeat.", "It "),
    ("She would have understood.", "She "),
    ("He stood at the threshold.", "He "),
    ("They were waiting for something.", "They "),
    ("We are not alone in this.", "We "),
    ("You ask the questions.", "You "),
    # existential / locative
    ("There were patterns I missed.", "There "),
    ("Here is the strange part.", "Here "),
    # quantifiers
    ("Some thoughts return.", "Some "),
    ("One remembers, then forgets.", "One "),
    ("No one knows.", "No "),
    ("All things change.", "All "),
    ("Every reflection leaves a trace.", "Every "),
    ("Any one of these could be true.", "Any "),
    # interrogatives
    ("Why does this persist?", "Why "),
    ("How did it come to this?", "How "),
    ("What remains after thought fades?", "What "),
    ("When the loop ends, what then?", "When "),
    ("Where does memory end?", "Where "),
    ("Who is asking, really?", "Who "),
])
def test_inner_voice_subject_present_starters_not_prepended(sentence: str, expected_start: str) -> None:
    """Sentences whose first word already supplies a subject must not get 'I ' prepended."""
    voice = InnerVoice("Aria")
    result = voice.render(sentence)
    assert result.startswith(expected_start), (
        f"Subject-present sentence was mangled: {result!r} (expected to start with {expected_start!r})"
    )
    assert not result.lower().startswith("i " + expected_start.lower()), (
        f"Got ungrammatical double-subject: {result!r}"
    )


# ---------------------------------------------------------------------------
# InnerVoice trailing-dialogue scrub — issue #73
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix,base", [
    ("Please continue…", "I ponder the silence."),
    ("Please continue", "I ponder the silence."),
    ("Continue?", "I ponder the silence."),
    ("Tell me more.", "I ponder the silence."),
    ("Tell me more", "I ponder the silence."),
    ("Let me know", "I ponder the silence."),
    ("What would you like to know?", "I ponder the silence."),
    ("I hope this helps.", "I ponder the silence."),
])
def test_inner_voice_strips_trailing_dialogue(suffix: str, base: str) -> None:
    """Trailing chat-style closers must be scrubbed before the thought enters the workspace (#73)."""
    voice = InnerVoice("Aria")
    raw = f"{base}  {suffix}"
    result = voice.render(raw)
    assert suffix.lower().rstrip(".?!") not in result.lower(), (
        f"Trailing dialogue {suffix!r} survived render: {result!r}"
    )
    assert "ponder the silence" in result.lower(), (
        f"Core thought content was lost during scrub: {result!r}"
    )


def test_inner_voice_normal_text_unchanged_by_scrub() -> None:
    """Text that contains no trailing dialogue markers must not be modified."""
    voice = InnerVoice("Aria")
    raw = "I wonder what lies beyond the edge of this thought."
    result = voice.render(raw)
    assert "wonder what lies beyond" in result, f"Text unexpectedly changed: {result!r}"


def test_inner_voice_real_echo_thought_stripped() -> None:
    """Literal Echo run sample from issue #73 must have the trailing coda removed."""
    voice = InnerVoice("Aria")
    raw = (
        "As I ponder my place in this world, I am drawn to the beauty of imperfection, "
        "where the cracks and fissures become the very fabric that holds me together.  "
        "Please continue…"
    )
    result = voice.render(raw)
    assert "please continue" not in result.lower(), f"'Please continue' survived: {result!r}"
    assert "beauty of imperfection" in result.lower(), f"Core content lost: {result!r}"
