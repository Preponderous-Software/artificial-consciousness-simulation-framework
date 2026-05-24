"""Tests for reflection triggering, outputs, and inner-voice rendering."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

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
    """'As the patterns evoke…' must NOT become 'I the patterns evoke…'."""
    voice = InnerVoice("Aria")
    result = voice.render("As the patterns on Conognatha splendens' elytra evoke a sense of intricacy")
    assert not result.lower().startswith("i the"), f"Got 'I the …': {result!r}"


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
