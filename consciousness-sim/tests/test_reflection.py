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
