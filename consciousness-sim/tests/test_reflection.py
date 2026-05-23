"""Tests for reflection triggering and outputs."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

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
