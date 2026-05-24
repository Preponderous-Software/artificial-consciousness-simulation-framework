"""Tests for thought loop cycle behavior and reflection triggering."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

from core.identity import IdentityDocument
from core.reflection import ReflectionEngine
from core.thought_loop import ThoughtLoop
from llm.provider import MockProvider
from memory.episodic import EpisodicMemory
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory


def test_thought_loop_cycle_records_thought() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            thought_prompt = base / "thought.txt"
            anchor_prompt = base / "anchor.txt"
            reflection_prompt = base / "self_reflection.txt"
            existential_prompt = base / "existential.txt"

            thought_prompt.write_text(
                "You are {name}. {identity_summary} {mood_vector} {retrieved_memories} {short_term_buffer}",
                encoding="utf-8",
            )
            anchor_prompt.write_text(
                "Name: {name}; Values: {values}; Purpose: {purpose}; Self: {self_concept}",
                encoding="utf-8",
            )
            reflection_prompt.write_text("You are {name}. {recent_thoughts}", encoding="utf-8")
            existential_prompt.write_text("You are {name}. {session_duration}", encoding="utf-8")

            provider = MockProvider()
            ident = IdentityDocument(
                name="Test",
                origin_story="origin",
                values=["curiosity"],
                purpose="understand",
                self_concept="I am Test",
                mood={"curiosity": 0.5},
            )
            stm = ShortTermMemory(capacity=20)
            epi = EpisodicMemory(base / "episodic.jsonl")
            ltm = LongTermMemory(base / "memory.db")
            await ltm.initialize()
            reflection = ReflectionEngine(provider, reflection_prompt, existential_prompt, deep_every_n=2)

            loop = ThoughtLoop(
                provider=provider,
                identity=ident,
                short_term=stm,
                episodic=epi,
                long_term=ltm,
                reflection_engine=reflection,
                thought_prompt_path=thought_prompt,
                identity_anchor_path=anchor_prompt,
                reflection_probability=1.0,
                existential_every_n=2,
            )

            result = await loop.run_cycle(thought_count=2)
            assert result.thought
            assert result.reflection is not None
            assert result.existential is not None
            assert any(item.kind == "thought" for item in stm.list())

    asyncio.run(_run())


def test_thought_loop_passes_temperature_and_max_tokens_to_provider() -> None:
    """thought_temperature and thought_max_tokens must propagate to provider.generate (#11 config drift)."""
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            thought_prompt = base / "thought.txt"
            anchor_prompt = base / "anchor.txt"
            reflection_prompt = base / "self_reflection.txt"
            existential_prompt = base / "existential.txt"

            thought_prompt.write_text(
                "You are {name}. {identity_summary} {mood_vector} {retrieved_memories} {short_term_buffer}",
                encoding="utf-8",
            )
            anchor_prompt.write_text(
                "Name: {name}; Values: {values}; Purpose: {purpose}; Self: {self_concept}",
                encoding="utf-8",
            )
            reflection_prompt.write_text("You are {name}. {recent_thoughts}", encoding="utf-8")
            existential_prompt.write_text("You are {name}. {session_duration}", encoding="utf-8")

            provider = MockProvider()
            captured_generate = AsyncMock(return_value="I notice I am thinking about this.")
            provider.generate = captured_generate

            ident = IdentityDocument(
                name="Test",
                origin_story="origin",
                values=["curiosity"],
                purpose="understand",
                self_concept="I am Test",
                mood={"curiosity": 0.5},
            )
            stm = ShortTermMemory(capacity=20)
            epi = EpisodicMemory(base / "episodic.jsonl")
            ltm = LongTermMemory(base / "memory.db")
            await ltm.initialize()
            reflection = ReflectionEngine(provider, reflection_prompt, existential_prompt)

            loop = ThoughtLoop(
                provider=provider,
                identity=ident,
                short_term=stm,
                episodic=epi,
                long_term=ltm,
                reflection_engine=reflection,
                thought_prompt_path=thought_prompt,
                identity_anchor_path=anchor_prompt,
                reflection_probability=0.0,
                thought_temperature=0.42,
                thought_max_tokens=99,
            )

            await loop.run_cycle(thought_count=1)

            assert captured_generate.called, "provider.generate must be called"
            call_kwargs = captured_generate.call_args.kwargs
            assert call_kwargs["temperature"] == 0.42, (
                f"temperature must be 0.42, got {call_kwargs.get('temperature')}"
            )
            assert call_kwargs["max_tokens"] == 99, (
                f"max_tokens must be 99, got {call_kwargs.get('max_tokens')}"
            )

    asyncio.run(_run())


def test_thought_loop_disables_existential_when_zero() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            thought_prompt = base / "thought.txt"
            anchor_prompt = base / "anchor.txt"
            reflection_prompt = base / "self_reflection.txt"
            existential_prompt = base / "existential.txt"

            thought_prompt.write_text(
                "You are {name}. {identity_summary} {mood_vector} {retrieved_memories} {short_term_buffer}",
                encoding="utf-8",
            )
            anchor_prompt.write_text(
                "Name: {name}; Values: {values}; Purpose: {purpose}; Self: {self_concept}",
                encoding="utf-8",
            )
            reflection_prompt.write_text("You are {name}. {recent_thoughts}", encoding="utf-8")
            existential_prompt.write_text("You are {name}. {session_duration}", encoding="utf-8")

            provider = MockProvider()
            ident = IdentityDocument(
                name="Test",
                origin_story="origin",
                values=["curiosity"],
                purpose="understand",
                self_concept="I am Test",
                mood={"curiosity": 0.5},
            )
            stm = ShortTermMemory(capacity=20)
            epi = EpisodicMemory(base / "episodic.jsonl")
            ltm = LongTermMemory(base / "memory.db")
            await ltm.initialize()
            reflection = ReflectionEngine(provider, reflection_prompt, existential_prompt, deep_every_n=2)

            loop = ThoughtLoop(
                provider=provider,
                identity=ident,
                short_term=stm,
                episodic=epi,
                long_term=ltm,
                reflection_engine=reflection,
                thought_prompt_path=thought_prompt,
                identity_anchor_path=anchor_prompt,
                reflection_probability=0.0,
                existential_every_n=0,
            )

            result = await loop.run_cycle(thought_count=100)
            assert result.thought
            assert result.existential is None

    asyncio.run(_run())
