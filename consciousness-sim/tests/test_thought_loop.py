"""Tests for thought loop cycle behavior and reflection triggering."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from core.identity import IdentityDocument
from core.metacognition import MetacognitiveMonitor
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


def _make_loop(base: Path, provider: MockProvider, reflection_probability: float) -> ThoughtLoop:
    thought_prompt = base / "thought.txt"
    anchor_prompt = base / "anchor.txt"
    reflection_prompt = base / "reflection.txt"
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
    ident = IdentityDocument(
        name="Test", origin_story="o", values=["curiosity"],
        purpose="understand", self_concept="I am Test", mood={"curiosity": 0.5},
    )
    return ThoughtLoop(
        provider=provider,
        identity=ident,
        short_term=ShortTermMemory(capacity=20),
        episodic=EpisodicMemory(base / "episodic.jsonl"),
        long_term=None,  # type: ignore[arg-type]  # patched below
        reflection_engine=ReflectionEngine(provider, reflection_prompt, existential_prompt),
        thought_prompt_path=thought_prompt,
        identity_anchor_path=anchor_prompt,
        reflection_probability=reflection_probability,
        existential_every_n=0,
    )


def test_reflection_boost_fires_when_thought_is_noisy() -> None:
    """label='noise' boosts effective reflection prob: base 0.15 + 0.30 = 0.45 > random 0.29."""
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            provider = MockProvider()
            from unittest.mock import MagicMock
            ltm = MagicMock()
            ltm.similarity_search = AsyncMock(return_value=[])

            loop = _make_loop(base, provider, reflection_probability=0.15)
            loop.long_term = ltm

            # Patch the scorer to return 'noise' regardless of content so the test
            # is not coupled to MockProvider's vocabulary.
            with patch.object(MetacognitiveMonitor, "score", return_value="noise"), \
                 patch("core.thought_loop.random.random", return_value=0.29):
                result = await loop.run_cycle(thought_count=1)

            assert result.reflection is not None, (
                "reflection should fire when effective_prob (0.45) > random value (0.29)"
            )

    asyncio.run(_run())


def test_reflection_does_not_fire_without_boost_at_same_random_value() -> None:
    """label='high' gives no boost: effective=0.15 < random 0.29 → reflection does not fire."""
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            provider = MockProvider()
            from unittest.mock import MagicMock
            ltm = MagicMock()
            ltm.similarity_search = AsyncMock(return_value=[])

            loop = _make_loop(base, provider, reflection_probability=0.15)
            loop.long_term = ltm
            with patch.object(MetacognitiveMonitor, "score", return_value="high"), \
                 patch("core.thought_loop.random.random", return_value=0.29):
                result = await loop.run_cycle(thought_count=1)

            assert result.reflection is None, (
                "reflection should not fire when effective_prob (0.15) < random value (0.29)"
            )

    asyncio.run(_run())


def test_effective_reflection_prob_clamped_to_one() -> None:
    """reflection_probability=1.0 + noise boost is clamped to 1.0 — always fires, no overflow."""
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            provider = MockProvider()
            from unittest.mock import MagicMock
            ltm = MagicMock()
            ltm.similarity_search = AsyncMock(return_value=[])

            loop = _make_loop(base, provider, reflection_probability=1.0)
            loop.long_term = ltm
            with patch.object(MetacognitiveMonitor, "score", return_value="noise"), \
                 patch("core.thought_loop.random.random", return_value=0.99):
                result = await loop.run_cycle(thought_count=1)

            assert result.reflection is not None, (
                "reflection should always fire when reflection_probability=1.0"
            )

    asyncio.run(_run())


def test_perf_log_emitted_at_correct_interval() -> None:
    """perf_log_every_n=5 should emit an INFO log on cycles 5, 10, … but not 1, 2, 3, 4."""
    import logging

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            provider = MockProvider()
            from unittest.mock import MagicMock
            ltm = MagicMock()
            ltm.similarity_search = AsyncMock(return_value=[])

            loop = _make_loop(base, provider, reflection_probability=0.0)
            loop.long_term = ltm
            loop.perf_log_every_n = 5

            logged_cycles: list[int] = []

            class _Capture(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    if "perf —" in record.getMessage():
                        # extract cycle number from "Cycle N perf —"
                        parts = record.getMessage().split()
                        logged_cycles.append(int(parts[1]))

            handler = _Capture()
            log = logging.getLogger()
            log.addHandler(handler)
            old_level = log.level
            log.setLevel(logging.INFO)
            try:
                for n in range(1, 12):
                    await loop.run_cycle(thought_count=n)
            finally:
                log.removeHandler(handler)
                log.setLevel(old_level)

            assert logged_cycles == [5, 10], (
                f"expected perf logs at cycles [5, 10], got {logged_cycles}"
            )

    asyncio.run(_run())
