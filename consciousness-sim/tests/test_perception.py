"""Tests for the PerceptionProvider specialist (issue #53).

Covers:
- MockPerception cycles through its fixed corpus
- WikipediaPerception parses a happy-path response
- HTTP timeout / 5xx / malformed responses yield None + a WARNING (no raise)
- Cache rejects recently-seen titles, returns None after 3 cache hits
- build_perception_provider factory + render_perception_block formatter
- ThoughtLoop fetches perception exactly every Nth cycle and skips on None
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from unittest.mock import AsyncMock

import pytest

from llm.perception import (
    MockPerception,
    Perception,
    WikipediaPerception,
    build_perception_provider,
    render_perception_block,
)


# ---------------------------------------------------------------------------
# Helpers — fake httpx module
# ---------------------------------------------------------------------------

def install_fake_httpx(monkeypatch, *, json_payload=None, status_code: int = 200,
                       raise_on_get: Exception | None = None):
    """Install a fake httpx module on sys.modules so WikipediaPerception sees it.

    Mirrors the pattern in tests/test_provider.py for the ollama module.
    """
    captured: dict[str, object] = {}

    class _Response:
        def __init__(self) -> None:
            self.status_code = status_code

        def json(self):
            return json_payload

        def raise_for_status(self):
            if status_code >= 400:
                raise RuntimeError(f"HTTP {status_code}")

    class _FakeClient:
        def __init__(self, timeout=None) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers=None):
            captured.setdefault("calls", []).append((url, headers))
            if raise_on_get is not None:
                raise raise_on_get
            return _Response()

    fake_module = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_module)
    return captured


# ---------------------------------------------------------------------------
# MockPerception
# ---------------------------------------------------------------------------

def test_mock_perception_cycles_through_corpus() -> None:
    mock = MockPerception(corpus=(("A", "alpha"), ("B", "beta")))
    p1 = asyncio.run(mock.fetch())
    p2 = asyncio.run(mock.fetch())
    p3 = asyncio.run(mock.fetch())
    assert (p1.title, p1.content) == ("A", "alpha")
    assert (p2.title, p2.content) == ("B", "beta")
    assert (p3.title, p3.content) == ("A", "alpha")  # wraps
    assert p1.source == "mock"


# ---------------------------------------------------------------------------
# WikipediaPerception
# ---------------------------------------------------------------------------

WIKIPEDIA_HAPPY = {
    "title": "Sodium chlorate",
    "extract": "Sodium chlorate is an inorganic compound with the chemical formula NaClO3.",
    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Sodium_chlorate"}},
}


def test_wikipedia_perception_happy_path(monkeypatch) -> None:
    captured = install_fake_httpx(monkeypatch, json_payload=WIKIPEDIA_HAPPY)
    wp = WikipediaPerception(timeout_seconds=5.0, cache_last_n=3)

    p = asyncio.run(wp.fetch())

    assert p is not None
    assert p.source == "wikipedia"
    assert p.title == "Sodium chlorate"
    assert p.content.startswith("Sodium chlorate is an inorganic compound")
    assert p.url == "https://en.wikipedia.org/wiki/Sodium_chlorate"
    assert captured["timeout"] == 5.0
    # User-Agent identifies us, per Wikipedia API etiquette
    assert "consciousness-sim" in captured["calls"][0][1]["User-Agent"]


def test_wikipedia_perception_returns_none_on_http_error(monkeypatch, caplog) -> None:
    install_fake_httpx(monkeypatch, json_payload={}, status_code=503)
    wp = WikipediaPerception(timeout_seconds=1.0, cache_last_n=0)

    with caplog.at_level(logging.WARNING, logger="llm.perception"):
        result = asyncio.run(wp.fetch())

    assert result is None
    assert any("Wikipedia" in r.message for r in caplog.records)


def test_wikipedia_perception_returns_none_on_timeout(monkeypatch, caplog) -> None:
    install_fake_httpx(monkeypatch, raise_on_get=TimeoutError("timed out"))
    wp = WikipediaPerception(timeout_seconds=1.0, cache_last_n=0)

    with caplog.at_level(logging.WARNING, logger="llm.perception"):
        result = asyncio.run(wp.fetch())

    assert result is None
    assert any("timed out" in r.message.lower() or "Wikipedia" in r.message
               for r in caplog.records)


def test_wikipedia_perception_returns_none_when_extract_missing(monkeypatch) -> None:
    install_fake_httpx(monkeypatch, json_payload={"title": "X", "extract": ""})
    wp = WikipediaPerception(timeout_seconds=1.0)

    assert asyncio.run(wp.fetch()) is None


def test_wikipedia_perception_caches_recent_titles(monkeypatch) -> None:
    """After fetching a title, three retry attempts returning the same title yield None."""
    install_fake_httpx(monkeypatch, json_payload=WIKIPEDIA_HAPPY)
    wp = WikipediaPerception(timeout_seconds=1.0, cache_last_n=3)

    first = asyncio.run(wp.fetch())
    assert first is not None and first.title == "Sodium chlorate"

    # Server keeps returning the same article — provider should give up after 3 attempts
    second = asyncio.run(wp.fetch())
    assert second is None


# ---------------------------------------------------------------------------
# Factory + formatter
# ---------------------------------------------------------------------------

def test_build_perception_provider_factory() -> None:
    assert isinstance(build_perception_provider("wikipedia"), WikipediaPerception)
    assert isinstance(build_perception_provider("Mock"), MockPerception)
    with pytest.raises(ValueError, match="Unsupported perception provider"):
        build_perception_provider("doesnotexist")


def test_render_perception_block_omits_block_when_none() -> None:
    assert render_perception_block(None) == ""


def test_render_perception_block_formats_with_source_and_title() -> None:
    p = Perception(source="wikipedia", title="X", content="Y is a thing.")
    block = render_perception_block(p)
    assert "SOMETHING YOU JUST ENCOUNTERED" in block
    assert "[wikipedia: X]" in block
    assert "Y is a thing." in block


# ---------------------------------------------------------------------------
# ThoughtLoop integration
# ---------------------------------------------------------------------------

def test_thought_loop_fetches_perception_every_n_cycles(monkeypatch) -> None:
    """ThoughtLoop calls perception.fetch() exactly on cycles divisible by N."""
    from core.identity import IdentityDocument
    from core.reflection import ReflectionEngine
    from core.thought_loop import ThoughtLoop
    from llm.provider import MockProvider
    from memory.episodic import EpisodicMemory
    from memory.long_term import LongTermMemory
    from memory.short_term import ShortTermMemory

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        identity = IdentityDocument(
            name="Test", origin_story="o", values=["v"], purpose="p",
            self_concept="I am Test.", personality_traits=[], mood={"curiosity": 0.5},
        )
        short_term = ShortTermMemory(capacity=5)
        episodic = EpisodicMemory(tmpdir / "e.jsonl")
        long_term = LongTermMemory(tmpdir / "lt.db")
        asyncio.run(long_term.initialize())
        provider = MockProvider()
        root = Path(__file__).resolve().parents[1]
        reflection = ReflectionEngine(
            provider=provider,
            self_reflection_prompt=root / "llm" / "prompts" / "self_reflection.txt",
            existential_prompt=root / "llm" / "prompts" / "existential_inquiry.txt",
        )

        perception = MockPerception(corpus=(("X", "content X"),))
        # Wrap fetch so we can count invocations
        original_fetch = perception.fetch
        call_count = {"n": 0}

        async def counting_fetch() -> Perception:
            call_count["n"] += 1
            return await original_fetch()

        perception.fetch = counting_fetch  # type: ignore[method-assign]

        loop = ThoughtLoop(
            provider=provider,
            identity=identity,
            short_term=short_term,
            episodic=episodic,
            long_term=long_term,
            reflection_engine=reflection,
            thought_prompt_path=root / "llm" / "prompts" / "thought_generation.txt",
            identity_anchor_path=root / "llm" / "prompts" / "identity_anchoring.txt",
            reflection_probability=0.0,
            existential_every_n=0,
            perception_provider=perception,
            perception_every_n=3,
        )

        # Run 9 cycles: should fetch on 3, 6, 9 — three total
        for n in range(1, 10):
            result = asyncio.run(loop.run_cycle(n))
            if n % 3 == 0:
                assert result.perception is not None, f"cycle {n} expected perception"
            else:
                assert result.perception is None, f"cycle {n} expected no perception"

        assert call_count["n"] == 3


def test_thought_loop_skips_perception_when_fetch_returns_none(monkeypatch) -> None:
    """A provider that returns None must not break the cycle — perception is just absent."""
    from core.identity import IdentityDocument
    from core.reflection import ReflectionEngine
    from core.thought_loop import ThoughtLoop
    from llm.perception import PerceptionProvider
    from llm.provider import MockProvider
    from memory.episodic import EpisodicMemory
    from memory.long_term import LongTermMemory
    from memory.short_term import ShortTermMemory

    import tempfile
    from pathlib import Path

    class _NullProvider(PerceptionProvider):
        async def fetch(self):
            return None

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        identity = IdentityDocument(
            name="Test", origin_story="o", values=["v"], purpose="p",
            self_concept="I am Test.", personality_traits=[], mood={"curiosity": 0.5},
        )
        short_term = ShortTermMemory(capacity=5)
        episodic = EpisodicMemory(tmpdir / "e.jsonl")
        long_term = LongTermMemory(tmpdir / "lt.db")
        asyncio.run(long_term.initialize())
        provider = MockProvider()
        root = Path(__file__).resolve().parents[1]
        reflection = ReflectionEngine(
            provider=provider,
            self_reflection_prompt=root / "llm" / "prompts" / "self_reflection.txt",
            existential_prompt=root / "llm" / "prompts" / "existential_inquiry.txt",
        )

        loop = ThoughtLoop(
            provider=provider,
            identity=identity,
            short_term=short_term,
            episodic=episodic,
            long_term=long_term,
            reflection_engine=reflection,
            thought_prompt_path=root / "llm" / "prompts" / "thought_generation.txt",
            identity_anchor_path=root / "llm" / "prompts" / "identity_anchoring.txt",
            reflection_probability=0.0,
            existential_every_n=0,
            perception_provider=_NullProvider(),
            perception_every_n=1,   # fetch every cycle
        )

        result = asyncio.run(loop.run_cycle(1))
        assert result.perception is None
        assert result.thought  # thought still generated cleanly
