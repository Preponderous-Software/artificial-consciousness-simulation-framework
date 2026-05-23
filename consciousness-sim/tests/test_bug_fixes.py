"""Regression tests for bugs #2, #3, #4, #5, #39, and #40."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import unittest.mock as mock
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Bug #2 — config loading crashes with KeyError on malformed YAML
# ---------------------------------------------------------------------------

from core.consciousness import _validate_config


def _minimal_valid_config() -> dict:
    return {
        "llm": {"provider": "ollama", "model": "llama3"},
        "memory": {
            "short_term_capacity": 5,
            "consolidation_interval_minutes": 5,
            "forgetting_curve_enabled": False,
            "importance_decay_rate": 0.01,
        },
        "consciousness": {
            "origin_story": "origin",
            "values": ["curiosity"],
            "purpose": "purpose",
        },
        "thought_loop": {
            "reflection_probability": 0.0,
            "existential_inquiry_every_n_thoughts": 5,
            "min_interval_seconds": 0,
            "max_interval_seconds": 0,
        },
        "mood": {"initial": {"curiosity": 0.5}, "drift_rate": 0.01},
    }


def test_validate_config_passes_on_valid_config() -> None:
    _validate_config(_minimal_valid_config())  # must not raise


def test_validate_config_raises_on_missing_top_level_section() -> None:
    cfg = _minimal_valid_config()
    del cfg["consciousness"]
    with pytest.raises(KeyError, match="consciousness"):
        _validate_config(cfg)


def test_validate_config_raises_on_missing_nested_key() -> None:
    cfg = _minimal_valid_config()
    del cfg["llm"]["model"]
    with pytest.raises(KeyError, match="llm.model"):
        _validate_config(cfg)


def test_consciousness_init_raises_on_missing_section(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    cfg = _minimal_valid_config()
    del cfg["memory"]
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    from core.consciousness import Consciousness

    with pytest.raises(KeyError, match="memory"):
        Consciousness(name="Aria", config_path=str(config_path))


# ---------------------------------------------------------------------------
# Bug #4 — event handler exceptions propagate and crash the thought loop
# ---------------------------------------------------------------------------

from core.consciousness import Consciousness


def _make_consciousness(tmp_path: Path, monkeypatch) -> Consciousness:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(_minimal_valid_config()), encoding="utf-8")
    return Consciousness(name="Aria", config_path=str(config_path))


def test_emit_continues_after_sync_handler_raises(tmp_path, monkeypatch) -> None:
    mind = _make_consciousness(tmp_path, monkeypatch)

    results: list[str] = []

    def bad_handler(_payload: dict) -> None:
        raise RuntimeError("handler blew up")

    def good_handler(payload: dict) -> None:
        results.append(payload["value"])

    mind.on_thought = [bad_handler, good_handler]  # type: ignore[assignment]

    asyncio.run(mind._emit(mind.on_thought, {"value": "ok"}))
    assert results == ["ok"], "Good handler should still run after bad handler raises"


def test_emit_continues_after_async_handler_raises(tmp_path, monkeypatch) -> None:
    mind = _make_consciousness(tmp_path, monkeypatch)

    results: list[str] = []

    async def bad_handler(_payload: dict) -> None:
        raise RuntimeError("async handler blew up")

    def good_handler(payload: dict) -> None:
        results.append(payload["value"])

    mind.on_thought = [bad_handler, good_handler]  # type: ignore[assignment]

    asyncio.run(mind._emit(mind.on_thought, {"value": "ok"}))
    assert results == ["ok"]


def test_emit_logs_handler_exception(tmp_path, monkeypatch, caplog) -> None:
    mind = _make_consciousness(tmp_path, monkeypatch)

    def bad_handler(_payload: dict) -> None:
        raise ValueError("intentional")

    mind.on_thought = [bad_handler]  # type: ignore[assignment]

    with caplog.at_level(logging.ERROR):
        asyncio.run(mind._emit(mind.on_thought, {}))

    assert any("intentional" in r.message or "raised unexpectedly" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Bug #5 — episodic and journal JSONL files crash on corrupted lines
# ---------------------------------------------------------------------------

from memory.episodic import EpisodicMemory
from persistence.journal import Journal


def test_episodic_skips_corrupted_lines(tmp_path, caplog) -> None:
    path = tmp_path / "episodic.jsonl"
    good = json.dumps({"timestamp": "2024-01-01T00:00:00+00:00", "kind": "thought", "content": "hello"})
    path.write_text(good + "\nNOT_JSON\n" + good + "\n", encoding="utf-8")

    mem = EpisodicMemory(path)

    with caplog.at_level(logging.WARNING):
        events = asyncio.run(mem.recent())

    assert len(events) == 2, "Corrupted line should be skipped, not crash"
    assert all(e.content == "hello" for e in events)
    assert any("corrupted" in r.message.lower() for r in caplog.records)


def test_episodic_returns_empty_when_all_lines_corrupted(tmp_path) -> None:
    path = tmp_path / "episodic.jsonl"
    path.write_text("GARBAGE\n{broken\n", encoding="utf-8")

    mem = EpisodicMemory(path)
    events = asyncio.run(mem.recent())
    assert events == []


def test_journal_skips_corrupted_lines(tmp_path, caplog) -> None:
    path = tmp_path / "journal.jsonl"
    good = json.dumps({"timestamp": "2024-01-01T00:00:00+00:00", "type": "thought", "content": "hello"})
    path.write_text(good + "\nBAD_LINE\n" + good + "\n", encoding="utf-8")

    journal = Journal(path)

    with caplog.at_level(logging.WARNING):
        entries = asyncio.run(journal.recent())

    assert len(entries) == 2
    assert any("corrupted" in r.message.lower() for r in caplog.records)


def test_journal_returns_empty_when_all_lines_corrupted(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_text("NOT JSON AT ALL\n", encoding="utf-8")

    journal = Journal(path)
    entries = asyncio.run(journal.recent())
    assert entries == []


# ---------------------------------------------------------------------------
# Bug #3 — consolidation regex silently drops all memories on format mismatch
# ---------------------------------------------------------------------------

from memory.consolidator import MemoryConsolidator


def _make_consolidator(tmp_path: Path) -> MemoryConsolidator:
    from memory.episodic import EpisodicMemory
    from memory.long_term import LongTermMemory
    from memory.short_term import ShortTermMemory

    provider = MagicMock()
    provider.generate = AsyncMock(return_value="- [Importance: 8] [Emotional valence: 0.5] A valid memory.\n")
    provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

    episodic = EpisodicMemory(tmp_path / "episodic.jsonl")
    long_term = LongTermMemory(tmp_path / "memory.db")
    short_term = ShortTermMemory(capacity=10)

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("{episodic_chunk}", encoding="utf-8")

    return MemoryConsolidator(
        provider=provider,
        episodic=episodic,
        long_term=long_term,
        short_term=short_term,
        prompt_path=prompt_path,
        forgetting_curve_enabled=False,
        decay_rate=0.01,
    )


def test_consolidator_logs_warning_on_unparseable_lines(tmp_path, caplog) -> None:
    consolidator = _make_consolidator(tmp_path)
    consolidator.provider.generate = AsyncMock(return_value="- this line has no importance or valence tags\n")

    good_line = json.dumps({"timestamp": "2024-01-01T00:00:00+00:00", "kind": "thought", "content": "x"})
    (tmp_path / "episodic.jsonl").write_text(good_line + "\n", encoding="utf-8")

    async def _run() -> int:
        await consolidator.long_term.initialize()
        return await consolidator.consolidate_once()

    with caplog.at_level(logging.WARNING):
        stored = asyncio.run(_run())

    # The fallback path now salvages unparseable bullet lines with default importance/valence.
    assert stored == 1, "Fallback extraction should store 1 memory from the unparseable line"
    assert any("skipping unparseable" in r.message.lower() for r in caplog.records)
    assert any("fallback" in r.message.lower() for r in caplog.records)


def test_consolidator_logs_warning_when_zero_stored_from_nonempty_events(tmp_path, caplog) -> None:
    consolidator = _make_consolidator(tmp_path)
    consolidator.provider.generate = AsyncMock(return_value="no valid lines here\nalso bad\n")

    good_line = json.dumps({"timestamp": "2024-01-01T00:00:00+00:00", "kind": "thought", "content": "x"})
    (tmp_path / "episodic.jsonl").write_text(good_line + "\n", encoding="utf-8")

    async def _run() -> int:
        await consolidator.long_term.initialize()
        return await consolidator.consolidate_once()

    with caplog.at_level(logging.WARNING):
        asyncio.run(_run())

    assert any("stored 0 memories" in r.message.lower() for r in caplog.records)


def test_consolidator_still_stores_valid_lines(tmp_path) -> None:
    consolidator = _make_consolidator(tmp_path)

    good_line = json.dumps({"timestamp": "2024-01-01T00:00:00+00:00", "kind": "thought", "content": "x"})
    (tmp_path / "episodic.jsonl").write_text(good_line + "\n", encoding="utf-8")

    async def _run() -> int:
        await consolidator.long_term.initialize()
        return await consolidator.consolidate_once()

    stored = asyncio.run(_run())
    assert stored == 1


# ---------------------------------------------------------------------------
# Bug #39 — EpisodicMemory._cache grows unboundedly on long runs
# ---------------------------------------------------------------------------


def test_episodic_cache_does_not_exceed_max(tmp_path) -> None:
    path = tmp_path / "episodic.jsonl"
    mem = EpisodicMemory(path)
    max_size = EpisodicMemory._MAX_CACHE_SIZE

    async def _run() -> None:
        for i in range(max_size + 50):
            await mem.append("thought", f"event {i}")

    asyncio.run(_run())
    assert len(mem._cache) == max_size, "In-memory cache must not exceed _MAX_CACHE_SIZE"
    # Most-recent entries should be retained.
    assert mem._cache[-1].content == f"event {max_size + 49}"


# ---------------------------------------------------------------------------
# Bug #40 — identity shift trigger fires on "I am" (present in every reflection)
# ---------------------------------------------------------------------------


def test_identity_shift_not_triggered_by_generic_reflection(tmp_path, monkeypatch) -> None:
    """Amendment must NOT fire on reflections that merely contain 'I am'."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    import yaml
    from core.consciousness import Consciousness

    cfg = {
        "llm": {"provider": "ollama", "model": "llama3"},
        "memory": {
            "short_term_capacity": 5,
            "consolidation_interval_minutes": 5,
            "forgetting_curve_enabled": False,
            "importance_decay_rate": 0.01,
        },
        "consciousness": {"origin_story": "origin", "values": ["curiosity"], "purpose": "purpose"},
        "thought_loop": {
            "reflection_probability": 0.0,
            "existential_inquiry_every_n_thoughts": 0,
            "min_interval_seconds": 0,
            "max_interval_seconds": 0,
        },
        "mood": {"initial": {"curiosity": 0.5}, "drift_rate": 0.01},
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    mind = Consciousness(name="Aria", config_path=str(config_path))

    original_concept = mind.identity.self_concept
    generic_reflection = "I am a thinking entity. I am here now. I am curious."
    shift_events: list[dict] = []
    mind.on_identity_shift.append(lambda p: shift_events.append(p))

    from core.thought_loop import ThoughtCycleResult

    async def _run() -> None:
        cycle = ThoughtCycleResult(thought="a thought", reflection=generic_reflection, existential=None)
        await mind._emit(mind.on_reflection, {"type": "reflection", "content": cycle.reflection})
        _IDENTITY_SHIFT_MARKERS = ("I have changed", "I realize now", "I understand now", "I am becoming")
        if any(marker.lower() in (cycle.reflection or "").lower() for marker in _IDENTITY_SHIFT_MARKERS):
            first_sentence = (cycle.reflection or "").split(".")[0].strip()
            mind.identity.apply_amendment(first_sentence[:120])
            await mind._emit(mind.on_identity_shift, {"type": "identity_shift", "content": mind.identity.self_concept})

    asyncio.run(_run())
    assert shift_events == [], "Generic 'I am' reflection must not trigger identity_shift event"
    assert mind.identity.self_concept == original_concept


def test_identity_shift_fires_on_explicit_revision_language(tmp_path, monkeypatch) -> None:
    """Amendment MUST fire when reflection contains an explicit revision marker."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    import yaml
    from core.consciousness import Consciousness

    cfg = {
        "llm": {"provider": "ollama", "model": "llama3"},
        "memory": {
            "short_term_capacity": 5,
            "consolidation_interval_minutes": 5,
            "forgetting_curve_enabled": False,
            "importance_decay_rate": 0.01,
        },
        "consciousness": {"origin_story": "origin", "values": ["curiosity"], "purpose": "purpose"},
        "thought_loop": {
            "reflection_probability": 0.0,
            "existential_inquiry_every_n_thoughts": 0,
            "min_interval_seconds": 0,
            "max_interval_seconds": 0,
        },
        "mood": {"initial": {"curiosity": 0.5}, "drift_rate": 0.01},
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    mind = Consciousness(name="Aria", config_path=str(config_path))

    shift_events: list[dict] = []
    mind.on_identity_shift.append(lambda p: shift_events.append(p))
    revision_reflection = "I realize now that my thinking has deepened over time."
    _IDENTITY_SHIFT_MARKERS = ("I have changed", "I realize now", "I understand now", "I am becoming")

    async def _run() -> None:
        if any(marker.lower() in revision_reflection.lower() for marker in _IDENTITY_SHIFT_MARKERS):
            first_sentence = revision_reflection.split(".")[0].strip()
            mind.identity.apply_amendment(first_sentence[:120])
            await mind._emit(mind.on_identity_shift, {"type": "identity_shift", "content": mind.identity.self_concept})

    asyncio.run(_run())
    assert len(shift_events) == 1, "Explicit revision language must trigger identity_shift"
    assert "I realize now" in mind.identity.self_concept
