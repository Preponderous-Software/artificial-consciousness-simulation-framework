"""Regression tests for bugs #2, #3, #4, #5, and #39."""

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
        "perception": {
            "enabled": False,
            "provider": "mock",
            "every_n_cycles": 0,
            "timeout_seconds": 1.0,
            "cache_last_n": 0,
        },
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
# #138 — _expand_env_vars() has no test coverage
# ---------------------------------------------------------------------------

from core.consciousness import _expand_env_vars


def test_expand_env_vars_substitutes_plain_string(monkeypatch) -> None:
    monkeypatch.setenv("MY_SECRET", "hunter2")
    assert _expand_env_vars("${MY_SECRET}") == "hunter2"


def test_expand_env_vars_substitutes_nested_in_dict(monkeypatch) -> None:
    monkeypatch.setenv("WEBHOOK_URL", "https://example.invalid/hook")
    value = {"discord": {"webhook_url": "${WEBHOOK_URL}"}}
    assert _expand_env_vars(value) == {"discord": {"webhook_url": "https://example.invalid/hook"}}


def test_expand_env_vars_substitutes_nested_in_list(monkeypatch) -> None:
    monkeypatch.setenv("TAG_A", "alpha")
    monkeypatch.setenv("TAG_B", "beta")
    value = ["${TAG_A}", "${TAG_B}", "literal"]
    assert _expand_env_vars(value) == ["alpha", "beta", "literal"]


def test_expand_env_vars_unset_variable_resolves_to_empty_string(monkeypatch) -> None:
    monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
    assert _expand_env_vars("${DEFINITELY_UNSET_VAR}") == ""


def test_expand_env_vars_leaves_value_without_placeholder_unchanged() -> None:
    assert _expand_env_vars("plain value") == "plain value"
    assert _expand_env_vars({"a": 1, "b": [1, 2]}) == {"a": 1, "b": [1, 2]}
    assert _expand_env_vars(42) == 42


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


# ---------------------------------------------------------------------------
# EpisodicMemory._load_from_disk — bounded memory + tail correctness (#110)
# ---------------------------------------------------------------------------

def test_episodic_load_returns_last_max_cache_size_when_file_large(tmp_path) -> None:
    """Lazy load on a journal larger than _MAX_CACHE_SIZE must populate the cache
    with exactly _MAX_CACHE_SIZE most-recent events, regardless of file size."""
    path = tmp_path / "episodic.jsonl"
    n_events = EpisodicMemory._MAX_CACHE_SIZE + 100
    with path.open("w", encoding="utf-8") as f:
        for i in range(n_events):
            f.write(json.dumps({
                "timestamp": f"2026-01-01T00:00:{i:06d}+00:00",
                "kind": "thought",
                "content": f"e{i}",
            }) + "\n")

    mem = EpisodicMemory(path)
    events = asyncio.run(mem.recent(limit=EpisodicMemory._MAX_CACHE_SIZE))
    assert len(events) == EpisodicMemory._MAX_CACHE_SIZE
    # Should be the LAST _MAX_CACHE_SIZE, in order
    assert events[0].content == f"e{n_events - EpisodicMemory._MAX_CACHE_SIZE}"
    assert events[-1].content == f"e{n_events - 1}"


def test_episodic_load_preserves_n_valid_when_corruption_near_tail(tmp_path) -> None:
    """Corruption clustered at the tail of the journal must not reduce the cached
    event count below _MAX_CACHE_SIZE when enough valid events exist earlier."""
    path = tmp_path / "episodic.jsonl"
    valid_events = EpisodicMemory._MAX_CACHE_SIZE + 10
    with path.open("w", encoding="utf-8") as f:
        for i in range(valid_events):
            f.write(json.dumps({
                "timestamp": f"2026-01-01T00:00:{i:06d}+00:00",
                "kind": "thought",
                "content": f"v{i}",
            }) + "\n")
        for _ in range(5):
            f.write("NOT_JSON\n")

    mem = EpisodicMemory(path)
    events = asyncio.run(mem.recent(limit=EpisodicMemory._MAX_CACHE_SIZE))
    # Contract: corrupted tail lines must not crowd out earlier valid events
    assert len(events) == EpisodicMemory._MAX_CACHE_SIZE
    assert all(e.kind == "thought" for e in events)


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
# Journal.recent() — bounded memory + correct tail semantics (#108)
# ---------------------------------------------------------------------------

def test_journal_recent_returns_last_n_when_file_exceeds_limit(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(1000):
            f.write(json.dumps({
                "timestamp": f"2026-01-01T00:00:{i:04d}+00:00",
                "type": "thought",
                "content": f"thought {i}",
            }) + "\n")

    journal = Journal(path)
    entries = asyncio.run(journal.recent(limit=10))
    assert len(entries) == 10
    # Last 10 of 1000 are thoughts 990..999, in order
    assert [e["content"] for e in entries] == [f"thought {i}" for i in range(990, 1000)]


def test_journal_recent_returns_n_valid_when_corruption_near_tail(tmp_path) -> None:
    """Contract preserved from pre-#108 behavior: corruption in the last N raw
    lines must not reduce the returned dict count below N when more valid
    events exist earlier in the file."""
    path = tmp_path / "journal.jsonl"
    good = lambda i: json.dumps({  # noqa: E731
        "timestamp": f"2026-01-01T00:00:{i:04d}+00:00",
        "type": "thought",
        "content": f"thought {i}",
    })
    with path.open("w", encoding="utf-8") as f:
        # 50 good lines, then 5 corrupted at the tail
        for i in range(50):
            f.write(good(i) + "\n")
        for _ in range(5):
            f.write("NOT JSON\n")

    journal = Journal(path)
    entries = asyncio.run(journal.recent(limit=10))
    # The last 10 RAW lines include 5 corrupted; old contract returned 10 valid
    # by walking past corruption — new code must too.
    assert len(entries) == 10
    assert [e["content"] for e in entries] == [f"thought {i}" for i in range(40, 50)]


def test_journal_recent_returns_all_when_limit_exceeds_size(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"type": "thought", "content": f"t{i}"}) + "\n")
    journal = Journal(path)
    entries = asyncio.run(journal.recent(limit=100))
    assert len(entries) == 3


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


def test_episodic_cache_does_not_exceed_max_on_append(tmp_path) -> None:
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


def test_episodic_cache_does_not_exceed_max_on_load(tmp_path) -> None:
    path = tmp_path / "episodic.jsonl"
    max_size = EpisodicMemory._MAX_CACHE_SIZE

    # Write more entries than the cap directly to disk (simulates a large existing file).
    with path.open("w", encoding="utf-8") as f:
        import json as _json
        for i in range(max_size + 100):
            f.write(_json.dumps({"timestamp": "2024-01-01T00:00:00+00:00", "kind": "thought", "content": f"disk event {i}"}) + "\n")

    mem = EpisodicMemory(path)
    events = asyncio.run(mem.recent(limit=max_size))
    assert len(mem._cache) == max_size, "_load_from_disk must cap cache to _MAX_CACHE_SIZE"
    # The tail (most recent) entries are retained.
    assert mem._cache[-1].content == f"disk event {max_size + 99}"

