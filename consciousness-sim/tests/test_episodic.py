"""Tests for memory/episodic.py — the append path and its cache.

Covers:
- append() before the first recent() keeps events already on disk (#230)
- append() writes one JSONL object per event and returns that event
- a fresh instance reads back what an earlier one appended
- recent(limit) returns the newest ``limit`` events, oldest first
- a missing file reads as empty, and blank lines are ignored

Corruption-skip and cache-cap coverage lives in test_bug_fixes.py (#5, #39,
#110, #175).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from memory.episodic import EpisodicMemory


def _write_events(path: Path, contents: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, content in enumerate(contents):
            f.write(json.dumps({
                "timestamp": f"2026-01-01T00:00:{i:02d}+00:00",
                "kind": "thought",
                "content": content,
            }) + "\n")


def test_append_before_recent_keeps_events_already_on_disk(tmp_path: Path) -> None:
    # A resumed process appends on its first cycle, long before the consolidator
    # first calls recent(); the previous process's events must still be there.
    path = tmp_path / "episodic.jsonl"
    _write_events(path, ["pre-restart-0", "pre-restart-1", "pre-restart-2"])

    mem = EpisodicMemory(path)

    async def _run() -> list[str]:
        await mem.append("thought", "post-restart")
        return [e.content for e in await mem.recent()]

    assert asyncio.run(_run()) == ["pre-restart-0", "pre-restart-1", "pre-restart-2", "post-restart"]


def test_append_before_recent_respects_the_cache_cap(tmp_path: Path) -> None:
    path = tmp_path / "episodic.jsonl"
    max_size = EpisodicMemory._MAX_CACHE_SIZE
    _write_events(path, [f"disk-{i}" for i in range(max_size)])

    mem = EpisodicMemory(path)

    async def _run() -> list[str]:
        await mem.append("thought", "newest")
        return [e.content for e in await mem.recent(limit=max_size)]

    contents = asyncio.run(_run())
    assert len(contents) == max_size
    assert contents[0] == "disk-1"
    assert contents[-1] == "newest"


def test_append_writes_one_json_object_per_event(tmp_path: Path) -> None:
    path = tmp_path / "episodic.jsonl"
    mem = EpisodicMemory(path)

    async def _run() -> None:
        await mem.append("thought", "first")
        await mem.append("reflection", "second")

    asyncio.run(_run())
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [(r["kind"], r["content"]) for r in rows] == [("thought", "first"), ("reflection", "second")]
    assert all(set(r) == {"timestamp", "kind", "content"} for r in rows)


def test_append_returns_the_event_with_a_utc_timestamp(tmp_path: Path) -> None:
    mem = EpisodicMemory(tmp_path / "episodic.jsonl")

    event = asyncio.run(mem.append("perception", "a stimulus"))

    assert (event.kind, event.content) == ("perception", "a stimulus")
    # The consolidator's watermark compares these strings lexicographically,
    # which only orders correctly when every timestamp is in UTC.
    assert datetime.fromisoformat(event.timestamp).utcoffset().total_seconds() == 0


def test_fresh_instance_reads_back_appended_events(tmp_path: Path) -> None:
    path = tmp_path / "episodic.jsonl"

    async def _write() -> None:
        writer = EpisodicMemory(path)
        await writer.append("thought", "one")
        await writer.append("thought", "two")

    asyncio.run(_write())
    events = asyncio.run(EpisodicMemory(path).recent())
    assert [e.content for e in events] == ["one", "two"]


def test_recent_returns_the_newest_limit_events_oldest_first(tmp_path: Path) -> None:
    path = tmp_path / "episodic.jsonl"
    _write_events(path, [f"e{i}" for i in range(10)])

    events = asyncio.run(EpisodicMemory(path).recent(limit=3))
    assert [e.content for e in events] == ["e7", "e8", "e9"]


def test_recent_on_missing_file_is_empty_and_creates_nothing(tmp_path: Path) -> None:
    path = tmp_path / "episodic.jsonl"

    assert asyncio.run(EpisodicMemory(path).recent()) == []
    assert not path.exists()


def test_recent_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "episodic.jsonl"
    good = json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "kind": "thought", "content": "kept"})
    path.write_text("\n" + good + "\n   \n\n" + good + "\n", encoding="utf-8")

    events = asyncio.run(EpisodicMemory(path).recent())
    assert [e.content for e in events] == ["kept", "kept"]
