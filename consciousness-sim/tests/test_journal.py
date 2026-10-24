"""Tests for persistence/journal.py — the append-only JSONL event journal.

Corrupted-line handling and the bounded tail read of ``recent()`` are covered
in tests/test_bug_fixes.py (#5, #108, #175). This file covers the rest:
- the constructor creates missing parent directories
- append() writes one JSON object per line with timestamp/type/content
- append() merges structured extras into the payload
- append() never truncates existing lines
- recent() on a missing file, with a non-positive limit, and past blank lines
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from persistence.journal import Journal


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_constructor_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "journal.jsonl"
    Journal(path)
    assert path.parent.is_dir()
    assert not path.exists()


def test_append_writes_timestamp_type_and_content(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    before = datetime.now(timezone.utc)
    asyncio.run(Journal(path).append("thought", "a river bends"))
    after = datetime.now(timezone.utc)

    [event] = _lines(path)
    assert set(event) == {"timestamp", "type", "content"}
    assert event["type"] == "thought"
    assert event["content"] == "a river bends"
    stamp = datetime.fromisoformat(str(event["timestamp"]))
    assert stamp.tzinfo is not None
    assert stamp.utcoffset() == timezone.utc.utcoffset(None)
    assert before <= stamp <= after


def test_append_merges_extras_into_the_payload(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    asyncio.run(
        Journal(path).append(
            "consolidation",
            "stored 3 memories",
            long_term_count=42,
            details={"stored": 3, "error": None},
        )
    )

    [event] = _lines(path)
    assert event["type"] == "consolidation"
    assert event["content"] == "stored 3 memories"
    assert event["long_term_count"] == 42
    assert event["details"] == {"stored": 3, "error": None}


def test_append_adds_lines_without_truncating_existing_ones(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_text(json.dumps({"type": "thought", "content": "pre-existing"}) + "\n", encoding="utf-8")

    journal = Journal(path)
    asyncio.run(journal.append("thought", "second"))
    asyncio.run(Journal(path).append("reflection", "third"))

    assert [e["content"] for e in _lines(path)] == ["pre-existing", "second", "third"]


def test_append_rejects_non_json_serializable_extras(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    with pytest.raises(TypeError):
        asyncio.run(Journal(path).append("thought", "x", bad=object()))
    assert not path.exists() or path.read_text(encoding="utf-8") == ""


def test_append_then_recent_round_trips_in_order(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.jsonl")

    async def _write() -> None:
        for i in range(5):
            await journal.append("thought", f"t{i}", cycle=i)

    asyncio.run(_write())
    entries = asyncio.run(journal.recent(limit=3))
    assert [e["content"] for e in entries] == ["t2", "t3", "t4"]
    assert [e["cycle"] for e in entries] == [2, 3, 4]


def test_recent_returns_empty_when_file_missing(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "never-written.jsonl")
    assert asyncio.run(journal.recent()) == []


@pytest.mark.parametrize("limit", [0, -5])
def test_recent_returns_empty_for_non_positive_limit(tmp_path: Path, limit: int) -> None:
    journal = Journal(tmp_path / "journal.jsonl")
    asyncio.run(journal.append("thought", "present"))
    assert asyncio.run(journal.recent(limit=limit)) == []


def test_recent_skips_blank_lines_without_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "journal.jsonl"
    good = json.dumps({"type": "thought", "content": "kept"})
    path.write_text(f"\n{good}\n   \n\n{good}\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        entries = asyncio.run(Journal(path).recent())

    assert [e["content"] for e in entries] == ["kept", "kept"]
    assert not caplog.records
