"""Tests for the standalone web dashboard's journal tailer.

Pins the invariants that keep the polling tailer (interfaces/web/journal_tail.py)
healthy across instance lifecycle and journal-file corruption:

- First sighting starts at EOF (no historical replay through the live queue)
- Subsequent polls pick up only newly appended lines
- File truncation (rotation / manual clear) resets the offset
- Partial trailing lines are deferred until the next complete line lands
- Corrupted JSON is skipped with a warning (CLAUDE.md JSONL invariant)
- Unknown event types are filtered before dispatch
- on_event handler exceptions never kill the tailer
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest

from interfaces.web.journal_tail import JournalTailer, _KNOWN_EVENT_TYPES


def _write_event(path: Path, event_type: str, content: str, **extra) -> None:
    payload = {"type": event_type, "content": content, **extra}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path


def _make_tailer(home: Path, collected: list[tuple[str, dict]]) -> JournalTailer:
    return JournalTailer(home, lambda inst, payload: collected.append((inst, payload)))


def test_first_sighting_seeks_to_eof(home: Path) -> None:
    inst_dir = home / "Aria"
    inst_dir.mkdir()
    journal = inst_dir / "journal.jsonl"
    _write_event(journal, "thought", "pre-existing")

    collected: list[tuple[str, dict]] = []
    tailer = _make_tailer(home, collected)
    tailer._poll_once()

    assert collected == []
    assert tailer._offsets["Aria"] == journal.stat().st_size


def test_subsequent_poll_picks_up_new_lines(home: Path) -> None:
    inst_dir = home / "Aria"
    inst_dir.mkdir()
    journal = inst_dir / "journal.jsonl"
    _write_event(journal, "thought", "first")

    collected: list[tuple[str, dict]] = []
    tailer = _make_tailer(home, collected)
    tailer._poll_once()

    _write_event(journal, "thought", "second")
    _write_event(journal, "reflection", "third")
    tailer._poll_once()

    assert [c["content"] for _, c in collected] == ["second", "third"]
    assert all(inst == "Aria" for inst, _ in collected)


def test_truncation_resets_offset(home: Path) -> None:
    inst_dir = home / "Aria"
    inst_dir.mkdir()
    journal = inst_dir / "journal.jsonl"
    _write_event(journal, "thought", "alpha")
    _write_event(journal, "thought", "beta")

    collected: list[tuple[str, dict]] = []
    tailer = _make_tailer(home, collected)
    tailer._poll_once()  # seeks to EOF

    # Rotate: clear the file and write a single fresh line
    journal.write_text("")
    _write_event(journal, "thought", "gamma")
    tailer._poll_once()

    assert [c["content"] for _, c in collected] == ["gamma"]


def test_partial_trailing_line_is_deferred(home: Path) -> None:
    inst_dir = home / "Aria"
    inst_dir.mkdir()
    journal = inst_dir / "journal.jsonl"
    _write_event(journal, "thought", "complete")

    collected: list[tuple[str, dict]] = []
    tailer = _make_tailer(home, collected)
    tailer._poll_once()

    # Write a complete line and start a second line without a trailing newline
    _write_event(journal, "thought", "next")
    with journal.open("a", encoding="utf-8") as f:
        f.write('{"type": "thought", "content": "partial"')  # no newline, no closing brace
    tailer._poll_once()

    assert [c["content"] for _, c in collected] == ["next"]

    # Finish the partial line — next poll picks it up
    with journal.open("a", encoding="utf-8") as f:
        f.write("}\n")
    tailer._poll_once()

    assert [c["content"] for _, c in collected] == ["next", "partial"]


def test_corrupted_line_is_skipped_with_warning(home: Path, caplog) -> None:
    inst_dir = home / "Aria"
    inst_dir.mkdir()
    journal = inst_dir / "journal.jsonl"
    _write_event(journal, "thought", "before")

    collected: list[tuple[str, dict]] = []
    tailer = _make_tailer(home, collected)
    tailer._poll_once()

    with journal.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    _write_event(journal, "thought", "after")

    with caplog.at_level(logging.WARNING, logger="interfaces.web.journal_tail"):
        tailer._poll_once()

    assert [c["content"] for _, c in collected] == ["after"]
    assert any("skipping corrupted line" in rec.message for rec in caplog.records)


def test_unknown_event_type_is_filtered(home: Path) -> None:
    inst_dir = home / "Aria"
    inst_dir.mkdir()
    journal = inst_dir / "journal.jsonl"
    _write_event(journal, "thought", "seed")

    collected: list[tuple[str, dict]] = []
    tailer = _make_tailer(home, collected)
    tailer._poll_once()

    _write_event(journal, "lifecycle", "ignored")
    _write_event(journal, "thought", "kept")
    tailer._poll_once()

    assert [c["content"] for _, c in collected] == ["kept"]
    # Sanity: confirm the filter set is the source of truth.
    assert "lifecycle" not in _KNOWN_EVENT_TYPES
    assert "thought" in _KNOWN_EVENT_TYPES


def test_on_event_exception_does_not_kill_tailer(home: Path, caplog) -> None:
    inst_dir = home / "Aria"
    inst_dir.mkdir()
    journal = inst_dir / "journal.jsonl"
    _write_event(journal, "thought", "seed")

    calls: list[str] = []

    def handler(inst: str, payload: dict) -> None:
        calls.append(payload["content"])
        if payload["content"] == "boom":
            raise RuntimeError("handler exploded")

    tailer = JournalTailer(home, handler)
    tailer._poll_once()

    _write_event(journal, "thought", "boom")
    _write_event(journal, "thought", "after")

    with caplog.at_level(logging.ERROR, logger="interfaces.web.journal_tail"):
        tailer._poll_once()

    assert calls == ["boom", "after"]
    assert any("on_event handler raised" in rec.message for rec in caplog.records)


def test_discovers_new_instance_directories_dynamically(home: Path) -> None:
    collected: list[tuple[str, dict]] = []
    tailer = _make_tailer(home, collected)
    tailer._poll_once()  # no instances yet

    inst_dir = home / "Wren"
    inst_dir.mkdir()
    journal = inst_dir / "journal.jsonl"
    _write_event(journal, "thought", "born")

    tailer._poll_once()  # first sighting → seeks to EOF
    assert collected == []

    _write_event(journal, "thought", "after-first-sighting")
    tailer._poll_once()
    assert [c["content"] for _, c in collected] == ["after-first-sighting"]
    assert collected[0][0] == "Wren"


def test_ignores_hidden_and_non_directory_entries(home: Path) -> None:
    (home / ".hidden").mkdir()
    (home / "not-an-instance.txt").write_text("nope")
    (home / "Aria").mkdir()
    journal = home / "Aria" / "journal.jsonl"
    _write_event(journal, "thought", "seed")

    collected: list[tuple[str, dict]] = []
    tailer = _make_tailer(home, collected)
    tailer._poll_once()
    _write_event(journal, "thought", "next")
    tailer._poll_once()

    assert [inst for inst, _ in collected] == ["Aria"]
    assert ".hidden" not in tailer._offsets
