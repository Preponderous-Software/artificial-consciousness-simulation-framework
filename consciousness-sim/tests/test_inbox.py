"""Tests for persistence/inbox.py — messages people send to an instance (#198).

Covers:
- messages are heard oldest first, each exactly once
- the read position survives a new Inbox object (a restarted instance)
- corrupt, non-object and text-less lines are skipped with a warning
- a partially written last line is not consumed until it is complete
- a truncated inbox restarts from the beginning
- conversation() / unread_count() report heard status
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from persistence.inbox import Inbox


def _drain(inbox: Inbox) -> list[str]:
    texts = []
    while (m := asyncio.run(inbox.next_unread())) is not None:
        texts.append(m.text)
    return texts


def test_messages_are_heard_oldest_first_exactly_once(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path / "inbox.jsonl")
    first = inbox.append("Dan", "hello")
    inbox.append("Dan", "what is the Concho River?")

    heard = asyncio.run(inbox.next_unread())
    assert heard is not None
    assert (heard.id, heard.sender, heard.text) == (first.id, "Dan", "hello")
    assert _drain(inbox) == ["what is the Concho River?"]
    assert asyncio.run(inbox.next_unread()) is None


def test_read_position_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "inbox.jsonl"
    Inbox(path).append("Dan", "one")
    Inbox(path).append("Dan", "two")
    assert asyncio.run(Inbox(path).next_unread()).text == "one"  # type: ignore[union-attr]

    restarted = Inbox(path)
    assert _drain(restarted) == ["two"]


def test_bad_lines_are_skipped_with_warning(tmp_path: Path, caplog) -> None:
    path = tmp_path / "inbox.jsonl"
    inbox = Inbox(path)
    inbox.append("Dan", "before")
    with path.open("a", encoding="utf-8") as f:
        f.write("{not json\n")
        f.write("[1, 2]\n")
        f.write(json.dumps({"sender": "Dan", "text": "   "}) + "\n")
    inbox.append("Dan", "after")

    with caplog.at_level(logging.WARNING):
        assert _drain(inbox) == ["before", "after"]
    assert "corrupted" in caplog.text
    assert "non-object" in caplog.text
    assert "without text" in caplog.text


def test_partial_last_line_waits_until_complete(tmp_path: Path) -> None:
    path = tmp_path / "inbox.jsonl"
    inbox = Inbox(path)
    inbox.append("Dan", "whole")
    record = json.dumps({"id": "x", "timestamp": "t", "sender": "Dan", "text": "split"})
    with path.open("a", encoding="utf-8") as f:
        f.write(record[:10])

    assert _drain(inbox) == ["whole"]
    with path.open("a", encoding="utf-8") as f:
        f.write(record[10:] + "\n")
    assert _drain(inbox) == ["split"]


def test_truncated_inbox_restarts_from_the_beginning(tmp_path: Path) -> None:
    path = tmp_path / "inbox.jsonl"
    inbox = Inbox(path)
    inbox.append("Dan", "a long first message that moves the offset well along")
    assert _drain(inbox)
    path.write_text("")
    inbox.append("Dan", "new")
    assert _drain(inbox) == ["new"]


def test_conversation_and_unread_count_report_heard_status(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path / "inbox.jsonl")
    assert inbox.conversation() == []
    assert inbox.unread_count() == 0
    inbox.append("Dan", "one")
    inbox.append("Dan", "two")
    asyncio.run(inbox.next_unread())

    assert [(m.text, heard) for m, heard in inbox.conversation()] == [("one", True), ("two", False)]
    assert inbox.unread_count() == 1
