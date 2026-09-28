"""Append-only JSONL inbox of messages people send to an instance (#198).

Theory mapping — RPT-1 (perceptual input) / AE-2 (action–perception loop): a
message is external stimulus, delivered to the thought loop as a `speech`
perception; the instance's reply (an `utterance`) is an action whose
consequence can arrive as the next message.
Gap: the loop is closed but not modelled — nothing represents how an
utterance changes the next message, which AE-2 requires.

The web dashboard (a separate process) appends; the running instance reads.
A persisted byte offset records how far the instance has read, so a restart
neither re-delivers heard messages nor drops unheard ones.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class InboxMessage:
    id: str
    timestamp: str
    sender: str
    text: str


def _parse(line: bytes) -> InboxMessage | None:
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        logging.warning("Inbox: skipping corrupted line: %r", line[:200])
        return None
    if not isinstance(payload, dict):
        logging.warning("Inbox: skipping non-object line: %r", line[:200])
        return None
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        logging.warning("Inbox: skipping message without text: %r", line[:200])
        return None
    return InboxMessage(
        id=str(payload.get("id") or ""),
        timestamp=str(payload.get("timestamp") or ""),
        sender=str(payload.get("sender") or "someone"),
        text=text,
    )


class Inbox:
    """Messages addressed to one instance, with a persisted read position."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset_path = path.with_name(path.name + ".offset")

    def append(self, sender: str, text: str) -> InboxMessage:
        message = InboxMessage(
            id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender=sender,
            text=text,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"id": message.id, "timestamp": message.timestamp, "sender": sender, "text": text}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return message

    def _read_offset(self) -> int:
        try:
            return max(0, int(self.offset_path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            return 0

    def _write_offset(self, offset: int) -> None:
        tmp = self.offset_path.with_name(self.offset_path.name + ".tmp")
        tmp.write_text(str(offset), encoding="utf-8")
        tmp.replace(self.offset_path)

    def _entries(self) -> list[tuple[InboxMessage | None, int]]:
        """Every complete line as (message or None if corrupt, byte offset after it)."""
        if not self.path.exists():
            return []
        entries: list[tuple[InboxMessage | None, int]] = []
        with self.path.open("rb") as f:
            position = 0
            for line in f:
                if not line.endswith(b"\n"):
                    # A writer is mid-append; the rest arrives on a later read.
                    break
                position += len(line)
                stripped = line.strip()
                if stripped:
                    entries.append((_parse(stripped), position))
        return entries

    def _consumed_offset(self) -> int:
        offset = self._read_offset()
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        # A truncated or replaced inbox restarts from the beginning.
        return offset if offset <= size else 0

    def conversation(self) -> list[tuple[InboxMessage, bool]]:
        """All valid messages in order, each with whether the instance has heard it."""
        consumed = self._consumed_offset()
        return [(m, end <= consumed) for m, end in self._entries() if m is not None]

    def unread_count(self) -> int:
        return sum(1 for _, heard in self.conversation() if not heard)

    async def next_unread(self) -> InboxMessage | None:
        """Return the oldest unheard message and mark it heard, or None."""

        def _next() -> InboxMessage | None:
            consumed = self._consumed_offset()
            for message, end in self._entries():
                if end <= consumed:
                    continue
                # Corrupt lines are passed over (and logged by _parse) rather
                # than blocking every message behind them.
                self._write_offset(end)
                if message is not None:
                    return message
            return None

        return await asyncio.to_thread(_next)
