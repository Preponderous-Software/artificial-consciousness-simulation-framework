"""Append-only JSONL journal of internal experience for replay and inspection.

Theory mapping — RPT-1 / AE-1 (agency via temporal record): The journal
implements narrative continuity — an ordered log of all internal events that
supports post-hoc inspection and replay. Analogous to episodic memory encoding
in cognitive neuroscience: each event is timestamped and typed, enabling
reconstruction of the agent's history.
Gap: journal entries are not read back by the agent at runtime (episodic.py
serves that role); the journal is inspection-only and does not feed the thought
loop directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class Journal:
    """Writes and reads chronological consciousness events."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, event_type: str, content: str) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "content": content,
        }

        def _write() -> None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

        await asyncio.to_thread(_write)

    async def recent(self, limit: int = 50) -> list[dict[str, str]]:
        if not self.path.exists():
            return []

        def _read() -> list[dict[str, str]]:
            rows: list[dict[str, str]] = []
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(dict(json.loads(line)))
                        except json.JSONDecodeError:
                            logging.warning("Journal: skipping corrupted line: %r", line)
            return rows[-limit:]

        return await asyncio.to_thread(_read)
