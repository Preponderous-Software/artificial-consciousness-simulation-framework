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
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Journal:
    """Writes and reads chronological consciousness events."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, event_type: str, content: str, **extras: Any) -> None:
        """Append an event. ``extras`` are merged into the payload alongside
        the canonical ``timestamp``/``type``/``content`` fields so consumers
        like the standalone dashboard's journal tailer can deliver structured
        data (e.g. ``long_term_count``) without parsing the content string."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "content": content,
            **extras,
        }

        def _write() -> None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

        await asyncio.to_thread(_write)

    async def recent(self, limit: int = 50) -> list[dict[str, str]]:
        if not self.path.exists():
            return []

        def _read() -> list[dict[str, str]]:
            # Stream the file and retain only the last `limit` PARSED dicts.
            # Memory stays O(limit) regardless of journal size, while still
            # honouring the contract that `recent(limit=N)` returns N valid
            # events even when corruption is concentrated near the tail (#108).
            rows: deque[dict[str, str]] = deque(maxlen=max(0, limit))
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(dict(json.loads(line)))
                    except json.JSONDecodeError:
                        logging.warning("Journal: skipping corrupted line: %r", line)
            return list(rows)

        return await asyncio.to_thread(_read)
