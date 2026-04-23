"""Append-only JSONL journal of internal experience for replay and inspection."""

from __future__ import annotations

import asyncio
import json
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
                        rows.append(dict(json.loads(line)))
            return rows[-limit:]

        return await asyncio.to_thread(_read)
