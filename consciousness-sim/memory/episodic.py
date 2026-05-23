"""Append-only episodic memory log for timestamped narrative continuity.

Theory mapping — RPT (Lamme 2006) / PP (Friston 2010): stores the raw
experience stream that consolidation compresses into long-term memory,
analogous to the episodic trace that recurrent processing leaves in
sensory areas. Partially implements RPT-1 (organised perceptual record).
Gap: no recurrent feedback within the write path (RPT-2).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class EpisodicEvent:
    timestamp: str
    kind: str
    content: str


class EpisodicMemory:
    """Stores raw narrative experience as JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: list[EpisodicEvent] = []
        self._cache_loaded: bool = False

    async def append(self, kind: str, content: str) -> EpisodicEvent:
        event = EpisodicEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            kind=kind,
            content=content,
        )

        def _write() -> None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event)) + "\n")

        await asyncio.to_thread(_write)
        self._cache.append(event)
        self._cache_loaded = True
        return event

    async def recent(self, limit: int = 50) -> list[EpisodicEvent]:
        if not self._cache_loaded:
            await self._load_from_disk()
        return self._cache[-limit:]

    async def _load_from_disk(self) -> None:
        if not self.path.exists():
            self._cache_loaded = True
            return

        def _read() -> list[EpisodicEvent]:
            rows: list[EpisodicEvent] = []
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(EpisodicEvent(**json.loads(line)))
                    except json.JSONDecodeError:
                        logging.warning("EpisodicMemory: skipping corrupted line: %r", line)
            return rows

        self._cache = await asyncio.to_thread(_read)
        self._cache_loaded = True
