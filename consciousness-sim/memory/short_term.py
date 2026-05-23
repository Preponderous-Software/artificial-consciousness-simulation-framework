"""Short-term working memory with a bounded sliding window.

Theory mapping — GWT (Baars 1988): approximates the global workspace buffer.
Capacity limit partially implements GWT-1 (limited-capacity workspace).
Gap: no competitive selection between specialists (GWT-2); items enter by
recency only. See issue #23 for importance-weighted eviction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(slots=True)
class MemoryItem:
    kind: str
    content: str
    timestamp: str


class ShortTermMemory:
    """Stores recent thought-related events for prompt context."""

    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self._items: list[MemoryItem] = []

    def add(self, kind: str, content: str) -> MemoryItem:
        item = MemoryItem(kind=kind, content=content, timestamp=datetime.now(timezone.utc).isoformat())
        self._items.append(item)
        self.prune_to_capacity()
        return item

    def list(self) -> list[MemoryItem]:
        return list(self._items)

    def render_for_prompt(self) -> str:
        if not self._items:
            return "(no recent thoughts yet)"
        return "\n".join(f"- [{i.kind}] {i.content}" for i in self._items)

    def prune_to_capacity(self) -> None:
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity:]
