"""Short-term working memory with a bounded sliding window.

Theory mapping — GWT (Baars 1988): approximates the global workspace buffer.
Capacity limit implements GWT-1 (limited-capacity workspace). Importance-
weighted eviction partially implements GWT-2 (selective attention controlling
workspace entry) by preferring higher-salience items when the buffer is full.
Gap: no parallel specialist competition for workspace writes (GWT-2 fully
requires competitive selection, not just importance-biased eviction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

# Salience weights by event kind — higher = survives eviction longer.
_KIND_IMPORTANCE: dict[str, float] = {
    "existential": 3.0,
    "reflection": 2.0,
    "thought": 1.0,
}
_DEFAULT_IMPORTANCE: float = 1.0


@dataclass(slots=True)
class MemoryItem:
    kind: str
    content: str
    timestamp: str
    importance: float = field(default=_DEFAULT_IMPORTANCE)


class ShortTermMemory:
    """Stores recent thought-related events for prompt context."""

    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self._items: list[MemoryItem] = []

    def add(self, kind: str, content: str, importance: float | None = None) -> MemoryItem:
        resolved = importance if importance is not None else _KIND_IMPORTANCE.get(kind, _DEFAULT_IMPORTANCE)
        item = MemoryItem(
            kind=kind,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            importance=resolved,
        )
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
        """Evict the lowest-importance item when over capacity (GWT-2 analog)."""
        while len(self._items) > self.capacity:
            min_idx = min(range(len(self._items)), key=lambda i: self._items[i].importance)
            self._items.pop(min_idx)
