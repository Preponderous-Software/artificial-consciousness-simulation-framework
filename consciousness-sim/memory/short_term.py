"""Short-term working memory with a bounded sliding window."""

from __future__ import annotations

from collections import deque
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
        self._items: deque[MemoryItem] = deque(maxlen=capacity)

    def add(self, kind: str, content: str) -> MemoryItem:
        item = MemoryItem(kind=kind, content=content, timestamp=datetime.now(timezone.utc).isoformat())
        self._items.append(item)
        return item

    def list(self) -> list[MemoryItem]:
        return list(self._items)

    def render_for_prompt(self) -> str:
        if not self._items:
            return "(no recent thoughts yet)"
        return "\n".join(f"- [{i.kind}] {i.content}" for i in self._items)

    def prune_to_capacity(self) -> None:
        while len(self._items) > self.capacity:
            self._items.popleft()
