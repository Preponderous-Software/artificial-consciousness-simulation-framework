"""Short-term working memory with a bounded sliding window.

Theory mapping — GWT (Baars 1988): approximates the global workspace buffer.
Capacity limit implements GWT-1 (limited-capacity workspace). Importance-
weighted eviction partially implements GWT-2 (selective attention controlling
workspace entry) by preferring higher-salience items when the buffer is full.
The rendered workspace is held to a character budget, newest items first
(#205), so what every prompt rereads stays inside the model's context.
Importance decays with age (#201): an item's eviction score halves every
``half_life`` later additions, so salience biases the competition without
letting any kind occupy the workspace permanently.
Gap: no parallel specialist competition for workspace writes (GWT-2 fully
requires competitive selection, not just importance-biased eviction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Salience weights by event kind — higher = survives eviction longer.
# perception sits above thought (it's an external stimulus, more salient than
# self-generated text) but below reflection (which is meta about content).
_KIND_IMPORTANCE: dict[str, float] = {
    "existential": 3.0,
    "reflection": 2.0,
    "perception": 1.5,
    "thought": 1.0,
}
_DEFAULT_IMPORTANCE: float = 1.0
# Items added after which an entry's eviction score has halved (#201). Half of
# the default capacity: a reflection (2.0) is outranked by a fresh thought
# (1.0) once ten newer items have arrived. 0 or None = importance-only
# eviction, the pre-#201 behaviour under which reflections accumulated until
# they filled the workspace and each new thought was evicted on the next add.
DEFAULT_HALF_LIFE: float = 10.0
# Character budget for render_for_prompt() (#205). Every prompt and the
# retrieval embedding reread the rendered workspace; counted in items alone it
# grew to ~19k chars of long reflections, past llama3.2:3b's 4,096-token
# context, and embedding it outran the 120 s provider timeout. ~8k chars is
# roughly half that context. 0 or None = unbounded, the pre-#205 behaviour.
DEFAULT_PROMPT_CHARS: int = 8000
_TRUNCATION_MARK = " …"


@dataclass(slots=True)
class MemoryItem:
    kind: str
    content: str
    timestamp: str
    importance: float = field(default=_DEFAULT_IMPORTANCE)
    # Insertion order within this buffer; ages items for eviction (#201).
    seq: int = 0


class ShortTermMemory:
    """Stores recent thought-related events for prompt context."""

    def __init__(
        self,
        capacity: int = 20,
        half_life: float | None = DEFAULT_HALF_LIFE,
        prompt_chars: int | None = DEFAULT_PROMPT_CHARS,
    ) -> None:
        self.capacity = capacity
        self.half_life = half_life if half_life else None
        self.prompt_chars = prompt_chars if prompt_chars else None
        self._items: list[MemoryItem] = []
        self._next_seq = 0

    def add(self, kind: str, content: str, importance: float | None = None) -> MemoryItem:
        resolved = importance if importance is not None else _KIND_IMPORTANCE.get(kind, _DEFAULT_IMPORTANCE)
        item = MemoryItem(
            kind=kind,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            importance=resolved,
            seq=self._next_seq,
        )
        self._next_seq += 1
        self._items.append(item)
        self.prune_to_capacity()
        return item

    def list(self) -> list[MemoryItem]:
        return list(self._items)

    def render_for_prompt(self) -> str:
        """Render the workspace oldest-first within the ``prompt_chars`` budget.

        Items are admitted newest-first, so when the budget runs out it is the
        oldest that are left out; the newest item is always kept, truncated if
        it alone exceeds the budget.
        """
        if not self._items:
            return "(no recent thoughts yet)"
        lines = [f"- [{i.kind}] {i.content}" for i in self._items]
        if self.prompt_chars is None:
            return "\n".join(lines)
        kept: list[str] = []
        used = 0
        for line in reversed(lines):
            cost = len(line) + (1 if kept else 0)
            if used + cost <= self.prompt_chars:
                kept.append(line)
                used += cost
                continue
            if not kept:
                kept.append(line[: max(0, self.prompt_chars - len(_TRUNCATION_MARK))].rstrip() + _TRUNCATION_MARK)
            break
        if len(kept) < len(lines):
            logging.debug(
                "Workspace render: %d of %d items within the %d-char prompt budget",
                len(kept), len(lines), self.prompt_chars,
            )
        return "\n".join(reversed(kept))

    def effective_importance(self, item: MemoryItem) -> float:
        """Eviction score: importance halved every ``half_life`` later additions."""
        if self.half_life is None:
            return item.importance
        age = (self._next_seq - 1) - item.seq
        return float(item.importance * 0.5 ** (age / self.half_life))

    def prune_to_capacity(self) -> None:
        """Evict the lowest-scoring item when over capacity (GWT-2 analog).

        Ties go to the oldest item, since min() returns the first minimum.
        """
        while len(self._items) > self.capacity:
            min_idx = min(range(len(self._items)), key=lambda i: self.effective_importance(self._items[i]))
            self._items.pop(min_idx)
