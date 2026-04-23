"""Tests for memory storage and retrieval behavior."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory


def test_short_term_sliding_window() -> None:
    stm = ShortTermMemory(capacity=2)
    stm.add("thought", "one")
    stm.add("thought", "two")
    stm.add("thought", "three")
    items = stm.list()
    assert len(items) == 2
    assert items[0].content == "two"
    assert items[1].content == "three"


def test_long_term_similarity_retrieval() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            await ltm.add_memory("I feel alone", 0.2, 7.0, [1.0, 0.0, 0.0])
            await ltm.add_memory("I am curious", 0.8, 8.0, [0.0, 1.0, 0.0])
            hits = await ltm.similarity_search([0.0, 1.0, 0.0], limit=1)
            assert len(hits) == 1
            assert "curious" in hits[0].summary

    asyncio.run(_run())
