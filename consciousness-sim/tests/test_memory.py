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


def test_long_term_index_exists_after_initialize() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            import aiosqlite
            db_path = Path(d) / "mem.db"
            ltm = LongTermMemory(db_path)
            await ltm.initialize()
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute("PRAGMA index_list(memories)")
                indexes = [row[1] for row in await cursor.fetchall()]
            assert "idx_memories_dim_importance" in indexes, (
                f"Expected index idx_memories_dim_importance, got: {indexes}"
            )

    asyncio.run(_run())


def test_long_term_rejects_dimension_mismatch_on_store() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            await ltm.add_memory("seed", 0.0, 1.0, [1.0, 0.0, 0.0])
            try:
                await ltm.add_memory("bad", 0.0, 1.0, [1.0, 0.0])
                assert False, "Expected ValueError for dimension mismatch."
            except ValueError as exc:
                assert "dimension mismatch" in str(exc).lower()

    asyncio.run(_run())
