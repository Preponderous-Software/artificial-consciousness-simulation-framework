"""Tests for memory storage and retrieval behavior."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from memory.long_term import DEFAULT_MAX_ROWS, LongTermMemory
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


def test_long_term_evicts_lowest_importance_over_bound() -> None:
    """Inserting past max_rows leaves exactly max_rows, keeping the most important."""

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db", max_rows=3)
            await ltm.initialize()
            for i in range(6):
                await ltm.add_memory(f"memory {i}", 0.0, float(i), [float(i), 1.0, 0.0])
            assert await ltm.count() == 3
            kept = {
                item.summary
                for item in await ltm.similarity_search([1.0, 1.0, 1.0], limit=10)
            }
            assert kept == {"memory 3", "memory 4", "memory 5"}, kept

    asyncio.run(_run())


def test_long_term_eviction_breaks_importance_ties_by_age() -> None:
    """Among equal-importance rows the oldest is evicted first."""

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db", max_rows=2)
            await ltm.initialize()
            await ltm.add_memory("oldest", 0.0, 5.0, [1.0, 0.0, 0.0])
            await ltm.add_memory("middle", 0.0, 5.0, [0.0, 1.0, 0.0])
            await ltm.add_memory("newest", 0.0, 5.0, [0.0, 0.0, 1.0])
            kept = {
                item.summary
                for item in await ltm.similarity_search([1.0, 1.0, 1.0], limit=10)
            }
            assert kept == {"middle", "newest"}, kept

    asyncio.run(_run())


def test_long_term_max_rows_zero_is_unbounded() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db", max_rows=0)
            await ltm.initialize()
            for i in range(8):
                await ltm.add_memory(f"memory {i}", 0.0, float(i), [float(i), 1.0, 0.0])
            assert await ltm.count() == 8

    asyncio.run(_run())


def test_long_term_rejects_negative_max_rows() -> None:
    with tempfile.TemporaryDirectory() as d:
        try:
            LongTermMemory(Path(d) / "mem.db", max_rows=-1)
            assert False, "Expected ValueError for negative max_rows."
        except ValueError as exc:
            assert "max_rows" in str(exc)


def test_long_term_initialize_trims_preexisting_overflow() -> None:
    """A store grown unbounded is trimmed on the open that first applies a bound."""

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "mem.db"
            unbounded = LongTermMemory(db_path, max_rows=0)
            await unbounded.initialize()
            for i in range(5):
                await unbounded.add_memory(f"memory {i}", 0.0, float(i), [float(i), 1.0, 0.0])
            assert await unbounded.count() == 5

            bounded = LongTermMemory(db_path, max_rows=2)
            await bounded.initialize()
            assert await bounded.count() == 2
            # The highest-importance rows are the ones kept.
            kept = {
                item.summary
                for item in await bounded.similarity_search([1.0, 1.0, 1.0], limit=10)
            }
            assert kept == {"memory 3", "memory 4"}, kept

    asyncio.run(_run())


def test_long_term_enforce_retention_applies_a_lowered_bound() -> None:
    """The standalone sweep trims when the bound is lowered after open."""

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "mem.db"
            ltm = LongTermMemory(db_path, max_rows=5)
            await ltm.initialize()
            for i in range(5):
                await ltm.add_memory(f"memory {i}", 0.0, float(i), [float(i), 1.0, 0.0])
            assert await ltm.count() == 5

            ltm.max_rows = 2
            assert await ltm.enforce_retention() == 3
            assert await ltm.count() == 2
            # A second sweep is a no-op once inside the bound.
            assert await ltm.enforce_retention() == 0

    asyncio.run(_run())


def test_long_term_enforce_retention_is_noop_when_unbounded() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db", max_rows=0)
            await ltm.initialize()
            for i in range(4):
                await ltm.add_memory(f"memory {i}", 0.0, float(i), [float(i), 1.0, 0.0])
            assert await ltm.enforce_retention() == 0
            assert await ltm.count() == 4

    asyncio.run(_run())


def test_long_term_default_bound_is_applied_without_explicit_max_rows() -> None:
    ltm = LongTermMemory(Path(tempfile.gettempdir()) / "unused-mem.db")
    assert ltm.max_rows == DEFAULT_MAX_ROWS
    assert DEFAULT_MAX_ROWS > 0


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
