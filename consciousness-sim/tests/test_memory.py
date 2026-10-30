"""Tests for memory storage and retrieval behavior."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

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


def test_short_term_newest_thoughts_survive_a_workspace_full_of_reflections() -> None:
    """Regression for #201: with importance-only eviction a buffer of
    reflections/perceptions evicted every new thought on the next add."""
    stm = ShortTermMemory(capacity=20)
    for i in range(13):
        stm.add("reflection", f"reflection {i}")
    for i in range(7):
        stm.add("perception", f"perception {i}")
    for i in range(6):
        stm.add("thought", f"thought {i}")
        stm.add("perception", f"new perception {i}")

    kinds = [item.kind for item in stm.list()]
    thoughts = [item.content for item in stm.list() if item.kind == "thought"]
    assert len(kinds) == 20
    assert thoughts[-3:] == ["thought 3", "thought 4", "thought 5"]
    assert kinds.count("reflection") < 13


def test_short_term_old_reflection_is_outranked_after_one_half_life_per_importance_doubling() -> None:
    stm = ShortTermMemory(capacity=100, half_life=10)
    reflection = stm.add("reflection", "r")
    for i in range(10):
        stm.add("thought", f"t{i}")
    newest = stm.list()[-1]
    assert stm.effective_importance(reflection) == pytest.approx(1.0)
    assert stm.effective_importance(newest) == pytest.approx(1.0)
    stm.add("thought", "one more")
    assert stm.effective_importance(reflection) < stm.effective_importance(stm.list()[-1])


def test_short_term_half_life_zero_restores_importance_only_eviction() -> None:
    stm = ShortTermMemory(capacity=3, half_life=0)
    for i in range(3):
        stm.add("reflection", f"r{i}")
    stm.add("thought", "t")
    assert [item.kind for item in stm.list()] == ["reflection", "reflection", "reflection"]
    assert stm.effective_importance(stm.list()[0]) == 2.0


def test_short_term_equal_scores_evict_oldest_first() -> None:
    stm = ShortTermMemory(capacity=2, half_life=None)
    stm.add("thought", "a")
    stm.add("thought", "b")
    stm.add("thought", "c")
    assert [item.content for item in stm.list()] == ["b", "c"]


def test_short_term_render_keeps_a_full_workspace_of_long_items_within_budget() -> None:
    """Regression for #205: 20 long reflections rendered to ~19k chars, past the
    model's context, and embedding that text timed out."""
    stm = ShortTermMemory(capacity=20, prompt_chars=8000)
    for i in range(20):
        stm.add("reflection", f"reflection {i:02d} " + "x" * 920)

    rendered = stm.render_for_prompt()

    assert len(rendered) <= 8000
    lines = rendered.split("\n")
    assert lines[-1].startswith("- [reflection] reflection 19 ")
    numbers = [int(line.split()[3]) for line in lines]
    assert numbers == sorted(numbers)
    assert numbers[0] > 0


def test_short_term_render_truncates_the_newest_item_if_it_alone_is_over_budget() -> None:
    stm = ShortTermMemory(capacity=5, prompt_chars=100)
    stm.add("thought", "older")
    stm.add("reflection", "y" * 500)

    rendered = stm.render_for_prompt()

    assert rendered.startswith("- [reflection] yyy")
    assert rendered.endswith(" …")
    assert len(rendered) <= 100
    assert "older" not in rendered


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_short_term_render_never_exceeds_a_budget_smaller_than_the_truncation_mark(budget) -> None:
    stm = ShortTermMemory(capacity=5, prompt_chars=budget)
    stm.add("thought", "some content")

    assert len(stm.render_for_prompt()) <= budget


def test_short_term_render_within_budget_is_unchanged() -> None:
    stm = ShortTermMemory(capacity=5, prompt_chars=8000)
    stm.add("thought", "one")
    stm.add("perception", "two")
    assert stm.render_for_prompt() == "- [thought] one\n- [perception] two"


@pytest.mark.parametrize("budget", [None, 0])
def test_short_term_render_budget_disabled_renders_everything(budget) -> None:
    stm = ShortTermMemory(capacity=20, prompt_chars=budget)
    for i in range(20):
        stm.add("reflection", "z" * 1000)
    assert len(stm.render_for_prompt()) > 20_000


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


def test_long_term_latest_timestamp_tracks_newest_insert() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            assert await ltm.latest_timestamp() is None
            await ltm.add_memory("first", 0.0, 5.0, [1.0, 0.0, 0.0])
            first = await ltm.latest_timestamp()
            await ltm.add_memory("second", 0.0, 5.0, [0.0, 1.0, 0.0])
            second = await ltm.latest_timestamp()
            assert first is not None and second is not None
            assert second > first

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


def test_long_term_rejects_empty_embedding_on_store() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            with pytest.raises(ValueError, match="must not be empty"):
                await ltm.add_memory("empty", 0.0, 1.0, [])
            assert await ltm.count() == 0

    asyncio.run(_run())


async def _insert_raw_row(db_path: Path, embedding: str, embedding_dim: int | None, summary: str) -> None:
    """Write a row directly, bypassing add_memory's validation — models a
    pre-``embedding_dim`` legacy row or an on-disk corruption."""
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO memories (timestamp, embedding, embedding_dim, summary, emotional_valence, importance_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("2026-01-01T00:00:00+00:00", embedding, embedding_dim, summary, 0.0, 5.0),
        )
        await db.commit()


def test_long_term_similarity_search_returns_stored_fields() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            new_id = await ltm.add_memory("the tide turns", 0.25, 6.5, [0.0, 1.0, 0.0])
            [hit] = await ltm.similarity_search([0.0, 1.0, 0.0], limit=5)
            assert hit.id == new_id
            assert hit.summary == "the tide turns"
            assert hit.emotional_valence == 0.25
            assert hit.importance_score == 6.5
            assert hit.embedding == [0.0, 1.0, 0.0]
            assert hit.timestamp == await ltm.latest_timestamp()

    asyncio.run(_run())


def test_long_term_similarity_search_ranks_by_cosine_not_importance() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            await ltm.add_memory("important but far", 0.0, 9.0, [1.0, 0.0, 0.0])
            await ltm.add_memory("minor but close", 0.0, 1.0, [0.0, 1.0, 0.1])
            await ltm.add_memory("in between", 0.0, 5.0, [0.5, 0.5, 0.0])
            hits = await ltm.similarity_search([0.0, 1.0, 0.0], limit=3)
            assert [h.summary for h in hits] == ["minor but close", "in between", "important but far"]

    asyncio.run(_run())


@pytest.mark.parametrize("query, limit", [([], 3), ([1.0, 0.0, 0.0], 0), ([1.0, 0.0, 0.0], -1)])
def test_long_term_similarity_search_empty_query_or_nonpositive_limit_returns_nothing(query, limit) -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            await ltm.add_memory("seed", 0.0, 5.0, [1.0, 0.0, 0.0])
            assert await ltm.similarity_search(query, limit=limit) == []

    asyncio.run(_run())


def test_long_term_similarity_search_on_empty_store_returns_nothing() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            assert await ltm.similarity_search([1.0, 0.0, 0.0]) == []

    asyncio.run(_run())


def test_long_term_similarity_search_ignores_rows_of_another_dimension() -> None:
    """A query of a different dimension than the store matches no candidates
    rather than raising on a shape mismatch."""

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            await ltm.add_memory("three-dim", 0.0, 5.0, [1.0, 0.0, 0.0])
            assert await ltm.similarity_search([1.0, 0.0]) == []

    asyncio.run(_run())


def test_long_term_similarity_search_zero_norm_query_does_not_divide_by_zero() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            await ltm.add_memory("a", 0.0, 5.0, [1.0, 0.0, 0.0])
            await ltm.add_memory("b", 0.0, 4.0, [0.0, 1.0, 0.0])
            hits = await ltm.similarity_search([0.0, 0.0, 0.0], limit=5)
            assert {h.summary for h in hits} == {"a", "b"}

    asyncio.run(_run())


def test_long_term_similarity_search_includes_legacy_null_dim_row_of_matching_shape() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "mem.db"
            ltm = LongTermMemory(db_path)
            await ltm.initialize()
            await _insert_raw_row(db_path, "[0.0, 1.0, 0.0]", None, "legacy")
            hits = await ltm.similarity_search([0.0, 1.0, 0.0], limit=5)
            assert [h.summary for h in hits] == ["legacy"]

    asyncio.run(_run())


@pytest.mark.parametrize(
    "raw_embedding",
    ["not json", '{"a": 1}', "[]", "[1.0, 0.0]", '["x", "y", "z"]'],
)
def test_long_term_similarity_search_skips_unusable_stored_embeddings(raw_embedding) -> None:
    """Corrupt, non-list, empty, wrong-shape, or non-numeric embeddings on a
    NULL-dim row are skipped; the valid row beside them is still returned."""

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "mem.db"
            ltm = LongTermMemory(db_path)
            await ltm.initialize()
            await ltm.add_memory("valid", 0.0, 1.0, [0.0, 1.0, 0.0])
            await _insert_raw_row(db_path, raw_embedding, None, "broken")
            hits = await ltm.similarity_search([0.0, 1.0, 0.0], limit=5)
            assert [h.summary for h in hits] == ["valid"]

    asyncio.run(_run())


def test_long_term_similarity_search_scores_only_top_importance_candidates() -> None:
    """candidate_limit restricts scoring to the most important rows, so a
    perfect cosine match outside that window is not returned."""

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            await ltm.add_memory("exact but minor", 0.0, 1.0, [0.0, 1.0, 0.0])
            await ltm.add_memory("major", 0.0, 9.0, [1.0, 0.0, 0.0])
            await ltm.add_memory("middling", 0.0, 5.0, [0.0, 0.0, 1.0])
            hits = await ltm.similarity_search([0.0, 1.0, 0.0], limit=5, candidate_limit=2)
            assert {h.summary for h in hits} == {"major", "middling"}

    asyncio.run(_run())


def test_long_term_similarity_search_candidate_limit_is_capped() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            for i in range(4):
                await ltm.add_memory(f"memory {i}", 0.0, float(i), [float(i), 1.0, 0.0])
            ltm._MAX_CANDIDATES = 2
            hits = await ltm.similarity_search([1.0, 1.0, 0.0], limit=10, candidate_limit=10)
            assert {h.summary for h in hits} == {"memory 2", "memory 3"}

    asyncio.run(_run())


def test_long_term_forgetting_curve_decays_importance_and_floors_at_zero() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            await ltm.add_memory("strong", 0.0, 5.0, [1.0, 0.0, 0.0])
            await ltm.add_memory("weak", 0.0, 0.3, [0.0, 1.0, 0.0])
            await ltm.apply_forgetting_curve(0.5)
            scores = {
                h.summary: h.importance_score
                for h in await ltm.similarity_search([1.0, 1.0, 0.0], limit=10)
            }
            assert scores == {"strong": pytest.approx(4.5), "weak": 0.0}
            assert await ltm.count() == 2

    asyncio.run(_run())


def test_long_term_forgetting_curve_zero_rate_leaves_importance_unchanged() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db")
            await ltm.initialize()
            await ltm.add_memory("kept", 0.0, 3.0, [1.0, 0.0, 0.0])
            await ltm.apply_forgetting_curve(0.0)
            [hit] = await ltm.similarity_search([1.0, 0.0, 0.0])
            assert hit.importance_score == 3.0

    asyncio.run(_run())


def test_long_term_forgetting_curve_shifts_which_rows_eviction_drops() -> None:
    """Decay lowers importance uniformly, so it does not reorder retention on
    its own; a fresh insert above the decayed scores then outranks old rows."""

    async def _run() -> None:
        with tempfile.TemporaryDirectory() as d:
            ltm = LongTermMemory(Path(d) / "mem.db", max_rows=2)
            await ltm.initialize()
            await ltm.add_memory("old high", 0.0, 4.0, [1.0, 0.0, 0.0])
            await ltm.add_memory("old low", 0.0, 3.0, [0.0, 1.0, 0.0])
            await ltm.apply_forgetting_curve(2.0)
            await ltm.add_memory("fresh", 0.0, 2.5, [0.0, 0.0, 1.0])
            kept = {h.summary for h in await ltm.similarity_search([1.0, 1.0, 1.0], limit=10)}
            assert kept == {"fresh", "old high"}

    asyncio.run(_run())
