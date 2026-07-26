"""Long-term semantic memory backed by SQLite with embedding similarity search.

Theory mapping — PP/FEP (Friston 2010) / GWT (Baars 1988): functions as
the prior belief store retrieved to contextualise each thought cycle.
Cosine similarity retrieval partially implements GWT-3 (global broadcast
content drawn from prior experience) and PP (top-down priors shaping
generation). Importance score + forgetting curve implements a simple
form of the FEP relevance-weighted prior; the row-count retention bound
(#135) adds the complementary discard step — low-importance, old priors
are dropped rather than accumulated indefinitely, so the store's size is
independent of run length.
Gap: no true Bayesian update — retrieval is similarity-based, not
posterior-weighted by prediction error. Retention is a hard row cap, not
a decay-driven forgetting process: importance drives *which* rows go, but
the trigger is capacity, not time-since-recall.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np


@dataclass(slots=True)
class LongTermMemoryItem:
    id: int
    timestamp: str
    embedding: list[float]
    summary: str
    emotional_valence: float
    importance_score: float


#: Row cap applied when the config omits ``memory.long_term_max_rows``.
#: At llama3.2:3b's 3072-dim embeddings a row costs ~40 KB of JSON, so this
#: bound plateaus ``memory.db`` at roughly 80 MB. The default consolidation
#: cadence stores ~3 rows every 5 minutes, so an instance reaches the bound
#: after ~2.5 days of continuous running and evicts from then on.
DEFAULT_MAX_ROWS: int = 2000


class LongTermMemory:
    """Persistent memory store with cosine similarity retrieval.

    ``max_rows`` bounds the store: once an insert pushes the row count above
    it, the lowest-importance (tie-broken by oldest) rows are deleted so the
    count returns to the bound (#135). ``max_rows=0`` disables the bound and
    restores the previous unbounded-growth behavior.
    """

    def __init__(self, db_path: Path, max_rows: int = DEFAULT_MAX_ROWS) -> None:
        if max_rows < 0:
            raise ValueError(f"max_rows must be >= 0 (0 disables the bound), got {max_rows}")
        self.db_path = db_path
        self.max_rows = int(max_rows)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    embedding_dim INTEGER,
                    summary TEXT NOT NULL,
                    emotional_valence REAL NOT NULL,
                    importance_score REAL NOT NULL
                )
                """
            )
            cursor = await db.execute("PRAGMA table_info(memories)")
            columns = [str(row[1]) for row in await cursor.fetchall()]
            if "embedding_dim" not in columns:
                await db.execute("ALTER TABLE memories ADD COLUMN embedding_dim INTEGER")
            # Compound index lets ORDER BY importance_score DESC, timestamp DESC LIMIT N
            # use an index scan instead of a full-table sort — avoids O(table) cost each
            # cycle as the LTM grows (issue #72).
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_dim_importance
                ON memories (embedding_dim, importance_score, timestamp)
                """
            )
            # Sweep once at open so a store that grew past the bound under an
            # older config (or before #135) is trimmed immediately, rather than
            # only once the next consolidation pass happens to insert.
            evicted = await self._evict_over_bound(db)
            await db.commit()
        if evicted:
            logging.info(
                "Long-term memory opened above its bound (%d); evicted %d row(s).",
                self.max_rows,
                evicted,
            )

    async def _evict_over_bound(self, db: aiosqlite.Connection) -> int:
        """Delete the lowest-importance/oldest rows above ``max_rows``.

        Runs inside the caller's open connection and transaction; the caller
        commits. Returns the number of rows deleted (0 when unbounded or
        already within the bound).
        """
        if self.max_rows <= 0:
            return 0
        cursor = await db.execute("SELECT COUNT(*) FROM memories")
        row = await cursor.fetchone()
        overflow = (int(row[0]) if row else 0) - self.max_rows
        if overflow <= 0:
            return 0
        # Ordering mirrors the retention priority of similarity_search's
        # candidate selection (importance first, recency as tie-break),
        # reversed: the rows least likely to ever be retrieved go first.
        await db.execute(
            """
            DELETE FROM memories WHERE id IN (
                SELECT id FROM memories
                ORDER BY importance_score ASC, timestamp ASC, id ASC
                LIMIT ?
            )
            """,
            (overflow,),
        )
        return overflow

    async def enforce_retention(self) -> int:
        """Apply the row bound as a standalone sweep. Returns rows deleted.

        ``initialize`` sweeps once at open and ``add_memory`` enforces the bound
        on every insert, so this is only needed when ``max_rows`` is lowered
        mid-run and no further insert is imminent.
        """
        if self.max_rows <= 0:
            return 0
        async with aiosqlite.connect(self.db_path) as db:
            evicted = await self._evict_over_bound(db)
            await db.commit()
        if evicted:
            logging.info(
                "Long-term memory retention sweep evicted %d row(s) (bound=%d).",
                evicted,
                self.max_rows,
            )
        return evicted

    async def add_memory(
        self,
        summary: str,
        emotional_valence: float,
        importance_score: float,
        embedding: list[float],
    ) -> int:
        """Store one memory and enforce the retention bound.

        Returns the new row's id. Note that when the store is at its bound and
        the new row is the least important of all rows, the eviction pass can
        remove the row that was just inserted — the returned id is then no
        longer present, by design: retention keeps the most important rows,
        not the most recent write.
        """
        embedding_dim = len(embedding)
        if embedding_dim == 0:
            raise ValueError("Embedding must not be empty.")

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT embedding_dim FROM memories WHERE embedding_dim IS NOT NULL LIMIT 1"
            )
            existing = await cursor.fetchone()
            if existing and int(existing[0]) != embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {int(existing[0])}, got {embedding_dim}"
                )

            cursor = await db.execute(
                """
                INSERT INTO memories (timestamp, embedding, embedding_dim, summary, emotional_valence, importance_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(embedding),
                    embedding_dim,
                    summary,
                    float(emotional_valence),
                    float(importance_score),
                ),
            )
            new_id = int(cursor.lastrowid)
            evicted = await self._evict_over_bound(db)
            await db.commit()
        if evicted:
            logging.debug(
                "Long-term memory at bound (%d rows); evicted %d low-importance row(s).",
                self.max_rows,
                evicted,
            )
        return new_id

    async def count(self) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM memories")
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    _MAX_CANDIDATES: int = 50

    async def similarity_search(
        self,
        query_embedding: list[float],
        limit: int = 3,
        candidate_limit: int | None = None,
    ) -> list[LongTermMemoryItem]:
        if not query_embedding or limit <= 0:
            return []
        q_dim = len(query_embedding)
        candidate_limit = min(candidate_limit or self._MAX_CANDIDATES, self._MAX_CANDIDATES)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT id, timestamp, embedding, summary, emotional_valence, importance_score
                FROM memories
                WHERE embedding_dim = ? OR embedding_dim IS NULL
                ORDER BY importance_score DESC, timestamp DESC
                LIMIT ?
                """,
                (q_dim, int(candidate_limit)),
            )
            rows = await cursor.fetchall()

        if not rows:
            return []

        q = np.array(query_embedding, dtype=float)
        if q.size == 0:
            return []
        q_norm = np.linalg.norm(q)
        scored: list[tuple[float, LongTermMemoryItem]] = []
        for row in rows:
            try:
                emb_raw: Any = json.loads(row[2])
                if not isinstance(emb_raw, list):
                    continue
                emb = np.array(emb_raw, dtype=float)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if emb.size == 0 or emb.shape != q.shape:
                continue
            denom = (np.linalg.norm(emb) * q_norm) or 1.0
            sim = float(np.dot(emb, q) / denom)
            scored.append(
                (
                    sim,
                    LongTermMemoryItem(
                        id=int(row[0]),
                        timestamp=str(row[1]),
                        embedding=emb.tolist(),
                        summary=str(row[3]),
                        emotional_valence=float(row[4]),
                        importance_score=float(row[5]),
                    ),
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    async def apply_forgetting_curve(self, decay_rate: float) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE memories SET importance_score = MAX(0.0, importance_score - ?)",
                (float(decay_rate),),
            )
            await db.commit()
