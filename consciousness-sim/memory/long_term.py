"""Long-term semantic memory backed by SQLite with embedding similarity search.

Theory mapping — PP/FEP (Friston 2010) / GWT (Baars 1988): functions as
the prior belief store retrieved to contextualise each thought cycle.
Cosine similarity retrieval partially implements GWT-3 (global broadcast
content drawn from prior experience) and PP (top-down priors shaping
generation). Importance score + forgetting curve implements a simple
form of the FEP relevance-weighted prior.
Gap: no true Bayesian update — retrieval is similarity-based, not
posterior-weighted by prediction error.
"""

from __future__ import annotations

import json
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


class LongTermMemory:
    """Persistent memory store with cosine similarity retrieval."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
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
            await db.commit()

    async def add_memory(
        self,
        summary: str,
        emotional_valence: float,
        importance_score: float,
        embedding: list[float],
    ) -> int:
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
            await db.commit()
            return int(cursor.lastrowid)

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
