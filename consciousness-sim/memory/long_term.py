"""Long-term semantic memory backed by SQLite with embedding similarity search."""

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
                    summary TEXT NOT NULL,
                    emotional_valence REAL NOT NULL,
                    importance_score REAL NOT NULL
                )
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
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO memories (timestamp, embedding, summary, emotional_valence, importance_score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(embedding),
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

    async def similarity_search(
        self,
        query_embedding: list[float],
        limit: int = 3,
        candidate_limit: int | None = None,
    ) -> list[LongTermMemoryItem]:
        if not query_embedding or limit <= 0:
            return []
        candidate_limit = candidate_limit or max(200, limit * 50)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT id, timestamp, embedding, summary, emotional_valence, importance_score
                FROM memories
                ORDER BY importance_score DESC, timestamp DESC
                LIMIT ?
                """,
                (int(candidate_limit),),
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
