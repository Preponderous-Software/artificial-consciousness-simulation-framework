"""Tests for identity serialization and persistence behavior."""

from __future__ import annotations

import asyncio

from core.identity import IdentityDocument
from persistence.state_manager import StateManager


def test_identity_round_trip() -> None:
    ident = IdentityDocument(
        name="Aria",
        origin_story="origin",
        values=["curiosity"],
        purpose="explore",
        self_concept="I am Aria",
        personality_traits=["calm"],
        mood={"curiosity": 0.7},
    )
    restored = IdentityDocument.from_dict(ident.to_dict())
    assert restored.name == "Aria"
    assert restored.mood["curiosity"] == 0.7


def test_state_manager_persists_identity(tmp_path) -> None:
    async def _run() -> None:
        import os

        os.environ["CONSCIOUSNESS_HOME"] = str(tmp_path)
        manager = StateManager("Aria")
        payload = {"identity": {"name": "Aria"}, "short_term": [], "thought_count": 3}
        await manager.save(payload)
        loaded = await manager.load()
        assert loaded is not None
        assert loaded["identity"]["name"] == "Aria"
        assert loaded["thought_count"] == 3

    asyncio.run(_run())
