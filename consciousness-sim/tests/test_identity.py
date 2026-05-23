"""Tests for identity serialization and persistence behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.identity import IdentityDocument
from persistence.paths import sanitize_consciousness_name
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


def test_state_manager_persists_identity(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
        manager = StateManager("Aria")
        payload = {"identity": {"name": "Aria"}, "short_term": [], "thought_count": 3}
        await manager.save(payload)
        loaded = await manager.load()
        assert loaded is not None
        assert loaded["identity"]["name"] == "Aria"
        assert loaded["thought_count"] == 3

    asyncio.run(_run())


def test_state_manager_sanitizes_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    manager = StateManager("../escape")
    assert manager.path == Path(tmp_path) / "escape" / "state.json"
    assert sanitize_consciousness_name("../escape") == "escape"


def test_apply_amendment_caps_at_max() -> None:
    ident = IdentityDocument(
        name="Test",
        origin_story="origin",
        values=["curiosity"],
        purpose="explore",
        self_concept="I am Test",
        mood={},
    )
    max_amt = IdentityDocument._MAX_AMENDMENTS
    for i in range(max_amt + 5):
        ident.apply_amendment(f"amendment {i}")
    assert len(ident.amendments) == max_amt, "amendments list must not exceed _MAX_AMENDMENTS"
    # Oldest entries should have been dropped; last entry is the most recent.
    assert ident.amendments[-1] == f"amendment {max_amt + 4}"
    assert ident.amendments[0] == f"amendment 5"


def test_apply_amendment_serialization_stays_bounded() -> None:
    ident = IdentityDocument(
        name="Test",
        origin_story="origin",
        values=["curiosity"],
        purpose="explore",
        self_concept="I am Test",
        mood={},
    )
    for i in range(50):
        ident.apply_amendment(f"update {i}")
    serialized = ident.to_dict()
    assert len(serialized["amendments"]) == IdentityDocument._MAX_AMENDMENTS
