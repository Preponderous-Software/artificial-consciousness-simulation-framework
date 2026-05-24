"""Tests for identity serialization and persistence behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.identity import AttentionSchema, IdentityDocument
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


def test_attention_schema_update() -> None:
    schema = AttentionSchema()
    schema.update("memory", "identity")
    assert schema.focus == "memory"
    assert schema.theme == "identity"
    assert schema.salience == 1.0
    assert schema.history == ["introspection"]


def test_attention_schema_decay() -> None:
    schema = AttentionSchema(salience=0.5)
    schema.decay(rate=0.1)
    assert abs(schema.salience - 0.4) < 1e-9
    schema.salience = 0.05
    schema.decay(rate=0.1)
    assert schema.salience == 0.0


def test_attention_schema_history_capped() -> None:
    schema = AttentionSchema()
    for i in range(AttentionSchema._MAX_HISTORY + 5):
        schema.update(f"focus{i}", "")
    assert len(schema.history) == AttentionSchema._MAX_HISTORY


def test_attention_schema_render() -> None:
    schema = AttentionSchema(focus="reflection", theme="time", salience=0.8)
    rendered = schema.render()
    assert "reflection" in rendered
    assert "time" in rendered
    assert "0.80" in rendered


def test_attention_schema_render_no_theme() -> None:
    schema = AttentionSchema(focus="introspection", theme="", salience=1.0)
    rendered = schema.render()
    assert "introspection" in rendered
    assert ":" not in rendered.split("(")[0]


def test_identity_round_trip_with_attention() -> None:
    ident = IdentityDocument(
        name="Aria",
        origin_story="origin",
        values=["curiosity"],
        purpose="explore",
        self_concept="I am Aria",
        mood={"curiosity": 0.7},
    )
    ident.attention_schema.update("memory", "language")
    restored = IdentityDocument.from_dict(ident.to_dict())
    assert restored.attention_schema.focus == "memory"
    assert restored.attention_schema.theme == "language"
    assert restored.attention_schema.salience == 1.0


def test_identity_from_dict_missing_attention_schema() -> None:
    """Backward compat: old state.json without attention_schema key."""
    payload = {
        "name": "Aria",
        "origin_story": "",
        "values": ["curiosity"],
        "purpose": "explore",
        "self_concept": "I am Aria",
        "mood": {},
    }
    ident = IdentityDocument.from_dict(payload)
    assert ident.attention_schema.focus == "introspection"
    assert ident.attention_schema.salience == 1.0


def test_anchor_payload_includes_attention_state() -> None:
    ident = IdentityDocument(
        name="Aria",
        origin_story="",
        values=["curiosity"],
        purpose="explore",
        self_concept="I am Aria",
        mood={},
    )
    ident.attention_schema.update("perception", "ocean")
    payload = ident.anchor_payload()
    assert "attention_state" in payload
    assert "perception" in payload["attention_state"]
    assert "ocean" in payload["attention_state"]


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
