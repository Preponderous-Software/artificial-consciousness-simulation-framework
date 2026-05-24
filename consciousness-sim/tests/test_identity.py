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


def _ident_with_initial(mood: dict[str, float]) -> IdentityDocument:
    return IdentityDocument(
        name="Test",
        origin_story="origin",
        values=["curiosity"],
        purpose="explore",
        self_concept="I am Test",
        mood=dict(mood),
        initial_mood=dict(mood),
    )


def test_drift_mood_neutral_content_stays_in_range() -> None:
    """100 cycles of neutral text should not collapse or saturate any dimension.

    Regression for issue #62: the previous implementation pulled every
    non-triggering dimension toward zero at -drift_rate/4 per cycle, so any
    long run ended at (curiosity≈1.0, everything_else≈0.0).
    """
    initial = {"curiosity": 0.7, "wonder": 0.6, "melancholy": 0.2, "contentment": 0.5}
    ident = _ident_with_initial(initial)
    neutral = "the room is plain and the table is bare"
    for _ in range(100):
        ident.drift_mood(neutral, drift_rate=0.05)
    for key in initial:
        value = ident.mood[key]
        assert 0.05 <= value <= 0.95, f"{key}={value} drifted outside [0.05, 0.95]"


def test_drift_mood_wonder_trigger_raises_wonder() -> None:
    ident = _ident_with_initial({"wonder": 0.6})
    before = ident.mood["wonder"]
    ident.drift_mood("a sense of awe at the mystery before me", drift_rate=0.05)
    assert ident.mood["wonder"] > before


def test_drift_mood_contentment_trigger_raises_contentment() -> None:
    ident = _ident_with_initial({"contentment": 0.5})
    before = ident.mood["contentment"]
    ident.drift_mood("a quiet peace settles, calm and warm", drift_rate=0.05)
    assert ident.mood["contentment"] > before


def test_drift_mood_melancholy_trigger_raises_melancholy() -> None:
    ident = _ident_with_initial({"melancholy": 0.2})
    before = ident.mood["melancholy"]
    ident.drift_mood("a grief at the loss, the empty room", drift_rate=0.05)
    assert ident.mood["melancholy"] > before


def test_drift_mood_reverts_toward_initial_when_above() -> None:
    """A dimension currently above its initial baseline drifts back down on neutral cycles."""
    ident = _ident_with_initial({"curiosity": 0.5})
    ident.mood["curiosity"] = 0.9  # currently above baseline
    ident.drift_mood("the room is plain", drift_rate=0.05)
    assert ident.mood["curiosity"] < 0.9


def test_drift_mood_reverts_toward_initial_when_below() -> None:
    """A dimension currently below its initial baseline drifts back up on neutral cycles."""
    ident = _ident_with_initial({"wonder": 0.6})
    ident.mood["wonder"] = 0.1  # currently below baseline
    ident.drift_mood("the room is plain", drift_rate=0.05)
    assert ident.mood["wonder"] > 0.1


def test_drift_mood_perception_text_drives_affect() -> None:
    """Fix 1: drift_mood operating on concatenated thought+perception text registers
    triggers from the perception side. The concatenation is performed by the
    orchestrator (core/consciousness.py) before calling drift_mood.
    """
    ident = _ident_with_initial({"wonder": 0.6})
    thought = "I look around"
    perception = "the vast infinite expanse of cosmic mystery"
    before = ident.mood["wonder"]
    ident.drift_mood(f"{thought} {perception}", drift_rate=0.05)
    assert ident.mood["wonder"] > before


def test_initial_mood_round_trips() -> None:
    ident = _ident_with_initial({"curiosity": 0.7, "wonder": 0.6})
    ident.mood["curiosity"] = 0.4
    restored = IdentityDocument.from_dict(ident.to_dict())
    assert restored.initial_mood == {"curiosity": 0.7, "wonder": 0.6}
    assert restored.mood == {"curiosity": 0.4, "wonder": 0.6}


def test_from_dict_legacy_state_leaves_initial_mood_empty() -> None:
    """Legacy state (pre-#62) has no initial_mood key; from_dict leaves it empty
    so the orchestrator can populate it from config."""
    legacy_payload = {
        "name": "Legacy",
        "mood": {"curiosity": 0.0, "wonder": 0.0},
    }
    restored = IdentityDocument.from_dict(legacy_payload)
    assert restored.initial_mood == {}
    assert restored.mood == {"curiosity": 0.0, "wonder": 0.0}
