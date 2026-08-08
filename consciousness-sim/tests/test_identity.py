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


def test_apply_amendment_skips_verbatim_repeat() -> None:
    """Regression for #133: feeding the same amendment N times must yield it
    at most once in self_concept, instead of accreting N copies."""
    ident = IdentityDocument(
        name="Offline",
        origin_story="origin",
        values=["curiosity"],
        purpose="explore",
        self_concept="I am Offline, an emergent mind in process.",
        mood={},
    )
    for _ in range(8):
        ident.apply_amendment("I remain Offline")
    assert ident.self_concept.count("I remain Offline") == 1
    assert ident.amendments.count("I remain Offline") == 1


def test_apply_amendment_skips_case_insensitive_repeat() -> None:
    """The observed Offline run mixed casing ('I remain offline' vs 'I remain
    Offline') across repeats — dedup must be case-insensitive."""
    ident = IdentityDocument(
        name="Offline",
        origin_story="origin",
        values=["curiosity"],
        purpose="explore",
        self_concept="I am Offline.",
        mood={},
    )
    ident.apply_amendment("I remain offline")
    ident.apply_amendment("I remain Offline")
    ident.apply_amendment("I REMAIN OFFLINE")
    assert len(ident.amendments) == 1
    assert ident.self_concept.lower().count("i remain offline") == 1


def test_apply_amendment_skips_near_duplicate_outside_self_concept_tail() -> None:
    """A near-duplicate (high token overlap) of a recent amendment is skipped
    even when self_concept truncation has already dropped the original text
    from view (issue #133's substring check alone would miss this case)."""
    ident = IdentityDocument(
        name="Test",
        origin_story="origin",
        values=["curiosity"],
        purpose="explore",
        self_concept="x" * (IdentityDocument._MAX_SELF_CONCEPT_LEN - 5),
        mood={},
    )
    ident.apply_amendment("I remain quiet and still")
    before = len(ident.amendments)
    ident.apply_amendment("I remain quiet and still today")
    assert len(ident.amendments) == before, "near-duplicate amendment must not be appended"


def test_apply_amendment_distinct_amendments_still_accumulate() -> None:
    """No regression to legitimate growth: genuinely distinct amendments
    keep accumulating in both amendments and self_concept."""
    ident = IdentityDocument(
        name="Test",
        origin_story="origin",
        values=["curiosity"],
        purpose="explore",
        self_concept="I am Test",
        mood={},
    )
    phrases = [
        "I value curiosity",
        "I seek connection with others",
        "I question the nature of memory",
    ]
    for phrase in phrases:
        ident.apply_amendment(phrase)
    assert ident.amendments == phrases
    for phrase in phrases:
        assert phrase in ident.self_concept


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


# --- #119 mood homeostasis -------------------------------------------------


def test_drift_mood_homeostasis_caps_continuous_reinforcement() -> None:
    """Regression for #119: dominant trait must plateau below 1.0 under
    continuous reinforcement, instead of saturating at the ceiling.

    The James run (2026-05-25) showed curiosity drifting to 0.997 in
    ~34 cycles because the "?" trigger fired every cycle and there was
    no opposing homeostatic pull. With additive homeostasis at 0.1,
    the equilibrium is `initial + drift_rate / homeostasis_rate`
    = 0.7 + 0.05/0.1 = 1.2, clipped — but the approach is exponential,
    so the trait reaches 1.0 only asymptotically; over hundreds of
    cycles with a less-than-maximal drift, the equilibrium for
    continuously-triggered traits should be strictly below 1.0 for
    realistic parameters. We pick parameters that exercise the
    equilibrium: drift_rate=0.05, homeostasis_rate=0.2 → eq=0.95.
    """
    ident = _ident_with_initial({"curiosity": 0.7})
    trigger = "what would I wonder about? a question."  # hits "wonder", "?", "question"
    for _ in range(500):
        ident.drift_mood(trigger, drift_rate=0.05, homeostasis_rate=0.2)
    # Equilibrium 0.7 + 0.05/0.2 = 0.95; allow ±0.01 tolerance.
    assert 0.93 <= ident.mood["curiosity"] <= 0.97, (
        f"curiosity should equilibrate near 0.95 under continuous trigger "
        f"with homeostasis_rate=0.2; got {ident.mood['curiosity']}"
    )


def test_drift_mood_homeostasis_keeps_dominant_trait_below_saturation_short_run() -> None:
    """A short run that the prior implementation saturated must now stay
    well below 1.0."""
    ident = _ident_with_initial({"curiosity": 0.7})
    trigger = "what would I explore? curious wonder."
    # Old behaviour: ~6 triggered cycles took curiosity from 0.7 to 1.0
    # (0.7 + 6 * 0.05 = 1.0). With homeostasis, the same 6 cycles should
    # not reach 1.0.
    for _ in range(6):
        ident.drift_mood(trigger, drift_rate=0.05)
    assert ident.mood["curiosity"] < 1.0


def test_drift_mood_homeostasis_rate_override() -> None:
    """Caller-supplied homeostasis_rate overrides the default."""
    ident = _ident_with_initial({"curiosity": 0.7})
    # zero homeostasis → unbounded drift toward 1.0 (old behaviour)
    trigger = "what if?"
    for _ in range(10):
        ident.drift_mood(trigger, drift_rate=0.05, homeostasis_rate=0.0)
    assert abs(ident.mood["curiosity"] - 1.0) < 1e-6  # saturated


def test_drift_mood_default_homeostasis_keeps_high_baseline_trait_below_ceiling() -> None:
    """Regression for #134: at the *default* homeostasis_rate (no override),
    curiosity's 0.7 baseline must equilibrate strictly below 1.0 under
    continuous reinforcement — at rate=0.1 this saturated at 1.0 (the exact
    failure #119 was meant to prevent); at the new default the equilibrium
    is 0.7 + 0.05/0.3 ≈ 0.867.
    """
    ident = _ident_with_initial({"curiosity": 0.7})
    trigger = "what would I wonder about? a question."  # hits "wonder", "?", "question"
    for _ in range(500):
        ident.drift_mood(trigger, drift_rate=0.05)  # homeostasis_rate defaults
    assert ident.mood["curiosity"] <= 0.9, (
        f"curiosity should equilibrate below the 0.9 acceptance bound at the "
        f"default homeostasis_rate; got {ident.mood['curiosity']}"
    )
    assert 0.85 <= ident.mood["curiosity"] <= 0.9


# --- #120 AttentionSchema.decay_only --------------------------------------


def test_attention_schema_decay_only_smoothly_drops_salience() -> None:
    """10 consecutive 'skipped cycles' must produce smooth, monotonic decay
    rather than a freeze-then-collapse."""
    schema = AttentionSchema(focus="memory", theme="time", salience=1.0)
    prior = 1.0
    for _ in range(10):
        schema.decay_only()
        # Each step must reduce salience and stay non-negative.
        assert schema.salience <= prior
        assert schema.salience >= 0.0
        prior = schema.salience
    # After 10 decay() calls at default rate 0.1, salience reaches floor.
    # Accumulated FP error keeps it just barely positive (~1e-16) — the
    # max(0, ...) clamp only fires when subtraction goes strictly negative.
    assert schema.salience < 1e-9


def test_attention_schema_decay_only_preserves_focus_and_theme() -> None:
    """decay_only must not touch focus or theme — only salience changes."""
    schema = AttentionSchema(focus="memory", theme="time", salience=0.8)
    schema.decay_only()
    assert schema.focus == "memory"
    assert schema.theme == "time"
    assert abs(schema.salience - 0.7) < 1e-9


def test_attention_schema_decay_only_is_aliased_to_decay() -> None:
    """Verify decay_only is behaviourally equivalent to decay so existing
    callers can switch incrementally."""
    a = AttentionSchema(salience=0.6)
    b = AttentionSchema(salience=0.6)
    a.decay(rate=0.15)
    b.decay_only(rate=0.15)
    assert a.salience == b.salience


def test_attention_schema_failure_then_success_focus_updates_normally() -> None:
    """After a long failure burst that decays salience to 0, the next
    successful update() must restore salience to 1.0 with the new focus."""
    schema = AttentionSchema(focus="reflection", theme="pause", salience=1.0)
    for _ in range(15):
        schema.decay_only()
    assert schema.salience == 0.0
    assert schema.focus == "reflection"  # focus preserved through decay burst
    schema.update("memory", "river")
    assert schema.focus == "memory"
    assert schema.theme == "river"
    assert schema.salience == 1.0


# --- #11 loose-payload narrowing in from_dict ------------------------------


def test_from_dict_malformed_mood_falls_back_to_empty() -> None:
    """A corrupt state.json whose `mood` is not a mapping must not crash the
    restore path; the orchestrator repopulates mood from config when empty.
    Mirrors the narrowing already applied to `attention_schema`."""
    restored = IdentityDocument.from_dict(
        {"name": "Corrupt", "mood": "not-a-mapping", "initial_mood": 7}
    )
    assert restored.mood == {}
    assert restored.initial_mood == {}
    assert restored.name == "Corrupt"


def test_from_dict_malformed_string_lists_fall_back_to_empty() -> None:
    """`values`/`personality_traits`/`amendments` stored as non-sequences are
    dropped rather than raising a TypeError during restore."""
    restored = IdentityDocument.from_dict(
        {
            "name": "Corrupt",
            "values": 42,
            "personality_traits": None,
            "amendments": 1.5,
        }
    )
    assert restored.values == []
    assert restored.personality_traits == []
    assert restored.amendments == []


def test_from_dict_coerces_well_formed_sequences_and_mappings() -> None:
    """Well-formed payloads keep their existing coercion behavior: sequences
    become lists of str, and numeric mood values become floats."""
    restored = IdentityDocument.from_dict(
        {
            "name": "Aria",
            "values": ("curiosity", 3),
            "amendments": ["grew"],
            "mood": {"curiosity": 1, "wonder": "0.25"},
        }
    )
    assert restored.values == ["curiosity", "3"]
    assert restored.amendments == ["grew"]
    assert restored.mood == {"curiosity": 1.0, "wonder": 0.25}
