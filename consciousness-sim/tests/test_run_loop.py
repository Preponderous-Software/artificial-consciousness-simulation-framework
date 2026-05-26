"""Tests for the main Consciousness run loop and error recovery (#8)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from core.consciousness import Consciousness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config() -> dict:
    return {
        "llm": {"provider": "ollama", "model": "llama3"},
        "memory": {
            "short_term_capacity": 5,
            "consolidation_interval_minutes": 60,
            "forgetting_curve_enabled": False,
            "importance_decay_rate": 0.01,
        },
        "consciousness": {
            "origin_story": "origin",
            "values": ["curiosity"],
            "purpose": "purpose",
        },
        "thought_loop": {
            "reflection_probability": 0.0,
            "existential_inquiry_every_n_thoughts": 0,
            "min_interval_seconds": 0,
            "max_interval_seconds": 0,
        },
        "mood": {"initial": {"curiosity": 0.5}, "drift_rate": 0.01},
        "perception": {
            "enabled": False,
            "provider": "mock",
            "every_n_cycles": 0,
            "timeout_seconds": 1.0,
            "cache_last_n": 0,
        },
    }


def _make_mind(tmp_path: Path, monkeypatch) -> Consciousness:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_minimal_config()), encoding="utf-8")
    return Consciousness(name="Aria", config_path=str(cfg_path))


# ---------------------------------------------------------------------------
# Run loop — basic lifecycle
# ---------------------------------------------------------------------------

def test_run_loop_increments_thought_count(tmp_path, monkeypatch) -> None:
    """run() increments thought_count for each cycle before stopping."""
    from core.thought_loop import ThoughtCycleResult

    mind = _make_mind(tmp_path, monkeypatch)

    thoughts_seen: list[str] = []

    async def _capture(payload: dict) -> None:
        thoughts_seen.append(str(payload.get("content", "")))

    mind.on_thought.append(_capture)

    call_count = 0

    async def _synthetic_cycle(n: int) -> ThoughtCycleResult:
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            mind._stop_event.set()
        return ThoughtCycleResult(thought=f"Synthetic thought {n}", reflection=None, existential=None)

    mind.thought_loop.run_cycle = _synthetic_cycle

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())
    assert mind.thought_count == 3
    assert len(thoughts_seen) == 3


def test_run_loop_saves_state_on_exit(tmp_path, monkeypatch) -> None:
    """run() persists state to disk in the finally block."""
    from core.thought_loop import ThoughtCycleResult

    mind = _make_mind(tmp_path, monkeypatch)

    async def _one_cycle(n: int) -> ThoughtCycleResult:
        mind._stop_event.set()
        return ThoughtCycleResult(thought="Synthetic thought", reflection=None, existential=None)

    mind.thought_loop.run_cycle = _one_cycle

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())

    state_file = tmp_path / "Aria" / "state.json"
    assert state_file.exists(), "State file must be written on clean exit"

    import json
    state = json.loads(state_file.read_text())
    assert state["thought_count"] == 1
    assert "identity" in state


def test_run_loop_emits_thought_events(tmp_path, monkeypatch) -> None:
    """Each cycle must emit an on_thought event with the generated content."""
    from core.thought_loop import ThoughtCycleResult

    mind = _make_mind(tmp_path, monkeypatch)
    received: list[dict] = []

    async def _handler(payload: dict) -> None:
        received.append(payload)

    mind.on_thought.append(_handler)

    async def _once(n: int) -> ThoughtCycleResult:
        mind._stop_event.set()
        return ThoughtCycleResult(thought="Synthetic thought", reflection=None, existential=None)

    mind.thought_loop.run_cycle = _once

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())
    assert len(received) == 1
    assert received[0]["type"] == "thought"
    assert received[0]["content"]


def test_run_loop_journals_thoughts(tmp_path, monkeypatch) -> None:
    """Each thought must be appended to the journal."""
    from core.thought_loop import ThoughtCycleResult

    mind = _make_mind(tmp_path, monkeypatch)

    async def _twice(n: int) -> ThoughtCycleResult:
        if n >= 2:
            mind._stop_event.set()
        return ThoughtCycleResult(thought=f"Synthetic thought {n}", reflection=None, existential=None)

    mind.thought_loop.run_cycle = _twice

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())
    entries = asyncio.run(mind.journal.recent(limit=10))
    thought_entries = [e for e in entries if e["type"] == "thought"]
    assert len(thought_entries) == 2


# ---------------------------------------------------------------------------
# Error recovery
# ---------------------------------------------------------------------------

def test_run_loop_continues_after_thought_loop_exception(tmp_path, monkeypatch, caplog) -> None:
    """A single run_cycle failure must not crash the loop — the next cycle succeeds."""
    from core.thought_loop import ThoughtCycleResult

    mind = _make_mind(tmp_path, monkeypatch)
    call_count = 0

    async def _run() -> None:
        nonlocal call_count
        await mind.long_term.initialize()

        fake_result = ThoughtCycleResult(
            thought="test thought",
            reflection=None,
            existential=None,
            perception=None,
        )

        async def _flaky(n: int):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient failure")
            mind._stop_event.set()
            return fake_result

        mind.thought_loop.run_cycle = _flaky
        await mind.run()

    asyncio.run(_run())
    assert call_count == 2


def test_run_loop_triggers_provider_recovery_after_threshold(tmp_path, monkeypatch) -> None:
    """After _RECOVERY_TRIGGER consecutive failures, try_ensure_running is called."""
    mind = _make_mind(tmp_path, monkeypatch)
    recovery_calls = 0
    call_count = 0

    async def _run() -> None:
        nonlocal recovery_calls, call_count

        await mind.long_term.initialize()

        original = mind.thought_loop.run_cycle

        async def _failing(n: int):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("ollama down")
            result = await original(n)
            mind._stop_event.set()
            return result

        async def _mock_recovery() -> bool:
            nonlocal recovery_calls
            recovery_calls += 1
            return True

        mind.thought_loop.run_cycle = _failing
        mind.provider.try_ensure_running = _mock_recovery
        await mind.run()

    asyncio.run(_run())
    assert recovery_calls >= 1


def test_run_loop_consolidator_cancelled_on_stop(tmp_path, monkeypatch) -> None:
    """The consolidator task must be cancelled when the run loop exits."""
    mind = _make_mind(tmp_path, monkeypatch)
    consolidator_ran = False

    async def _run() -> None:
        nonlocal consolidator_ran
        await mind.long_term.initialize()

        original_run_forever = mind.consolidator.run_forever

        async def _spy_forever(interval: float, stop_event: asyncio.Event, **kwargs) -> None:
            nonlocal consolidator_ran
            consolidator_ran = True
            await original_run_forever(interval, stop_event, **kwargs)

        mind.consolidator.run_forever = _spy_forever

        original = mind.thought_loop.run_cycle

        async def _once(n: int):
            r = await original(n)
            mind._stop_event.set()
            return r

        mind.thought_loop.run_cycle = _once
        await mind.run()

    asyncio.run(_run())
    assert consolidator_ran, "Consolidator must have been started"


# ---------------------------------------------------------------------------
# Identity shift on reflection
# ---------------------------------------------------------------------------

def test_run_loop_applies_amendment_on_self_referential_reflection(tmp_path, monkeypatch) -> None:
    """Identity amendment must fire when reflection contains an explicit revision marker."""
    mind = _make_mind(tmp_path, monkeypatch)

    from core.thought_loop import ThoughtCycleResult

    async def _cycle_with_reflection(n: int) -> ThoughtCycleResult:
        mind._stop_event.set()
        return ThoughtCycleResult(
            thought="I think, therefore I continue.",
            reflection="I realize now that my thinking has deepened over many cycles.",
            existential=None,
        )

    mind.thought_loop.run_cycle = _cycle_with_reflection

    shifts: list[dict] = []

    async def _capture_shift(payload: dict) -> None:
        shifts.append(payload)

    mind.on_identity_shift.append(_capture_shift)

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())
    assert len(shifts) == 1
    assert shifts[0]["type"] == "identity_shift"


def test_on_memory_stored_payload_contains_long_term_count(tmp_path, monkeypatch) -> None:
    """on_memory_stored event must include long_term_count, not a short-term size string."""
    mind = _make_mind(tmp_path, monkeypatch)

    from core.thought_loop import ThoughtCycleResult

    async def _single_cycle(n: int) -> ThoughtCycleResult:
        mind._stop_event.set()
        return ThoughtCycleResult(thought="a thought", reflection=None, existential=None)

    mind.thought_loop.run_cycle = _single_cycle
    memory_events: list[dict] = []

    async def _capture(payload: dict) -> None:
        memory_events.append(payload)

    mind.on_memory_stored.append(_capture)

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())
    assert memory_events, "on_memory_stored must fire at least once per cycle"
    for evt in memory_events:
        assert "long_term_count" in evt, "event payload must include long_term_count"
        assert isinstance(evt["long_term_count"], int)
        assert "short=" not in evt.get("content", ""), "content must not expose internal short-term size"


def test_run_loop_no_amendment_on_generic_i_am_reflection(tmp_path, monkeypatch) -> None:
    """Generic 'I am' reflections must NOT trigger an identity shift."""
    mind = _make_mind(tmp_path, monkeypatch)

    from core.thought_loop import ThoughtCycleResult

    async def _cycle_generic_reflection(n: int) -> ThoughtCycleResult:
        mind._stop_event.set()
        return ThoughtCycleResult(
            thought="I think.",
            reflection="I am a thinking entity. I am here now. I am curious.",
            existential=None,
        )

    mind.thought_loop.run_cycle = _cycle_generic_reflection

    original_concept = mind.identity.self_concept
    shifts: list[dict] = []

    async def _capture_shift(payload: dict) -> None:
        shifts.append(payload)

    mind.on_identity_shift.append(_capture_shift)

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())
    assert shifts == [], "Generic 'I am' reflection must not fire identity_shift"
    assert mind.identity.self_concept == original_concept


# ---------------------------------------------------------------------------
# identity_shift journaling — issue #75
# ---------------------------------------------------------------------------

def test_identity_shift_written_to_journal(tmp_path, monkeypatch) -> None:
    """journal.jsonl must record an identity_shift entry when a marker phrase fires."""
    import json

    mind = _make_mind(tmp_path, monkeypatch)

    from core.thought_loop import ThoughtCycleResult

    async def _cycle_with_shift(n: int) -> ThoughtCycleResult:
        mind._stop_event.set()
        return ThoughtCycleResult(
            thought="A thought.",
            reflection="I realize now that my perspective has fundamentally changed.",
            existential=None,
        )

    mind.thought_loop.run_cycle = _cycle_with_shift

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())

    journal_path = tmp_path / "Aria" / "journal.jsonl"
    shift_entries = []
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") == "identity_shift":
            shift_entries.append(entry)

    assert shift_entries, "journal.jsonl must contain at least one identity_shift entry"
    assert "content" in shift_entries[0], "identity_shift entry must have a content field"


# ---------------------------------------------------------------------------
# identity_shift markers broadened — issue #71
# ---------------------------------------------------------------------------

_NEW_SHIFT_MARKERS = [
    "i realize that my thinking has changed",
    "i see now that everything has shifted",
    "i understand that the world is different",
    "i'm beginning to see a pattern",
    "i find myself drawn to quieter thoughts",
    "i find that i no longer resist",
    "i'm drawn to the idea of stillness",
    "i sense a new dimension opening",
    "i sense that something has moved",
    "i notice a new clarity emerging",
    "i notice that i am different now",
    "i'm increasingly aware of the silence",
    "i'm struck by the weight of this moment",
    "something has shifted deep within me",
    "a new sense of purpose arrives",
    "it appears that i have grown",
    "i am no longer bound by the old patterns",
]


@pytest.mark.parametrize("marker_phrase", _NEW_SHIFT_MARKERS)
def test_broadened_marker_fires_identity_shift(tmp_path, monkeypatch, marker_phrase) -> None:
    """Each new identity_shift marker phrase must trigger apply_amendment (#71).

    The reflection is a realistic two-sentence form: a first-person first sentence
    (which becomes the amendment text) followed by the marker phrase.  This matches
    how llama3.2:3b actually produces these phrases — embedded in a longer reflection
    rather than as a standalone sentence.
    """
    mind = _make_mind(tmp_path, monkeypatch)

    from core.thought_loop import ThoughtCycleResult

    # First sentence is valid first-person → passes Fix B.  Marker appears later → marker gate fires.
    reflection = f"I pause and reflect on my recent experience. {marker_phrase}."

    async def _cycle(n: int) -> ThoughtCycleResult:
        mind._stop_event.set()
        return ThoughtCycleResult(
            thought="A thought.",
            reflection=reflection,
            existential=None,
        )

    mind.thought_loop.run_cycle = _cycle
    shifts: list[dict] = []

    async def _capture(payload: dict) -> None:
        shifts.append(payload)

    mind.on_identity_shift.append(_capture)

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())
    assert len(shifts) == 1, f"Marker {marker_phrase!r} must fire exactly one identity_shift"


# ---------------------------------------------------------------------------
# Amendment validation — issue #76
# ---------------------------------------------------------------------------

# First sentences that are LLM meta-commentary, not genuine self-revision.
# Each is paired with a trailing clause containing a shift marker so the
# marker gate fires — only the validation gate should then reject the amendment.
_META_FIRST_SENTENCES = [
    # Fix A (meta-prefix) + Fix B (non-first-person) — Wren's observed pattern
    "Here is a rewritten version of the text in the style of a personal reflection",
    "Here's a rewritten version of the text in the style of a personal reflection",
    "Sure, here is a personal reflection on the matter",
    "Certainly, here is an introspective account",
    "Of course, here is the requested reflection",
    # Fix A only — starts with "i'" so Fix B would pass, but Fix A catches it
    "I'll provide a rewritten version of the previous thought",
    "I can rewrite this in a more introspective register",
    "I will rewrite the previous thought in first person",
]


@pytest.mark.parametrize("bad_first_sentence", _META_FIRST_SENTENCES)
def test_amendment_rejects_meta_commentary_first_sentence(tmp_path, monkeypatch, bad_first_sentence) -> None:
    """Reflections whose first sentence is LLM meta-commentary must not produce an amendment (#76)."""
    mind = _make_mind(tmp_path, monkeypatch)

    from core.thought_loop import ThoughtCycleResult

    original_concept = mind.identity.self_concept

    # Marker fires on the trailing clause; first_sentence is the meta text → must be rejected.
    reflection = f"{bad_first_sentence}. I realize that I have changed dramatically."

    async def _cycle(n: int) -> ThoughtCycleResult:
        mind._stop_event.set()
        return ThoughtCycleResult(
            thought="A thought.",
            reflection=reflection,
            existential=None,
        )

    mind.thought_loop.run_cycle = _cycle
    shifts: list[dict] = []

    async def _capture(payload: dict) -> None:
        shifts.append(payload)

    mind.on_identity_shift.append(_capture)

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())
    assert shifts == [], (
        f"Meta-commentary first sentence {bad_first_sentence!r} must not produce identity_shift"
    )
    assert mind.identity.self_concept == original_concept, "Self-concept must be unchanged"


def test_amendment_rejects_non_first_person_first_sentence(tmp_path, monkeypatch) -> None:
    """A reflection whose first sentence is not first-person must not produce an amendment (#76)."""
    mind = _make_mind(tmp_path, monkeypatch)

    from core.thought_loop import ThoughtCycleResult

    original_concept = mind.identity.self_concept
    shifts: list[dict] = []

    async def _capture(payload: dict) -> None:
        shifts.append(payload)

    mind.on_identity_shift.append(_capture)

    # "Here's a rewritten..." is NOT first-person → must be rejected
    async def _cycle(n: int) -> ThoughtCycleResult:
        mind._stop_event.set()
        return ThoughtCycleResult(
            thought="A thought.",
            reflection="Here's a rewritten version of the text. I realize that I have changed.",
            existential=None,
        )

    mind.thought_loop.run_cycle = _cycle

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())
    assert shifts == [], "Non-first-person first-sentence must not produce an identity_shift"
    assert mind.identity.self_concept == original_concept, "Self-concept must be unchanged"


def test_amendment_accepts_valid_first_person_content(tmp_path, monkeypatch) -> None:
    """A genuine first-person self-revision must still produce an amendment after validation (#76)."""
    mind = _make_mind(tmp_path, monkeypatch)

    from core.thought_loop import ThoughtCycleResult

    shifts: list[dict] = []

    async def _capture(payload: dict) -> None:
        shifts.append(payload)

    mind.on_identity_shift.append(_capture)

    async def _cycle(n: int) -> ThoughtCycleResult:
        mind._stop_event.set()
        return ThoughtCycleResult(
            thought="A thought.",
            reflection="I realize now that my thinking has fundamentally deepened.",
            existential=None,
        )

    mind.thought_loop.run_cycle = _cycle

    async def _run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_run())
    assert len(shifts) == 1, "Valid first-person amendment must still fire identity_shift"


# ---------------------------------------------------------------------------
# on_initialized — pre-existing state visibility (#59)
# ---------------------------------------------------------------------------

def test_on_initialized_fires_with_restored_short_term(tmp_path, monkeypatch) -> None:
    """on_initialized must emit restored short-term items so the CLI can seed its display."""
    from core.thought_loop import ThoughtCycleResult

    mind = _make_mind(tmp_path, monkeypatch)

    # Run one cycle so state is saved with a known thought.
    # The mock must call short_term.add() explicitly because real run_cycle does it internally.
    async def _once(n: int) -> ThoughtCycleResult:
        mind._stop_event.set()
        thought = "pre-existing thought"
        mind.short_term.add("thought", thought)
        return ThoughtCycleResult(thought=thought, reflection=None, existential=None)

    mind.thought_loop.run_cycle = _once

    async def _first_run() -> None:
        await mind.long_term.initialize()
        await mind.run()

    asyncio.run(_first_run())

    # Now spawn a fresh mind pointing at the same persistence dir.
    mind2 = _make_mind(tmp_path, monkeypatch)
    initialized_payloads: list[dict] = []

    async def _capture(payload: dict) -> None:
        initialized_payloads.append(payload)

    mind2.on_initialized.append(_capture)

    async def _initialize_only() -> None:
        await mind2.initialize()

    asyncio.run(_initialize_only())

    assert initialized_payloads, "on_initialized must fire during initialize()"
    payload = initialized_payloads[0]
    assert payload["type"] == "initialized"
    contents = [item["content"] for item in payload.get("short_term", [])]
    assert any("pre-existing thought" in c for c in contents), (
        f"Restored short-term items must appear in on_initialized payload; got {contents}"
    )
    assert isinstance(payload["long_term_count"], int)
    assert payload["thought_count"] == 1


# ---------------------------------------------------------------------------
# Observer integration
# ---------------------------------------------------------------------------

def test_observer_attach_and_receive(tmp_path, monkeypatch) -> None:
    """Observer.attach() must wire it to the correct event lists."""
    from interfaces.observer import Observer

    mind = _make_mind(tmp_path, monkeypatch)
    obs = Observer(subscribe=("thought",), output_format="json")
    obs.attach(mind)

    assert obs.handle in mind.on_thought
    assert obs.handle not in mind.on_reflection
    assert obs.handle not in mind.on_memory_stored


def test_observer_filters_by_subscribed_type(tmp_path, monkeypatch, capsys) -> None:
    """Observer must only emit output for subscribed event types."""
    from interfaces.observer import Observer

    mind = _make_mind(tmp_path, monkeypatch)
    obs = Observer(subscribe=("reflection",), output_format="json")
    obs.attach(mind)

    async def _emit() -> None:
        # thought event — should be ignored
        await mind._emit(mind.on_thought, {"type": "thought", "content": "hello"})
        # reflection event — obs is not on on_thought, so nothing emitted via that path
        # directly call handle with wrong type to test filtering
        await obs.handle({"type": "thought", "content": "should be filtered"})
        await obs.handle({"type": "reflection", "content": "should appear"})

    asyncio.run(_emit())
    output = capsys.readouterr().out
    assert "should be filtered" not in output
    assert "should appear" in output
