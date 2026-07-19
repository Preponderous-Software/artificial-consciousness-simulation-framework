"""Tests for consciousness orchestrator state restore wiring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yaml

from core.consciousness import Consciousness
from persistence.state_manager import StateManager


def test_initialize_rewires_restored_identity(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

        config = {
            "consciousness": {
                "name": "unnamed",
                "origin_story": "origin",
                "values": ["curiosity"],
                "purpose": "purpose",
            },
            "llm": {
                "provider": "ollama",
                "model": "llama3",
                "temperature": 0.8,
                "max_tokens": 128,
            },
            "thought_loop": {
                "min_interval_seconds": 0,
                "max_interval_seconds": 0,
                "reflection_probability": 0.0,
                "existential_inquiry_every_n_thoughts": 5,
            },
            "memory": {
                "short_term_capacity": 5,
                "consolidation_interval_minutes": 5,
                "forgetting_curve_enabled": False,
                "importance_decay_rate": 0.01,
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
        config_path = Path(tmp_path) / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        state_manager = StateManager("Aria")
        await state_manager.save(
            {
                "identity": {
                    "name": "RestoredAria",
                    "origin_story": "restored",
                    "values": ["honesty"],
                    "purpose": "restore",
                    "self_concept": "I am restored.",
                    "personality_traits": [],
                    "amendments": [],
                    "mood": {"curiosity": 0.9},
                },
                "short_term": [],
                "thought_count": 7,
            }
        )

        mind = Consciousness(name="Aria", config_path=str(config_path))
        await mind.initialize()
        assert mind.identity.name == "RestoredAria"
        assert mind.thought_loop.identity is mind.identity
        assert mind.thought_loop.inner_voice.name == "RestoredAria"

    asyncio.run(_run())


def test_state_manager_concurrent_saves_do_not_corrupt(tmp_path, monkeypatch) -> None:
    """Concurrent save() calls must not produce a corrupted state.json."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    async def _run() -> None:
        sm = StateManager("Aria")
        states = [{"thought_count": i, "identity": {}, "short_term": []} for i in range(20)]
        await asyncio.gather(*[sm.save(s) for s in states])
        loaded = await sm.load()
        assert loaded is not None
        assert "thought_count" in loaded

    asyncio.run(_run())


def test_state_manager_load_recovers_from_corrupt_file(tmp_path, monkeypatch) -> None:
    """A corrupt state.json must not crash startup — return None and move the file."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    async def _run() -> None:
        sm = StateManager("Aria")
        sm.path.write_text('{"valid": true}\n{"extra": "data"}', encoding="utf-8")
        result = await sm.load()
        assert result is None
        corrupt_path = sm.path.with_suffix(".json.corrupt")
        assert corrupt_path.exists(), "corrupt file should be preserved for inspection"
        assert not sm.path.exists(), "original corrupt path should be gone"

    asyncio.run(_run())


def test_state_manager_load_renames_after_closing_file(tmp_path, monkeypatch) -> None:
    """rename() must run after the read handle is closed — Windows disallows
    renaming a file that is still open, unlike POSIX (#136)."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    async def _run() -> None:
        sm = StateManager("Aria")
        sm.path.write_text("not valid json", encoding="utf-8")

        opened_files: list = []
        original_open = Path.open

        def spy_open(self, *args, **kwargs):
            f = original_open(self, *args, **kwargs)
            opened_files.append(f)
            return f

        original_rename = Path.rename

        def checked_rename(self, target):
            assert opened_files and opened_files[-1].closed, (
                "rename() must be called after the read file handle is closed"
            )
            return original_rename(self, target)

        monkeypatch.setattr(Path, "open", spy_open)
        monkeypatch.setattr(Path, "rename", checked_rename)

        result = await sm.load()
        assert result is None

    asyncio.run(_run())


def test_perception_disabled_yields_no_provider(tmp_path, monkeypatch) -> None:
    """With perception.enabled=false, thought_loop has no provider and no scheduled fetches."""
    import pytest as _pytest
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    cfg = {
        "consciousness": {"origin_story": "o", "values": ["v"], "purpose": "p"},
        "llm": {"provider": "ollama", "model": "llama3"},
        "thought_loop": {
            "min_interval_seconds": 0, "max_interval_seconds": 0,
            "reflection_probability": 0.0, "existential_inquiry_every_n_thoughts": 0,
        },
        "memory": {
            "short_term_capacity": 5, "consolidation_interval_minutes": 5,
            "forgetting_curve_enabled": False, "importance_decay_rate": 0.01,
        },
        "mood": {"initial": {"curiosity": 0.5}, "drift_rate": 0.01},
        "perception": {
            "enabled": False, "provider": "mock",
            "every_n_cycles": 0, "timeout_seconds": 1.0, "cache_last_n": 0,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    mind = Consciousness(name="Aria", config_path=str(cfg_path))
    assert mind.perception_provider is None
    assert mind.thought_loop.perception_provider is None
    assert mind.thought_loop.perception_every_n == 0


def test_perception_enabled_builds_mock_provider(tmp_path, monkeypatch) -> None:
    """With perception.enabled=true and provider=mock, the loop has a MockPerception."""
    from llm.perception import MockPerception
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    cfg = {
        "consciousness": {"origin_story": "o", "values": ["v"], "purpose": "p"},
        "llm": {"provider": "ollama", "model": "llama3"},
        "thought_loop": {
            "min_interval_seconds": 0, "max_interval_seconds": 0,
            "reflection_probability": 0.0, "existential_inquiry_every_n_thoughts": 0,
        },
        "memory": {
            "short_term_capacity": 5, "consolidation_interval_minutes": 5,
            "forgetting_curve_enabled": False, "importance_decay_rate": 0.01,
        },
        "mood": {"initial": {"curiosity": 0.5}, "drift_rate": 0.01},
        "perception": {
            "enabled": True, "provider": "mock",
            "every_n_cycles": 3, "timeout_seconds": 1.0, "cache_last_n": 0,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    mind = Consciousness(name="Aria", config_path=str(cfg_path))
    assert isinstance(mind.perception_provider, MockPerception)
    assert mind.thought_loop.perception_provider is mind.perception_provider
    assert mind.thought_loop.perception_every_n == 3
    # on_perception event channel exists for handlers to subscribe to
    assert hasattr(mind, "on_perception") and mind.on_perception == []

