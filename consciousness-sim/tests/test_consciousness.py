"""Tests for consciousness orchestrator state restore wiring."""

from __future__ import annotations

import asyncio
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

