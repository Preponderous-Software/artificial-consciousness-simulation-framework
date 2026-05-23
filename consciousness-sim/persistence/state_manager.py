"""Persistence manager for saving/loading full consciousness state snapshots.

Theory mapping — AE-1 (agency) / HOT-3 (agentive consumer): StateManager
implements identity persistence across run boundaries — the agent's self-model
(identity, mood, short-term buffer, thought count) survives process restarts.
This is a prerequisite for long-horizon agency: goals and self-concept must
outlive individual sessions to influence future behaviour.
Gap: no direct theory mapping beyond enabling the conditions for agency;
state is inert JSON until loaded and acted upon by the consciousness run loop.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from persistence.paths import consciousness_dir


class StateManager:
    """Persists recoverable state to ~/.consciousness/<name>/state.json."""

    def __init__(self, name: str) -> None:
        self.path = consciousness_dir(name) / "state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def save(self, state: dict[str, Any]) -> None:
        def _write() -> None:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

        await asyncio.to_thread(_write)

    async def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None

        def _read() -> dict[str, Any]:
            with self.path.open("r", encoding="utf-8") as f:
                return dict(json.load(f))

        return await asyncio.to_thread(_read)
