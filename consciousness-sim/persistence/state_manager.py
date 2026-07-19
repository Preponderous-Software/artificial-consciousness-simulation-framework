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
import logging
import os
from typing import Any

from persistence.paths import consciousness_dir

logger = logging.getLogger(__name__)


class StateManager:
    """Persists recoverable state to ~/.consciousness/<name>/state.json."""

    def __init__(self, name: str) -> None:
        self.path = consciousness_dir(name) / "state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def save(self, state: dict[str, Any]) -> None:
        # Lock prevents concurrent saves from racing on the same .tmp path,
        # which would interleave writes and corrupt the JSON.
        async with self._lock:
            def _write() -> None:
                # Per-PID temp name avoids collisions if two processes ever
                # share the same CONSCIOUSNESS_HOME directory.
                tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                tmp.replace(self.path)

            await asyncio.to_thread(_write)

    async def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None

        async with self._lock:
            def _read() -> dict[str, Any] | None:
                # read_text() opens, reads, and closes the file before returning,
                # so by the time we get here for the except branch the handle is
                # already closed — rename() must happen after that: Windows raises
                # PermissionError when renaming a file that is still open.
                try:
                    return dict(json.loads(self.path.read_text(encoding="utf-8")))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "state.json is corrupt (%s) — starting fresh; "
                        "corrupt file moved to %s.corrupt",
                        exc,
                        self.path,
                    )
                    self.path.rename(self.path.with_suffix(".json.corrupt"))
                    return None

            return await asyncio.to_thread(_read)
