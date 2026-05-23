"""Passive read-only observer stream for following consciousness events."""

from __future__ import annotations

import json
from typing import Literal

from rich.console import Console

EventType = Literal["thought", "reflection", "memory", "identity_shift"]
_ALL_EVENTS: tuple[EventType, ...] = ("thought", "reflection", "memory", "identity_shift")


class Observer:
    """Subscribes to selected consciousness event types and emits structured output."""

    def __init__(
        self,
        subscribe: tuple[EventType, ...] | None = None,
        output_format: Literal["rich", "json"] = "rich",
    ) -> None:
        self.subscribe: tuple[EventType, ...] = subscribe if subscribe is not None else _ALL_EVENTS
        self.output_format = output_format
        self.console = Console()

    def attach(self, consciousness: object) -> None:
        """Register handlers on a Consciousness instance for all subscribed event types."""
        mapping = {
            "thought": "on_thought",
            "reflection": "on_reflection",
            "memory": "on_memory_stored",
            "identity_shift": "on_identity_shift",
        }
        for event_type in self.subscribe:
            attr = mapping.get(event_type)
            if attr and hasattr(consciousness, attr):
                getattr(consciousness, attr).append(self.handle)

    async def handle(self, payload: dict[str, object]) -> None:
        event_type = str(payload.get("type", ""))
        if self.subscribe and event_type not in self.subscribe:
            return
        if self.output_format == "json":
            self.console.print(json.dumps(payload))
        else:
            self.console.print(f"[dim]{event_type}[/dim]: {payload.get('content')}")
