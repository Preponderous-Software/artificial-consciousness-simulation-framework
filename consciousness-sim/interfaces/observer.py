"""Passive read-only observer stream for following consciousness events."""

from __future__ import annotations

from rich.console import Console


class Observer:
    """Simple read-only stream consumer."""

    def __init__(self) -> None:
        self.console = Console()

    async def handle(self, payload: dict[str, object]) -> None:
        self.console.print(f"[dim]{payload.get('type')}[/dim]: {payload.get('content')}")
