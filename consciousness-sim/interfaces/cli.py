"""Rich terminal interface for live consciousness observation and control."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from core.consciousness import Consciousness


class ConsciousnessCLI:
    """Live dashboard with thought stream, mood, and control shortcuts."""

    def __init__(self, consciousness: Consciousness) -> None:
        self.consciousness = consciousness
        self.console = Console()
        self.thoughts: list[str] = []
        self.memories: list[str] = []
        self.last_reflection = "never"
        self.started_at = datetime.now(timezone.utc)

        self.consciousness.on_thought.append(self._on_thought)
        self.consciousness.on_memory_stored.append(self._on_memory)
        self.consciousness.on_reflection.append(self._on_reflection)

    async def _on_thought(self, payload: dict[str, object]) -> None:
        self.thoughts.append(str(payload.get("content", "")))
        self.thoughts = self.thoughts[-20:]

    async def _on_memory(self, payload: dict[str, object]) -> None:
        self.memories.append(str(payload.get("content", "")))
        self.memories = self.memories[-8:]

    async def _on_reflection(self, payload: dict[str, object]) -> None:
        self.last_reflection = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.thoughts.append(f"[reflection] {payload.get('content', '')}")

    def _layout(self) -> Table:
        table = Table.grid(expand=True)
        table.add_column(ratio=3)
        table.add_column(ratio=1)

        mood_panel = Panel(
            "\n".join(f"{k}: {v:.2f}" for k, v in self.consciousness.identity.mood.items()),
            title=f"{self.consciousness.name} mood",
        )
        stream_panel = Panel("\n\n".join(self.thoughts[-10:]) or "Waiting for first thought...", title="Stream")
        memory_panel = Panel("\n".join(self.memories[-6:]) or "No memory events yet", title="Memory")

        uptime = datetime.now(timezone.utc) - self.started_at
        import logging
        log_level = logging.getLevelName(logging.root.level)
        status = (
            f"Uptime: {uptime}\n"
            f"Thoughts: {self.consciousness.thought_count}\n"
            f"Memory count: {len(self.memories)}\n"
            f"Last reflection: {self.last_reflection}\n"
            f"Log level: {log_level}\n"
            "Controls: r=reflect now, j=journal preview, q=quit"
        )
        status_panel = Panel(status, title="Status")

        left = Table.grid(expand=True)
        left.add_row(stream_panel)
        left.add_row(memory_panel)

        right = Table.grid(expand=True)
        right.add_row(mood_panel)
        right.add_row(status_panel)

        table.add_row(left, right)
        return table

    async def _keyboard_loop(self) -> None:
        while True:
            line = await asyncio.to_thread(input, "")
            cmd = line.strip().lower()
            if cmd == "r":
                await self.consciousness.request_reflection()
            elif cmd == "j":
                events = await self.consciousness.journal.recent(limit=5)
                self.console.print(Panel("\n".join(f"{e['timestamp']} {e['type']}: {e['content']}" for e in events), title="Journal"))
            elif cmd == "q":
                self.consciousness._stop_event.set()
                return

    async def run(self) -> None:
        thinker = asyncio.create_task(self.consciousness.run())
        keys = asyncio.create_task(self._keyboard_loop())
        try:
            with Live(self._layout(), console=self.console, refresh_per_second=4) as live:
                while not self.consciousness._stop_event.is_set():
                    live.update(self._layout())
                    await asyncio.sleep(0.25)
        finally:
            self.consciousness._stop_event.set()
            await asyncio.gather(thinker, keys, return_exceptions=True)
