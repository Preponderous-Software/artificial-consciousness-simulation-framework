"""Rich terminal interface for live consciousness observation and control.

Theory mapping — AST-1 (attention schema) / GWT-3 (global broadcast): The
CLI is the external attention schema display — it renders mood, thought stream,
and memory events in a structured dashboard, approximating the system's
self-model being made legible to an observer. The live refresh loop mirrors
the broadcast cycle: each workspace update propagates to the display.
Gap: The dashboard is read-only for an external human, not an internal model
of attention guiding the agent's own processing (true AST requires the schema
to be causally efficacious, not just displayed).
"""

from __future__ import annotations

import asyncio
import logging
import os
import select
import sys
import termios
import threading
import tty
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from core.consciousness import Consciousness


def _as_int(value: object, default: int) -> int:
    """Narrow a loosely-typed event-payload value to an int for display.

    Event payloads are ``dict[str, object]``, so counts have to be narrowed
    before rendering. Everything ``int()`` already accepted is still accepted;
    values that would previously have raised fall back to ``default`` instead,
    because a raise here is swallowed by ``_emit()`` and would silently freeze
    the displayed counter. ``bool`` is excluded deliberately — rendering a
    long-term count of 1 because a payload carried a flag is worse than
    keeping the previous value.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except ValueError:
        return default


class ConsciousnessCLI:
    """Live dashboard with thought stream, mood, and control shortcuts."""

    def __init__(self, consciousness: Consciousness) -> None:
        self.consciousness = consciousness
        self.console = Console()
        self.thoughts: list[str] = []
        self.memories: list[str] = []
        self.long_term_count: int = 0
        self.last_reflection = "never"
        self.started_at = datetime.now(timezone.utc)
        self._detach_requested: bool = False

        self.consciousness.on_initialized.append(self._on_initialized)
        self.consciousness.on_thought.append(self._on_thought)
        self.consciousness.on_memory_stored.append(self._on_memory)
        self.consciousness.on_reflection.append(self._on_reflection)

    async def _on_initialized(self, payload: dict[str, object]) -> None:
        short_term = payload.get("short_term", [])
        if isinstance(short_term, (list, tuple)):
            for item in short_term:
                if isinstance(item, dict):
                    self.thoughts.append(str(item.get("content", "")))
        self.thoughts = self.thoughts[-20:]
        self.long_term_count = _as_int(payload.get("long_term_count", 0), 0)

    async def _on_thought(self, payload: dict[str, object]) -> None:
        self.thoughts.append(str(payload.get("content", "")))
        self.thoughts = self.thoughts[-20:]

    async def _on_memory(self, payload: dict[str, object]) -> None:
        self.long_term_count = _as_int(
            payload.get("long_term_count", self.long_term_count), self.long_term_count
        )
        content = str(payload.get("content", ""))
        if content:
            self.memories.append(content)
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
        log_level = logging.getLevelName(logging.root.level)
        status = (
            f"Uptime: {uptime}\n"
            f"Thoughts: {self.consciousness.thought_count}\n"
            f"Long-term memories: {self.long_term_count}\n"
            f"Last reflection: {self.last_reflection}\n"
            f"Log level: {log_level}\n"
            "Controls: r=reflect now, j=journal preview, d=detach, q=quit"
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
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stop_r, stop_w = os.pipe()

        fd = sys.stdin.fileno()
        is_tty = sys.stdin.isatty()
        old_settings = termios.tcgetattr(fd) if is_tty else None
        if is_tty:
            # cbreak: keys arrive immediately without Enter; signals (Ctrl+C) still work.
            tty.setcbreak(fd)

        def _reader() -> None:
            try:
                while True:
                    # Use raw fd ints — passing sys.stdin (TextIOWrapper) lets
                    # Python's read-ahead buffer hide data from select in cbreak mode.
                    ready, _, _ = select.select([fd, stop_r], [], [])
                    if stop_r in ready:
                        break
                    ch = os.read(fd, 1)
                    if not ch:
                        loop.call_soon_threadsafe(queue.put_nowait, None)
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, ch.decode(errors="replace"))
            except OSError:
                loop.call_soon_threadsafe(queue.put_nowait, None)
            finally:
                try:
                    os.close(stop_r)
                except OSError:
                    pass

        threading.Thread(target=_reader, daemon=True, name="kbd-reader").start()

        try:
            while not self.consciousness._stop_event.is_set():
                stop_task = asyncio.create_task(self.consciousness._stop_event.wait())
                get_task = asyncio.create_task(queue.get())
                done, pending = await asyncio.wait(
                    {stop_task, get_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                if stop_task in done or self.consciousness._stop_event.is_set():
                    return
                if get_task in done:
                    ch = get_task.result()
                    if ch is None:
                        logger.debug("stdin closed — keyboard loop exiting")
                        return
                    cmd = ch.lower()
                    if cmd == "r":
                        await self.consciousness.request_reflection()
                    elif cmd == "j":
                        events = await self.consciousness.journal.recent(limit=5)
                        self.console.print(Panel("\n".join(f"{e['timestamp']} {e['type']}: {e['content']}" for e in events), title="Journal"))
                    elif cmd == "d":
                        self._detach_requested = True
                        self.consciousness._stop_event.set()
                        return
                    elif cmd == "q":
                        self.consciousness._stop_event.set()
                        return
        finally:
            if is_tty and old_settings is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except termios.error:
                    pass
            try:
                os.write(stop_w, b"\x00")
                os.close(stop_w)
            except OSError:
                pass

    def _spawn_detached_background(self) -> None:
        """Re-launch as a headless background process after detach.

        State is already saved (Consciousness.run() finally block ran before
        this is called), so the new process will restore from disk.
        """
        import subprocess
        from pathlib import Path as _Path

        from persistence.paths import consciousness_dir

        name = self.consciousness.name
        agent_dir = consciousness_dir(name)
        agent_dir.mkdir(parents=True, exist_ok=True)
        log_path = agent_dir / "run.log"
        pid_path = agent_dir / "pid"

        script = str(_Path(__file__).resolve().parents[1] / "scripts" / "spawn.py")
        cfg = self.consciousness.config["llm"]
        args = [
            sys.executable,
            script,
            "--name", name,
            "--provider", str(cfg["provider"]),
            "--model", str(cfg["model"]),
            "--headless",
        ]
        with open(log_path, "a") as log_fh:
            proc = subprocess.Popen(
                args,
                stdout=log_fh,
                stderr=log_fh,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ},
            )
        pid_path.write_text(str(proc.pid))
        self.console.print(
            f"Detached — '{name}' running in background (PID {proc.pid})\n"
            f"Attach : python scripts/attach.py --name {name}\n"
            f"Stop   : python scripts/stop.py --name {name}"
        )

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
            thinker.cancel()  # interrupt any in-flight LLM call, same as Ctrl+C
            keys.cancel()
            await asyncio.gather(thinker, keys, return_exceptions=True)
            if self._detach_requested:
                self._spawn_detached_background()
