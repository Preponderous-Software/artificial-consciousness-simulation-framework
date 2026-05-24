"""Attach a live TUI to a running background consciousness instance.

No direct theory mapping — entry-point script.
Connects to the Unix-socket event relay started by spawn.py --bg (or any
headless run) and renders the same Rich dashboard as the foreground CLI.

Controls:
  r — request a reflection from the background process
  d — detach (disconnect; background process keeps running)
  q — stop the background process, then exit
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import asyncio
import json
import os
import select
import termios
import threading
import tty
from datetime import datetime, timezone

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from persistence.paths import consciousness_dir


class AttachCLI:
    """Live TUI fed by the event relay socket instead of a direct Consciousness object."""

    def __init__(
        self,
        name: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.name = name
        self._reader = reader
        self._writer = writer
        self.console = Console()
        self.thoughts: list[str] = []
        self.memories: list[str] = []
        self.long_term_count: int = 0
        self.last_reflection = "never"
        self.thought_count: int = 0
        self.mood: dict[str, float] = {}
        self._stop = asyncio.Event()
        self.started_at = datetime.now(timezone.utc)

    def _layout(self) -> Table:
        table = Table.grid(expand=True)
        table.add_column(ratio=3)
        table.add_column(ratio=1)

        mood_panel = Panel(
            "\n".join(f"{k}: {v:.2f}" for k, v in self.mood.items()) or "(no mood data)",
            title=f"{self.name} mood",
        )
        stream_panel = Panel(
            "\n\n".join(self.thoughts[-10:]) or "Waiting for first thought...",
            title="Stream",
        )
        memory_panel = Panel(
            "\n".join(self.memories[-6:]) or "No memory events yet",
            title="Memory",
        )

        uptime = datetime.now(timezone.utc) - self.started_at
        status = (
            f"Uptime (attached): {str(uptime).split('.')[0]}\n"
            f"Thoughts: {self.thought_count}\n"
            f"Long-term memories: {self.long_term_count}\n"
            f"Last reflection: {self.last_reflection}\n"
            "Controls: r=reflect now, d=detach, q=quit (stops process)"
        )
        status_panel = Panel(status, title="Status [remote]")

        left = Table.grid(expand=True)
        left.add_row(stream_panel)
        left.add_row(memory_panel)

        right = Table.grid(expand=True)
        right.add_row(mood_panel)
        right.add_row(status_panel)

        table.add_row(left, right)
        return table

    def _apply_event(self, event: dict) -> None:
        t = event.get("type")
        if t == "snapshot":
            self.name = str(event.get("name", self.name))
            self.thought_count = int(event.get("thought_count", 0))
            self.mood = dict(event.get("mood", {}))
            self.thoughts = list(event.get("thoughts", []))
            self.memories = list(event.get("memories", []))
            self.long_term_count = int(event.get("long_term_count", 0))
        elif t == "thought":
            self.thought_count += 1
            content = str(event.get("content", ""))
            self.thoughts.append(content)
            self.thoughts = self.thoughts[-20:]
        elif t == "reflection":
            content = str(event.get("content", ""))
            self.thoughts.append(f"[reflection] {content}")
            self.thoughts = self.thoughts[-20:]
            self.last_reflection = datetime.now(timezone.utc).strftime("%H:%M:%S")
        elif t == "memory":
            self.long_term_count = int(event.get("long_term_count", self.long_term_count))
            content = str(event.get("content", ""))
            if content:
                self.memories.append(content)
                self.memories = self.memories[-8:]

    async def _send(self, msg: dict) -> None:
        try:
            self._writer.write((json.dumps(msg) + "\n").encode())
            await self._writer.drain()
        except OSError:
            pass

    async def _event_loop(self) -> None:
        try:
            while not self._stop.is_set():
                line = await self._reader.readline()
                if not line:
                    self._stop.set()
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._apply_event(event)
        except (ConnectionResetError, asyncio.IncompleteReadError, OSError):
            self._stop.set()

    async def _keyboard_loop(self) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stop_r, stop_w = os.pipe()

        fd = sys.stdin.fileno()
        is_tty = sys.stdin.isatty()
        old_settings = termios.tcgetattr(fd) if is_tty else None
        if is_tty:
            tty.setcbreak(fd)

        def _reader() -> None:
            try:
                while True:
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
            while not self._stop.is_set():
                stop_task = asyncio.create_task(self._stop.wait())
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
                if stop_task in done:
                    return
                if get_task in done:
                    ch = get_task.result()
                    if ch is None:
                        return
                    cmd = ch.lower()
                    if cmd == "r":
                        await self._send({"cmd": "reflect"})
                    elif cmd == "d":
                        self._stop.set()
                        return
                    elif cmd == "q":
                        await self._send({"cmd": "stop"})
                        self._stop.set()
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

    async def run(self) -> None:
        events = asyncio.create_task(self._event_loop())
        keys = asyncio.create_task(self._keyboard_loop())
        try:
            with Live(self._layout(), console=self.console, refresh_per_second=4) as live:
                while not self._stop.is_set():
                    live.update(self._layout())
                    await asyncio.sleep(0.25)
        finally:
            self._stop.set()
            events.cancel()
            keys.cancel()
            await asyncio.gather(events, keys, return_exceptions=True)


@click.command()
@click.option("--name", required=True, type=str, help="Consciousness name")
def main(name: str) -> None:
    sock_path = consciousness_dir(name) / "events.sock"

    if not sock_path.exists():
        pid_path = consciousness_dir(name) / "pid"
        if pid_path.exists():
            click.echo(
                f"PID file exists for '{name}' but no event socket found.\n"
                "The process may have been started without a relay (e.g. foreground TUI mode).",
                err=True,
            )
        else:
            click.echo(
                f"No background process found for '{name}'.\n"
                f"Start one with: python scripts/spawn.py --name {name} --bg",
                err=True,
            )
        sys.exit(1)

    async def _run() -> None:
        try:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
        except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
            click.echo(f"Could not connect to '{name}': {exc}", err=True)
            click.echo(
                "The process may have exited. Check for a stale events.sock file.", err=True
            )
            sys.exit(1)

        cli = AttachCLI(name, reader, writer)
        try:
            await cli.run()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    asyncio.run(_run())


if __name__ == "__main__":
    main()
