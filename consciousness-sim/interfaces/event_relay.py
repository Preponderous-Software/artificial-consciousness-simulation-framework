"""Unix-socket event relay enabling detach/attach for headless instances.

No direct theory mapping — infrastructure module.
Registers as an observer on a running Consciousness and relays all events
to connected clients over a Unix domain socket. This allows a separate
terminal (attach.py) to display a live TUI without sharing a process.

Socket location: <CONSCIOUSNESS_HOME>/<name>/events.sock

Protocol (newline-delimited JSON, UTF-8):
  Server → client on connect:
    {"type": "snapshot", "name": str, "thought_count": int,
     "mood": {...}, "thoughts": [...], "memories": [...], "long_term_count": int}
  Server → client, ongoing:
    {"type": "thought"|"reflection"|"memory"|"perception", "content": str, ...}
  Client → server (optional):
    {"cmd": "reflect"}   — trigger a shallow reflection
    {"cmd": "stop"}      — request graceful shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EventRelay:
    def __init__(self, consciousness: object, sock_path: Path) -> None:
        self._mind = consciousness  # type: ignore[assignment]
        self._sock_path = sock_path
        self._clients: list[asyncio.StreamWriter] = []
        self._recent_thoughts: list[str] = []
        self._recent_memories: list[str] = []

        self._mind.on_thought.append(self._on_thought)
        self._mind.on_reflection.append(self._on_reflection)
        self._mind.on_memory_stored.append(self._on_memory_stored)
        self._mind.on_perception.append(self._on_perception)

    async def _on_thought(self, payload: dict) -> None:
        content = str(payload.get("content", ""))
        self._recent_thoughts.append(content)
        self._recent_thoughts = self._recent_thoughts[-20:]
        await self._broadcast(payload)

    async def _on_reflection(self, payload: dict) -> None:
        content = str(payload.get("content", ""))
        self._recent_thoughts.append(f"[reflection] {content}")
        self._recent_thoughts = self._recent_thoughts[-20:]
        await self._broadcast(payload)

    async def _on_memory_stored(self, payload: dict) -> None:
        content = str(payload.get("content", ""))
        if content:
            self._recent_memories.append(content)
            self._recent_memories = self._recent_memories[-8:]
        await self._broadcast(payload)

    async def _on_perception(self, payload: dict) -> None:
        await self._broadcast(payload)

    async def _broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        line = (json.dumps(payload) + "\n").encode()
        dead = []
        for writer in list(self._clients):
            try:
                writer.write(line)
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError, OSError):
                dead.append(writer)
        for w in dead:
            if w in self._clients:
                self._clients.remove(w)

    async def _snapshot(self) -> dict:
        lt_count = await self._mind.long_term.count()
        return {
            "type": "snapshot",
            "name": self._mind.name,
            "thought_count": self._mind.thought_count,
            "mood": dict(self._mind.identity.mood),
            "thoughts": list(self._recent_thoughts),
            "memories": list(self._recent_memories),
            "long_term_count": lt_count,
        }

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername", "unknown")
        logger.debug("Relay: client connected (%s)", peer)
        self._clients.append(writer)
        try:
            snap = await self._snapshot()
            writer.write((json.dumps(snap) + "\n").encode())
            await writer.drain()

            while True:
                try:
                    line = await reader.readline()
                except (asyncio.IncompleteReadError, OSError):
                    break
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cmd = msg.get("cmd")
                if cmd == "reflect":
                    asyncio.create_task(self._mind.request_reflection())
                elif cmd == "stop":
                    self._mind._stop_event.set()
                    break
        except (ConnectionResetError, OSError):
            pass
        finally:
            if writer in self._clients:
                self._clients.remove(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass
            logger.debug("Relay: client disconnected (%s)", peer)

    async def serve(self) -> None:
        """Serve until the consciousness stop event fires."""
        self._sock_path.unlink(missing_ok=True)
        server = await asyncio.start_unix_server(
            self._handle_client, path=str(self._sock_path)
        )
        logger.info("Event relay listening on %s", self._sock_path)
        try:
            async with server:
                await self._mind._stop_event.wait()
        finally:
            self._sock_path.unlink(missing_ok=True)
            logger.info("Event relay stopped")
