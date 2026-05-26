"""Tests for interfaces/event_relay.py — the Unix-socket event relay (#91).

The relay was introduced in PR #59 (detach/attach for --bg instances) and
had zero dedicated test coverage prior to this file. It is on the critical
attach/detach path: any regression would silently break --bg users'
ability to inspect a running instance.

The tests below mock the Consciousness object (its on_* event channels +
long_term.count + identity.mood + name/thought_count attrs) and exercise
EventRelay end-to-end against a real Unix socket in a tmp_path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from interfaces.event_relay import EventRelay


# ---------------------------------------------------------------------------
# Fake Consciousness — just enough surface for EventRelay to subscribe + serve
# ---------------------------------------------------------------------------


class _FakeLongTerm:
    def __init__(self, count_value: int = 0) -> None:
        self._count = count_value

    async def count(self) -> int:
        return self._count


class _FakeMind:
    """Minimum surface area EventRelay reads from Consciousness."""

    def __init__(self, name: str = "Aria") -> None:
        self.name = name
        self.thought_count = 0
        self.identity = SimpleNamespace(mood={"curiosity": 0.7})
        self.long_term = _FakeLongTerm()
        self.on_thought: list = []
        self.on_reflection: list = []
        self.on_memory_stored: list = []
        self.on_perception: list = []
        self._stop_event = asyncio.Event()
        self.reflection_requests = 0

    async def request_reflection(self) -> str:
        self.reflection_requests += 1
        return "reflected"


@pytest.fixture
def mind() -> _FakeMind:
    return _FakeMind()


@pytest.fixture
def sock_path(tmp_path: Path) -> Path:
    return tmp_path / "events.sock"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _read_one_message(reader: asyncio.StreamReader) -> dict:
    """Read one newline-delimited JSON message from the stream."""
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    assert line, "expected a message but got EOF"
    return json.loads(line)


async def _start_relay_and_connect(
    mind: _FakeMind, sock_path: Path
) -> tuple[EventRelay, asyncio.Task, asyncio.StreamReader, asyncio.StreamWriter]:
    """Build a relay, start serving on sock_path, open a client connection."""
    relay = EventRelay(mind, sock_path)
    server_task = asyncio.create_task(relay.serve())
    # Wait for the socket to appear before connecting.
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    assert sock_path.exists(), "relay never created socket"
    reader, writer = await asyncio.open_unix_connection(str(sock_path))
    return relay, server_task, reader, writer


async def _shutdown(mind: _FakeMind, server_task: asyncio.Task) -> None:
    mind._stop_event.set()
    try:
        await asyncio.wait_for(server_task, timeout=2.0)
    except asyncio.TimeoutError:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_relay_subscribes_to_event_channels(mind: _FakeMind, sock_path: Path) -> None:
    """Constructor must register itself on all four mind.on_* channels."""
    EventRelay(mind, sock_path)
    assert len(mind.on_thought) == 1
    assert len(mind.on_reflection) == 1
    assert len(mind.on_memory_stored) == 1
    assert len(mind.on_perception) == 1


def test_relay_sends_snapshot_on_connect(mind: _FakeMind, sock_path: Path) -> None:
    """First message after a client connects must be the snapshot frame."""

    async def _run() -> None:
        mind.thought_count = 7
        mind.long_term._count = 12
        _, server_task, reader, writer = await _start_relay_and_connect(mind, sock_path)
        try:
            snap = await _read_one_message(reader)
            assert snap["type"] == "snapshot"
            assert snap["name"] == "Aria"
            assert snap["thought_count"] == 7
            assert snap["long_term_count"] == 12
            assert snap["mood"] == {"curiosity": 0.7}
        finally:
            writer.close()
            await writer.wait_closed()
            await _shutdown(mind, server_task)

    asyncio.run(_run())


def test_relay_broadcasts_thought_to_connected_clients(
    mind: _FakeMind, sock_path: Path
) -> None:
    """A thought event fired on mind.on_thought must reach all connected clients."""

    async def _run() -> None:
        relay, server_task, reader, writer = await _start_relay_and_connect(mind, sock_path)
        try:
            # drain snapshot
            await _read_one_message(reader)
            # Give the relay a tick to register the writer as a client.
            await asyncio.sleep(0.05)
            await mind.on_thought[0]({"type": "thought", "content": "hello"})
            msg = await _read_one_message(reader)
            assert msg["type"] == "thought"
            assert msg["content"] == "hello"
        finally:
            writer.close()
            await writer.wait_closed()
            await _shutdown(mind, server_task)

    asyncio.run(_run())


def test_relay_broadcasts_to_multiple_clients(mind: _FakeMind, sock_path: Path) -> None:
    """Two simultaneous clients each receive an event."""

    async def _run() -> None:
        relay = EventRelay(mind, sock_path)
        server_task = asyncio.create_task(relay.serve())
        for _ in range(50):
            if sock_path.exists():
                break
            await asyncio.sleep(0.02)

        ra, wa = await asyncio.open_unix_connection(str(sock_path))
        rb, wb = await asyncio.open_unix_connection(str(sock_path))
        try:
            await _read_one_message(ra)  # snapshot
            await _read_one_message(rb)
            await asyncio.sleep(0.05)
            await mind.on_thought[0]({"type": "thought", "content": "x"})
            ma = await _read_one_message(ra)
            mb = await _read_one_message(rb)
            assert ma["content"] == "x"
            assert mb["content"] == "x"
        finally:
            for w in (wa, wb):
                w.close()
                try:
                    await w.wait_closed()
                except OSError:
                    pass
            await _shutdown(mind, server_task)

    asyncio.run(_run())


def test_relay_disconnected_client_does_not_break_others(
    mind: _FakeMind, sock_path: Path
) -> None:
    """One client closing must NOT prevent the other from receiving events."""

    async def _run() -> None:
        relay = EventRelay(mind, sock_path)
        server_task = asyncio.create_task(relay.serve())
        for _ in range(50):
            if sock_path.exists():
                break
            await asyncio.sleep(0.02)

        ra, wa = await asyncio.open_unix_connection(str(sock_path))
        rb, wb = await asyncio.open_unix_connection(str(sock_path))
        try:
            await _read_one_message(ra)
            await _read_one_message(rb)
            await asyncio.sleep(0.05)
            # Hard-drop client A.
            wa.close()
            try:
                await wa.wait_closed()
            except OSError:
                pass
            await asyncio.sleep(0.05)
            # Broadcast — relay should detect A is dead and only feed B.
            await mind.on_thought[0]({"type": "thought", "content": "after-drop"})
            mb = await _read_one_message(rb)
            assert mb["content"] == "after-drop"
        finally:
            wb.close()
            try:
                await wb.wait_closed()
            except OSError:
                pass
            await _shutdown(mind, server_task)

    asyncio.run(_run())


def test_relay_handles_reflect_command(mind: _FakeMind, sock_path: Path) -> None:
    """An inbound `{cmd: reflect}` triggers mind.request_reflection."""

    async def _run() -> None:
        _, server_task, reader, writer = await _start_relay_and_connect(mind, sock_path)
        try:
            await _read_one_message(reader)
            writer.write((json.dumps({"cmd": "reflect"}) + "\n").encode())
            await writer.drain()
            # request_reflection is scheduled as a task — yield to the loop.
            for _ in range(20):
                if mind.reflection_requests >= 1:
                    break
                await asyncio.sleep(0.05)
            assert mind.reflection_requests == 1
        finally:
            writer.close()
            await writer.wait_closed()
            await _shutdown(mind, server_task)

    asyncio.run(_run())


def test_relay_handles_stop_command(mind: _FakeMind, sock_path: Path) -> None:
    """`{cmd: stop}` sets mind._stop_event, ending the serve loop."""

    async def _run() -> None:
        relay = EventRelay(mind, sock_path)
        server_task = asyncio.create_task(relay.serve())
        for _ in range(50):
            if sock_path.exists():
                break
            await asyncio.sleep(0.02)
        reader, writer = await asyncio.open_unix_connection(str(sock_path))
        try:
            await _read_one_message(reader)
            writer.write((json.dumps({"cmd": "stop"}) + "\n").encode())
            await writer.drain()
            # Server should observe _stop_event and exit serve() within ~1s.
            await asyncio.wait_for(server_task, timeout=2.0)
            assert mind._stop_event.is_set()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    asyncio.run(_run())


def test_relay_ignores_malformed_command_line(mind: _FakeMind, sock_path: Path) -> None:
    """Garbage on the inbound stream must not crash the relay."""

    async def _run() -> None:
        _, server_task, reader, writer = await _start_relay_and_connect(mind, sock_path)
        try:
            await _read_one_message(reader)
            # Send malformed JSON, then a valid reflect command. The relay
            # should skip the bad line and process the good one.
            writer.write(b"not json at all\n")
            await writer.drain()
            writer.write((json.dumps({"cmd": "reflect"}) + "\n").encode())
            await writer.drain()
            for _ in range(20):
                if mind.reflection_requests >= 1:
                    break
                await asyncio.sleep(0.05)
            assert mind.reflection_requests == 1, (
                "valid command after malformed one must still be processed"
            )
        finally:
            writer.close()
            await writer.wait_closed()
            await _shutdown(mind, server_task)

    asyncio.run(_run())


def test_relay_snapshot_includes_recent_thoughts_after_broadcasts(
    mind: _FakeMind, sock_path: Path
) -> None:
    """Late-joining clients see prior thoughts in the snapshot, not an empty buffer."""

    async def _run() -> None:
        relay, server_task, reader_a, writer_a = await _start_relay_and_connect(
            mind, sock_path
        )
        try:
            await _read_one_message(reader_a)  # snapshot
            await asyncio.sleep(0.05)
            # Broadcast some thoughts to populate the recent_thoughts buffer.
            await mind.on_thought[0]({"type": "thought", "content": "first"})
            await mind.on_thought[0]({"type": "thought", "content": "second"})
            await _read_one_message(reader_a)
            await _read_one_message(reader_a)

            # Now a late client joins — its snapshot should include the buffer.
            reader_b, writer_b = await asyncio.open_unix_connection(str(sock_path))
            snap_b = await _read_one_message(reader_b)
            assert snap_b["thoughts"] == ["first", "second"]
            writer_b.close()
            await writer_b.wait_closed()
        finally:
            writer_a.close()
            await writer_a.wait_closed()
            await _shutdown(mind, server_task)

    asyncio.run(_run())


def test_relay_cleans_up_socket_file_on_shutdown(
    mind: _FakeMind, sock_path: Path
) -> None:
    """The socket file must be removed when the relay's serve loop exits."""

    async def _run() -> None:
        _, server_task, reader, writer = await _start_relay_and_connect(mind, sock_path)
        try:
            await _read_one_message(reader)
            assert sock_path.exists()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            await _shutdown(mind, server_task)
        # Server has shut down — socket should be gone.
        assert not sock_path.exists()

    asyncio.run(_run())
