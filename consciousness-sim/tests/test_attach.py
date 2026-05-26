"""Tests for scripts/attach.py — the live TUI client for --bg instances (#91).

Covers the AttachCLI's _apply_event state machine (its main logic surface)
plus end-to-end Unix-socket exchange with a stand-in server.

The Rich Live rendering loop itself is not tested here (per the issue's
non-goals); instead we exercise the bits that determine what the renderer
would receive.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

# Make scripts/ importable as in spawn.py / attach.py.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.attach import AttachCLI, main as attach_main  # noqa: E402


# ---------------------------------------------------------------------------
# AttachCLI._apply_event — pure unit tests, no socket required
# ---------------------------------------------------------------------------


def _make_cli() -> AttachCLI:
    # reader/writer never used by _apply_event — pass None and ignore the type.
    return AttachCLI(name="Aria", reader=None, writer=None)  # type: ignore[arg-type]


def test_apply_event_snapshot_populates_initial_state() -> None:
    cli = _make_cli()
    cli._apply_event(
        {
            "type": "snapshot",
            "name": "Aria",
            "thought_count": 3,
            "mood": {"curiosity": 0.7, "wonder": 0.6},
            "thoughts": ["first", "second"],
            "memories": ["mem1"],
            "long_term_count": 5,
        }
    )
    assert cli.thought_count == 3
    assert cli.mood == {"curiosity": 0.7, "wonder": 0.6}
    assert cli.thoughts == ["first", "second"]
    assert cli.memories == ["mem1"]
    assert cli.long_term_count == 5


def test_apply_event_thought_increments_count_and_appends() -> None:
    cli = _make_cli()
    cli.thought_count = 4
    cli._apply_event({"type": "thought", "content": "new thought"})
    assert cli.thought_count == 5
    assert cli.thoughts[-1] == "new thought"


def test_apply_event_thoughts_capped_at_20() -> None:
    cli = _make_cli()
    for i in range(25):
        cli._apply_event({"type": "thought", "content": f"t{i}"})
    assert len(cli.thoughts) == 20
    assert cli.thoughts[0] == "t5"
    assert cli.thoughts[-1] == "t24"


def test_apply_event_reflection_prefixes_and_updates_timestamp() -> None:
    cli = _make_cli()
    before = cli.last_reflection
    cli._apply_event({"type": "reflection", "content": "I am uncertain"})
    assert any("[reflection] I am uncertain" in t for t in cli.thoughts)
    assert cli.last_reflection != before  # timestamp updated


def test_apply_event_memory_updates_long_term_count() -> None:
    cli = _make_cli()
    cli._apply_event(
        {"type": "memory", "content": "consolidated X", "long_term_count": 11}
    )
    assert cli.long_term_count == 11
    assert cli.memories[-1] == "consolidated X"


def test_apply_event_memory_without_content_still_updates_count() -> None:
    cli = _make_cli()
    cli._apply_event({"type": "memory", "long_term_count": 8})
    assert cli.long_term_count == 8
    assert cli.memories == []  # no content appended


def test_apply_event_unknown_type_does_not_raise() -> None:
    cli = _make_cli()
    # Should silently ignore unknown types.
    cli._apply_event({"type": "unknown", "content": "anything"})
    assert cli.thoughts == []
    assert cli.memories == []


# ---------------------------------------------------------------------------
# End-to-end: in-process Unix-socket server + AttachCLI._event_loop / _send
# ---------------------------------------------------------------------------


async def _fake_relay_server(
    sock_path: Path,
    snapshot: dict,
    *,
    received: list[dict],
    ready_event: asyncio.Event,
    follow_up_events: list[dict] | None = None,
) -> asyncio.AbstractServer:
    """Spin up a Unix-socket server that mimics EventRelay.

    Sends ``snapshot`` then any ``follow_up_events`` on connect, and records
    every client→server message in ``received``.
    """

    async def _handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        writer.write((json.dumps(snapshot) + "\n").encode())
        await writer.drain()
        for ev in follow_up_events or []:
            writer.write((json.dumps(ev) + "\n").encode())
            await writer.drain()
        ready_event.set()
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    received.append(json.loads(line))
                except json.JSONDecodeError:
                    received.append({"_raw": line.decode(errors="replace")})
        except (ConnectionResetError, OSError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    return await asyncio.start_unix_server(_handle, path=str(sock_path))


def test_attach_cli_event_loop_consumes_snapshot_and_thought(tmp_path: Path) -> None:
    """AttachCLI._event_loop reads the snapshot and any follow-up events,
    feeding them into _apply_event."""
    sock_path = tmp_path / "events.sock"

    async def _run() -> None:
        received: list[dict] = []
        ready = asyncio.Event()
        snapshot = {
            "type": "snapshot",
            "name": "Aria",
            "thought_count": 0,
            "mood": {"curiosity": 0.5},
            "thoughts": [],
            "memories": [],
            "long_term_count": 0,
        }
        follow_ups = [{"type": "thought", "content": "live thought"}]
        server = await _fake_relay_server(
            sock_path, snapshot, received=received, ready_event=ready,
            follow_up_events=follow_ups,
        )
        try:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            cli = AttachCLI("Aria", reader, writer)
            event_task = asyncio.create_task(cli._event_loop())
            await asyncio.wait_for(ready.wait(), timeout=2.0)
            # Give the event loop a moment to consume both messages.
            for _ in range(50):
                if cli.thoughts:
                    break
                await asyncio.sleep(0.02)
            assert cli.thoughts == ["live thought"]
            assert cli.mood == {"curiosity": 0.5}
            cli._stop.set()
            writer.close()
            await writer.wait_closed()
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_run())


def test_attach_cli_send_forwards_command_to_server(tmp_path: Path) -> None:
    """AttachCLI._send writes its JSON payload onto the socket as a newline-delimited line."""
    sock_path = tmp_path / "events.sock"

    async def _run() -> None:
        received: list[dict] = []
        ready = asyncio.Event()
        snapshot = {
            "type": "snapshot",
            "name": "Aria",
            "thought_count": 0,
            "mood": {},
            "thoughts": [],
            "memories": [],
            "long_term_count": 0,
        }
        server = await _fake_relay_server(
            sock_path, snapshot, received=received, ready_event=ready
        )
        try:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            cli = AttachCLI("Aria", reader, writer)
            await asyncio.wait_for(ready.wait(), timeout=2.0)
            await cli._send({"cmd": "reflect"})
            # Close the writer so the server's readline loop sees EOF and
            # the received list is finalised.
            writer.close()
            await writer.wait_closed()
            # Give the server a moment to flush.
            for _ in range(50):
                if received:
                    break
                await asyncio.sleep(0.02)
            assert {"cmd": "reflect"} in received
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_run())


def test_attach_cli_event_loop_handles_server_disconnect(tmp_path: Path) -> None:
    """When the server hangs up, _event_loop must set the stop event and exit cleanly."""
    sock_path = tmp_path / "events.sock"

    async def _handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # Send snapshot then close — simulates a server-side disconnect.
        snapshot = {
            "type": "snapshot",
            "name": "Aria",
            "thought_count": 0,
            "mood": {},
            "thoughts": [],
            "memories": [],
            "long_term_count": 0,
        }
        writer.write((json.dumps(snapshot) + "\n").encode())
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass

    async def _run() -> None:
        server = await asyncio.start_unix_server(_handle, path=str(sock_path))
        try:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            cli = AttachCLI("Aria", reader, writer)
            event_task = asyncio.create_task(cli._event_loop())
            # Server closed after snapshot — client should observe EOF and exit.
            await asyncio.wait_for(event_task, timeout=2.0)
            assert cli._stop.is_set()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass
            server.close()
            await server.wait_closed()

    asyncio.run(_run())


def test_attach_cli_event_loop_skips_malformed_lines(tmp_path: Path) -> None:
    """Garbage on the inbound stream is silently skipped without crashing."""
    sock_path = tmp_path / "events.sock"

    async def _handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        writer.write(b"not json\n")
        await writer.drain()
        writer.write((json.dumps({"type": "thought", "content": "after-bad"}) + "\n").encode())
        await writer.drain()
        try:
            await asyncio.sleep(0.5)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    async def _run() -> None:
        server = await asyncio.start_unix_server(_handle, path=str(sock_path))
        try:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            cli = AttachCLI("Aria", reader, writer)
            event_task = asyncio.create_task(cli._event_loop())
            for _ in range(50):
                if cli.thoughts:
                    break
                await asyncio.sleep(0.02)
            assert cli.thoughts == ["after-bad"], (
                "valid event after a malformed line must still be applied"
            )
            cli._stop.set()
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# CLI entry point — error paths
# ---------------------------------------------------------------------------


def test_attach_main_errors_when_socket_missing(monkeypatch, tmp_path: Path) -> None:
    """When events.sock doesn't exist, the CLI must exit non-zero with a clear
    error pointing the user at scripts/spawn.py --bg."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(attach_main, ["--name", "NoSuchAgent"])
    assert result.exit_code != 0
    assert "No background process" in result.output or "no event socket" in result.output


def test_attach_main_errors_when_socket_missing_but_pid_present(
    monkeypatch, tmp_path: Path
) -> None:
    """When pid exists but socket doesn't, error wording differs to point at
    foreground-mode launches."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    agent_dir = tmp_path / "Ghost"
    agent_dir.mkdir()
    (agent_dir / "pid").write_text("99999")  # arbitrary PID
    runner = CliRunner()
    result = runner.invoke(attach_main, ["--name", "Ghost"])
    assert result.exit_code != 0
    assert "no event socket found" in result.output or "PID file exists" in result.output
